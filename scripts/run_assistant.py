"""欢乐麻将实时 AI 助手 — 事件流 + 全局归属解码管道。

捕获 → 检测/跟踪(ScreenVision) → 事件提取(EventExtractor) →
窗口解码(WindowDecoder, 软归属) → 状态投影(GameView) →
手牌建议 + 客户端显示。事件日志 data/settle_logs/game_*.jsonl
(复盘/核对用)。

用法:
    python scripts/run_assistant.py               # 实时模式
    python scripts/run_assistant.py --no-dets     # 关闭悬浮框
    python scripts/run_assistant.py --imgsz 1280  # 全精度
"""

import argparse
import json
import signal
import sys
import threading
import time
import traceback
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mahjong_ai.efficiency.discard_selector import DiscardRecommendation
from src.mahjong_ai.efficiency.ukeire_selector import recommend_by_ukeire
from src.mahjong_ai.inference.soft_observer import MeldObs, SoftObserver
from src.mahjong_ai.pipeline.decoder import EvidenceModel, WindowDecoder
from src.mahjong_ai.pipeline.events import (
    MeldFormed,
    TileAppeared,
    TileClaimed,
    event_to_dict,
)
from src.mahjong_ai.pipeline.extractor import EventExtractor
from src.mahjong_ai.pipeline.meld_resolver import resolve_opponent_meld
from src.mahjong_ai.pipeline.state import GameView, project
from src.mahjong_ai.session import Advice, InferenceResult
from src.mahjong_ai.state.snapshot import (  # 仅数据类复用(旧代码只读)
    GameSnapshot,
    Meld,
    PlayerState,
)
from src.mahjong_ai.strategy import StrategyEngine
from src.mahjong_core import Hand
from src.mahjong_core.tile import tile_display
from src.mahjong_cv.capture.win32 import Win32Capture
from src.mahjong_cv.detections import TileDet
from src.mahjong_cv.game_area import detect_game_area, map_regions_to_frame
from src.mahjong_cv.screen_vision import ScreenVision
from src.mahjong_engine import get as get_rules
from src.mahjong_engine.judges.tenpai_judge import WaitingTile
from src.mahjong_engine.rules.interface import IRuleSet
from src.mahjong_ui.client import MahjongClient
from src.mahjong_ui.det_overlay import DetOverlay

MODEL = 'data/models/screen/mahjong_screen_detector/weights/best.pt'


class InferWorker:
    """后台推理线程(与旧入口同构): 最新帧优先, 结果取最新。"""

    def __init__(self, recognize: Callable[[np.ndarray], list[TileDet]]):
        self._recognize = recognize
        self._lock = threading.Lock()
        self._pending: np.ndarray | None = None
        self._results: list[tuple[tuple[int, int], list[TileDet]]] = []
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name='infer-worker')
        self._thread.start()

    def submit(self, frame: np.ndarray) -> None:
        with self._lock:
            self._pending = frame

    def poll(self) -> tuple[tuple[int, int], list[TileDet], float] | None:
        with self._lock:
            if not self._results:
                return None
            size, dets, dt = self._results[-1]
            self._results.clear()
            return size, dets, dt

    def _run(self) -> None:
        last_err = -float('inf')
        while True:
            with self._lock:
                frame = self._pending
                self._pending = None
            if frame is None:
                time.sleep(0.005)
                continue
            t0 = time.perf_counter()
            try:
                dets = self._recognize(frame)
            except Exception:  # noqa: BLE001 — 单帧异常不杀检测线程
                now = time.time()
                if now - last_err > 5.0:  # 同一异常 5s 只记一次
                    traceback.print_exc()
                    last_err = now
                continue
            dt = time.perf_counter() - t0
            with self._lock:
                self._results.append(((frame.shape[1], frame.shape[0]),
                                      dets, dt))


def load_seed_regions() -> tuple[dict[str, tuple[float, float, float, float]],
                                 dict[str, tuple[float, float, float, float]]]:
    """框选区域(config/river_regions.json, 相对画面区域归一化)。

    返回 (牌河种子, 副露种子); 无配置 → 空 dict(空间退化为弱证据)。
    副露键做 *_meld → *_river 重映射: 配置键是 *_meld, EvidenceModel
    只认 PLAYERS 名, 不映射则副露种子全失效(全部误归 my_river)。
    """
    p = Path('config/river_regions.json')
    if not p.exists():
        return {}, {}
    data = json.loads(p.read_text(encoding='utf-8'))
    rivers = {k: tuple(v) for k, v in data.get('regions', {}).items()}
    melds = {k.removesuffix('_meld') + '_river': tuple(v)
             for k, v in data.get('meld_regions', {}).items()}
    return rivers, melds


def _should_reseed(seeded: bool, area: tuple[int, int, int, int] | None,
                   last_area: tuple[int, int, int, int] | None) -> bool:
    """从未播种 或 画面区域变化 → 需要重播种。"""
    return not seeded or area != last_area


def build_advice(view: GameView, engine: StrategyEngine, rules: IRuleSet,
                 inference: InferenceResult | None = None,
                 ) -> tuple[Advice, dict[str, list[TileDet]],
                            dict[str, list[int]]]:
    """GameView → (Advice, boxes, provisional)。

    discard: 14 张推荐 / 副露局(手牌 13-3k 形态)推荐 / ≥2 张漏检也推荐;
    waiting: 手牌 13 张或 手牌+副露展开=13 的听牌提示。
    inference: 对手推断缓存(SoftObserver) — M3 接入, 可 None。
    """
    n = len(view.my_hand)
    my_melds = view.players['my_river'].melds
    k = len(my_melds)
    meld_tiles = {t for m in my_melds for t in m.tiles}
    melds_list = [[m.tiles[0]] * 3 for m in my_melds]
    discard: tuple[int, str] | None = None
    waiting: list[int] | None = None
    # 可用张数(4 − 我手牌 − 各河 − 已亮副露)+ 对手副露内容推断
    # (唯一解 → 确定碰/吃; 与 M3 扣池同源同函数) — 纯牌效推荐的
    # available 输入; 先算, 推荐块直接用
    available = Counter(dict.fromkeys(range(34), 4))
    for t in view.my_hand:
        available[t] -= 1
    for pv in view.players.values():
        for t in pv.river:
            available[t] -= 1
        for m in pv.melds:
            for t in m.tiles:
                available[t] -= 1
    players = {}
    for p, pv in view.players.items():
        resolved = []
        for m in pv.melds:
            tiles, kind = m.tiles, m.kind
            if p != 'my_river' and len(tiles) == 1:
                combos = resolve_opponent_meld(tiles[0], available)
                if len(combos) == 1:
                    tiles = combos[0]
                    kind = 'pong' if len(set(tiles)) == 1 else 'chi'
                    for t in tiles:  # 后续副露的可用计数同步
                        available[t] -= 1
            resolved.append(Meld(kind=kind, tile=tiles[0], tiles=tiles))
        players[p] = PlayerState(
            melds=resolved,
            river=list(pv.visible_river))  # 显示用可见牌河(被碰走已排除)
    # 出牌推荐(纯牌效: 向听数优先 + 有效进张枚数 — 新算法)
    enabled = frozenset(rules.get_tile_set().enabled_tiles)
    if k == 0:
        if n == 14:
            rec = recommend_by_ukeire(list(view.my_hand), available, enabled)
            discard = (rec.tile, rec.reason)
        elif n == 13:
            wt = rules.get_waiting_tiles(Hand(view.my_hand))
            if wt:
                waiting = [w.tile for w in wt]
        elif n >= 2:
            rec = recommend_by_ukeire(list(view.my_hand), available, enabled)
            discard = (rec.tile, rec.reason + '(手牌可能漏检)')
    else:
        # 副露局(M3): 听牌优先 — 手牌 + 副露展开 = 13 且已听 →
        # 显示听牌(不吃后手牌 10 的"你已听牌"被推荐掩盖);
        # 未听才推荐: 混入(n≥13)→ 副露 tile 排除; 剔除(n<13)→
        # melds 展开合并补足评估
        expanded = list(view.my_hand) + \
            [t for m in my_melds for t in m.tiles]
        tenpai = False
        if n + 3 * k == 13:
            try:
                wt = rules.get_waiting_tiles(Hand(expanded))
                if wt:
                    waiting = [w.tile for w in wt]
                    tenpai = True
            except ValueError:
                pass
        if not tenpai:
            if n >= 13:
                try:
                    rec = recommend_by_ukeire(list(view.my_hand), available,
                                              enabled, exclude=meld_tiles)
                    discard = (rec.tile, rec.reason)
                except ValueError:
                    pass
            elif n >= 2:
                try:
                    rec = recommend_by_ukeire(list(view.my_hand), available,
                                              enabled, melds=melds_list)
                    discard = (rec.tile, rec.reason)
                except ValueError:
                    rec = recommend_by_ukeire(list(view.my_hand), available,
                                              enabled)
                    discard = (rec.tile, rec.reason)
    snap = GameSnapshot(my_hand=view.my_hand,
                        my_hand_fallback=view.my_hand_fallback,
                        players=players)
    advice = Advice(snapshot=snap)
    if discard is not None:
        advice.discard = DiscardRecommendation(tile=discard[0],
                                               reason=discard[1])
    if waiting:
        advice.waiting = [WaitingTile(t, 0) for t in waiting]
    if inference is not None:
        advice.inference = inference
    return advice, {}, dict(view.provisional)


class AssistantRunner:
    """实时循环: 捕获 → 推理 → 事件 → 解码 → 投影 → 显示/日志。"""

    def __init__(self, cap: Win32Capture,
                 recognize: Callable[[np.ndarray], list[TileDet]],
                 client: MahjongClient,
                 det_overlay: DetOverlay | None, log_dir: str) -> None:
        self._cap = cap
        self._worker = InferWorker(recognize)
        self._client = client
        self._det_overlay = det_overlay
        self._extractor = EventExtractor()
        self._river_ev = EvidenceModel()
        self._meld_ev = EvidenceModel()
        self._decoder = WindowDecoder(self._river_ev, self._meld_ev)
        rules = get_rules('huanyu')
        self._rules = rules
        self._engine = StrategyEngine(rules)
        self._river_rel, self._meld_rel = load_seed_regions()
        self._last_area: tuple[int, int, int, int] | None = None
        self._seeded = False
        self._region_ticks = 0
        self._river_px: dict[str, tuple[float, float, float, float]] | None = None
        self._meld_px: dict[str, tuple[float, float, float, float]] | None = None
        self._frame_idx = 0
        self._frozen_count = 0
        self._meld_count = 0
        self._last_err_ts = 0.0
        self._last_discard_tile: int | None = None
        # M3: 软观测推断(后台线程; 主循环只入队/读缓存)
        self._observer = SoftObserver(rules)
        self._obs_pushed: set[int] = set()
        self._obs_map: dict[int, str] = {}   # eid -> 观测时 MAP(翻案比对)
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        name = f'game_{time.strftime("%Y%m%d_%H%M%S")}.jsonl'
        self._log = (Path(log_dir) / name).open('a', encoding='utf-8')

    def tick(self) -> None:
        t0 = time.perf_counter()
        frame = self._cap.capture()
        if frame is None:
            return
        w, h = frame.shape[1], frame.shape[0]
        if self._region_ticks <= 0:
            area = detect_game_area(frame)
            # 从未播种 或 画面区域变化 → 重播种。注意: area 为 None
            # (无黑边)时也必须播种一次 — 初值 last_area=None 与
            # area=None 相等会跳过(真实对局曾因此四家高斯停在原点、
            # 全部归属我的 bug); 用独立 _seeded 旗标防此回归
            if _should_reseed(self._seeded, area, self._last_area):
                river_px = (map_regions_to_frame(self._river_rel, area, w, h)
                            if self._river_rel else None)
                meld_px = (map_regions_to_frame(self._meld_rel, area, w, h)
                           if self._meld_rel else None)
                self._river_ev.reseed(river_px, w, h)
                self._meld_ev.reseed(meld_px, w, h)
                # 提取器区域门禁 + 区域副露检测用同一份重映射区域
                self._extractor.set_regions(river_px, meld_px)
                # 画面区域变化 → 手牌行像素位置可能整体移动,
                # 手牌带重锚(否则带外拒绝把手牌全拒了)
                self._extractor.reset_hand_band()
                self._river_px = river_px
                self._meld_px = meld_px
                self._last_area = area
                self._seeded = True
            self._region_ticks = 30
        self._region_ticks -= 1
        t1 = time.perf_counter()
        result = self._worker.poll()
        t2 = time.perf_counter()
        if result is not None:
            _size, dets, rec_dt = result
            try:
                self._handle(dets)
            except Exception:  # noqa: BLE001 — 单帧异常不杀主循环
                now = time.time()
                if now - self._last_err_ts > 2.0:
                    traceback.print_exc()
                    self._last_err_ts = now
        self._worker.submit(frame)
        t3 = time.perf_counter()
        # 帧耗时诊断(每 5 帧): 截图/检测/主循环处理/观测积压 —
        # 帧率随对局退化(4.4→0.7fps)的定位数据
        if self._frame_idx % 5 == 0 and result is not None:
            self._write_log({
                'type': 'frame_time',
                'ts': time.time(),
                'frame': self._frame_idx,
                't_capture': round(t1 - t0, 3),
                't_recognize': round(rec_dt, 3),
                't_handle': round(t3 - t2, 3),
                'queue': self._observer.queue_len(),
            })

    def _handle(self, dets: list[TileDet]) -> None:
        # 新一局(空桌后发牌完成): 重建提取器与解码器 — 上一局的
        # 未冻结窗口/相位/副露池全部作废; 证据模型保留(桌面布局
        # 跨局不变, 已学的高斯继续有效)。冻结记录在日志里永久。
        if self._extractor.consume_new_game():
            self._extractor = EventExtractor()
            if self._river_px is not None or self._meld_px is not None:
                self._extractor.set_regions(self._river_px, self._meld_px)
            self._decoder = WindowDecoder(self._river_ev, self._meld_ev)
            self._frozen_count = 0
            self._meld_count = 0
            self._obs_pushed = set()
            self._obs_map = {}
            self._write_log({'type': 'new_game', 'ts': time.time()})
            # M3 开局: 牌池 = 总数 − 我手牌 − 明牌(开局牌河为空)
            self._observer.reset(
                self._extractor.my_hand(),
                {p: [] for p in ('my_river', 'right_river', 'top_river',
                                 'left_river')},
                {p: [] for p in ('my_river', 'right_river', 'top_river',
                                 'left_river')},
                {'right_river': 13, 'top_river': 13, 'left_river': 13})
        frame, ts = self._frame_idx, time.time()
        self._frame_idx += 1
        new_events = self._extractor.process(dets, frame, ts)
        claims: dict[int, int | None] = {}  # meld eid -> 被碰事件 eid
        for ev in new_events:
            self._write_log(event_to_dict(ev))
            self._decoder.add(ev)
            if isinstance(ev, TileClaimed):
                claims[ev.meld] = ev.claimed
        # 时间冻结: 超过阈值的事件即使窗口未满也冻结 —
        # 显示滞后从 ~1-2 分钟压到 ~15s; 对局结束末 k 张也能冻上
        self._decoder.age_freeze(time.time())
        frozen = self._decoder.frozen()
        attribs = {a.eid: a for a in self._decoder.attributions()}
        for at in frozen[self._frozen_count:]:
            self._write_log({'type': 'freeze', 'eid': at.eid,
                             'tile': at.tile, 'player': at.player,
                             'entropy': round(at.entropy(), 3),
                             'logits': {p: round(v, 2)
                                        for p, v in at.logits.items()},
                             'ts': time.time()})
            # 翻案: 冻结 MAP ≠ 观测时 MAP → 快照回放修正
            if (at.eid in self._obs_map
                    and at.player != self._obs_map[at.eid]):
                self._observer.correct(at.eid, at.probs())
        self._frozen_count = len(frozen)
        # M3 观测推送(事件到达即观察, 当前归一化分布)
        for ev in new_events:
            if isinstance(ev, TileAppeared) and ev.eid not in self._obs_pushed:
                obs_at = attribs.get(ev.eid)
                if obs_at is not None:
                    self._observer.push_observation(ev.eid, ev.tile,
                                                    obs_at.probs())
                    self._obs_map[ev.eid] = obs_at.map()
                    self._obs_pushed.add(ev.eid)
        # M3 副露扣池
        for ev in new_events:
            if isinstance(ev, MeldFormed):
                claimed = claims.get(ev.eid)
                claimed_tile = None
                if claimed is not None and claimed in attribs:
                    claimed_tile = attribs[claimed].tile
                self._observer.push_meld(MeldObs(
                    meld_id=ev.eid, tiles=ev.tiles,
                    claimed_tile=claimed_tile,
                    melder=self._decoder.meld_player(ev.eid)))
        # 快照剪枝(冻结事件出窗)
        self._observer.prune({at.eid for at in frozen})
        melds = self._decoder.melds()
        for player, ev in melds[self._meld_count:]:
            self._write_log({'type': 'meld_assign', 'eid': ev.eid,
                             'player': player, 'ts': time.time()})
        self._meld_count = len(melds)
        view = project(self._decoder, self._extractor.my_hand(),
                       self._extractor.fallback_hand(time.time()))
        inference = self._observer.cached_inference()
        advice, _boxes, provisional = build_advice(
            view, self._engine, self._rules, inference=inference)
        self._update_risk(advice)
        boxes = {'hand': self._extractor.hand_dets()}
        self._client.update(advice, boxes, provisional=provisional)
        if self._det_overlay is not None:
            self._det_overlay.set_dets(boxes)
        self._log_inference(advice)
        self._log_hand_diag()

    def _log_hand_diag(self) -> None:
        """手牌诊断(每 5 帧): 聚类→剔除→带判定→逐张 conf — 塌陷根因。"""
        if self._frame_idx % 5 != 0:
            return
        self._write_log({'type': 'hand_diag', 'ts': time.time(),
                         'frame': self._frame_idx,
                         **self._extractor.hand_diag()})

    def _update_risk(self, advice: Advice) -> None:
        """候选牌放铳率(推荐 + 备选, 三家最大)+ 弃和提示。

        节流: 每 3 帧重算; 推荐牌变化时立即重算(复用旧 session 思路)。
        """
        if advice.discard is None:
            return
        if advice.discard.tile == self._last_discard_tile:
            if self._frame_idx % 3 != 0:
                return
        self._last_discard_tile = advice.discard.tile
        cands = [advice.discard.tile] + \
            [t for t, _ in advice.discard.alternatives[:5]]
        # 批量快照: 单次粒子快照算全部候选 — 逐牌实时读会在后台
        # 线程更新粒子时抖动(客户端放铳行闪烁, 真实对局反馈)
        risks = self._observer.discard_risks(list(dict.fromkeys(cands)))
        if advice.inference is None:
            advice.inference = InferenceResult()
        advice.inference.discard_risk = risks
        advice.inference.risk_tiles = list(risks)
        # 诊断日志(放铳率每次实际计算都留痕 — 核对 risks 空不空)
        self._write_log({'type': 'risk', 'ts': time.time(),
                         'tile': advice.discard.tile,
                         'risks': {str(t): round(r, 3)
                                   for t, r in risks.items()}})
        risk = risks.get(advice.discard.tile, 0.0)
        if risk >= 0.15:
            safe = [t for t, r in risks.items() if r < 0.05]
            hint = f'打出 {tile_display(advice.discard.tile)} ' \
                   f'放铳率 {risk:.0%}!'
            if safe:
                hint += f' 安全候选: {" ".join(tile_display(t) for t in safe)}'
            advice.risk_warning = hint

    def _log_inference(self, advice: Advice) -> None:
        """推断快照(每 10 帧 — 结算核对/复盘用)。"""
        if advice.inference is None or self._frame_idx % 10 != 0:
            return
        self._write_log({
            'type': 'infer',
            'ts': time.time(),
            'my_hand': advice.snapshot.my_hand,
            'events_total': len(self._obs_pushed),  # settle_check 阶段曲线用
            'discard': advice.discard.tile if advice.discard else None,
            # 手牌塌陷诊断: 带(最后已知手牌行 y 范围) + 带内探针
            # (该区域所有检测, 不限 conf) — 塌陷时区分"检测不到"
            # 与"检测到但聚类/置信度丢"
            'hand_band': self._extractor.hand_band(),
            'band_dets': [[t, round(cx), round(cy), round(conf, 2)]
                          for t, cx, cy, conf in self._extractor.band_dets()],
            'tenpai': {p: round(v, 3)
                       for p, v in advice.inference.tenpai_probs.items()},
            'waiting': dict(advice.inference.waiting),
            'risks': {str(t): round(r, 3) for t, r in
                      advice.inference.discard_risk.items()},
        })

    def _write_log(self, record: dict[str, object]) -> None:
        self._log.write(json.dumps(record, ensure_ascii=False) + '\n')
        self._log.flush()


def _build_capture(args: argparse.Namespace) -> Win32Capture:
    from src.mahjong_cv.capture.win32 import (  # noqa: PLC0415
        DEFAULT_TITLE_CANDIDATES,
        list_window_titles,
    )

    candidates = (DEFAULT_TITLE_CANDIDATES if args.title == '欢乐麻将'
                  else (args.title,))
    cap = Win32Capture(args.title, candidates)
    if cap.client_rect() is None:
        print('[窗口] 未找到游戏窗口, 当前可见窗口:')
        for t in list_window_titles():
            print(f'   - {t}')
        print('  请先启动欢乐麻将, 或使用 --title 指定标题')
        sys.exit(1)
    return cap


def main() -> None:
    parser = argparse.ArgumentParser(description='欢乐麻将实时 AI 助手')
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--log-dir', default='data/settle_logs')
    parser.add_argument('--no-dets', action='store_true',
                        help='关闭游戏上的悬浮检测框')
    parser.add_argument('--imgsz', type=int, default=960)
    parser.add_argument('--title', default='欢乐麻将')
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f'模型不存在: {args.model}')
        sys.exit(1)
    pipe = ScreenVision(args.model)

    def recognize(frame: np.ndarray) -> list[TileDet]:
        return pipe.track(frame, imgsz=args.imgsz)

    from PySide6.QtCore import QTimer  # noqa: PLC0415
    from PySide6.QtWidgets import QApplication  # noqa: PLC0415

    app = QApplication(sys.argv)
    cap = _build_capture(args)
    client = MahjongClient()
    det_overlay = None if args.no_dets else DetOverlay(cap.client_rect)
    client.set_status('模型加载中...')
    # 冷启动预热: 首次推理含模型加载/CUDA 初始化, 实测可达 100s —
    # 不预热则对局首帧(或停顿后首帧)撞上冷启动, 真实日志表现为
    # 识别 P99=2.5s 长尾。预热一次后对局内每帧耗时稳定。
    warm_frame = np.zeros((824, 1409, 3), dtype=np.uint8)
    pipe.warmup(warm_frame)
    client.set_status('就绪')
    runner = AssistantRunner(cap=cap, recognize=recognize, client=client,
                      det_overlay=det_overlay, log_dir=args.log_dir)

    def _on_sigint(_signum: int, _frame: object) -> None:
        print('\n正在退出...')
        app.quit()

    signal.signal(signal.SIGINT, _on_sigint)
    timer = QTimer()
    timer.timeout.connect(runner.tick)
    timer.start(66)
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
