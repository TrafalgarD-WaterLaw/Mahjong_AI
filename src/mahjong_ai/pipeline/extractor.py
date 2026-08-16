"""事件提取器 — 检测帧序列 → 观测事件(不带归属结论)。

复用 det_cluster 的聚类(手牌/副露组)做分组; 这里只做"出现/成组/
手牌变化"的确认, 不做归属(归属是解码器的事)。提取规则独立可单测。
"""

from __future__ import annotations

from collections import Counter
from statistics import mean

from src.mahjong_ai.pipeline.events import (
    Event,
    FlowerShown,
    HandChanged,
    MeldFormed,
    TileAppeared,
    TileVanished,
)
from src.mahjong_ai.pipeline.hand_tracker import HandTracker, StepSig
from src.mahjong_ai.pipeline.meld_inferrer import (
    MeldInferrer,
    VisualMeld,
    meld_valid,
)
from src.mahjong_ai.pipeline.river_watcher import RiverWatcher
from src.mahjong_cv.det_cluster import cluster_dets
from src.mahjong_cv.detections import TileDet

#: 手牌置信度门槛(与现行去抖口径一致)
HAND_CONF = 0.55
#: 手牌去抖窗口/majority(现行口径, 自包含复制 — 不 import 旧 tracker)
_DEBOUNCE_WINDOW = 5
_DEBOUNCE_MAJORITY = 3
#: 新轨迹确认: 累计出现 ≥ 此帧数且位置稳定才发 TileAppeared
_APPEAR_FRAMES = 2
#: 位置稳定: 距首现位置位移 < 此值 × 框宽
_STABLE_RATIO = 0.5
#: 新轨迹超时(帧): 迟迟不稳定的候选丢弃(误检)
_APPEAR_TTL = 30
#: 视觉副露确认帧数(合法组持续 ≥ 此帧数才交给推断器)
_VISUAL_CONFIRM = 5
#: 副露候选位置匹配距离(像素)
_MELD_MATCH_PX = 40.0
#: 手牌带外扩(带内新轨迹 = 手牌, 不是打出; 带锚定同口径)
_BAND_PAD = 10.0
#: 手牌带探针外扩(诊断: 带内 ±此值的所有检测, 不限 conf —
#: 手牌塌陷时判断是"检测不到"还是"检测到但聚类/置信度丢")
_BAND_PROBE = 50.0
#: 花牌位置去重粒度(像素/格)
_FLOWER_GRID = 20
#: 重复事件位置去重半径(像素 — 同牌重识别不重发)
_DUP_RADIUS = 30.0
#: 牌河区域门禁的外扩边距(像素 — 框边界处/第 3 行的打出不因框紧而丢;
#: 20px 不会碰到对手手牌行(上家手牌在区域上方 ~30px 外))
_REGION_MARGIN = 20.0
#: 手牌在场门槛: 手牌簇 ≥ 此张数才算"对局进行中" — 大厅/进局前
#: 屏幕误检不产事件(发牌动画前半段也在此门之下)
_HAND_PRESENT_MIN = 4
#: 副露区域外扩边距(像素 — 区域副露检测/手牌剔除用)
_MELD_REGION_MARGIN = 10.0
#: 打出缓冲/台阶缓冲保留窗口(秒)
_SIGNAL_WINDOW = 10.0
#: 手牌数变化提交迟滞(帧): 新值持续 ≥ 此帧数才发 HandChanged —
#: 手牌簇 9-14 振荡的 1-2 帧闪烁被吞掉(P1: 虚假 13→14 锚/显示抖动)
_HAND_HOLD_FRAMES = 2
#: 空桌判定帧数: 手牌持续为 0 达到此帧数 → 桌空(结算/局间) —
#: 之后手牌再达 ≥13 = 新一局(P0: 跨局状态重建信号)。
#: 30 帧(≈4s): 局间结算实测 0 持续 80+ 帧; 中局检测抽风实测
#: 2-4s(15:15 局 f1150/f1199)不能误判成空桌 — 门闩与事件保持
_EMPTY_TABLE_FRAMES = 30


class EventExtractor:
    """检测帧 → 事件。手牌去抖自包含(与旧 tracker 解耦)。"""

    def __init__(self) -> None:
        self._eid = 0
        self._hand_history: list[Counter[int]] = []
        self._hand_n = 0
        self._hand_pos: dict[int, float] = {}
        self._hand_band: tuple[float, float] | None = None
        self._last_hand: list[TileDet] = []
        self._band_dets: list[tuple[int, float, float, float]] = []
        self._tracks: dict[int, dict[str, float]] = {}
        self._seen: set[int] = set()             # 基线已见轨迹(不发事件)
        self._emitted: set[int] = set()
        self._watcher = RiverWatcher()
        self._tracker = HandTracker()
        self._inferrer = MeldInferrer(self._next_eid)
        self._visual_pool: list[dict[str, float | bool]] = []
        self._discard_buf: list[tuple[str | None, int, float, int]] = []
        self._step_buf: list[StepSig] = []
        self._dup_table: dict[int, tuple[int, float, float]] = {}
        self._meld_seen = False
        self._gate_was_open = False
        self._flower_seen: set[tuple[int, int]] = set()
        #: 每帧手牌诊断(塌陷根因定位: 聚类→剔除→带判定→逐张conf)
        self._hand_diag: dict = {}
        # 手牌数迟滞: 候选新值与连续出现帧数(持续 ≥2 帧才提交)
        self._pending_n: int | None = None
        self._pending_streak = 0
        # 新一局信号: 空桌(手牌 0 持续 ≥_EMPTY_TABLE_FRAMES)后
        # 手牌再达 ≥13 → consume_new_game() 返回 True 一次
        self._zero_streak = 0
        self._saw_empty_table = False
        self._new_game_pending = False
        self._in_game = False
        #: 牌河区域(像素, 运行器随画面区域重映射后注入) — 非空时
        #: 区域门禁生效(见 process 第 3 步)
        self._regions: dict[str, tuple[float, float, float, float]] = {}
        #: 副露区域(像素) — 区域副露检测/手牌剔除用(见 process 第 4 步)
        self._meld_regions: dict[str, tuple[float, float, float, float]] = {}

    def set_regions(
        self,
        regions: dict[str, tuple[float, float, float, float]] | None,
        meld_regions: dict[str, tuple[float, float, float, float]] | None = None,
    ) -> None:
        """注入牌河区域与副露区域(像素, 由运行器随画面区域重映射)。

        regions 非空 → 区域门禁: 新轨迹必须落在某区域内才发
        TileAppeared — 手牌数振荡(副露并簇导致)不再卡住事件;
        对手手牌/发牌残留都在区域外, 自然被过滤。
        空/None → 回退旧张数门禁(手牌≥13 或已见副露)。

        meld_regions 非空 → 区域副露检测: 聚类路径漏检时兜底
        (我的副露与手牌并簇 / 对手副露透视缩小被判成牌河)。
        """
        self._regions = dict(regions) if regions else {}
        self._meld_regions = dict(meld_regions) if meld_regions else {}
        self._watcher.set_regions(self._regions)
        self._inferrer.set_meld_regions(self._meld_regions)

    def _next_eid(self) -> int:
        """全局自增事件 id(提取器内唯一)。"""
        self._eid += 1
        return self._eid

    def consume_new_game(self) -> bool:
        """运行器查询: 检测到新一局 → 返回 True 一次(运行器据此
        重建提取器与解码器, 证据模型保留 — 桌面布局跨局不变)。"""
        if not self._new_game_pending:
            return False
        self._new_game_pending = False
        self._saw_empty_table = False
        return True

    # ---- 主入口 ----

    def process(self, dets: list[TileDet], frame: int, ts: float) -> list[Event]:
        """一帧检测 → 本帧新事件(打出/消失在前, 推断副露随后, 手牌变化最后)。"""
        events: list[Event] = []
        hand, melds, _rivers = cluster_dets(dets)
        cluster_n = len(hand)
        # 副露区域内的牌从手牌簇剔除: 我的副露与手牌并簇时, 手牌数
        # 虚高(16 = 13+3)且副露组丢失 — 区域是固定 UI 区, 比聚类可靠
        stripped: list[tuple[int, float, float]] = []
        if self._meld_regions:
            kept = []
            for d in hand:
                cx, cy = (d.x1 + d.x2) / 2, (d.y1 + d.y2) / 2
                if self._in_any_meld_region(cx, cy):
                    stripped.append((d.tile, round(cy), round(d.conf, 2)))
                else:
                    kept.append(d)
            hand = kept
        # 手牌带锚定: 带已知时, 聚类手牌簇必须落在带内(±_BAND_PAD) —
        # 真实对局回归: 手牌塌陷后, 横排牌河/副露行满足(横排 + 尺寸
        # 锚 + cy 最大)被聚类判成手牌, 显示乱成牌河牌。带 = 手牌行
        # 最后已知 y 范围, 带外一律拒绝(宁可空, 不可错); 窗口移动/
        # 新局由 reset_hand_band / 重建提取器重锚。
        band_pass = True
        if self._hand_band is not None and hand:
            cy = mean((d.y1 + d.y2) / 2 for d in hand)
            if not (self._hand_band[0] - _BAND_PAD <= cy
                    <= self._hand_band[1] + _BAND_PAD):
                band_pass = False
                hand = []
        self._last_hand = [d for d in hand
                           if d.conf >= HAND_CONF and d.tile != 34]
        low_conf = [(d.tile, round(d.conf, 2))
                    for d in hand if d.conf < HAND_CONF or d.tile == 34]
        # 诊断: 带内探针(±_BAND_PROBE 的所有检测, 不限 conf) —
        # 手牌塌陷时区分"检测不到"(探针也空)与"聚类/置信度丢"(探针有牌)
        if self._hand_band is not None:
            self._band_dets = [(d.tile, (d.x1 + d.x2) / 2, (d.y1 + d.y2) / 2,
                                d.conf)
                               for d in dets
                               if self._hand_band[0] - _BAND_PROBE
                               <= (d.y1 + d.y2) / 2
                               <= self._hand_band[1] + _BAND_PROBE]
        else:
            self._band_dets = []
        # 1) 手牌去抖 + HandChanged(最后追加: 相位锚作用于本帧之后)
        counts: Counter[int] = Counter()
        for d in self._last_hand:
            counts[d.tile] += 1
            self._hand_pos[d.tile] = (d.x1 + d.x2) / 2
        self._hand_history.append(counts)
        if len(self._hand_history) > _DEBOUNCE_WINDOW:
            self._hand_history.pop(0)
        hand_changed = None
        n = len(self._debounced_hand())
        # 每帧诊断(塌陷根因: 聚类原始数 → 区域剔除 → 带判定 →
        # 低置信过滤 → 去抖结果; 运行器写入 jsonl 供离线分析)
        self._hand_diag = {
            'cluster_n': cluster_n,
            'stripped': stripped,
            'band_pass': band_pass,
            'low_conf': low_conf,
            'hand_n': len(self._last_hand),
            'debounced_n': n,
            'hand_confs': [(d.tile, round(d.conf, 2), round((d.y1 + d.y2) / 2))
                           for d in self._last_hand],
        }
        # 迟滞提交: 新值持续 ≥_HAND_HOLD_FRAMES 帧才发 HandChanged —
        # 1-2 帧的 9↔14 振荡被吞(虚假 13→14 锚/显示抖动, P1)
        if n != self._pending_n:
            self._pending_n = n
            self._pending_streak = 1
        else:
            self._pending_streak += 1
        if n != self._hand_n and self._pending_streak >= _HAND_HOLD_FRAMES:
            hand_changed = HandChanged(eid=self._next_eid(),
                                       n_old=self._hand_n, n_new=n,
                                       frame=frame, ts=ts)
            self._hand_n = n
        # 空桌/新局信号(P0: 跨局状态重建) + 局中闩锁:
        # 手牌 ≥4 出现过 → _in_game 闩上(中局手牌塌 0 不再关事件 —
        # 真实对局: 手牌全检测不到时四家事件全停); 只有空桌信号
        # (手牌 0 持续 ≥_EMPTY_TABLE_FRAMES)才解锁(结算/局间)
        if n == 0:
            self._zero_streak += 1
            if self._zero_streak >= _EMPTY_TABLE_FRAMES:
                self._saw_empty_table = True
                self._in_game = False
        else:
            self._zero_streak = 0
            if len(self._last_hand) >= _HAND_PRESENT_MIN:
                self._in_game = True
        if n >= 13 and self._saw_empty_table:
            self._new_game_pending = True
        if self._last_hand:
            self._hand_band = (min(d.y1 for d in self._last_hand),
                               max(d.y2 for d in self._last_hand))
        # 2) 花牌 → 亮花事件(位置去重; 不进归属)
        for d in dets:
            if d.tile != 34:
                continue
            key = (round((d.x1 + d.x2) / 2 / _FLOWER_GRID),
                   round((d.y1 + d.y2) / 2 / _FLOWER_GRID))
            if key in self._flower_seen:
                continue
            self._flower_seen.add(key)
            events.append(FlowerShown(
                eid=self._next_eid(), tile=d.tile,
                cx=(d.x1 + d.x2) / 2, cy=(d.y1 + d.y2) / 2,
                frame=frame, ts=ts))
        # 3) 新轨迹 → TileAppeared(记录牌河归属 + 喂消失检测/打出缓冲)
        meld_tids = {d.track_id for m in melds for d in m
                     if d.track_id is not None}
        region_meld = [d for d in dets
                       if d.conf >= HAND_CONF
                       and d.tile != 34
                       and self._in_any_meld_region((d.x1 + d.x2) / 2,
                                                    (d.y1 + d.y2) / 2)]
        meld_tids |= {d.track_id for d in region_meld
                      if d.track_id is not None}
        gate_open = self._hand_n >= 13 or self._meld_seen
        if self._regions:
            gate_open = self._in_game
        if gate_open and not self._gate_was_open:
            hand_tids = {d.track_id for d in hand if d.track_id is not None}
            for d in dets:
                if d.track_id is not None and d.track_id not in hand_tids:
                    self._seen.add(d.track_id)
            self._gate_was_open = True
        if gate_open:
            for d in dets:
                self._check_appeared(d, frame, ts, events, meld_tids)
        # 4) 视觉副露候选(聚类组 + 区域组; ≥3 张且组合合法才进池)
        visual_confirmed: list[VisualMeld] = []
        for g in [m for m in melds if self._group_valid(m)]:
            visual_confirmed += self._visual_track(g)
        for g in self._split_groups(region_meld):
            if self._group_valid(g):
                visual_confirmed += self._visual_track(g)
        # 5) 牌河消失检测
        vanished: list[TileVanished] = []
        for v in self._watcher.tick(dets, frame, ts, self._next_eid):
            vanished.append(v)
            events.append(v)
        # 6) 手牌账本 + 台阶
        step = self._tracker.tick(self._debounced_hand(), ts)
        if step is not None:
            self._step_buf.append(step)
        while self._step_buf and ts - self._step_buf[0].ts > _SIGNAL_WINDOW:
            self._step_buf.pop(0)
        # 7) 副露推断(消费本帧全部信号)
        for ev in self._inferrer.tick(vanished, self._step_buf,
                                      self._discard_buf, visual_confirmed,
                                      self._tracker, frame, ts):
            events.append(ev)
            if isinstance(ev, MeldFormed):
                self._meld_seen = True
                if ev.player == 'my_river':
                    self._tracker.on_my_meld(list(ev.tiles[1:]),
                                             max(1, len(ev.tiles) - 1))
        if hand_changed is not None:
            events.append(hand_changed)
        return events

    # ---- 手牌 ----

    def my_hand(self) -> list[int]:
        """去抖手牌(位置序, 与检测框显示一致)。"""
        return self._debounced_hand()

    def hand_dets(self) -> list[TileDet]:
        """本帧手牌框(显示用)。"""
        return self._last_hand

    def hand_band(self) -> tuple[float, float] | None:
        """手牌行最后已知 y 范围(带锚定/诊断用)。"""
        return self._hand_band

    def band_dets(self) -> list[tuple[int, float, float, float]]:
        """带内探针检测 [(tile, cx, cy, conf)] — 诊断手牌塌陷用。"""
        return self._band_dets

    def hand_diag(self) -> dict:
        """本帧手牌诊断(聚类/剔除/带判定/去抖) — 塌陷根因定位。"""
        return self._hand_diag

    def reset_hand_band(self) -> None:
        """手牌带重锚: 窗口移动/画面区域变化后调用(带外拒绝失效)。"""
        self._hand_band = None

    def fallback_hand(self, ts: float) -> list[int | None] | None:
        """塌陷兜底手牌(未知槽 = None); 检测健康 → None(显示用)。"""
        return self._tracker.fallback_hand(ts)

    def _debounced_hand(self) -> list[int]:
        counts: Counter[int] = Counter()
        for c in self._hand_history:
            counts.update(c)
        result: list[int] = []
        for t, n in counts.items():
            if n >= _DEBOUNCE_MAJORITY:
                mult = max(c[t] for c in self._hand_history)
                result.extend([t] * mult)
        result.sort(key=lambda t: self._hand_pos.get(t, 0.0))
        return result[:14]

    # ---- 新轨迹 ----

    def _check_appeared(self, d: TileDet, frame: int, ts: float,
                        events: list[Event], meld_tids: set[int]) -> None:
        if d.track_id is None or d.tile == 34:
            return
        tid = d.track_id
        if tid in self._emitted or tid in self._seen or tid in meld_tids:
            return
        cx, cy = (d.x1 + d.x2) / 2, (d.y1 + d.y2) / 2
        t = self._tracks.get(tid)
        if t is None:
            self._tracks[tid] = {'lx': cx, 'ly': cy, 'f0': frame,
                                 'stable': 1}
            return
        # 稳定性锚定最近位置: 相邻两次出现位移 < 0.5×框宽 = 已落定。
        # 锚定首现位置的旧逻辑会把"飞行中首现"的牌判不稳定而丢弃 —
        # 区域门禁常开后大量打出因此丢失(真实对局回归: 整局 9 事件)
        w = d.x2 - d.x1
        moved = ((cx - t['lx']) ** 2 + (cy - t['ly']) ** 2) ** 0.5
        if moved < _STABLE_RATIO * w:
            t['stable'] += 1
        else:
            t['stable'] = 1  # 还在动(飞行中): 重新计数
        t['lx'], t['ly'] = cx, cy
        if (t['stable'] >= _APPEAR_FRAMES
                and not self._in_hand_band(d)
                and not self._recent_dup(d.tile, cx, cy)
                and (not self._regions
                     or self._in_any_river_region(cx, cy))):
            self._emitted.add(tid)
            self._tracks.pop(tid)
            eid = self._next_eid()
            events.append(TileAppeared(
                eid=eid, tile=d.tile, track_id=tid,
                cx=cx, cy=cy, conf=d.conf, frame=frame, ts=ts))
            reg = self._river_of(cx, cy)
            if reg is not None:
                self._watcher.on_appeared(reg, eid, d.tile, cx, cy, d.conf)
                self._discard_buf.append((reg, d.tile, ts, eid))
                if reg == 'my_river':
                    self._tracker.on_my_discard(d.tile, ts)
            while self._discard_buf and \
                    ts - self._discard_buf[0][2] > _SIGNAL_WINDOW:
                self._discard_buf.pop(0)
        elif frame - t['f0'] > _APPEAR_TTL:
            self._tracks.pop(tid)  # 迟迟不落定(误检) → 丢弃

    def _recent_dup(self, tile: int, cx: float, cy: float) -> bool:
        """同 tile 近位置已发过事件 → 重复(跟踪重识别), 不重发。"""
        for t, x, y in self._dup_table.values():
            if t == tile and abs(x - cx) < _DUP_RADIUS \
                    and abs(y - cy) < _DUP_RADIUS:
                return True
        self._dup_table[len(self._dup_table)] = (tile, cx, cy)
        return False

    def _in_hand_band(self, d: TileDet) -> bool:
        """框中心是否在手牌带内(±_BAND_PAD) — 带内框是手牌, 不是打出。"""
        if self._hand_band is None:
            return False
        cy = (d.y1 + d.y2) / 2
        return self._hand_band[0] - _BAND_PAD <= cy \
            <= self._hand_band[1] + _BAND_PAD

    def _in_any_river_region(self, cx: float, cy: float) -> bool:
        """落点是否在某牌河区域内(±_REGION_MARGIN)。

        区域门禁: 打出 = 落进牌河区域; 对手手牌/发牌残留都在区域外。
        """
        for rx1, ry1, rx2, ry2 in self._regions.values():
            if (rx1 - _REGION_MARGIN <= cx <= rx2 + _REGION_MARGIN
                    and ry1 - _REGION_MARGIN <= cy <= ry2 + _REGION_MARGIN):
                return True
        return False

    def _in_any_meld_region(self, cx: float, cy: float) -> bool:
        """落点是否在某副露区域内(±_MELD_REGION_MARGIN)。"""
        for rx1, ry1, rx2, ry2 in self._meld_regions.values():
            if (rx1 - _MELD_REGION_MARGIN <= cx <= rx2 + _MELD_REGION_MARGIN
                    and ry1 - _MELD_REGION_MARGIN <= cy
                    <= ry2 + _MELD_REGION_MARGIN):
                return True
        return False

    def _split_groups(self, dets: list[TileDet]) -> list[list[TileDet]]:
        """副露区域内的牌按紧邻分组成 2-4 张的候选组(间隙 > 2×框宽 分组)。

        同一副露区域可放多组副露(第 2/3 组紧贴第 1 组), 按 x 排序
        后间隙分组; 单张/超 4 张的组不确认(单张可能是飞行动画)。
        """
        groups: list[list[TileDet]] = []
        cur: list[TileDet] = []
        prev_cx: float | None = None
        prev_w: float = 0.0
        for d in sorted(dets, key=lambda x: (x.x1 + x.x2) / 2):
            cx = (d.x1 + d.x2) / 2
            w = d.x2 - d.x1
            if prev_cx is not None and cx - prev_cx > 2.0 * min(w, prev_w):
                groups.append(cur)
                cur = []
            cur.append(d)
            prev_cx, prev_w = cx, w
        if cur:
            groups.append(cur)
        return [g for g in groups if 2 <= len(g) <= 4]

    # ---- 副露 ----

    def _group_valid(self, g: list[TileDet]) -> bool:
        """视觉副露组合法性: 清洗花牌后 ≥3 张且组合合法。"""
        clean = tuple(sorted(d.tile for d in g if d.tile != 34))
        return len(clean) >= 3 and meld_valid(clean)

    def _visual_track(self, g: list[TileDet]) -> list[VisualMeld]:
        """视觉副露候选池: 位置匹配去重, 累计 ≥_VISUAL_CONFIRM 帧 → 确认。"""
        cx = sum((d.x1 + d.x2) / 2 for d in g) / len(g)
        cy = sum((d.y1 + d.y2) / 2 for d in g) / len(g)
        cand = next((c for c in self._visual_pool
                     if abs(float(c['cx']) - cx) < _MELD_MATCH_PX
                     and abs(float(c['cy']) - cy) < _MELD_MATCH_PX), None)
        if cand is None:
            cand = {'cx': cx, 'cy': cy, 'count': 0.0, 'done': False}
            self._visual_pool.append(cand)
        if bool(cand['done']):
            return []
        cand['count'] = float(cand['count']) + 1.0
        if cand['count'] < _VISUAL_CONFIRM:
            return []
        cand['done'] = True
        tiles = tuple(sorted(d.tile for d in g if d.tile != 34))
        bbox = (min(d.x1 for d in g), min(d.y1 for d in g),
                max(d.x2 for d in g), max(d.y2 for d in g))
        return [VisualMeld(player=self._meld_owner(cx, cy),
                           tiles=tiles, cx=cx, cy=cy, bbox=bbox)]

    def _meld_owner(self, cx: float, cy: float) -> str | None:
        """组中心所在副露区域的归属(PLAYERS 名); 不在任何区域 → None。"""
        for name, (rx1, ry1, rx2, ry2) in self._meld_regions.items():
            if (rx1 - _MELD_REGION_MARGIN <= cx <= rx2 + _MELD_REGION_MARGIN
                    and ry1 - _MELD_REGION_MARGIN <= cy
                    <= ry2 + _MELD_REGION_MARGIN):
                return name
        return None

    def _river_of(self, cx: float, cy: float) -> str | None:
        """落点所在牌河区域(PLAYERS 名); 不在任何区域 → None。"""
        for name, (rx1, ry1, rx2, ry2) in self._regions.items():
            if (rx1 - _REGION_MARGIN <= cx <= rx2 + _REGION_MARGIN
                    and ry1 - _REGION_MARGIN <= cy <= ry2 + _REGION_MARGIN):
                return name
        return None
