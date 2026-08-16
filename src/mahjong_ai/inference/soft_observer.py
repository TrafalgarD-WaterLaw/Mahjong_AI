"""软观测推断器 — 归属分布 → 粒子滤波(软更新 + 翻案快照回放)。

设计: docs/superpowers/specs/2026-08-15-m3-soft-inference-design.md

- 软更新: 观测 (obs_id, tile, dist) 按权重做部分粒子更新
  (w×n 粒子走 discard 似然重采样, 其余原样) — 贝叶斯混合
- 翻案纠正: 归属 MAP 改变 → 回退到该观测的快照, 按新分布
  确定性重放之后全部观测(RNG 状态在快照里)
- 公开牌扣池(observe_public)是确定的 — 牌已打出, 与归属无关,
  三家每张公开牌恰好扣一次
- worker=False 同步模式: 单测直接 apply_all()
"""

from __future__ import annotations

import threading
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, cast

from src.mahjong_ai.inference.opponent_inference import OpponentInference
from src.mahjong_ai.pipeline.events import PLAYERS
from src.mahjong_ai.pipeline.meld_resolver import resolve_opponent_meld
from src.mahjong_ai.session import InferenceResult

#: 单家推断状态快照(粒子表/牌池/RNG 状态/hand_count)
_InferState = tuple[list[Counter[int]], Counter[int], tuple[Any, ...], int]

#: 三家对手(自己不算)
OPPONENTS = ('right_river', 'top_river', 'left_river')
#: 软更新权重下限: 低于此值的玩家跳过(低置信不喂粒子)
_MIN_WEIGHT = 0.05
#: 十判批量重算最小间隔(秒)
_TENPAI_MIN_INTERVAL = 0.5
#: 队列软上限: 积压超过此值 → 观测降级(池记账逐条保留, 粒子更新
#: 只留每家最新)。主循环入队零阻塞, worker 消化慢时队列无界 →
#: 听牌延迟无界(审查发现)。降级把最坏延迟压到 ~_MAX_QUEUE 条观测。
_MAX_QUEUE = 32


@dataclass
class MeldObs:
    """副露观测(扣池用)。tiles 保多重度; claimed_tile = 被碰/吃走
    的那张(该张已随打出事件扣过, 不再重复扣); melder = MAP 碰家。"""

    meld_id: int
    tiles: tuple[int, ...]
    claimed_tile: int | None
    melder: str | None


@dataclass
class _Entry:
    """观测日志条目: 快照在该条应用前拍摄 — 翻案回放从这里还原。"""

    obs_id: int
    tile: int
    dist: dict[str, float]
    snapshots: dict[str, _InferState] = field(default_factory=dict)


class SoftObserver:
    """三家粒子滤波 + 软观测 + 翻案回放(后台线程; worker=False 同步)。"""

    def __init__(self, rules: Any, n_particles: int = 400, seed: int = 42,
                 worker: bool = True) -> None:
        self._rules = rules
        self._n = n_particles
        self._infers = {p: OpponentInference(rules, n_particles, seed + i)
                        for i, p in enumerate(OPPONENTS)}
        self._log: list[_Entry] = []
        self._queue: list[tuple[object, ...]] = []
        self._lock = threading.Lock()
        self._dirty = threading.Event()
        self._cache: InferenceResult | None = None
        self._last_tenpai = 0.0
        #: 每家已确认副露内容(十判/放铳展开用; 未知内容 = (X,) 按碰近似)
        self._melds: dict[str, list[tuple[int, ...]]] = {p: []
                                                         for p in OPPONENTS}
        if worker:
            self._thread = threading.Thread(target=self._loop, daemon=True,
                                            name='soft-infer')
            self._thread.start()

    # ---- 主循环接口(只入队, 零阻塞) ----

    def reset(self, my_hand: list[int], rivers: dict[str, list[int]],
              meld_tiles: dict[str, list[int]],
              hand_counts: dict[str, int]) -> None:
        """开局: 牌池 = 总数 − 我手牌 − 明牌; 粒子初始化。

        meld_tiles 应排除"被碰走的那张"(它在打出家 river 里已扣)。
        """
        with self._lock:
            self._queue.append(('reset', my_hand, rivers, meld_tiles,
                                hand_counts))
        self._dirty.set()

    def push_observation(self, obs_id: int, tile: int,
                         dist: dict[str, float]) -> None:
        with self._lock:
            self._queue.append(('obs', obs_id, tile, dict(dist)))
        self._dirty.set()

    def correct(self, obs_id: int, dist: dict[str, float]) -> None:
        """翻案: 归属分布变了 → 回退快照按新分布重放。"""
        with self._lock:
            self._queue.append(('correct', obs_id, dict(dist)))
        self._dirty.set()

    def push_meld(self, meld: MeldObs) -> None:
        with self._lock:
            self._queue.append(('meld', meld))
        self._dirty.set()

    def prune(self, frozen_eids: set[int]) -> None:
        """冻结事件出窗: 其日志条目与快照可丢。"""
        with self._lock:
            self._queue.append(('prune', set(frozen_eids)))
        self._dirty.set()

    def cached_inference(self) -> InferenceResult | None:
        with self._lock:
            return self._cache

    def queue_len(self) -> int:
        """观测队列积压(诊断: worker 消化速度)。"""
        with self._lock:
            return len(self._queue)

    def discard_risk(self, tile: int) -> float:
        """候选牌放铳率(三家最大) — 含副露展开(P0 修复后与批量一致)。"""
        return self.discard_risks([tile])[tile]

    def discard_risks(self, tiles: list[int]) -> dict[int, float]:
        """候选牌放铳率批量(三家最大, 单次粒子快照)。

        快照一次算全部候选: 帧内一致, 观测更新才变 —
        客户端放铳行闪烁的修复(逐牌实时读会抖动)。
        """
        snaps = {p: inf.particles_snapshot()
                 for p, inf in self._infers.items()}
        return {t: round(max(inf.discard_risk_from(
                             snaps[p], t, tuple(self._melds.get(p, ())))
                             for p, inf in self._infers.items()), 3)
                for t in tiles}

    def log(self) -> list[_Entry]:
        """观测日志(含翻案审计) — 复盘/调试用。"""
        return list(self._log)

    # ---- 内部(后台线程) ----

    def _loop(self) -> None:
        while True:
            self._dirty.wait()
            self._dirty.clear()
            self._drain(swallow=True)
            self._recompute_tenpai()

    def apply_all(self) -> None:
        """同步模式: 清空队列并应用(单测用; worker 模式下勿调)。"""
        self._drain(swallow=False)
        self._recompute_tenpai(force=True)

    def _drain(self, swallow: bool) -> None:
        """清空队列并逐条应用。

        积压超过 _MAX_QUEUE 时降级: 每条观测仍做公开扣池 — 牌池
        是下游记账基准(副露解析/补牌抽样), 必须逐条精确; 粒子
        更新只对每家最后一个全权重观测做(最新观测信息量最大)。
        被降级的观测不入日志 → 翻案目标缺失时静默跳过 — 有界
        延迟优先于积压下的翻案精度(翻案发生在 8s 冻结窗内, 正常
        消化下不受影响)。
        """
        with self._lock:
            items, self._queue = self._queue, []
        if len(items) <= _MAX_QUEUE:
            for item in items:
                if swallow:
                    try:
                        self._dispatch(item)
                    except Exception:  # noqa: BLE001 — 单条异常不杀线程
                        traceback.print_exc()
                else:
                    self._dispatch(item)
                # GIL 让渡: 向听/十判是纯 Python 计算, 占着 GIL 会把
                # 主循环饿死(真实对局实测被拖到 ~1.25fps — 推断日志
                # 间隔 8s)。每条观测后让 2ms, 主循环恢复响应。
                time.sleep(0.002)
            return
        last_full: dict[str, int] = {}
        for i, item in enumerate(items):
            if item[0] == 'obs':
                for p in OPPONENTS:
                    if cast(dict[str, float], item[3]).get(p, 0.0) \
                            >= _MIN_WEIGHT:
                        last_full[p] = i
        full_idx = set(last_full.values())
        for i, item in enumerate(items):
            if item[0] == 'obs' and i not in full_idx:
                # 降级: 只做公开扣池(不碰粒子似然/快照 — 瓶颈所在)
                tile = cast(int, item[2])
                for p in OPPONENTS:
                    self._infers[p].observe_public(tile)
            elif swallow:
                try:
                    self._dispatch(item)
                except Exception:  # noqa: BLE001
                    traceback.print_exc()
            else:
                self._dispatch(item)
            time.sleep(0.002)

    def _dispatch(self, item: tuple[object, ...]) -> None:
        kind = item[0]
        if kind == 'reset':
            self._apply_reset(cast(list[int], item[1]),
                              cast(dict[str, list[int]], item[2]),
                              cast(dict[str, list[int]], item[3]),
                              cast(dict[str, int], item[4]))
        elif kind == 'obs':
            self._apply_obs(cast(int, item[1]), cast(int, item[2]),
                            cast(dict[str, float], item[3]))
        elif kind == 'correct':
            self._apply_correct(cast(int, item[1]),
                                cast(dict[str, float], item[2]))
        elif kind == 'meld':
            self._apply_meld(cast(MeldObs, item[1]))
        elif kind == 'prune':
            self._apply_prune(cast(set[int], item[1]))

    def _apply_reset(self, my_hand: list[int],
                     rivers: dict[str, list[int]],
                     meld_tiles: dict[str, list[int]],
                     hand_counts: dict[str, int]) -> None:
        from src.mahjong_ai.state.snapshot import (  # noqa: PLC0415
            Meld,
            PlayerState,
        )

        # OpponentInference.reset 期望 PlayerState 形态(旧数据类复用);
        # meld_tiles 应排除"被碰走的那张"(它在打出家 river 里已扣)
        players = {}
        for p in PLAYERS:
            players[p] = PlayerState(
                river=list(rivers.get(p, [])),
                melds=[Meld(kind='pong', tile=t)
                       for t in meld_tiles.get(p, [])],
                hand_count=hand_counts.get(p, 13))
        for p in OPPONENTS:
            self._infers[p].reset(my_hand, players,
                                  hand_count=players[p].hand_count)
        self._log.clear()
        self._melds = {p: [] for p in OPPONENTS}

    def _apply_obs(self, obs_id: int, tile: int,
                   dist: dict[str, float]) -> None:
        entry = _Entry(obs_id=obs_id, tile=tile, dist=dict(dist))
        entry.snapshots = {p: self._infers[p].get_state()
                           for p in OPPONENTS}
        self._replay_one(entry, entry.dist)
        self._log.append(entry)

    def _apply_correct(self, obs_id: int,
                       new_dist: dict[str, float]) -> None:
        idx = next((i for i, e in enumerate(self._log)
                    if e.obs_id == obs_id), None)
        if idx is None:
            return  # 已剪枝: 冻结前才会翻案, 正常路径日志里都有
        entry = self._log[idx]
        for p in OPPONENTS:
            self._infers[p].set_state(entry.snapshots[p])
        for e in self._log[idx:]:
            dist = new_dist if e.obs_id == obs_id else e.dist
            self._replay_one(e, dist)
            e.dist = dict(dist)  # 审计: 记录最终使用的分布

    def _replay_one(self, entry: _Entry,
                    dist: dict[str, float]) -> None:
        """一条观测的应用(公开扣池确定 + 各家按权重部分更新)。

        打出观测一律走反向过滤(粒子 = 打出后手牌, 刚打出的 tile
        不该在其中): 听牌家打掉的多是刚摸进的牌 — 正向过滤(要求
        粒子含该牌)会把真听牌手结构性滤掉, 评估实测模型概率挤在
        0-0.2 桶而实际听牌 48%(全员反向过滤后手牌质量 0.59 → 0.96,
        waiting 命中 14% → 29%)。旧的正向路径保留在 OpponentInference
        层(observe_discard*)供单测, 管道不再使用。
        """
        for p in OPPONENTS:
            self._infers[p].observe_public(entry.tile)
            w = dist.get(p, 0.0)
            if w >= _MIN_WEIGHT:
                self._infers[p].observe_discarded_partial(entry.tile, w)

    def _apply_meld(self, m: MeldObs) -> None:
        """副露扣池 + 副露者手牌缩小(P0 修复)。

        扣池: 被碰/吃走的那张已随打出事件扣过 → 只扣其余。
        内容未知(tiles 只有被碰牌)时用牌池收窄: 碰需要池中还有
        ≥2 张被碰牌, 吃需要顺子其余两张各 ≥1 — 唯一解全扣(对手
        手牌推断的关键约束: 这三张确定离开牌池), 歧义只扣被碰牌。
        粒子缩小: 副露者粒子 → 13−3k(副露后打出的手牌), 十判时
        展开副露回 13 — 副露后听牌/放铳不再恒 0。
        """
        tiles = m.tiles
        if (len(tiles) == 1 and m.claimed_tile is not None
                and m.melder in OPPONENTS):
            pool = self._infers[OPPONENTS[0]].get_state()[1]
            combos = resolve_opponent_meld(m.claimed_tile, pool)
            if len(combos) == 1:
                tiles = combos[0]
        counts = Counter(tiles)
        if m.claimed_tile is not None and counts.get(m.claimed_tile, 0) > 0:
            counts[m.claimed_tile] -= 1
        for t, n in counts.items():
            for _ in range(max(0, n)):
                for p in OPPONENTS:
                    self._infers[p].observe_public(t)
        if m.melder in OPPONENTS:
            # 记录副露内容(十判/放铳展开用)。内容未知存 (X,) → 按碰
            # 近似展开 (X,X,X): 粒子缩到 10 张后展开只有 1 张 →
            # 11 张 ≠13 十判全跳过, 听牌概率归零(真实对局: 对家假
            # 副露 (1,) 后听牌概率没了)
            stored = tiles if len(tiles) >= 3 else (tiles[0],) * 3
            self._melds[m.melder].append(stored)
            # 确定进副露的手牌 = 内容 − 被碰牌(一张)
            hand_tiles: list[int] = []
            claimed_left = 1 if m.claimed_tile is not None else 0
            for t in tiles:
                if claimed_left and t == m.claimed_tile:
                    claimed_left -= 1
                else:
                    hand_tiles.append(t)
            inf = self._infers[m.melder]
            inf.shrink_for_meld(tuple(hand_tiles), 3)
            particles, pool, rng_state, hand_count = inf.get_state()
            inf.set_state((particles, pool, rng_state,
                           max(0, hand_count - 3)))

    def _apply_prune(self, frozen_eids: set[int]) -> None:
        self._log = [e for e in self._log if e.obs_id not in frozen_eids]

    def _recompute_tenpai(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_tenpai < _TENPAI_MIN_INTERVAL:
            return
        snap = {p: self._infers[p].particles_snapshot()
                for p in OPPONENTS}
        results = {p: self._infers[p]._tenpai_of(  # noqa: SLF001
            snap[p], tuple(self._melds.get(p, ())))
            for p in OPPONENTS}
        probs = {p: v[0] for p, v in results.items()}
        waiting = {p: [t for t, _ in sorted(v[1].items(),
                                            key=lambda kv: -kv[1])[:3]]
                   for p, v in results.items()}
        with self._lock:
            self._cache = InferenceResult(tenpai_probs=probs,
                                          waiting=waiting,
                                          discard_risk={})
            self._last_tenpai = now


__all__ = ['MeldObs', 'OPPONENTS', 'SoftObserver']
