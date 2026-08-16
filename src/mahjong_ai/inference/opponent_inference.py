"""对家手牌/听牌/放铳率推断 — 粒子滤波。

输入: PlayerState(每家的手牌数/副露/出牌序列) + 自己的手牌(已知牌)
输出: 手牌分布 / 听牌概率与听什么 / 放铳率(我打 X 被胡的概率)

粒子 = 该家手牌的一个具体组合(Counter), 从剩余牌池采样。
出牌观测两条路径(SoftObserver 一律走反向):
  - observe_discard*(正向): 过滤必含该牌 → 似然加权 → 重采样 → 删牌
    补牌 — 建模"打掉手牌里的旧牌, 摸进的新牌留下"(保留作单测 API)
  - observe_discarded*(反向): 过滤不含该牌(粒子 = 打出后手牌) →
    似然(从粒子+该牌的打出前手牌) → 重采样 — 建模"打掉刚摸进的牌",
    听牌家的主流行为。评估证明反向过滤校准正确(见 M3 spec §4)。
所有推断基于已确认状态(PlayerState), 不读瞬时检测。

策略观测模型(第二版, 向听数牌效):
  打出 X 的似然 = 孤张因子 × exp(-α × Δ)
    Δ = shanten(打出后) - shanten(打出前)
      Δ > 0 → X 对向听结构有贡献, 理性玩家不打(毁牌效) → 低权重
      Δ = 0 → 孤张/多余 → 正常权重
    α 控制惩罚强度(玩家不完全理性: 防守拆牌/失误仍可能打 Δ>0 的牌)
"""

from __future__ import annotations

import math
import random
import time
from collections import Counter
from functools import lru_cache
from typing import Any

from src.mahjong_ai.efficiency.shanten import calculate_shanten
from src.mahjong_ai.state.snapshot import RIVER_NAMES, PlayerState
from src.mahjong_core.hand import Hand

#: 向听数惩罚系数: 打出让向听数变差 Δ 的牌, 权重 × exp(-α·Δ)。
#: Δ=1 → 0.30, Δ=2 → 0.09 — 明显抑制但保留防守/失误的余地
_ALPHA = 1.2
#: 结构化漂移系数: 重采样权重 × exp(-β·shanten) — "玩家朝听牌构建"
#: 的软先验(随机先验几乎全是散牌, 不漂移则听牌概率恒 0)。
#: β=1.0 实测更差: 假听牌结构劫持重采样, 置信度与真值负相关。
_BETA = 0.4


def _counter_to_list(c: Counter[int]) -> list[int]:
    return [t for t, n in c.items() for _ in range(n)]


#: GIL 让渡粒度: 纯 Python 大循环每 N 次迭代让一次 —
#: 后台线程的粒子更新/十判若连续占 GIL 0.3-2s, 主循环(检测/显示)
#: 会被饿死, 整个流程跟不上打牌(真实对局反馈)
_YIELD_EVERY = 50


def _gil_yield(i: int) -> None:
    if i % _YIELD_EVERY == 0:
        time.sleep(0)  # sleep(0) = 主动让出 GIL 给其他线程


@lru_cache(maxsize=8192)
def _shanten_cached(tiles: tuple[int, ...]) -> int:
    """向听数(排序 tuple 缓存 — 重采样粒子有重复, 省 175μs/次 × 800)。"""
    return calculate_shanten(list(tiles))


def _discard_likelihood(hand: Counter[int], tile: int) -> float:
    """打出 tile 的似然: 孤张因子 × 向听数边际价值。

    Δ = shanten(打出后) - shanten(打出前):
      Δ > 0 → tile 对手牌向听结构有贡献(打出毁牌效) → 低权重
      Δ = 0 → 孤张/多余 → 正常
    孤张因子(保留): 相邻 ±1/±2 同花色牌越多越不像孤张。
    """
    base = _shanten_cached(tuple(sorted(hand.elements())))
    reduced = hand.copy()
    reduced[tile] -= 1
    if reduced[tile] <= 0:
        del reduced[tile]
    after = _shanten_cached(tuple(sorted(reduced.elements())))
    delta = max(0, after - base)
    if tile > 26:  # 字牌: 无搭子概念(相邻因子恒 1), 但仍按向听数计价值
        lonely = 1.0
    else:
        n = sum(hand.get(t, 0)
                for t in (tile - 2, tile - 1, tile + 1, tile + 2)
                if 0 <= t <= 26)
        lonely = 1.0 / (1.0 + n)
    return lonely * math.exp(-_ALPHA * delta)


class OpponentInference:
    """单家粒子滤波(对家威胁最大, 先每家独立建模)。"""

    def __init__(
        self,
        rules: Any,
        n_particles: int = 800,
        seed: int = 42,
    ) -> None:
        self._rules = rules
        self._n = n_particles
        self._rng = random.Random(seed)
        self._enabled = set(rules.get_tile_set().enabled_tiles)
        self._particles: list[Counter[int]] = []
        self._pool: Counter[int] = Counter()  # 剩余牌池(每类未知张数)
        self._hand_count = 13

    # ---- 生命周期 ----

    def reset(
        self,
        my_hand: list[int],
        players: dict[str, PlayerState],
        hand_count: int = 13,
    ) -> None:
        """开局/重建: 计算牌池(总数 - 自己手牌 - 所有明牌)并初始化粒子。

        hand_count: 该家当前推断手牌数(副露确认后联动, PlayerState 提供)。
        """
        self._hand_count = hand_count
        pool: Counter[int] = Counter(dict.fromkeys(self._enabled, 4))
        for t in my_hand:  # 自己手牌(已知, 不可能在他家手里)
            pool[t] -= 1
        for p in RIVER_NAMES:
            ps = players.get(p, PlayerState())
            for t in ps.river:  # 已打出的明牌
                pool[t] -= 1
            for m in ps.melds:  # 副露明牌
                pool[m.tile] = pool.get(m.tile, 0) - 1
        self._pool = Counter({t: n for t, n in pool.items() if n > 0})
        self._init_particles()

    def _init_particles(self) -> None:
        self._particles = []
        for i in range(self._n):
            _gil_yield(i)
            self._particles.append(self._sample_hand())

    def _sample_hand(self) -> Counter[int]:
        """从牌池抽 hand_count 张, 结构化加权(见 _structured_draw)。

        与纯随机不同: 抽牌时倾向与已有手牌形成搭子/对子的牌 —
        模拟"玩家在朝听牌构建"的先验。否则随机 13 张几乎全是
        散牌, 粒子集里没有听牌结构, tenpai 概率恒 0。
        """
        tiles = list(self._pool.elements())
        hand: Counter[int] = Counter()
        for _ in range(self._hand_count):
            if not tiles:
                break
            hand[self._structured_draw(hand, tiles)] += 1
        return hand

    def _structured_draw(self, hand: Counter[int], tiles: list[int]) -> int:
        """按结构加权抽一张牌: 与手牌相邻(±1/±2 同花色)或成对的优先。

        权重 = 1 + 2×相邻数牌 + 3×对子(字牌只计对子)。
        有偏提议(非严格重要性采样), 方向正确: 真实玩家摸牌会
        倾向有用的牌(牌效), 粒子必须跟随这个分布。
        """
        weights: list[float] = []
        for t in tiles:
            w = 1.0
            if t <= 26:
                for d in (-2, -1, 1, 2):
                    if 0 <= t + d <= 26:
                        w += 2.0 * hand.get(t + d, 0)
            if hand.get(t, 0) >= 1:
                w += 3.0
            weights.append(w)
        total = sum(weights)
        r = self._rng.random() * total
        acc = 0.0
        for t, w in zip(tiles, weights, strict=True):
            acc += w
            if r <= acc:
                return t
        return tiles[-1]

    def particles_snapshot(self) -> list[Counter[int]]:
        """粒子快照(引用复制, 微秒级) — 后台线程计算读一致快照用。"""
        return list(self._particles)

    def _seed_particles(self, hands: list[Counter[int]]) -> None:
        """测试专用: 直接注入粒子(绕开随机初始化)。"""
        self._particles = [h.copy() for h in hands]

    # ---- 观测 ----

    def observe_public(self, tile: int) -> None:
        """公开牌(任意家打出/副露亮出): 牌池删一张 + 粒子中该牌移除补牌。

        非幂等: 每张公开牌调用一次(调用方保证增量 — session 按事件
        索引与副露去重); 重复调用会把粒子中同 tile 的牌再删一次
        (样本空间错误收缩)。

        公开牌不可能在任何人手牌里(已打出/亮出), 粒子中的残留
        必须同步删除, 否则手牌分布虚高该牌(牌池收缩被架空)。
        删后从剩余牌池补一张保持手牌张数。
        """
        if self._pool.get(tile, 0) > 0:
            self._pool[tile] -= 1
        avail = None
        for i, p in enumerate(self._particles):
            _gil_yield(i)
            if p.get(tile, 0) == 0:
                continue
            q = p.copy()
            q[tile] -= 1
            if q[tile] <= 0:
                del q[tile]
            if avail is None:
                avail = list(self._pool.elements())
            if avail:
                q[self._structured_draw(q, avail)] += 1
            self._particles[i] = q

    def observe_discard(self, tile: int) -> None:
        """该家打出 tile: 过滤必含 → 似然加权重采样 → 删牌补牌。

        权重 = 打出似然(向听数边际 + 孤张) × exp(-β·shanten) —
        后一项是"玩家在朝听牌构建"的持续软先验: 随机 13 张的先验
        几乎全是散牌(向听 4-5), 不施加结构化漂移则粒子永远采样
        不到听牌结构, 听牌概率恒 0。β 小(0.4)保留防守/失误余地。
        """
        self._particles = self._discard_update(self._particles, tile)

    def _discard_update(
        self, particles: list[Counter[int]], tile: int,
    ) -> list[Counter[int]]:
        """纯更新: 给定粒子表 → 打出 tile 后的新粒子表(不碰 self)。"""
        weighted = []
        for i, p in enumerate(particles):
            _gil_yield(i)
            if p.get(tile, 0) <= 0:
                continue
            weighted.append((p, _discard_likelihood(p, tile)
                             * math.exp(-_BETA
                                        * _shanten_cached(
                                            tuple(sorted(p.elements()))))))
        if not weighted:
            # 观测与全部粒子冲突(先验崩了) → 按当前牌池重建
            self._init_particles()
            return self._particles
        total = sum(w for _, w in weighted)
        new: list[Counter[int]] = []
        for i in range(len(particles)):
            _gil_yield(i)
            r = self._rng.random() * total
            acc = 0.0
            chosen = weighted[-1][0]
            for p, w in weighted:
                acc += w
                if r <= acc:
                    chosen = p
                    break
            q = chosen.copy()
            q[tile] -= 1
            if q[tile] <= 0:
                del q[tile]
            q[self._structured_draw(q, list(self._pool.elements()))] += 1
            new.append(q)
        return new

    def observe_discard_partial(self, tile: int, weight: float) -> None:
        """软观测(M3): weight×n 个粒子走 discard 更新, 其余原样。

        统计上 = 贝叶斯混合 P(后验) = Σ_p w_p·P(后验|p 打出) —
        归属不确定时按概率部分更新, 而不是全量硬更新。
        """
        k = round(weight * self._n)
        if k <= 0:
            return
        if k >= self._n:
            self.observe_discard(tile)
            return
        idxs = self._rng.sample(range(self._n), k)
        subset = [self._particles[i] for i in idxs]
        updated = self._discard_update(subset, tile)
        for i, j in enumerate(idxs):
            self._particles[j] = updated[i]

    def shrink_for_meld(self, hand_tiles: tuple[int, ...],
                        n_total: int) -> None:
        """副露缩小手牌(P0): 粒子 → 副露后打出的手牌(13−3k 张)。

        hand_tiles: 确定进了副露的手牌(内容 − 被碰牌); 不足 n_total
        的部分随机移除(副露后打出的那张/内容未知的副露 — 不可观测)。
        十判时粒子 + 副露展开 = 13, 与 get_waiting_tiles 口径对齐。
        """
        for i, p in enumerate(self._particles):
            _gil_yield(i)
            q = p.copy()
            n = n_total
            for t in hand_tiles:
                if n <= 0:
                    break
                if q.get(t, 0) > 0:
                    q[t] -= 1
                    if q[t] <= 0:
                        del q[t]
                    n -= 1
            for _ in range(n):  # 剩余 = 随机(不可观测的离开牌)
                tiles = [t for t, c in q.items() for _ in range(c)]
                if not tiles:
                    break
                t = self._rng.choice(tiles)
                q[t] -= 1
                if q[t] <= 0:
                    del q[t]
            self._particles[i] = q

    def observe_discarded(self, tile: int) -> None:
        """副露后状态的打出观测(反向过滤, P0): 粒子 = 打出后手牌,
        刚打出的 tile 不该在粒子中 — 过滤掉含 tile 的粒子; 权重 =
        从"粒子+tile"(打出前手牌)打出 tile 的似然(与正向口径一致),
        重采样保持张数(摸进的下一张不可观测, 不建模 — 近似)。
        """
        self._particles = self._discard_update_post(self._particles, tile)

    def observe_discarded_partial(self, tile: int, weight: float) -> None:
        """软观测版: weight×n 个粒子走反向过滤, 其余原样(贝叶斯混合)。"""
        k = round(weight * self._n)
        if k <= 0:
            return
        if k >= self._n:
            self.observe_discarded(tile)
            return
        idxs = self._rng.sample(range(self._n), k)
        subset = [self._particles[i] for i in idxs]
        updated = self._discard_update_post(subset, tile)
        for i, j in enumerate(idxs):
            self._particles[j] = updated[i]

    def _discard_update_post(
        self, particles: list[Counter[int]], tile: int,
    ) -> list[Counter[int]]:
        """纯更新: 反向过滤(含 tile 的粒子不可能 — 它刚被打出)。"""
        weighted = []
        for i, p in enumerate(particles):
            _gil_yield(i)
            if p.get(tile, 0) > 0:
                continue
            q = p.copy()
            q[tile] += 1  # 打出前的手牌(似然口径)
            weighted.append((p, _discard_likelihood(q, tile)
                             * math.exp(-_BETA
                                        * _shanten_cached(
                                            tuple(sorted(q.elements()))))))
        if not weighted:
            # 观测与全部粒子冲突(先验崩了) → 按当前牌池重建
            self._init_particles()
            return self._particles
        total = sum(w for _, w in weighted)
        new: list[Counter[int]] = []
        for i in range(len(particles)):
            _gil_yield(i)
            r = self._rng.random() * total
            acc = 0.0
            chosen = weighted[-1][0]
            for p, w in weighted:
                acc += w
                if r <= acc:
                    chosen = p
                    break
            new.append(chosen.copy())
        return new

    def get_state(self) -> tuple:
        """(粒子表副本, 牌池副本, RNG 状态, hand_count) — M3 快照回放。"""
        return ([p.copy() for p in self._particles],
                Counter(self._pool), self._rng.getstate(), self._hand_count)

    def set_state(self, state: tuple) -> None:
        """恢复快照(M3 翻案回放 — 确定性, RNG 状态随快照走)。"""
        particles, pool, rng_state, hand_count = state
        self._particles = particles
        self._pool = pool
        self._rng.setstate(rng_state)
        self._hand_count = hand_count

    # ---- 输出 ----

    def hand_distribution(self) -> dict[int, float]:
        """每张牌在该家手中的期望张数(粒子均值)。"""
        dist: dict[int, float] = {}
        for p in self._particles:
            for t, n in p.items():
                dist[t] = dist.get(t, 0.0) + n / self._n
        return dist

    def _tenpai_of(
        self,
        particles: list[Counter[int]],
        melds: tuple[tuple[int, ...], ...] = (),
    ) -> tuple[float, dict[int, float]]:
        """对给定粒子集计算听牌(纯计算, 不读 self._particles — 后台线程用)。

        副露展开(P0 修复): 粒子 = 副露后打出的手牌(13−3k 张) +
        各副露前 3 张(杠的第 4 张不影响面子结构) = 13 → 标准十判。
        内容未知的副露按碰展开(近似)。张数不符(快照与副露记录
        不一致)的粒子跳过 — 旧行为副露后恒 0 已废除。
        """
        expansion = [t for m in melds for t in m[:3]]
        waiting: Counter[int] = Counter()
        n_tenpai = 0
        for i, p in enumerate(particles):
            _gil_yield(i)
            try:
                ws = self._rules.get_waiting_tiles(
                    Hand(_counter_to_list(p) + expansion))
            except ValueError:
                continue
            if ws:
                n_tenpai += 1
                for w in ws:
                    waiting[w.tile] += 1
        prob = n_tenpai / self._n
        dist = ({t: c / n_tenpai for t, c in waiting.items()}
                if n_tenpai else {})
        return prob, dist

    def tenpai(self) -> tuple[float, dict[int, float]]:
        """听牌推断: (听牌概率, 听牌分布 {tile: 概率|听牌时})。

        粒子快照(引用复制, 微秒级)后计算 — 主线程读结果时, 后台
        observe_discard/observe_public 可能已替换粒子, 快照保证一致性。
        """
        return self._tenpai_of(list(self._particles))

    def discard_risk(self, tile: int) -> float:
        """放铳率: 我打出 tile, 该家能胡的概率(粒子中手牌+tile 成胡的比例)。"""
        return self.discard_risk_from(self._particles, tile)

    def discard_risk_from(self, particles: list[Counter[int]],
                          tile: int,
                          melds: tuple[tuple[int, ...], ...] = ()) -> float:
        """给定粒子快照的放铳率 — 主线程批量计算用(读快照, 非实时)。

        逐牌读实时粒子时, 后台线程在两次调用间更新粒子 → 同一帧
        候选值互相不一致且逐帧抖动, 客户端放铳行闪烁(真实对局反馈)。
        快照一次算全部候选, 帧内一致; 观测更新(0.5s 节流)才变。
        副露展开同 _tenpai_of: 展开 13 张 + tile = 14 胡判(P0: 副露后
        不再恒 0); 张数不符的粒子跳过。
        """
        expansion = [t for m in melds for t in m[:3]]
        n_win = 0
        for p in particles:
            try:
                if self._rules.is_winning_hand(
                        Hand(_counter_to_list(p) + expansion),
                        new_tile=tile).can_win:
                    n_win += 1
            except ValueError:
                continue
        return n_win / self._n
