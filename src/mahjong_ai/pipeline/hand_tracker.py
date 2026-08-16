"""我的手牌账本(塌陷兜底) + 副露台阶签名。

检测健康时手牌 = 检测(现行); 检测塌陷时用事件账本兜底:
最后已知手牌 − 我的打出 − 副露扣除, 摸进的新牌 = 未知槽。
台阶签名 = 稳定计数纪元掉 2(碰/吃)/3(明杠)且新纪元保持 ≥2 秒 —
抽风是振荡不落脚, 不构成台阶。手牌差 = 前后稳定手牌差(剔除
窗口内我打出的牌), 供 MeldInferrer 推断我的副露内容。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

#: 稳定纪元: 计数连续不变 ≥ 此帧数(≈0.7s @7fps)
_STABLE_TICKS = 5
#: 台阶保持: 新纪元持续 ≥ 此秒数才确认副露签名
_STEP_HOLD_S = 2.0
#: 快照/历史保留窗口(秒)
_SNAP_WINDOW = 10.0
#: 稳定期快照刷新间隔(秒 — 13 张手牌可能稳定几十秒, 快照必须新鲜)
_SNAP_REFRESH_S = 2.0
#: 台阶触发前纪元最小计数(排除发牌斜坡 10→13 等; 与消失配对后
#: 假台阶自然被过滤, 此阈值只省事)
_STEP_MIN_PREV = 8
#: 兜底判定: 近此秒数内有稳定快照 = 检测健康(兜底返回 None)
_HEALTHY_S = 2.0


@dataclass(frozen=True)
class StepSig:
    """副露台阶签名: 稳定计数掉 drop(2=碰/吃, 3=明杠), ts=台阶时刻。"""

    drop: int
    ts: float


class HandTracker:
    """手牌账本 + 台阶签名(纯 Python, 每帧 tick)。"""

    def __init__(self) -> None:
        self._last_good: Counter[int] = Counter()
        self._unknown = 0
        self._snapshots: list[tuple[float, Counter[int]]] = []
        self._discarded: list[tuple[int, float]] = []
        self._epoch_count: int | None = None
        self._epoch_ticks = 0
        self._pending: StepSig | None = None

    def tick(self, hand: list[int], ts: float) -> StepSig | None:
        """每帧: 纪元/台阶检测 + 稳定快照更新(可能返回台阶)。"""
        n = len(hand)
        if n != self._epoch_count:
            if (self._epoch_count is not None
                    and self._epoch_count >= _STEP_MIN_PREV
                    and self._epoch_count - n in (2, 3)):
                self._pending = StepSig(drop=self._epoch_count - n, ts=ts)
            self._epoch_count = n
            self._epoch_ticks = 0
        self._epoch_ticks += 1
        out = None
        if (self._pending is not None
                and ts - self._pending.ts >= _STEP_HOLD_S):
            out = self._pending
            self._pending = None
        if self._epoch_ticks == _STABLE_TICKS and hand:
            self._record(ts, hand)
        elif (self._epoch_ticks > _STABLE_TICKS and hand
              and self._snapshots
              and ts - self._snapshots[-1][0] >= _SNAP_REFRESH_S):
            self._record(ts, hand)  # 长稳定期快照刷新(手牌差要新鲜)
        return out

    def _record(self, ts: float, hand: list[int]) -> None:
        """稳定快照: 账本重建 + 快照历史(裁剪保留窗口)。"""
        self._last_good = Counter(hand)
        self._unknown = 0
        self._snapshots.append((ts, Counter(hand)))
        while self._snapshots and \
                ts - self._snapshots[0][0] > _SNAP_WINDOW:
            self._snapshots.pop(0)

    def on_my_discard(self, tile: int, ts: float) -> None:
        """我打出一张: 账本扣牌(已知扣已知, 未知扣未知槽)。"""
        self._discarded.append((tile, ts))
        while self._discarded and \
                ts - self._discarded[0][1] > _SNAP_WINDOW:
            self._discarded.pop(0)
        if self._last_good.get(tile, 0) > 0:
            self._last_good[tile] -= 1
            if self._last_good[tile] <= 0:
                del self._last_good[tile]
        else:
            self._unknown = max(0, self._unknown - 1)

    def on_my_meld(self, delta: list[int], drop: int) -> None:
        """我的副露扣除(塌陷兜底账本)。账本已被后续稳定检测重建时
        扣除已隐含 — 校验账本仍含 delta 才扣(防双扣)。内容未知
        (delta 为空)时: 先扣未知槽(新摸的牌最可能被副露用掉),
        不足部分从已知账本移除(优先重复多的 — 副露通常是刻子,
        同频取牌值大的; 具体哪张不可知, 只承诺数量)。"""
        if delta and all(self._last_good.get(t, 0) >= delta.count(t)
                         for t in set(delta)):
            for t in delta:
                self._last_good[t] -= 1
                if self._last_good[t] <= 0:
                    del self._last_good[t]
        elif not delta:
            rem = drop - self._unknown
            self._unknown = max(0, self._unknown - drop)
            while rem > 0 and self._last_good:
                t = max(self._last_good,
                        key=lambda x: (self._last_good[x], x))
                self._last_good[t] -= 1
                if self._last_good[t] <= 0:
                    del self._last_good[t]
                rem -= 1

    def hand_delta(self, ts: float, window: float = 3.0) -> list[int] | None:
        """ts 前后的稳定手牌差(净副露牌): 前 = 计数变化前的最后
        快照, 后 = 变化后的首张快照(设计: 前 = vanish 前最后稳定
        手牌, 后 = 台阶落地后首个稳定手牌 — 副露后手牌在消失事件
        之前就稳定, 消失检测滞后 ≈2s, 所以后窗口前伸 1.5s);
        差值剔除 [ts−window, ts+window] 内我打出的牌。任一缺失
        (检测塌陷)→ None(内容降级未知)。
        """
        before = after = None
        prev_t: float | None = None
        prev_c: Counter[int] | None = None
        for t, c in self._snapshots:
            if (prev_c is not None
                    and sum(c.values()) != sum(prev_c.values())
                    and prev_t is not None and prev_t <= ts
                    and ts - 1.5 <= t <= ts + window):
                before, after = prev_c, c  # 覆盖 = 取最近一次计数变化
            prev_t, prev_c = t, c
        if before is None or after is None:
            return None
        delta = Counter(before)
        for tt, cc in after.items():
            delta[tt] -= cc
        for t, dts in self._discarded:
            if ts - window <= dts <= ts + window and delta.get(t, 0) > 0:
                delta[t] -= 1
        out: list[int] = []
        for t, k in delta.items():
            if k > 0:
                out.extend([t] * k)
        return sorted(out)

    def fallback_hand(self, ts: float) -> list[int | None] | None:
        """塌陷兜底: 最后已知 + 未知槽(None); 检测健康 → None。"""
        if self._snapshots and ts - self._snapshots[-1][0] < _HEALTHY_S:
            return None
        return sorted(self._last_good.elements()) + [None] * self._unknown
