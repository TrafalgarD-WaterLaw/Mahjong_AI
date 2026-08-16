"""对手副露内容推断 — 用牌池(剩余未知张数)收窄碰/吃组合。

事件推断只确定被碰牌 X(来自牌河); 另外两张来自副露者的暗手牌,
内容未知。但可用张数给了硬约束:
  碰 = (X,X,X): 副露者手牌需 2 张 X → 池中 X ≥ 2
  吃 = 含 X 的同花顺子: 其余两张各需池中 ≥ 1
唯一解 = 内容确定(扣池/显示直接受益 — 这三张确定离开牌池,
对手手牌推断的关键约束); 多解 = 歧义(只扣确定的被碰牌)。

池语义: 每类牌剩余未知张数(总数 4 − 我手牌 − 各河打出 − 已亮副露,
被碰张随打出事件已扣)。显示层的"可用张数"与 M3 的池同源 —
两处消费同一函数, 结论一致。
"""

from __future__ import annotations

from collections import Counter


def resolve_opponent_meld(
    claimed: int, available: Counter[int],
) -> list[tuple[int, ...]]:
    """对手副露(被碰牌 claimed, 池 available) → 可能的组合列表。

    0 个 = 计数不一致(上游检测漏 — 调用方按无解处理, 不采用);
    1 个 = 确定; 多个 = 歧义(调用方只按确定的交集扣)。
    """
    out: list[tuple[int, ...]] = []
    if available.get(claimed, 0) >= 2:
        out.append((claimed,) * 3)
    if claimed <= 26:  # 字牌不吃
        suit = claimed // 9 * 9
        lo = max(claimed - 2, suit)
        hi = min(claimed, suit + 6)
        for a in range(lo, hi + 1):
            run = (a, a + 1, a + 2)
            if all(available.get(t, 0) >= (0 if t == claimed else 1)
                   for t in run):
                out.append(run)
    return out
