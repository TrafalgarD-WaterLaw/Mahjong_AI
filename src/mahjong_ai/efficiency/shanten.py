"""向听数计算。

Shanten（向听数）表示距离听牌还需换几张牌:
    -1: 已胡牌
     0: 听牌（再摸一张即可胡）
     1: 一向听（还需换一张牌才能听牌）
     ...
"""

from collections import Counter

from src.mahjong_core.tile import (
    B1,
    B9,
    BAI,
    BEI,
    BING_TILES,
    DONG,
    FA,
    FENG_TILES,
    HUA_TILES,
    JIAN_TILES,
    NAN,
    T1,
    T9,
    TIAO_TILES,
    W1,
    W9,
    WAN_TILES,
    XI,
    ZHONG,
)
from src.mahjong_engine.judges.win_judge import is_winning_hand

_TERMINAL_SET = frozenset({
    W1, W9, B1, B9, T1, T9,
    DONG, NAN, XI, BEI, ZHONG, FA, BAI,
})


def calculate_shanten(tiles: list[int]) -> int:
    """计算手牌的向听数。取标准型、七对型、十三幺型的最小值。

    Args:
        tiles: 13或14张手牌

    Returns:
        shanten值 (-1=胡牌, 0=听牌, 1=一向听, ...)
    """
    n = len(tiles)

    # 先检查是否已胡牌
    if n == 14:
        result = is_winning_hand(tiles)
        if result.can_win:
            return -1

    counts = Counter(tiles)

    shanten = min(
        _standard_shanten(counts, n),
        _seven_pairs_shanten(counts, n),
        _thirteen_orphans_shanten(counts, n),
    )
    return shanten


def _standard_shanten(counts: Counter[int], n: int) -> int:
    """标准型向听数: 4面子+1雀头。

    算法: 对每种可能的雀头位置，贪心计算最大面子数，
    取最小值: 8 - 2*melds - partials - has_pair
    """
    # 数牌三花色(顺子+刻子+搭子); 字牌单独处理(无顺子/搭子概念,
    # 东南西被当顺子、中发被当搭子会把向听算虚低 — 只有刻子/对子有价值)
    suits = [
        list(WAN_TILES),
        list(BING_TILES),
        list(TIAO_TILES),
    ]
    honor_tiles = list(FENG_TILES) + list(JIAN_TILES)

    best_m = 0
    best_p = 0

    # 尝试以每种牌为雀头（或不设雀头）
    pair_candidates = [t for t, c in counts.items() if c >= 2] + [None]

    for pair_tile in pair_candidates:
        total_m = 0
        total_p = 0
        has_pair = 0

        # 分配雀头
        local = dict(counts)
        if pair_tile is not None:
            local[pair_tile] -= 2
            has_pair = 1

        for suit in suits:
            suit_counts = [local.get(t, 0) for t in suit]
            m, p = _count_melds(suit_counts)
            total_m += m
            total_p += p

        # 字牌: 只计刻子与对子(顺子不存在)
        honor = [local.get(t, 0) for t in honor_tiles]
        total_m += sum(c // 3 for c in honor)
        total_p += sum(c // 2 for c in honor)

        # 面子数不能超过4
        total_m = min(total_m, 4)
        total_p = min(total_p, 4 - total_m)

        score = 8 - 2 * total_m - total_p - has_pair
        shanten = max(-1, score)

        if shanten <= 0:
            return shanten

        # 记录可用于后续计算的最佳值
        if total_m > best_m or (total_m == best_m and total_p > best_p):
            best_m = total_m
            best_p = total_p

    result = 8 - 2 * best_m - best_p - (1 if best_m > 0 or best_p > 0 else 0)
    return max(-1, result)


def _count_melds(suit_counts: list[int]) -> tuple[int, int]:
    """对一组同花色牌，计算(完整面子数, 有效搭子数)。

    顺子优先与刻子优先两种贪心都试取优: 刻子优先会破坏顺子组合
    (22234 万按刻贪心得 222+34, 最优是 234+22 — 只贪一种会高估向听)。
    剩余牌数**有效搭子**(对子/邻张/隔1 的贪心匹配), 不用 //2 高估
    (1357 万实际只有 1-2 个搭子, //2 会算 2 个 → 向听偏低)。
    """
    def run(seq_first: bool) -> tuple[int, list[int]]:
        counts = list(suit_counts)
        melds = 0
        n = len(counts)
        for first in (seq_first, not seq_first):
            if first:  # 顺子
                for i in range(n - 2):
                    while counts[i] >= 1 and counts[i + 1] >= 1 \
                            and counts[i + 2] >= 1:
                        counts[i] -= 1
                        counts[i + 1] -= 1
                        counts[i + 2] -= 1
                        melds += 1
            else:      # 刻子
                for i in range(n):
                    while counts[i] >= 3:
                        counts[i] -= 3
                        melds += 1
        return melds, counts

    def partials(counts: list[int]) -> int:
        """剩余牌的搭子数: 对子 > 隔1(嵌张) > 相邻(两面), 贪心匹配。"""
        c = list(counts)
        p = 0
        n = len(c)
        for i in range(n):
            while c[i] >= 2:
                c[i] -= 2
                p += 1
        for i in range(n - 2):
            while c[i] >= 1 and c[i + 2] >= 1:
                c[i] -= 1
                c[i + 2] -= 1
                p += 1
        for i in range(n - 1):
            while c[i] >= 1 and c[i + 1] >= 1:
                c[i] -= 1
                c[i + 1] -= 1
                p += 1
        return p

    m_seq, rest_seq = run(True)
    m_pung, rest_pung = run(False)
    p_seq, p_pung = partials(rest_seq), partials(rest_pung)
    if m_seq > m_pung or (m_seq == m_pung and p_seq >= p_pung):
        return m_seq, p_seq
    return m_pung, p_pung


def _seven_pairs_shanten(counts: Counter[int], n: int) -> int:
    """七对型向听数: 6 - 对子数。"""
    pairs = sum(1 for c in counts.values() if c >= 2)
    # 如果有超过2张相同的牌, 超出部分也算对子
    extra = sum(c // 2 - 1 for c in counts.values() if c >= 4)
    shanten = 6 - pairs - extra
    return max(-1, shanten)


def _thirteen_orphans_shanten(counts: Counter[int], n: int) -> int:
    """十三幺型向听数: 13 - 已有幺九牌种类数。"""
    present = sum(1 for t in _TERMINAL_SET if counts.get(t, 0) > 0)
    has_pair = any(counts.get(t, 0) >= 2 for t in _TERMINAL_SET)
    shanten = 13 - present - (1 if has_pair else 0)
    return max(-1, shanten)
