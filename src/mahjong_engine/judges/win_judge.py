"""回溯法胡牌判定。

支持三种牌型:
    - standard: 4面子(顺子或刻子)+1雀头
    - seven_pairs: 7种牌各2张
    - thirteen_orphans: 13种幺九牌各1张+1张重复
"""

from dataclasses import dataclass, field

from src.mahjong_core.tile import (
    B1,
    B9,
    BAI,
    BEI,
    BING,
    DONG,
    FA,
    NAN,
    T1,
    T9,
    TIAO,
    W1,
    W9,
    WAN,
    XI,
    ZHONG,
    tile_display,
    tile_number,
    tile_suit,
)

# 幺九牌: 1万,9万,1筒,9筒,1条,9条,东南西北中发白
_TERMINAL_AND_HONOR = frozenset({
    W1, W9, B1, B9, T1, T9,
    DONG, NAN, XI, BEI, ZHONG, FA, BAI,
})

TOTAL_TILE_TYPES = 42  # 支持全部42种牌索引


@dataclass
class WinResult:
    """胡牌判定结果。"""
    can_win: bool
    pattern_type: str = ""
    breakdown: list[str] = field(default_factory=list)


def is_winning_hand(tiles: list[int]) -> WinResult:
    """判断14张牌是否胡牌。

    Args:
        tiles: 14张牌的整数列表

    Returns:
        WinResult 含胡牌类型和面子拆解
    """
    if len(tiles) != 14:
        raise ValueError(f"胡牌判定需要14张牌，当前{len(tiles)}张")

    counts = _count_tiles(tiles)

    # 1. 七对判定
    result = _check_seven_pairs(counts)
    if result.can_win:
        return result

    # 2. 十三幺判定
    result = _check_thirteen_orphans(counts)
    if result.can_win:
        return result

    # 3. 标准胡牌: 回溯
    return _check_standard(counts)


def _count_tiles(tiles: list[int]) -> list[int]:
    """统计每种牌的数量，返回长度42的列表。"""
    counts = [0] * TOTAL_TILE_TYPES
    for t in tiles:
        counts[t] += 1
    return counts


def _check_seven_pairs(counts: list[int]) -> WinResult:
    """七对判定: 7种牌各2张。"""
    pair_count = 0
    pairs = []
    for t in range(TOTAL_TILE_TYPES):
        if counts[t] == 2:
            pair_count += 1
            pairs.append(t)
        elif counts[t] != 0:
            return WinResult(can_win=False)
    if pair_count == 7:
        breakdown = [tile_display(t) + tile_display(t) for t in pairs]
        return WinResult(can_win=True, pattern_type='seven_pairs', breakdown=breakdown)
    return WinResult(can_win=False)


def _check_thirteen_orphans(counts: list[int]) -> WinResult:
    """十三幺判定: 13种幺九牌各1张 + 1张重复。"""
    present = []
    pair_tile = None
    for t in _TERMINAL_AND_HONOR:
        if counts[t] == 1:
            present.append(t)
        elif counts[t] == 2:
            if pair_tile is not None:
                return WinResult(can_win=False)  # 两个对子，不是十三幺
            pair_tile = t
            present.append(t)
        elif counts[t] > 2:
            return WinResult(can_win=False)

    # 检查有没有非幺九牌
    for t in range(TOTAL_TILE_TYPES):
        if counts[t] > 0 and t not in _TERMINAL_AND_HONOR:
            return WinResult(can_win=False)

    if len(present) == 13 and pair_tile is not None:
        breakdown = [tile_display(t) for t in present]
        return WinResult(can_win=True, pattern_type='thirteen_orphans', breakdown=breakdown)
    return WinResult(can_win=False)


def _check_standard(counts: list[int]) -> WinResult:
    """标准胡牌: 4面子+1雀头，回溯法。"""
    total = sum(counts)
    if total != 14:
        return WinResult(can_win=False)

    # 尝试每种数量≥2的牌作雀头
    for pair_tile in range(TOTAL_TILE_TYPES):
        if counts[pair_tile] >= 2:
            counts[pair_tile] -= 2
            melds: list[list[int]] = []
            if _try_remove_melds(counts, melds):
                # 构建拆解描述
                breakdown = _format_breakdown(pair_tile, melds)
                counts[pair_tile] += 2  # 恢复
                return WinResult(can_win=True, pattern_type='standard', breakdown=breakdown)
            counts[pair_tile] += 2  # 恢复

    return WinResult(can_win=False)


def _try_remove_melds(counts: list[int], melds: list[list[int]]) -> bool:
    """递归尝试去除4个面子。找到最小非零牌，尝试刻子或顺子。"""
    # 检查是否全部去除（总数应为0）
    remaining = sum(counts)
    if remaining == 0:
        return len(melds) == 4

    if len(melds) >= 4:
        return remaining == 0

    # 找到最小的非零牌
    start = _find_first_nonzero(counts)
    if start < 0:
        return False

    # 尝试刻子
    if counts[start] >= 3:
        counts[start] -= 3
        melds.append([start, start, start])
        if _try_remove_melds(counts, melds):
            return True
        melds.pop()
        counts[start] += 3

    # 尝试顺子（仅万筒条）
    s = tile_suit(start)
    if s in (WAN, BING, TIAO):
        n = tile_number(start)
        if n is not None and n <= 7:  # 顺子起始数字1-7
            a, b, c = start, start + 1, start + 2
            if (tile_suit(a) == tile_suit(b) == tile_suit(c)
                    and counts[a] >= 1 and counts[b] >= 1 and counts[c] >= 1):
                counts[a] -= 1
                counts[b] -= 1
                counts[c] -= 1
                melds.append([a, b, c])
                if _try_remove_melds(counts, melds):
                    return True
                melds.pop()
                counts[a] += 1
                counts[b] += 1
                counts[c] += 1

    return False


def _find_first_nonzero(counts: list[int]) -> int:
    """找到第一个数量>0的牌的索引。"""
    for t in range(TOTAL_TILE_TYPES):
        if counts[t] > 0:
            return t
    return -1


def _format_breakdown(pair: int, melds: list[list[int]]) -> list[str]:
    """格式化拆解描述为可读字符串。"""
    result = []
    for m in melds:
        if m[0] == m[1] == m[2]:
            result.append(f"{tile_display(m[0])}{tile_display(m[0])}{tile_display(m[0])}")
        else:
            result.append(f"{tile_display(m[0])}{tile_display(m[1])}{tile_display(m[2])}")
    result.append(f"{tile_display(pair)}{tile_display(pair)}(雀头)")
    return result
