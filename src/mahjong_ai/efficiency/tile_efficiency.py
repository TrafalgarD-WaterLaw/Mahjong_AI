"""牌效评估 — 评估每张候选打出牌的价值。"""

from dataclasses import dataclass

from src.mahjong_ai.efficiency.shanten import calculate_shanten
from src.mahjong_engine.judges.tenpai_judge import get_waiting_tiles


def _lonely(tile: int, hand: list[int]) -> int:
    """该牌在手牌中的搭子潜力 = 参与的搭子进张种类数(打出后失去的价值)。

    层级(越小越孤 → 越先打):
      字牌孤张(-10) < 数牌边张孤张(-4..-2) < 字牌对子(-1)
      < 数牌中张孤张(0) < 搭子牌(进张种类 1-3, 边张搭减 1)
    数牌孤张按边张程度区分: 1/9 最孤(只有 12/13 或 78/89 潜力),
    中张 5 潜力最大(可成 4 种搭子)。
    """
    if tile > 26:  # 字牌: 孤张绝对优先, 对子可碰(价值高于数牌孤张)
        return -1 if hand.count(tile) >= 2 else -10
    waits: set[tuple[str, int]] = set()
    if hand.count(tile) >= 2:
        waits.add(('pong', tile))          # 对子: 碰/成刻
    for d in (-1, 1):                      # 相邻 → 两面/边张: 两端外延
        t = tile + d
        if 0 <= t <= 26 and t // 9 == tile // 9 and hand.count(t) > 0:
            for w in (tile + 2 * d, tile - d):
                if 0 <= w <= 26 and w // 9 == tile // 9:
                    waits.add(('ryanmen', w))
    for d in (-2, 2):                      # 隔1 → 嵌张: 中间一张
        t = tile + d
        if 0 <= t <= 26 and t // 9 == tile // 9 and hand.count(t) > 0:
            waits.add(('kanchan', (tile + t) // 2))
    if waits:
        # 搭子牌: 边张(1/9)作为搭子时价值更低(12/89 边搭) → 减 1
        n = tile % 9
        return len(waits) + (-1 if n in (0, 8) else 0)
    # 数牌孤张: 边张更孤(1/9 潜力最小), 中张潜力最大
    n = tile % 9
    return -(4 - min(n, 8 - n))


@dataclass
class TileScore:
    """单张牌的打分。"""
    tile: int
    shanten_after: int        # 打出后向听数
    acceptance: int = 0        # 听牌后的等待牌种类数(仅shanten=0时有效)
    is_tenpai: bool = False    # 打出后是否听牌
    lonely: int = 0            # 搭子潜力(越小越孤, 越该打)


def evaluate_discard(
    tiles: list[int],
    enabled_tiles: frozenset[int] | None = None,
) -> list[TileScore]:
    """评估手牌中每张牌的打出价值。

    Args:
        tiles: 14张手牌
        enabled_tiles: 牌库启用的牌(用于听牌分析)

    Returns:
        对每张唯一的牌的评估结果列表，按价值降序排列

    排序: 向听数(小优先) → 听牌面数(大优先) → 孤张价值(搭子少优先)。
    无最后一级时, 向听相同的候选会退化为手牌检测顺序(乱序) —
    孤张西风与搭子牌同分时可能被跳过, 推荐显得随机。
    """
    scores: list[TileScore] = []
    seen: set[int] = set()

    for tile in tiles:
        if tile in seen:
            continue
        seen.add(tile)

        # 模拟打出这张牌
        remaining = list(tiles)
        remaining.remove(tile)  # 移除一张

        shanten = calculate_shanten(remaining)
        score = TileScore(tile=tile, shanten_after=shanten,
                          lonely=_lonely(tile, tiles))

        # 听牌时计算等待牌数量(仅 13 张 — 十判引擎要求; 手牌不足时
        # 向听可能被低估成 0, 但听牌面数无意义)
        if shanten == 0 and enabled_tiles is not None and len(remaining) == 13:
            waiting = get_waiting_tiles(remaining, enabled_tiles)
            score.acceptance = len(waiting)
            score.is_tenpai = True

        scores.append(score)

    # 排序: 向听数 → 听牌面数 → 孤张价值(越孤越先打)
    scores.sort(key=lambda s: (s.shanten_after, -s.acceptance, s.lonely))
    return scores
