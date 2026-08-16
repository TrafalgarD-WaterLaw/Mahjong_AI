"""操作合法性判定: 碰、杠、吃。"""

from dataclasses import dataclass

from src.mahjong_core.tile import BING, TIAO, WAN, tile_number, tile_suit


@dataclass
class KongResult:
    """杠牌判定结果。"""
    can_kong: bool
    kong_type: str = ""  # "ming" | "an" | "jia"


def can_pong(hand_tiles: list[int], tile: int) -> bool:
    """判断是否可以碰: 手牌中至少有2张与tile相同。"""
    return hand_tiles.count(tile) >= 2


def can_kong(hand_tiles: list[int], tile: int, ponged_tiles: list[int] | None = None) -> KongResult:
    """判断是否可以杠。

    暗杠: 手牌中有4张相同
    加杠: 已碰过的牌中，手牌还有1张相同
    明杠: 手牌中有3张相同 (此处仅判定手牌条件)

    Args:
        hand_tiles: 当前手牌
        tile: 待杠的牌
        ponged_tiles: 已碰的牌列表(用于判断加杠)
    """
    count = hand_tiles.count(tile)

    # 暗杠: 手牌4张
    if count == 4:
        return KongResult(can_kong=True, kong_type='an')

    # 加杠: 已碰过这张牌
    if ponged_tiles and tile in ponged_tiles:
        if count >= 1:
            return KongResult(can_kong=True, kong_type='jia')

    # 明杠: 手牌3张 (别家打出1张)
    if count >= 3:
        return KongResult(can_kong=True, kong_type='ming')

    return KongResult(can_kong=False)


def can_chi(
    hand_tiles: list[int],
    tile: int,
    wild_tiles: frozenset[int] | None = None,
) -> list[list[int]]:
    """判断是否可以吃，返回可行的顺子组合。

    仅万筒条可以形成顺子。上家打出的tile可作为顺子的任一位(首/中/尾)。

    Args:
        hand_tiles: 当前手牌
        tile: 上家打出的牌
        wild_tiles: 万能牌集合(本规则为空)

    Returns:
        可行的顺子组合列表，如 [[tile, tile+1, tile+2], ...]
    """
    s = tile_suit(tile)
    if s not in (WAN, BING, TIAO):
        return []

    n = tile_number(tile)
    if n is None:
        return []

    options: list[list[int]] = []

    # tile 作为顺子首位 (n, n+1, n+2)
    if n <= 7:
        if hand_tiles.count(tile + 1) >= 1 and hand_tiles.count(tile + 2) >= 1:
            options.append([tile, tile + 1, tile + 2])

    # tile 作为顺子中位 (n-1, n, n+1)
    if 2 <= n <= 8:
        if hand_tiles.count(tile - 1) >= 1 and hand_tiles.count(tile + 1) >= 1:
            options.append([tile - 1, tile, tile + 1])

    # tile 作为顺子末位 (n-2, n-1, n)
    if n >= 3:
        if hand_tiles.count(tile - 2) >= 1 and hand_tiles.count(tile - 1) >= 1:
            options.append([tile - 2, tile - 1, tile])

    return options
