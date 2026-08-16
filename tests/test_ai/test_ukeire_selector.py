"""纯牌效出牌推荐 — 经典牌理用例(期望值经算法输出与牌理核对)。"""

from src.mahjong_ai.efficiency.ukeire_selector import (
    _shanten_cached,
    _ukeire,
    recommend_by_ukeire,
)
from src.mahjong_engine import get as get_rules

_ENABLED = frozenset(get_rules('huanyu').get_tile_set().enabled_tiles)
_FULL = dict.fromkeys(range(34), 4)


def _rec(tiles, available=None, melds=None, exclude=None):
    return recommend_by_ukeire(tiles, available or _FULL, _ENABLED,
                               melds=melds, exclude=exclude)


def test_honor_lonely_first():
    """字牌孤张最先打(结构完整时; 三张孤字等价, 任一都合理)。"""
    tiles = [0, 0, 0, 1, 1, 1, 2, 2, 2, 27, 30, 33, 5, 6]
    assert _rec(tiles).tile in (27, 30, 33)


def test_float_before_breaking_taatsu():
    """浮张先于拆搭: 边张搭(1,2) + 两面搭(4,5) + 浮张 30 → 打 30。"""
    tiles = [1, 2, 4, 5, 9, 9, 9, 18, 19, 20, 27, 27, 27, 30]
    assert _rec(tiles).tile == 30


def test_three_pairs_break_one():
    """三对子 + 搭子: 拆一对保两对 + 搭(进张面最大)。"""
    tiles = [0, 0, 5, 5, 9, 10, 18, 19, 20, 27, 27, 27, 30, 30]
    assert _rec(tiles).tile == 30


def test_acceptance_reflects_availability():
    """听牌时等待枚数 = 剩余可摸枚数(手牌已占与全见都少计)。"""
    # 3 面子 + 27 对 + 4 对 + 浮张 33(14 张) → 打 33 听牌, 等 4/27
    tiles = [0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 27, 4, 4, 33]
    avail = dict(_FULL)
    for t in tiles:
        avail[t] -= 1  # available = 4 − 自己手牌
    rec = _rec(tiles, avail)
    assert rec.tile == 33
    assert rec.acceptance == 4  # 4 剩 2 枚 + 27 剩 2 枚
    avail[4] = 0  # 4 全可见 → 只剩 27 两张
    assert _rec(tiles, avail).acceptance == 2


def test_ukeire_respects_availability():
    """进张枚数按剩余可摸枚数计(全见 → 不计)。

    3 面子 + 30 对 + 边张 1,2 = 13 张已听(等 0 和 3, 各 4 枚)。
    """
    remaining = [1, 2, 9, 9, 9, 18, 19, 20, 27, 27, 27, 30, 30]
    sa = _shanten_cached(tuple(sorted(remaining)))
    assert sa == 0
    improve, _keep = _ukeire(remaining, _FULL, sa)
    assert improve == 8  # 一万 4 枚 + 三万 4 枚
    avail = dict(_FULL)
    avail[3] = 0
    improve0, _ = _ukeire(remaining, avail, sa)
    assert improve0 == 4  # 三万全见 → 只剩一万 4 枚


def test_meld_expansion_and_exclude():
    """副露展开评估 + 副露牌排除(与旧接口同语义)。"""
    # 手牌 11 张(碰 9 后) + 碰展开 3 张 = 14 评估; 打浮张 5
    tiles = [1, 2, 5, 18, 19, 20, 27, 27, 27, 30, 30]
    rec = _rec(tiles, melds=[[9, 9, 9]])
    assert rec.tile == 5
    # 副露牌混入手牌时(旧口径: 不展开, 只排除 9 不能打)
    rec2 = _rec(tiles + [9, 9, 9], exclude={9})
    assert rec2.tile != 9
