"""听牌分析的单元测试。"""
from src.mahjong_core.config import TileSetConfig
from src.mahjong_core.tile import (
    B2,
    B4,
    B5,
    B6,
    DONG,
    NAN,
    T3,
    T4,
    T5,
    T7,
    T8,
    T9,
    W1,
    W2,
    W3,
    W4,
    W5,
    W6,
    XI,
    ZHONG,
)
from src.mahjong_engine.judges.tenpai_judge import get_waiting_tiles

_ENABLED = TileSetConfig.standard_136().enabled_tiles


def test_multi_sided_wait():
    """边张+单钓: 123万456筒789条+3456万 → 听6万"""
    tiles = [W1, W2, W3, B4, B5, B6, T7, T8, T9, W3, W4, W5, W6]
    result = get_waiting_tiles(tiles, _ENABLED)
    waited = {wt.tile for wt in result}
    # 3456万+6万 → 345万+66万 胡牌
    assert W6 in waited

def test_double_sided_wait():
    """两面听: 123万456筒789条+45万万 → 听3/6万"""
    tiles = [W1, W2, W3, B4, B5, B6, T7, T8, T9, W4, W5, W6, W6]
    result = get_waiting_tiles(tiles, _ENABLED)
    waited = {wt.tile for wt in result}
    # 4566万+3万 → 345+66万; +6万 → 456+66万
    assert W3 in waited
    assert W6 in waited


def test_pair_wait():
    """单钓: 4面子齐+单张 → 听该单张"""
    tiles = [W1, W2, W3, B4, B5, B6, T7, T8, T9, DONG, DONG, DONG, ZHONG]
    result = get_waiting_tiles(tiles, _ENABLED)
    waited = {wt.tile for wt in result}
    assert ZHONG in waited


def test_no_wait():
    """不听牌"""
    tiles = [W1, W3, W5, B2, B4, B6, T3, T5, T7, DONG, NAN, XI, ZHONG]
    result = get_waiting_tiles(tiles, _ENABLED)
    assert len(result) == 0


def test_seven_pairs_wait():
    """6对+1单 → 七对听"""
    tiles = [W1, W1, W3, W3, W5, W5, B2, B2, B6, B6, T4, T4, DONG]
    result = get_waiting_tiles(tiles, _ENABLED)
    waited = {wt.tile for wt in result}
    assert DONG in waited
