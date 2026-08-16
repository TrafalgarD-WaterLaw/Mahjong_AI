"""tile.py 辅助函数的单元测试。"""
import pytest

from src.mahjong_core.tile import (
    B1,
    B9,
    BAI,
    BEI,
    BING,
    BING_TILES,
    DONG,
    FA,
    FENG,
    FENG_TILES,
    HUA,
    HUA_TILES,
    JIAN,
    JIAN_TILES,
    NAN,
    T1,
    T9,
    TIAO,
    TIAO_TILES,
    TOTAL_TILES,
    W1,
    W9,
    WAN,
    WAN_TILES,
    XI,
    ZHONG,
    tile_display,
    tile_from_str,
    tile_number,
    tile_suit,
)


class TestTileSuit:
    def test_all_suits(self):
        for t in WAN_TILES:
            assert tile_suit(t) == WAN
        for t in BING_TILES:
            assert tile_suit(t) == BING
        for t in TIAO_TILES:
            assert tile_suit(t) == TIAO
        for t in FENG_TILES:
            assert tile_suit(t) == FENG
        for t in JIAN_TILES:
            assert tile_suit(t) == JIAN
        for t in HUA_TILES:
            assert tile_suit(t) == HUA

    def test_invalid_raises(self):
        for v in [-1, 42, 100]:
            with pytest.raises(ValueError):
                tile_suit(v)


class TestTileNumber:
    def test_numeric_tiles(self):
        assert tile_number(W1) == 1
        assert tile_number(W9) == 9
        assert tile_number(B1) == 1
        assert tile_number(B9) == 9
        assert tile_number(T1) == 1
        assert tile_number(T9) == 9

    def test_honor_and_hua_return_none(self):
        for t in list(FENG_TILES) + list(JIAN_TILES) + list(HUA_TILES):
            assert tile_number(t) is None


class TestTileDisplay:
    def test_numbered_tiles(self):
        assert tile_display(W1) == '一万'
        assert tile_display(W9) == '九万'
        assert tile_display(B1) == '一饼'
        assert tile_display(T1) == '一条'

    def test_honor_tiles(self):
        assert tile_display(DONG) == '东'
        assert tile_display(NAN) == '南'
        assert tile_display(XI) == '西'
        assert tile_display(BEI) == '北'
        assert tile_display(ZHONG) == '中'
        assert tile_display(FA) == '发'
        assert tile_display(BAI) == '白'

    def test_hua_display(self):
        # 花牌 8 种统一为 1 类(2026-08-09 合并), 35-41 不再是合法类
        assert tile_display(34) == "花"
        with pytest.raises(ValueError):
            tile_display(35)


class TestTileFromStr:
    def test_roundtrip_all_standard(self):
        for t in range(0, 34):
            assert tile_from_str(tile_display(t)) == t

    def test_invalid_raises(self):
        for s in ['', '不是牌', '十萬']:
            with pytest.raises(ValueError):
                tile_from_str(s)


class TestConstants:
    def test_total(self):
        assert TOTAL_TILES == 35

    def test_groups_disjoint(self):
        all_tiles = (list(WAN_TILES) + list(BING_TILES) + list(TIAO_TILES) +
                     list(FENG_TILES) + list(JIAN_TILES) + list(HUA_TILES))
        assert len(all_tiles) == 35
        assert len(set(all_tiles)) == 35
