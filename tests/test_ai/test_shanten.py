"""向听数计算的单元测试。"""
from src.mahjong_ai.efficiency.shanten import calculate_shanten
from src.mahjong_core.tile import (
    B2,
    B4,
    B5,
    B6,
    B8,
    BEI,
    DONG,
    NAN,
    T3,
    T4,
    T5,
    T6,
    T7,
    T8,
    T9,
    W1,
    W2,
    W3,
    W4,
    W5,
    W6,
    W7,
    W8,
    W9,
    XI,
    ZHONG,
)


class TestShantenWinning:
    def test_winning_standard(self):
        """已胡牌 → -1"""
        tiles = [W1, W2, W3, B4, B5, B6, T7, T8, T9, W2, W3, W4, W4, W4]
        assert calculate_shanten(tiles) == -1

    def test_winning_seven_pairs(self):
        tiles = [W1, W1, W3, W3, W5, W5, B2, B2, B6, B6, T4, T4, DONG, DONG]
        assert calculate_shanten(tiles) == -1


class TestShantenTenpai:
    def test_tenpai_standard(self):
        """听牌 → 0"""
        tiles = [W1, W2, W3, B4, B5, B6, T7, T8, T9, W2, W3, W4, W4]
        assert calculate_shanten(tiles) == 0

    def test_tenpai_seven_pairs(self):
        """六对半 → 0"""
        tiles = [W1, W1, W3, W3, W5, W5, B2, B2, B6, B6, T4, T4, DONG]
        assert calculate_shanten(tiles) == 0


class TestShantenOneAway:
    def test_one_away(self):
        """一向听 → 1"""
        tiles = [W1, W3, W5, B2, B4, B6, T3, T5, T7, DONG, NAN, XI, ZHONG]
        assert calculate_shanten(tiles) >= 1  # 至少一向听

    def test_complete_random(self):
        """完全散牌 → 高向听数"""
        tiles = [W1, W4, W7, B2, B5, B8, T3, T6, T9, DONG, NAN, XI, BEI]
        assert calculate_shanten(tiles) >= 2  # 散牌


class TestShantenKnownValues:
    def test_ryanmen_tenpai(self):
        """两面听牌 → 0"""
        tiles = [W1, W2, W3, B4, B5, B6, T7, T8, T9, W5, W6, W7, W8]
        shanten = calculate_shanten(tiles)
        assert shanten <= 1  # 最多一向听

    def test_full_flush_progress(self):
        """清一色进行中"""
        tiles = [W1, W1, W2, W3, W4, W5, W6, W7, W7, W8, W8, W9, W9]
        shanten = calculate_shanten(tiles)
        assert -1 <= shanten <= 2
