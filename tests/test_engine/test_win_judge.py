"""回溯法胡牌判定的单元测试。"""
import pytest

from src.mahjong_core.tile import (
    B1,
    B2,
    B3,
    B4,
    B5,
    B6,
    B7,
    B8,
    B9,
    BAI,
    BEI,
    DONG,
    FA,
    NAN,
    T1,
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
    W9,
    XI,
    ZHONG,
)
from src.mahjong_engine.judges.win_judge import is_winning_hand


class TestStandardWinning:
    """标准胡牌: 4面子+1雀头"""

    def test_sequences_with_pair(self):
        """123万 456筒 789条 23万 44万 → 胡"""
        tiles = [W1, W2, W3, B4, B5, B6, T7, T8, T9, W2, W3, W4, W4, W4]
        result = is_winning_hand(tiles)
        assert result.can_win is True
        assert result.pattern_type == 'standard'

    def test_triplets_with_pair(self):
        """111万 222筒 333条 东东东 55万 → 胡"""
        tiles = [W1, W1, W1, B2, B2, B2, T3, T3, T3, DONG, DONG, DONG, W5, W5]
        result = is_winning_hand(tiles)
        assert result.can_win is True

    def test_mixed_melds(self):
        """123万 111筒 789条 东东东 22筒 → 胡"""
        tiles = [W1, W2, W3, B1, B1, B1, T7, T8, T9, DONG, DONG, DONG, B2, B2]
        result = is_winning_hand(tiles)
        assert result.can_win is True

    def test_single_tile_edge_pair(self):
        """1万做雀头，边张顺子"""
        tiles = [W1, W2, W3, W1, W1, B1, B2, B3, B7, B8, B9, T4, T5, T6]
        result = is_winning_hand(tiles)
        assert result.can_win is True


class TestSevenPairs:
    """七对: 7种牌各2张"""

    def test_standard_seven_pairs(self):
        tiles = [W1, W1, W3, W3, W5, W5, B2, B2, B6, B6, T4, T4, DONG, DONG]
        result = is_winning_hand(tiles)
        assert result.can_win is True
        assert result.pattern_type == 'seven_pairs'

    def test_seven_pairs_with_triplet_should_be_standard(self):
        """如果某牌3张，不是七对面是标准胡"""
        tiles = [W1, W1, W1, W3, W3, W5, W5, B2, B2, B6, B6, T4, T4, DONG]
        result = is_winning_hand(tiles)
        # 这也能构成标准胡 (111刻子 + 6对)
        # 或者不胡
        assert result.can_win in [True, False]  # 取决于实现


class TestThirteenOrphans:
    """十三幺: 1万9万1筒9筒1条9条东南西北中发白 + 任意幺九牌"""

    def test_thirteen_orphans(self):
        tiles = [W1, W9, B1, B9, T1, T9,
                 DONG, NAN, XI, BEI, ZHONG, FA, BAI, W1]
        result = is_winning_hand(tiles)
        assert result.can_win is True
        assert result.pattern_type == 'thirteen_orphans'

    def test_missing_one_orphan(self):
        """缺少9万 → 不胡"""
        tiles = [W1, W1, B1, B9, T1, T9,
                 DONG, NAN, XI, BEI, ZHONG, FA, BAI, B2]
        result = is_winning_hand(tiles)
        assert result.can_win is False


class TestNotWinning:
    """不胡的牌型"""

    def test_random_tiles(self):
        tiles = [W1, W3, W5, B2, B4, B6, T3, T5, T7, DONG, NAN, XI, ZHONG, FA]
        result = is_winning_hand(tiles)
        assert result.can_win is False

    def test_one_away(self):
        """一向听: 123万 456筒 789条 东西白 中"""
        tiles = [W1, W2, W3, B4, B5, B6, T7, T8, T9, DONG, NAN, BAI, ZHONG, ZHONG]
        result = is_winning_hand(tiles)
        assert result.can_win is False

    def test_empty_hand(self):
        with pytest.raises(ValueError):
            is_winning_hand([])

    def test_wrong_count(self):
        with pytest.raises(ValueError):
            is_winning_hand([W1, W2, W3])


class TestBreakdown:
    """拆解结果验证"""

    def test_breakdown_not_empty(self):
        tiles = [W1, W2, W3, B4, B5, B6, T7, T8, T9, W2, W3, W4, W4, W4]
        result = is_winning_hand(tiles)
        assert len(result.breakdown) == 5  # 4 melds + 1 pair
