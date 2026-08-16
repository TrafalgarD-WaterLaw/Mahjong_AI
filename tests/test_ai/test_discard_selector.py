"""出牌推荐的单元测试。"""
from src.mahjong_ai.efficiency.discard_selector import recommend_discard
from src.mahjong_ai.efficiency.tile_efficiency import evaluate_discard
from src.mahjong_core.config import TileSetConfig
from src.mahjong_core.tile import (
    B1,
    B2,
    B3,
    B4,
    B5,
    B6,
    B9,
    T1,
    T2,
    T3,
    T4,
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
    BAI,
    DONG,
    FA,
    NAN,
    XI,
    ZHONG,
)

_ENABLED = TileSetConfig.standard_136().enabled_tiles


class TestEvaluateDiscard:
    def test_returns_scores(self):
        tiles = [W1, W2, W3, B4, B5, B6, T7, T8, T9, W2, W3, W4, W4, W4]
        scores = evaluate_discard(tiles, _ENABLED)
        assert len(scores) >= 1

    def test_best_is_tenpai_or_better(self):
        """已经听牌的14张，打出后仍听牌"""
        tiles = [W1, W2, W3, B4, B5, B6, T7, T8, T9, W2, W3, W4, W4, W4]
        scores = evaluate_discard(tiles, _ENABLED)
        assert scores[0].shanten_after <= 0


class TestRecommendDiscard:
    def test_recommends_tile(self):
        tiles = [W1, W2, W3, B4, B5, B6, T7, T8, T9, W2, W3, W4, W4, ZHONG]
        result = recommend_discard(tiles, _ENABLED)
        assert result.tile >= 0  # 有推荐
        assert len(result.reason) > 0

    def test_isolated_honor_recommended_first(self):
        """孤立的字牌应该是首选打出"""
        tiles = [W1, W2, W3, B4, B5, B6, T7, T8, T9, W2, W3, W4, W4, ZHONG]
        result = recommend_discard(tiles, _ENABLED)
        # 中是最孤立(不与任何牌形成搭子)，应该被推荐
        assert result.tile == ZHONG

    def test_alternatives_provided(self):
        tiles = [W1, W2, W3, B4, B5, B6, T7, T8, T9, W2, W3, W4, W4, ZHONG]
        result = recommend_discard(tiles, _ENABLED)
        assert len(result.alternatives) >= 1

    def test_lonely_honor_beats_tile_with_connector(self):
        # 回归: 孤张西风 vs 与 B3 相邻的 B4 — 打出后向听相同,
        # 必须按孤张价值选西风(旧实现退化为手牌顺序, 可能不打孤张)
        tiles = [W1, W2, W3, W4, W5, W6, W7, W8, W9,
                 B1, B2, B3, B4, XI]
        result = recommend_discard(tiles, _ENABLED)
        assert result.tile == XI

    def test_pair_not_broken_when_lonely_exists(self):
        # 回归: 散牌 + 一对二万 + 五个孤张字牌 + 单张 W7
        # 必须优先打孤张字牌(牌效: 字牌永远无法成搭), 绝不拆对子
        tiles = [W2, W2, W4, W5, B3, B4, T4, T8,
                 DONG, NAN, ZHONG, FA, XI, W7]
        result = recommend_discard(tiles, _ENABLED)
        assert result.tile in (DONG, NAN, ZHONG, FA, XI)  # 孤张字牌之一

    def test_two_tiles_edge_discarded_first(self):
        # 两张牌也能判断: 1万-3万 嵌张搭 → 打边张(1万)留中张(3万)
        tiles = [W1, W3]
        result = recommend_discard(tiles, _ENABLED)
        assert result.tile == W1

    def test_two_lonely_tiles_edge_first(self):
        # 两张孤张: 1万(边) vs 5万(中) → 打 1 万
        tiles = [W1, W5]
        result = recommend_discard(tiles, _ENABLED)
        assert result.tile == W1

    def test_kanchan_discarded_before_ryanmen(self):
        # 回归(真实对局暴露): 9万-7万是嵌张(进8万), 6条-7条是两面(进5/8条)
        # 拆搭时先拆嵌张, 保留两面 — 旧 lonely 只数相邻张数, 会拆错
        tiles = [W5, W6, W6, W7, W7, W7, W9, B5, B5,
                 T6, T7, T7, T9, T9]
        result = recommend_discard(tiles, _ENABLED)
        assert result.tile == W9

    def test_wrong_count_raises(self):
        import pytest
        with pytest.raises(ValueError):
            recommend_discard([W1], _ENABLED)  # 1 张没有选择可言

    def test_meld_hand_recommends_discard(self):
        # 碰后: 手牌 11 张 + 碰(中中中) = 14 → 应推荐拆 45 饼搭(而不是副露牌)
        tiles = [W1, W2, W3, W4, W5, W6, W7, W8, W9, B4, B5]
        melds = [[ZHONG, ZHONG, ZHONG]]
        result = recommend_discard(tiles, _ENABLED, melds=melds)
        assert result.tile in (B4, B5)  # 拆两面保留三面+刻
        assert '副露 1 组' in result.reason

    def test_meld_tiles_never_recommended(self):
        # 副露牌(中)不是候选: 即使它看起来孤张
        tiles = [W1, W2, W3, W4, W5, W6, W7, W8, W9, B1, B2]
        melds = [[ZHONG, ZHONG, ZHONG]]
        result = recommend_discard(tiles, _ENABLED, melds=melds)
        assert result.tile != ZHONG

    def test_all_meld_tiles_no_candidate(self):
        # 检测到的牌全是副露牌(真手牌全漏检) → 无可打候选, 拒绝
        import pytest
        with pytest.raises(ValueError):
            recommend_discard([ZHONG, ZHONG], _ENABLED,
                              melds=[[ZHONG, ZHONG, ZHONG]])

    def test_meld_wrong_count_raises(self):
        import pytest
        # 碰1次应 11 张手牌, 给 12 张 → 合并 15 张 → 拒绝
        tiles = [W1, W2, W3, W4, W5, W6, W7, W8, W9, B1, B2, B3]
        with pytest.raises(ValueError):
            recommend_discard(tiles, _ENABLED, melds=[[ZHONG] * 3])
