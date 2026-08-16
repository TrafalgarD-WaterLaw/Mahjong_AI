"""结算核对纯函数测试: 亮牌 vs 推断的核对逻辑/阶段曲线/累计统计。"""

from scripts.settle_check import (
    bucket_of,
    tenpai_curve,
    update_stats,
    verify_settle,
)
from src.mahjong_core.tile import (
    B1,
    B4,
    B5,
    B6,
    B7,
    B9,
    W1,
    W2,
    W3,
    W4,
    W5,
    W6,
    W7,
    W8,
    W9,
)
from src.mahjong_engine.rules.huanyu_rules import HuanyuRules

RULES = HuanyuRules()

#: 123456789万 + 456饼 + 9饼 = 听 9 饼(13 张)
TENPAI_HAND = [W1, W2, W3, W4, W5, W6, W7, W8, W9, B4, B5, B6, B9]


class TestVerifySettle:
    def test_top3_hit_when_predicted_includes_actual(self):
        result = verify_settle([B9, W5, B1], 0.6, TENPAI_HAND, RULES)
        assert result['actual_tenpai'] is True
        assert result['actual_waiting'] == [B9]
        assert result['top3_hit'] is True
        assert result['bucket'] == '60%-80%'  # 0.6 归下界桶

    def test_top3_miss(self):
        result = verify_settle([W5, W6, B1], 0.4, TENPAI_HAND, RULES)
        assert result['top3_hit'] is False
        assert result['actual_waiting'] == [B9]

    def test_requires_13_tiles(self):
        try:
            verify_settle([], 0.0, TENPAI_HAND + [B1], RULES)
            raise AssertionError('应拒绝 14 张亮牌')
        except ValueError:
            pass

    def test_no_tenpai_hand(self):
        # 完全散乱 13 张: 实际未听牌
        scattered = [W1, W4, W7, B1, B4, B7, 18, 21, 24, 27, 28, 29, 30]
        result = verify_settle([B9], 0.5, scattered, RULES)
        assert result['actual_tenpai'] is False
        assert result['top3_hit'] is False


class TestBucketOf:
    def test_edges(self):
        assert bucket_of(0.0) == '0%-20%'
        assert bucket_of(0.19) == '0%-20%'
        assert bucket_of(0.2) == '20%-40%'
        assert bucket_of(0.99) == '80%-100%'
        assert bucket_of(1.0) == '80%-100%'


class TestTenpaiCurve:
    def test_curve_by_events(self):
        records = [
            {'events_total': 5, 'tenpai': {'right_river': 0.05}},
            {'events_total': 12, 'tenpai': {'right_river': 0.3}},
            {'events_total': 20, 'tenpai': {'right_river': 0.6}},
        ]
        curve = tenpai_curve('right_river', records)
        assert curve == [(5, 0.05), (12, 0.3), (20, 0.6)]

    def test_missing_player_zero(self):
        records = [{'events_total': 3, 'tenpai': {}}]
        assert tenpai_curve('top_river', records) == [(3, 0.0)]


class TestUpdateStats:
    def test_accumulates_buckets(self):
        stats = {}
        stats = update_stats(stats, verify_settle([B9], 0.5, TENPAI_HAND, RULES))
        stats = update_stats(stats, verify_settle([W5], 0.1, TENPAI_HAND, RULES))
        assert stats['games'] == 2
        assert stats['top3_hit_n'] == 1  # 只有第一局 top3 含 B9
        assert stats['tenpai_n'] == 2    # 两局实际都听牌
        b1 = stats['buckets']['40%-60%']
        assert b1 == {'n': 1, 'tenpai': 1}
        b2 = stats['buckets']['0%-20%']
        assert b2 == {'n': 1, 'tenpai': 1}
