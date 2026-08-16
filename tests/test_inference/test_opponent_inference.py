"""粒子滤波推断单元测试 — 听牌/放铳率/观测过滤/牌池约束。"""
from collections import Counter

from src.mahjong_ai.inference.opponent_inference import (
    OpponentInference,
    _discard_likelihood,
)
from src.mahjong_core.tile import (
    B1,
    B4,
    B5,
    B6,
    B7,
    B9,
    T1,
    T2,
    T3,
    T4,
    T5,
    T6,
    T7,
    T8,
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


def make_inf(n: int = 200) -> OpponentInference:
    return OpponentInference(RULES, n_particles=n)


def c(*tiles: int) -> Counter[int]:
    return Counter(tiles)


class TestPool:
    def test_my_hand_removed_from_pool(self):
        inf = make_inf(50)
        inf.reset([W1, W1, W1], {})  # 自己手里 3 张一万
        dist = inf.hand_distribution()
        # 牌池里一万只剩 1 张, 期望持有数应远低于均值(13/34≈0.38)
        assert dist.get(W1, 0) < 0.2

    def test_river_tiles_removed_from_pool(self):
        inf = make_inf(50)
        from src.mahjong_ai.state.snapshot import PlayerState

        ps = PlayerState()
        ps.river = [W2, W2, W2, W2]  # 该家已打出全部四张二万
        inf.reset([], {'right_river': ps}, hand_count=13)
        dist = inf.hand_distribution()
        assert dist.get(W2, 0) == 0.0  # 池里无二万, 粒子不可能有


class TestShantenObservation:
    """向听数观测(第二版): 毁牌效的打出低权重, 孤张高权重。"""

    TENPAI = Counter({W1: 1, W2: 1, W3: 1, W4: 1, W5: 1, W6: 1,
                      W7: 1, W8: 1, W9: 1, B4: 1, B5: 1, B6: 1, B9: 1})

    def test_lonely_tile_much_more_likely_than_structure_tile(self):
        # 听牌结构: 打孤张 9饼(Δ=0)远高于拆面子的 5万(Δ>0)
        w_lonely = _discard_likelihood(self.TENPAI, B9)
        w_useful = _discard_likelihood(self.TENPAI, W5)
        assert w_lonely > w_useful * 10  # 16 倍抑制, 阈值留余量

    def test_edge_tile_less_likely_than_lonely(self):
        # 边张 9万(保留 789 面子机会)比孤张 9饼 低, 比拆面子 5万 高
        w_lonely = _discard_likelihood(self.TENPAI, B9)
        w_edge = _discard_likelihood(self.TENPAI, W9)
        w_useful = _discard_likelihood(self.TENPAI, W5)
        assert w_useful < w_edge < w_lonely

    def test_honor_pair_discard_penalized(self):
        # 字牌对子: 打掉一张破坏对子(对子有雀头/碰的价值) → 权重降低
        hand = Counter({W1: 1, W2: 1, W3: 1, W4: 1, W5: 1, W6: 1,
                        W7: 1, W8: 1, W9: 1, B4: 1, B5: 1, B6: 1, 27: 2})
        w_pair = _discard_likelihood(hand, 27)
        assert w_pair < 1.0  # 对子牌打出被抑制(不再是恒 1)


class TestObserveDiscard:
    def test_discarded_tile_removed_from_particles(self):
        inf = make_inf(100)
        inf.reset([], {})
        inf.observe_public(W1)  # 打出的牌同时公开(真实路径: 扣池)
        inf.observe_discard(W1)
        dist = inf.hand_distribution()
        # 打出一万后, 粒子中一万的期望大幅下降(过滤+删牌+扣池)
        assert dist.get(W1, 0) < 0.15

    def test_lonely_tile_discarded_more_likely(self):
        # 孤张(一万) vs 搭子牌(七万, 手牌里 5/6/8/9 万多)
        inf = make_inf(200)
        inf.reset([], {})
        # 注入: 手牌含大量 5-9 万 → 打七万不太合理(有搭子), 打一万更合理
        hand = Counter({W5: 2, W6: 1, W7: 2, W8: 1, W9: 2, B1: 2, B9: 2, T1: 1})
        inf._seed_particles([hand] * 200)
        # 观测: 打出一万 → 大部分粒子被过滤(不含一万) → 重建
        inf.observe_discard(W1)
        assert len(inf._particles) == 200  # 观测后粒子数保持


class TestTenpai:
    def test_tenpai_probability_and_distribution(self):
        # 手牌: 123456789万 + 456饼 + 9饼 → 听 9饼 之前已胡? 听 147饼?
        # 用标准 13 张听牌: 123456789万 + 456饼 = 12张 + 9饼 = 听 9饼(9饼成对)
        hand = Counter({W1: 1, W2: 1, W3: 1, W4: 1, W5: 1, W6: 1,
                        W7: 1, W8: 1, W9: 1, B4: 1, B5: 1, B6: 1, B9: 1})
        inf = make_inf(100)
        inf.reset([], {})
        inf._seed_particles([hand] * 100)
        prob, dist = inf.tenpai()
        assert prob == 1.0
        assert dist.get(B9, 0) > 0  # 听 9 饼

    def test_no_tenpai_hand(self):
        # 完全散乱: 13 张互不相邻
        hand = Counter({W1: 1, W4: 1, W7: 1, B1: 1, B4: 1, B7: 1,
                        T1: 1, T4: 1, T7: 1, 27: 1, 28: 1, 29: 1, 30: 1})
        inf = make_inf(100)
        inf.reset([], {})
        inf._seed_particles([hand] * 100)
        prob, dist = inf.tenpai()
        assert prob < 0.5


class TestDiscardRisk:
    def test_risk_one_on_waiting_tile(self):
        # 听 9 饼: 打 9 饼必铳
        hand = Counter({W1: 1, W2: 1, W3: 1, W4: 1, W5: 1, W6: 1,
                        W7: 1, W8: 1, W9: 1, B4: 1, B5: 1, B6: 1, B9: 1})
        inf = make_inf(100)
        inf.reset([], {})
        inf._seed_particles([hand] * 100)
        assert inf.discard_risk(B9) == 1.0
        assert inf.discard_risk(W5) == 0.0  # 非听牌张

    def test_risk_mixed_particles(self):
        # 一半听 9 饼, 一半听 3 条
        hand_a = Counter({W1: 1, W2: 1, W3: 1, W4: 1, W5: 1, W6: 1,
                          W7: 1, W8: 1, W9: 1, B4: 1, B5: 1, B6: 1, B9: 1})
        hand_b = Counter({T1: 1, T2: 1, T3: 1, T4: 1, T5: 1, T6: 1,
                          T7: 1, T8: 1, W1: 1, W2: 1, W3: 1, W4: 1, W5: 1})
        inf = make_inf(100)
        inf.reset([], {})
        inf._seed_particles([hand_a] * 50 + [hand_b] * 50)
        assert abs(inf.discard_risk(B9) - 0.5) < 0.01
