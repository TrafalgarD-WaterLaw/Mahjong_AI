"""欢乐推倒胡规则测试。"""
from src.mahjong_core import Hand
from src.mahjong_core.tile import B4, B5, B6, DONG, T7, T8, T9, W1, W2, W3, W4
from src.mahjong_engine import get, list_names
from src.mahjong_engine.rules.huanyu_rules import HuanyuRules


class TestHuanyuRules:
    def setup_method(self):
        self.rules = HuanyuRules()

    def test_name(self):
        assert self.rules.name() == 'huanyu'

    def test_registered(self):
        assert 'huanyu' in list_names()
        assert get('huanyu').name() == 'huanyu'

    def test_can_chi_as_tail(self):
        h = Hand([W1, W2])
        combos = self.rules.can_chi(h, W3)
        assert [W1, W2, W3] in combos

    def test_can_chi_as_middle(self):
        h = Hand([W1, W3])
        combos = self.rules.can_chi(h, W2)
        assert [W1, W2, W3] in combos

    def test_can_chi_not_allowed_for_honor(self):
        h = Hand([W1, W2])
        assert self.rules.can_chi(h, DONG) == []

    def test_shaanxi_rules_inherited(self):
        # 136张/七对/十三幺与陕西一致
        assert self.rules.get_tile_set().total_tiles == 136
        h = Hand([W1, W2, W3, B4, B5, B6, T7, T8, T9, W2, W3, W4, W4])
        assert self.rules.is_winning_hand(h, new_tile=W4).can_win is True
