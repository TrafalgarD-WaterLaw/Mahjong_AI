"""陕西推倒胡规则的集成测试。"""
from src.mahjong_core import Hand
from src.mahjong_core.tile import B4, B5, B6, DONG, T7, T8, T9, W1, W2, W3, W4
from src.mahjong_engine.rules.shaanxi_rules import ShaanxiRules


class TestShaanxiIntegration:
    def setup_method(self):
        self.rules = ShaanxiRules()

    def test_name(self):
        assert self.rules.name() == 'shaanxi'

    def test_tile_set_136(self):
        cfg = self.rules.get_tile_set()
        assert cfg.total_tiles == 136

    def test_winning_hand_with_new_tile(self):
        h = Hand([W1, W2, W3, B4, B5, B6, T7, T8, T9, W2, W3, W4, W4])
        result = self.rules.is_winning_hand(h, new_tile=W4)
        assert result.can_win is True

    def test_winning_hand_14_in_hand(self):
        h = Hand([W1, W2, W3, B4, B5, B6, T7, T8, T9, W2, W3, W4, W4, W4])
        result = self.rules.is_winning_hand(h)
        assert result.can_win is True

    def test_waiting_tiles(self):
        h = Hand([W1, W2, W3, B4, B5, B6, T7, T8, T9, DONG, DONG, DONG, W4])
        result = self.rules.get_waiting_tiles(h)
        assert len(result) > 0

    def test_can_pong(self):
        h = Hand([W1, W1, W2, W3])
        assert self.rules.can_pong(h, W1) is True

    def test_cannot_chi(self):
        h = Hand([W1, W3, W4])
        assert self.rules.can_chi(h, W2) == []

    def test_is_wild_false(self):
        assert self.rules.is_wild_tile(W1) is False
