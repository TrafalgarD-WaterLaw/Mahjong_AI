"""操作判定的单元测试。"""
from src.mahjong_core.tile import B1, B2, DONG, W1, W2, W3
from src.mahjong_engine.judges.action_judge import can_chi, can_kong, can_pong


class TestCanPong:
    def test_can_pong(self):
        assert can_pong([W1, W1, W2], W1) is True

    def test_cannot_pong(self):
        assert can_pong([W1, W2, W3], W1) is False

    def test_three_same(self):
        assert can_pong([W1, W1, W1], W1) is True


class TestCanKong:
    def test_an_kong(self):
        """暗杠: 手牌4张"""
        result = can_kong([W1, W1, W1, W1, W2], W1)
        assert result.can_kong is True
        assert result.kong_type == 'an'

    def test_ming_kong(self):
        """明杠: 手牌3张"""
        result = can_kong([W1, W1, W1, W2, W3], W1)
        assert result.can_kong is True
        assert result.kong_type == 'ming'

    def test_jia_kong(self):
        """加杠: 已碰的牌再摸到"""
        result = can_kong([W1, W2, W3], W1, ponged_tiles=[W1])
        assert result.can_kong is True
        assert result.kong_type == 'jia'

    def test_cannot_kong(self):
        result = can_kong([W1, W1, W2, W3], W1)
        assert result.can_kong is False


class TestCanChi:
    def test_chi_middle(self):
        """tile作顺子中位"""
        result = can_chi([W1, W3, B2], W2)
        assert len(result) == 1
        assert result[0] == [W1, W2, W3]

    def test_chi_start(self):
        """tile作顺子首位"""
        result = can_chi([W2, W3, B1], W1)
        assert len(result) == 1
        assert result[0] == [W1, W2, W3]

    def test_chi_end(self):
        """tile作顺子末位"""
        result = can_chi([W1, W2, B1], W3)
        assert len(result) == 1
        assert result[0] == [W1, W2, W3]

    def test_cannot_chi_honor(self):
        """字牌不能吃"""
        result = can_chi([W1, W2], DONG)
        assert len(result) == 0
