"""Hand 手牌类的单元测试。"""
import pytest

from src.mahjong_core.hand import Hand
from src.mahjong_core.tile import B1, B9, DONG, W1, W2, W3, ZHONG


class TestHandInit:
    def test_empty_hand(self):
        h = Hand()
        assert len(h) == 0
        assert list(h) == []

    def test_hand_sorts_on_init(self):
        h = Hand([W3, W1, W2])
        assert list(h) == [W1, W2, W3]

    def test_hand_with_duplicates(self):
        h = Hand([W1, W1, W2])
        assert list(h) == [W1, W1, W2]


class TestHandAdd:
    def test_add_maintains_sort(self):
        h = Hand([W1, W3])
        h.add(W2)
        assert list(h) == [W1, W2, W3]

    def test_add_to_empty(self):
        h = Hand()
        h.add(W1)
        assert len(h) == 1

    def test_add_duplicate(self):
        h = Hand([W1, W2])
        h.add(W1)
        assert list(h) == [W1, W1, W2]


class TestHandDiscard:
    def test_discard_removes_one(self):
        h = Hand([W1, W1, W2])
        h.discard(W1)
        assert list(h) == [W1, W2]

    def test_discard_not_present_raises(self):
        h = Hand([W1, W2])
        with pytest.raises(ValueError, match='not in hand'):
            h.discard(W3)

    def test_discard_last_occurrence(self):
        h = Hand([W1])
        h.discard(W1)
        assert len(h) == 0


class TestHandCount:
    def test_count(self):
        h = Hand([W1, W1, W2])
        assert h.count(W1) == 2
        assert h.count(W2) == 1
        assert h.count(W3) == 0


class TestHandDunder:
    def test_len(self):
        assert len(Hand([W1, W2, W3, B1, B9])) == 5

    def test_getitem(self):
        h = Hand([W2, W1])
        assert h[0] == W1
        assert h[1] == W2
        with pytest.raises(IndexError):
            _ = h[5]

    def test_iter(self):
        assert list(Hand([W3, W1, W2])) == [W1, W2, W3]

    def test_contains(self):
        h = Hand([W1, W2])
        assert W1 in h
        assert W3 not in h

    def test_repr(self):
        assert 'Hand' in repr(Hand([W1, DONG]))


class TestHandEdgeCases:
    def test_14_tiles(self):
        assert len(Hand([W1] * 14)) == 14

    def test_large_hand_sort(self):
        from src.mahjong_core.tile import B1 as t1
        from src.mahjong_core.tile import B9 as t9
        from src.mahjong_core.tile import W1 as w1
        from src.mahjong_core.tile import W9 as w9
        h = Hand([t9, t1, w9, w1, DONG, ZHONG])
        assert list(h) == [w1, w9, t1, t9, DONG, ZHONG]
