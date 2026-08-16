"""对手副露内容推断 — 牌池约束收窄碰/吃组合。"""

from collections import Counter

from src.mahjong_ai.pipeline.meld_resolver import resolve_opponent_meld

_pool = Counter


def test_pong_unique_when_chi_impossible():
    """池中 X ≥ 2 且顺子被池挡住 → 唯一解 = 碰。"""
    # 8(九万): 顺子只有 (6,7,8); 6 在池中为 0 → 吃不可能
    combos = resolve_opponent_meld(8, _pool({8: 3, 6: 0, 7: 2}))
    assert combos == [(8, 8, 8)]


def test_chi_unique_when_pong_impossible():
    """池中 X = 1(碰不可能)且只有一条顺子可行 → 唯一解 = 吃。"""
    combos = resolve_opponent_meld(8, _pool({8: 1, 6: 2, 7: 2}))
    assert combos == [(6, 7, 8)]


def test_ambiguous_when_both_possible():
    """碰与吃都可行 → 多解(歧义, 调用方只扣确定部分)。"""
    combos = resolve_opponent_meld(8, _pool({8: 3, 6: 2, 7: 2}))
    assert combos == [(8, 8, 8), (6, 7, 8)]


def test_mid_tile_two_chis():
    """被碰 7(八万): 顺子 (5,6,7) 和 (6,7,8) 都可行 → 歧义。"""
    combos = resolve_opponent_meld(7, _pool({7: 0, 5: 2, 6: 2, 8: 2}))
    assert (5, 6, 7) in combos
    assert (6, 7, 8) in combos


def test_edge_tiles():
    """边界牌: 0(一万)只可能 (0,1,2); 9(一饼)只可能 (9,10,11)。"""
    assert resolve_opponent_meld(0, _pool({0: 0, 1: 1, 2: 1})) == [(0, 1, 2)]
    assert resolve_opponent_meld(9, _pool({9: 0, 10: 1, 11: 1})) == [(9, 10, 11)]


def test_honor_tile_pong_only():
    """字牌(27 东): 无吃 → 池够 2 张 = 碰, 否则无解。"""
    assert resolve_opponent_meld(27, _pool({27: 2})) == [(27, 27, 27)]
    assert resolve_opponent_meld(27, _pool({27: 1})) == []


def test_no_solution_when_counts_inconsistent():
    """检测漏导致计数不一致 → 空(调用方不采用, 按未知处理)。"""
    assert resolve_opponent_meld(8, _pool({8: 1, 6: 0, 7: 0})) == []
