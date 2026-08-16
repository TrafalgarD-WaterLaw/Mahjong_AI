"""评估脚本的纯函数单测(模拟器辅助)。"""

from collections import Counter

from scripts.eval_inference import _meld_improves, claim_candidates


def test_claim_candidates_priority():
    """杠 > 碰 > 吃; 无候选返回 None。"""
    assert claim_candidates(Counter([8, 8, 8]), 8) == (8, 8, 8, 8)
    assert claim_candidates(Counter([8, 8]), 8) == (8, 8, 8)
    assert claim_candidates(Counter([6, 7]), 8) == (6, 7, 8)
    assert claim_candidates(Counter([6, 8]), 7) == (6, 7, 8)
    assert claim_candidates(Counter([5, 6]), 8) is None
    assert claim_candidates(Counter([27, 27, 28]), 27) == (27, 27, 27)
    assert claim_candidates(Counter([27, 28]), 27) is None  # 单张不能碰


def test_meld_improves_shanten():
    """副露决策: 已近听(向听 ≤1)不碰; 散牌阶段(向听 ≥2)才考虑。"""
    # 已听(3 面子 + 两对, 向听 0) → 不碰
    hand = [0, 1, 2, 9, 10, 11, 18, 19, 20, 8, 8, 27, 27]
    assert not _meld_improves(Counter(hand), (8, 8, 8))
    # 散牌(向听 3) → 可碰
    assert _meld_improves(Counter([8, 8, 1, 2, 4, 9, 18, 27, 30, 33, 5, 6, 7]),
                          (8, 8, 8))
