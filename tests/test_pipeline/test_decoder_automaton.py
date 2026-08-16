"""轮转自动机单测 — 跳家惩罚/副露截断/整轮重置。"""

from src.mahjong_ai.pipeline.decoder import (
    TurnPhase,
    next_in_rotation,
    phase_prior,
)


def test_rotation_order():
    assert next_in_rotation('my_river') == 'right_river'
    assert next_in_rotation('right_river') == 'top_river'
    assert next_in_rotation('top_river') == 'left_river'
    assert next_in_rotation('left_river') == 'my_river'


def test_advance_in_turn_no_penalty():
    ph = TurnPhase(actor='my_river', played=frozenset())
    nph, penalty = ph.advance('my_river')
    assert penalty == 0
    assert nph.actor == 'right_river'
    assert nph.played == frozenset({'my_river'})


def test_advance_skip_penalty():
    """actor=我 但 top 打出 → 跳过 right(其打出漏检), 罚 1×跳家惩罚。"""
    ph = TurnPhase(actor='my_river', played=frozenset())
    nph, penalty = ph.advance('top_river')
    assert penalty > 0
    assert nph.actor == 'left_river'
    assert nph.played == frozenset({'right_river', 'top_river'})


def test_meld_interrupt_keeps_unplayed():
    """碰截断: top 碰(right 还没打)→ top 接着打; right 本轮失去行动权。"""
    ph = TurnPhase(actor='right_river',
                   played=frozenset({'my_river'}))
    nph, penalty = ph.claim('top_river')
    assert penalty == 0            # 碰是合法截断, 不罚
    assert nph.actor == 'top_river'  # 碰家接着打
    assert nph.played == frozenset({'my_river', 'right_river', 'top_river'})
    # top 接着打出: 无惩罚, 轮转从 top 继续
    nph2, penalty2 = nph.advance('top_river')
    assert penalty2 == 0
    assert nph2.actor == 'left_river'


def test_claim_by_next_player_no_skip():
    """我打出后下家(右)直接碰: actor 已是右, 无跳家。"""
    ph = TurnPhase(actor='right_river',
                   played=frozenset({'my_river'}))
    nph, penalty = ph.claim('right_river')
    assert penalty == 0
    assert nph.actor == 'right_river'
    assert nph.played == frozenset({'my_river', 'right_river'})


def test_round_resets_after_all_four():
    ph = TurnPhase(actor='my_river', played=frozenset())
    for p in ('my_river', 'right_river', 'top_river'):
        ph, _ = ph.advance(p)
    assert len(ph.played) == 3
    ph, _ = ph.advance('left_river')
    assert ph.played == frozenset()  # 整轮重置
    assert ph.actor == 'my_river'


def test_phase_prior_favors_actor():
    ph = TurnPhase(actor='top_river', played=frozenset())
    assert phase_prior(ph, 'top_river') > phase_prior(ph, 'my_river')
    assert phase_prior(ph, 'my_river') == phase_prior(ph, 'right_river')
