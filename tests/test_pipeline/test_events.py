"""事件流数据类单测。"""

from src.mahjong_ai.pipeline.events import (
    HandChanged,
    MeldFormed,
    TileAppeared,
    TileClaimed,
    TileVanished,
    event_to_dict,
)


def test_tile_vanished_serializes():
    ev = TileVanished(eid=7, tile=8, river='right_river',
                      appeared_eid=3, frame=101, ts=12.5)
    d = event_to_dict(ev)
    assert d['type'] == 'TileVanished'
    assert d['eid'] == 7 and d['tile'] == 8
    assert d['river'] == 'right_river'
    assert d['appeared_eid'] == 3


def test_meld_formed_player_defaults_none():
    ev = MeldFormed(eid=1, tiles=(8, 8, 8), cx=0.0, cy=0.0,
                    bbox=(0.0, 0.0, 0.0, 0.0), frame=1, ts=0.1)
    assert ev.player is None
    d = event_to_dict(ev)
    assert d['player'] is None


def test_meld_formed_player_carried():
    ev = MeldFormed(eid=1, tiles=(8, 8, 8), cx=0.0, cy=0.0,
                    bbox=(0.0, 0.0, 0.0, 0.0), frame=1, ts=0.1,
                    player='my_river')
    assert event_to_dict(ev)['player'] == 'my_river'


def test_event_to_dict_includes_type():
    ev = TileAppeared(eid=1, tile=5, track_id=101, cx=700.0, cy=400.0,
                      conf=0.7, frame=10, ts=1.0)
    d = event_to_dict(ev)
    assert d['type'] == 'TileAppeared'
    assert d['eid'] == 1 and d['tile'] == 5 and d['cx'] == 700.0


def test_claimed_second_pong_supported():
    """秒碰: claimed=None(该牌从未作为 TileAppeared 出现)。"""
    meld = MeldFormed(eid=2, tiles=(5, 5), cx=300.0, cy=500.0,
                      bbox=(280, 480, 360, 560), frame=20, ts=2.0)
    claim = TileClaimed(eid=3, claimed=None, meld=2, frame=20, ts=2.0)
    assert claim.claimed is None and claim.meld == meld.eid


def test_hand_changed_draw_semantics():
    draw = HandChanged(eid=4, n_old=13, n_new=14, frame=30, ts=3.0)
    discard = HandChanged(eid=5, n_old=14, n_new=13, frame=31, ts=3.1)
    assert (draw.n_old, draw.n_new) == (13, 14)
    assert (discard.n_old, discard.n_new) == (14, 13)
