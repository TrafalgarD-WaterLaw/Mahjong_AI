"""状态投影单测 — 事件日志 → 玩家视图(可见/推断拆分)。"""

from src.mahjong_ai.pipeline.decoder import EvidenceModel, WindowDecoder
from src.mahjong_ai.pipeline.events import (
    MeldFormed,
    TileAppeared,
    TileClaimed,
)
from src.mahjong_ai.pipeline.state import project

_SEED = {
    'my_river': (600.0, 700.0, 800.0, 780.0),
    'right_river': (1100.0, 500.0, 1180.0, 600.0),
    'top_river': (500.0, 100.0, 700.0, 180.0),
    'left_river': (100.0, 500.0, 180.0, 600.0),
}


def _build() -> WindowDecoder:
    ev = EvidenceModel()
    ev.reseed(_SEED, 1280, 800)
    mev = EvidenceModel()
    mev.reseed(_SEED, 1280, 800)
    return WindowDecoder(ev, mev, k=1, beam=8)  # k=1: 逐张冻结


def test_river_visible_split_on_claim():
    """被碰走的牌: river 保留(打出历史), visible_river 排除(显示)。"""
    d = _build()
    d.add(TileAppeared(eid=1, tile=5, track_id=1001, cx=700.0, cy=740.0,
                       conf=0.8, frame=0, ts=0.0))
    d.add(TileAppeared(eid=2, tile=8, track_id=1002, cx=700.0, cy=742.0,
                       conf=0.8, frame=1, ts=0.1))
    d.add(TileClaimed(eid=50, claimed=2, meld=60, frame=2, ts=0.2))
    d.add(MeldFormed(eid=60, tiles=(8, 8, 8), cx=1150.0, cy=550.0,
                     bbox=(1100.0, 500.0, 1200.0, 600.0),
                     frame=2, ts=0.2))
    d.add(TileAppeared(eid=3, tile=9, track_id=1003, cx=700.0, cy=744.0,
                       conf=0.8, frame=3, ts=0.3))
    d.add(TileAppeared(eid=4, tile=2, track_id=1004, cx=1150.0, cy=550.0,
                       conf=0.8, frame=4, ts=0.4))
    d.add(TileAppeared(eid=5, tile=7, track_id=1005, cx=600.0, cy=140.0,
                       conf=0.8, frame=5, ts=0.5))  # 推 e4 出窗冻结
    view = project(d, my_hand=[1, 2, 3])
    me = view.players['my_river']
    assert me.river == [5, 8, 9]          # 打出历史含被碰走
    assert me.visible_river == [5, 9]     # 显示不含被碰走
    right = view.players['right_river']
    assert right.river == [2]
    assert right.melds and right.melds[0].kind == 'pong'
    assert right.melds[0].tiles == (8, 8, 8)
    assert view.my_hand == [1, 2, 3]


def test_kong_kind_from_multiplicity():
    """杠 tiles=(8,8,8,8)(多重度保留)→ kind == 'kong'。

    MeldFormed 落在 right 种子中心 → 归属右家; 追加 appeared 冻结
    (k=1)把副露硬更新消化进相位基底, 投影后断言 kind 按张数判定。
    """
    d = _build()
    d.add(MeldFormed(eid=60, tiles=(8, 8, 8, 8), cx=1150.0, cy=550.0,
                     bbox=(1100.0, 500.0, 1200.0, 600.0),
                     frame=0, ts=0.0))
    d.add(TileAppeared(eid=1, tile=5, track_id=1001, cx=1150.0, cy=552.0,
                       conf=0.8, frame=1, ts=0.1))
    d.add(TileAppeared(eid=2, tile=6, track_id=1002, cx=1150.0, cy=554.0,
                       conf=0.8, frame=2, ts=0.2))  # 推副露出窗冻结
    view = project(d, my_hand=[])
    assert view.players['right_river'].melds[0].kind == 'kong'
    assert view.players['right_river'].melds[0].tiles == (8, 8, 8, 8)


def test_chi_kind_distinct_tiles():
    """三张不同牌面 → chi(真实对局: 吃七万被判成碰七万的修复)。"""
    d = _build()
    d.add(MeldFormed(eid=60, tiles=(6, 7, 8), cx=1150.0, cy=550.0,
                     bbox=(1100.0, 500.0, 1200.0, 600.0),
                     frame=0, ts=0.0))
    d.add(TileAppeared(eid=1, tile=5, track_id=1001, cx=1150.0, cy=552.0,
                       conf=0.8, frame=1, ts=0.1))
    d.add(TileAppeared(eid=2, tile=6, track_id=1002, cx=1150.0, cy=554.0,
                       conf=0.8, frame=2, ts=0.2))
    view = project(d, my_hand=[])
    melds = view.players['right_river'].melds
    assert melds and melds[0].kind == 'chi'
    assert melds[0].tiles == (6, 7, 8)


def test_provisional_shows_unfrozen_map():
    """待定牌: 未冻结事件按当前 MAP 投影 — 打出后即时显示, 冻结转实。"""
    d = _build()
    d.add(TileAppeared(eid=1, tile=5, track_id=1001, cx=700.0, cy=740.0,
                       conf=0.8, frame=0, ts=0.0))
    view = project(d, my_hand=[])
    assert view.players['my_river'].visible_river == []  # 未冻结不进实河
    assert view.provisional['my_river'] == [5]           # 待定即时可见
    # 推 e1 出窗冻结后: 待定转实
    d.add(TileAppeared(eid=2, tile=6, track_id=1002, cx=700.0, cy=742.0,
                       conf=0.8, frame=1, ts=0.1))
    view = project(d, my_hand=[])
    assert view.players['my_river'].visible_river == [5]
    assert view.provisional['my_river'] == [6]  # e2 成为新的待定


def test_claim_without_event_not_in_visible():
    """秒碰(claimed=None): 不指向任何牌河事件 — 投影不受影响。"""
    d = _build()
    d.add(TileAppeared(eid=1, tile=5, track_id=1001, cx=700.0, cy=740.0,
                       conf=0.8, frame=0, ts=0.0))
    d.add(MeldFormed(eid=60, tiles=(8, 8, 8), cx=1150.0, cy=550.0,
                     bbox=(1100.0, 500.0, 1200.0, 600.0),
                     frame=1, ts=0.1))
    d.add(TileClaimed(eid=50, claimed=None, meld=60, frame=1, ts=0.1))
    d.add(TileAppeared(eid=2, tile=6, track_id=1002, cx=700.0, cy=742.0,
                       conf=0.8, frame=2, ts=0.2))  # 推 e1 出窗冻结
    view = project(d, my_hand=[])
    assert view.players['my_river'].river == [5]
    assert view.players['my_river'].visible_river == [5]
    assert view.players['right_river'].melds[0].tiles == (8, 8, 8)


def test_project_carries_hand_fallback():
    from src.mahjong_ai.pipeline.decoder import EvidenceModel, WindowDecoder

    ev = EvidenceModel()
    mev = EvidenceModel()
    ev.reseed({'my_river': (600.0, 700.0, 800.0, 780.0),
               'right_river': (1100.0, 500.0, 1180.0, 600.0),
               'top_river': (500.0, 100.0, 700.0, 180.0),
               'left_river': (100.0, 500.0, 180.0, 600.0)}, 1280, 800)
    mev.reseed(None, 1280, 800)
    d = WindowDecoder(ev, mev)
    view = project(d, [], [1, 2, None, 3])
    assert view.my_hand_fallback == [1, 2, None, 3]


def test_project_fallback_default_none():
    from src.mahjong_ai.pipeline.decoder import EvidenceModel, WindowDecoder

    ev = EvidenceModel()
    mev = EvidenceModel()
    ev.reseed(None, 1280, 800)
    mev.reseed(None, 1280, 800)
    d = WindowDecoder(ev, mev)
    view = project(d, [])
    assert view.my_hand_fallback is None
