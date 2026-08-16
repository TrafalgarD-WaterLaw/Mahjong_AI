"""牌河消失检测 — 最新落定牌连续缺失 → TileVanished(被碰/吃)。"""

from src.mahjong_ai.pipeline.river_watcher import RiverWatcher
from src.mahjong_cv.detections import TileDet

_REGION = {'right_river': (900.0, 300.0, 1200.0, 400.0)}


def _mk(tile, x1, y1, x2, y2, conf=0.8, tid=None) -> TileDet:
    return TileDet(tile=tile, x1=x1, y1=y1, x2=x2, y2=y2, conf=conf,
                   track_id=tid)


def test_no_vanish_while_present():
    w = RiverWatcher()
    w.set_regions(_REGION)
    w.on_appeared('right_river', 3, 8, 950.0, 350.0)
    dets = [_mk(5, 1000.0, 310.0, 1030.0, 360.0, tid=2),
            _mk(8, 925.0, 320.0, 975.0, 370.0, tid=3)]
    events = []
    for i in range(30):
        events += w.tick(dets, frame=i, ts=i * 0.1, next_eid=lambda: 99)
    assert events == []


def test_vanish_after_sustained_absence():
    w = RiverWatcher()
    w.set_regions(_REGION)
    w.on_appeared('right_river', 3, 8, 950.0, 350.0)
    dets = [_mk(5, 1000.0, 310.0, 1030.0, 360.0, tid=2),
            _mk(8, 925.0, 320.0, 975.0, 370.0, tid=3)]
    w.tick(dets, frame=0, ts=0.0, next_eid=lambda: 99)
    # 之后 20 帧: 8 消失, 但同河 5 还在(河活着)
    dets_without = [_mk(5, 1000.0, 310.0, 1030.0, 360.0, tid=2)]
    events = [w.tick(dets_without, frame=i, ts=i * 0.1,
                     next_eid=lambda: 99) for i in range(1, 21)]
    fired = [e for sub in events for e in sub]
    assert len(fired) == 1
    assert fired[0].tile == 8
    assert fired[0].river == 'right_river'
    assert fired[0].appeared_eid == 3


def test_low_conf_watched_tile_no_vanish():
    """低置信误检的消失不发事件(真实对局: 二万 conf 0.35 是误检,
    其淡出被当成被碰 → 对家假副露)。高置信正常发。"""
    alive = [_mk(5, 1000.0, 310.0, 1030.0, 360.0, tid=2)]
    w = RiverWatcher()
    w.set_regions(_REGION)
    w.on_appeared('right_river', 3, 1, 950.0, 350.0, conf=0.35)
    events = [w.tick(alive, frame=i, ts=i * 0.1, next_eid=lambda: 99)
              for i in range(20)]
    assert not [e for sub in events for e in sub]
    # 高置信对照
    w2 = RiverWatcher()
    w2.set_regions(_REGION)
    w2.on_appeared('right_river', 3, 1, 950.0, 350.0, conf=0.8)
    fired = [w2.tick(alive, frame=i, ts=i * 0.1, next_eid=lambda: 99)
             for i in range(20)]
    assert len([e for sub in fired for e in sub]) == 1


def test_brief_occlusion_no_vanish():
    """飞牌遮挡 2 帧不算消失(阈值 15 帧吸收)。"""
    w = RiverWatcher()
    w.set_regions(_REGION)
    w.on_appeared('right_river', 3, 8, 950.0, 350.0)
    w.tick([_mk(5, 1000.0, 310.0, 1030.0, 360.0, tid=2),
            _mk(8, 925.0, 320.0, 975.0, 370.0, tid=3)],
           frame=0, ts=0.0, next_eid=lambda: 99)
    events = []
    for i in range(1, 40):
        dets = ([_mk(5, 1000.0, 310.0, 1030.0, 360.0, tid=2)]
                if i % 10 >= 2 else
                [_mk(5, 1000.0, 310.0, 1030.0, 360.0, tid=2),
                 _mk(8, 925.0, 320.0, 975.0, 370.0, tid=3)])
        events += w.tick(dets, frame=i, ts=i * 0.1, next_eid=lambda: 99)
    assert events == []   # 每 10 帧只缺 2 帧, 从未连续 15 帧


def test_whole_river_collapse_no_vanish():
    """整河检测塌(连 5 都检测不到)不算消失 — 挡住抽风假消失。"""
    w = RiverWatcher()
    w.set_regions(_REGION)
    w.on_appeared('right_river', 3, 8, 950.0, 350.0)
    w.tick([_mk(5, 1000.0, 310.0, 1030.0, 360.0, tid=2),
            _mk(8, 925.0, 320.0, 975.0, 370.0, tid=3)],
           frame=0, ts=0.0, next_eid=lambda: 99)
    events = []
    for i in range(1, 40):
        events += w.tick([], frame=i, ts=i * 0.1, next_eid=lambda: 99)
    assert events == []


def test_newest_replaced_by_later_appear():
    """新打出落定后, 盯防对象换成那张; 旧牌消失不再触发。"""
    w = RiverWatcher()
    w.set_regions(_REGION)
    w.on_appeared('right_river', 3, 8, 950.0, 350.0)
    w.tick([_mk(8, 925.0, 320.0, 975.0, 370.0, tid=3)],
           frame=0, ts=0.0, next_eid=lambda: 99)
    w.on_appeared('right_river', 4, 2, 1050.0, 350.0)
    events = []
    for i in range(1, 30):
        # 旧牌 8 消失, 新牌 2 在场
        events += w.tick([_mk(2, 1025.0, 320.0, 1075.0, 370.0, tid=4)],
                         frame=i, ts=i * 0.1, next_eid=lambda: 99)
    assert events == []
