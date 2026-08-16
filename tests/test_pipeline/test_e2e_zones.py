"""端到端: 事件推断副露 → 解码 → 投影(被碰牌摘除 + 副露内容)。"""

from src.mahjong_ai.pipeline.decoder import EvidenceModel, WindowDecoder
from src.mahjong_ai.pipeline.events import (
    MeldFormed,
    TileAppeared,
    TileClaimed,
    TileVanished,
)
from src.mahjong_ai.pipeline.extractor import EventExtractor
from src.mahjong_ai.pipeline.state import project
from src.mahjong_cv.detections import TileDet


def _mk(tile, x1, y1, x2, y2, conf=0.8, tid=None) -> TileDet:
    return TileDet(tile=tile, x1=x1, y1=y1, x2=x2, y2=y2, conf=conf,
                   track_id=tid)


def _hand13(t0: float = 100.0) -> list[TileDet]:
    """我的手牌: 13 张(含两张 8), 底行 y 700-770。"""
    return [_mk(i if i <= 8 else i - 1, t0 + i * 40, 700.0,
                t0 + i * 40 + 40, 770.0, tid=1000 + i) for i in range(13)]


def _hand11() -> list[TileDet]:
    """副露后: 11 张(两张 8 进副露)。"""
    tiles = [i for i in range(12) if i != 8]
    return [_mk(t, 100.0 + i * 40, 700.0, 100.0 + i * 40 + 40, 770.0,
                tid=2000 + i) for i, t in enumerate(tiles)]


def test_my_pong_inferred_from_vanish_and_step():
    """我的碰: 右家打出 8 → 8 从牌河消失 + 我手牌 13→11 → 副露(8,8,8);
    被碰的 8 从右家 visible_river 摘除。"""
    ext = EventExtractor()
    ext.set_regions(
        {'my_river': (500.0, 300.0, 800.0, 450.0),
         'right_river': (900.0, 300.0, 1200.0, 450.0)},
        {'my_river': (800.0, 600.0, 1000.0, 700.0)})
    events = []
    # 阶段 1: 手牌 13 稳定(快照建立)
    for _ in range(8):
        events += ext.process(_hand13(), frame=0, ts=0.0)
    # 阶段 2: 右家先打 5(河中有其他牌 — 消失计数时河还活着),
    # 再打出 8(落进 right_river, 稳定 2 帧 → TileAppeared)。
    # 5 先于 8 落河: 其打出时间(1.1)落在推断器出牌配对窗
    # [vanish−2, vanish+5] 之外, 消失候选只能与我的手牌台阶配对
    # (事件推断路径 — 本测试要验证的目标)。
    other = _mk(5, 1000.0, 400.0, 1015.0, 415.0, tid=501)
    events += ext.process(_hand13() + [other], frame=1, ts=1.0)
    events += ext.process(_hand13() + [other], frame=2, ts=1.1)
    disc = _mk(8, 950.0, 400.0, 965.0, 415.0, tid=500)
    events += ext.process(_hand13() + [other, disc], frame=3, ts=2.0)
    events += ext.process(_hand13() + [other, disc], frame=4, ts=2.1)
    appeared = [e for e in events if isinstance(e, TileAppeared)]
    assert appeared and appeared[-1].tile == 8
    # 阶段 3: 8 消失(被碰走), 右河其他牌还在; 我手牌 13→11。
    # 消失 15 帧确认(≈2s)在 ts 4.4; 手牌台阶(纪元掉 2 + 保持 2s)
    # 在 ts 5.3 落定, 与消失 ts 差 0.9 ≤ 台阶窗口 2.0 — 配对成立
    vanish_ts = 3.0
    for i in range(25):
        ts = vanish_ts + i * 0.1
        events += ext.process(_hand11() + [other], frame=10 + i, ts=ts)
    vanished = [e for e in events if isinstance(e, TileVanished)]
    assert len(vanished) == 1
    melds = [e for e in events if isinstance(e, MeldFormed)]
    assert len(melds) == 1
    assert melds[0].tiles == (8, 8, 8)
    assert melds[0].player == 'my_river'
    claims = [e for e in events if isinstance(e, TileClaimed)]
    assert claims and claims[0].claimed == appeared[-1].eid
    # 阶段 4: 解码 + 投影 — 被碰的 8 从右家 visible_river 摘除
    river_ev = EvidenceModel()
    meld_ev = EvidenceModel()
    river_ev.reseed({'my_river': (500.0, 300.0, 800.0, 450.0),
                     'right_river': (900.0, 300.0, 1200.0, 450.0),
                     'top_river': (500.0, 100.0, 700.0, 180.0),
                     'left_river': (100.0, 500.0, 180.0, 600.0)}, 1280, 800)
    meld_ev.reseed(None, 1280, 800)
    d = WindowDecoder(river_ev, meld_ev)
    for ev in events:
        d.add(ev)
    d.age_freeze(100.0)
    view = project(d, ext.my_hand())
    assert 8 not in view.players['right_river'].visible_river
    assert view.players['my_river'].melds[0].tiles == (8, 8, 8)
