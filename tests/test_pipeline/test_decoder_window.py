"""窗口解码器单测 — 软归属/回看纠错/相位硬锚/冻结学习。"""

from src.mahjong_ai.pipeline.decoder import EvidenceModel, WindowDecoder
from src.mahjong_ai.pipeline.events import (
    PLAYERS,
    HandChanged,
    MeldFormed,
    TileAppeared,
)

_SEED = {
    'my_river': (600.0, 700.0, 800.0, 780.0),
    'right_river': (1100.0, 500.0, 1180.0, 600.0),
    'top_river': (500.0, 100.0, 700.0, 180.0),
    'left_river': (100.0, 500.0, 180.0, 600.0),
}
_CENTERS = {p: ((r[0] + r[2]) / 2, (r[1] + r[3]) / 2)
            for p, r in _SEED.items()}


def _make(player: str, eid: int, frame: int = 0,
          jitter: tuple[float, float] = (0.0, 0.0)) -> TileAppeared:
    cx, cy = _CENTERS[player]
    return TileAppeared(eid=eid, tile=eid % 34, track_id=1000 + eid,
                        cx=cx + jitter[0], cy=cy + jitter[1],
                        conf=0.8, frame=frame, ts=frame * 0.1)


def _decoder(k: int = 4) -> WindowDecoder:
    ev = EvidenceModel()
    ev.reseed(_SEED, 1280, 800)
    mev = EvidenceModel()
    mev.reseed(None, 1280, 800)
    return WindowDecoder(ev, mev, k=k, beam=16)


def test_clean_rotation_all_correct():
    """位置清晰 + 轮转正常 → 全部归对。"""
    d = _decoder()
    for i, p in enumerate(('my_river', 'right_river', 'top_river', 'left_river',
                           'my_river', 'right_river', 'top_river', 'left_river',
                           'my_river', 'right_river', 'top_river', 'left_river')):
        d.add(_make(p, eid=i + 1, frame=i))
    got = {a.eid: a.map() for a in d.frozen()}
    truth = {1: 'my_river', 2: 'right_river', 3: 'top_river', 4: 'left_river',
             5: 'my_river', 6: 'right_river', 7: 'top_river', 8: 'left_river'}
    for eid, p in truth.items():
        assert got.get(eid) == p, f'eid {eid}: {got.get(eid)}'
    assert len(d.frozen()) == 8  # k=4: 12 张 → 冻结前 8 张


def test_missed_event_skip_penalty_path():
    """top 的打出漏检: my,right,left 序列 → left 仍归 left(不归 top)。"""
    d = _decoder(k=3)
    d.add(_make('my_river', 1, 0))
    d.add(_make('right_river', 2, 1))
    d.add(_make('left_river', 3, 2))
    d.add(_make('my_river', 4, 3))
    d.add(_make('right_river', 5, 4))
    a3 = [a for a in d.attributions() if a.eid == 3][0]
    assert a3.map() == 'left_river'


def test_retro_correction_on_new_evidence():
    """首事件位置歧义(桌面中央)先按轮转先验, 后续证据回看修正。"""
    d = _decoder(k=5)
    # 真值: e1=right(位置漂移大), e2=top, e3=left, e4=my
    d.add(TileAppeared(eid=1, tile=1, track_id=1001,
                       cx=640.0, cy=400.0,  # 中央歧义点
                       conf=0.5, frame=0, ts=0.0))
    before = {a.eid: dict(a.probs()) for a in d.attributions()}[1]
    d.add(_make('top_river', 2, 1))
    d.add(_make('left_river', 3, 2))
    d.add(_make('my_river', 4, 3))
    d.add(_make('right_river', 5, 4))
    d.add(_make('top_river', 6, 5))
    after = {a.eid: dict(a.probs()) for a in d.attributions()}[1]
    assert after != before  # 分布被重解(回看纠错发生)


def test_hand_anchor_resets_phase():
    """我摸牌(13→14)后相位硬锚: 后续新牌空间歧义时轮转先验归我。"""
    d = _decoder(k=3)
    d.add(_make('right_river', 1, 0))
    d.add(HandChanged(eid=90, n_old=13, n_new=14, frame=1, ts=0.1))
    # 空间歧义点(我的牌河上方 — 距我锚较近), 相位锚定后 → 归我
    d.add(TileAppeared(eid=2, tile=2, track_id=1002,
                       cx=700.0, cy=600.0, conf=0.5, frame=2, ts=0.2))
    a2 = [a for a in d.attributions() if a.eid == 2][0]
    assert a2.map() == 'my_river'


def test_entropy_low_for_clear_high_for_ambiguous():
    d = _decoder(k=2)
    d.add(_make('my_river', 1, 0))
    d.add(TileAppeared(eid=2, tile=2, track_id=1002,
                       cx=640.0, cy=400.0, conf=0.5, frame=1, ts=0.1))
    clear, amb = (a for a in d.unfrozen())
    assert clear.entropy() < amb.entropy()


def test_age_freeze_freezes_stale_events():
    """时间冻结: 事件超过 _FREEZE_MAX_AGE 即使窗口未满也冻结。

    真实对局回归: 事件间隔 3-13s, 窗口溢出才冻结 → 显示滞后
    1-2 分钟; 对局结束不再有新事件 → 末 k 张永远不冻结。
    """
    d = _decoder(k=10)  # 窗口远未满
    d.add(_make('my_river', 1, 0))     # ts = 0
    assert not d.frozen()
    d.age_freeze(now=16.0)
    assert [a.eid for a in d.frozen()] == [1]
    d.add(_make('right_river', 2, 1))  # ts = 0.1
    d.age_freeze(now=5.0)              # 才 4.9s, 不冻结
    assert [a.eid for a in d.frozen()] == [1]
    d.age_freeze(now=16.0)             # 15.9s ≥ 阈值 → 冻结
    assert [a.eid for a in d.frozen()] == [1, 2]


def test_freeze_high_margin_learns_spatial():
    """高置信冻结 → 该家落点高斯在线学习(对实际落点似然上升)。"""
    ev = EvidenceModel()
    ev.reseed(_SEED, 1280, 800)
    d = WindowDecoder(ev, EvidenceModel(), k=1, beam=8)
    before = ev.spatial_logp('my_river', 700.0, 650.0)
    # 偏移落点(偏离种子中心): 连续 3 张我家的牌 → 冻结学习
    for i in range(3):
        d.add(TileAppeared(eid=i + 1, tile=i, track_id=2000 + i,
                           cx=700.0, cy=650.0,
                           conf=0.8, frame=i, ts=i * 0.1))
    after = ev.spatial_logp('my_river', 700.0, 650.0)
    assert after > before  # 均值向实际落点漂移(学习方向正确)


def test_meld_soft_probs_distribution():
    """副露软归属: meld_probs 返回四家归一化分布(M3 扣池加权用)。"""
    ev = EvidenceModel()
    ev.reseed(_SEED, 1280, 800)
    meld_ev = EvidenceModel()
    meld_ev.reseed({'my_river': (600.0, 700.0, 800.0, 780.0),
                    'right_river': (1100.0, 500.0, 1180.0, 600.0),
                    'top_river': (500.0, 100.0, 700.0, 180.0),
                    'left_river': (100.0, 500.0, 180.0, 600.0)},
                   1280, 800)
    d = WindowDecoder(ev, meld_ev, k=4, beam=8)
    d.add(MeldFormed(eid=60, tiles=(8, 8, 8), cx=1150.0, cy=550.0,
                     bbox=(1100.0, 500.0, 1200.0, 600.0),
                     frame=0, ts=0.0))
    probs = d.meld_probs(60)
    assert set(probs) == set(PLAYERS)
    assert abs(sum(probs.values()) - 1.0) < 1e-6
    assert max(probs, key=probs.__getitem__) == 'right_river'


def test_meld_unassigned_when_no_seed():
    """无副露种子(四家得分相同) → 副露不归属(中性), 不产生 my_river 偏置。"""
    d = WindowDecoder(EvidenceModel(), EvidenceModel(), k=4, beam=8)
    d.add(MeldFormed(eid=60, tiles=(8, 8, 8), cx=640.0, cy=400.0,
                     bbox=(600.0, 360.0, 680.0, 440.0),
                     frame=0, ts=0.0))
    assert d.melds() == []  # 不产生任何归属


def test_probs_stable_on_extreme_logits():
    """logit 全部远低于 -745 时 probs/entropy 不崩溃(数值稳定)。"""
    from src.mahjong_ai.pipeline.decoder import Attribution

    at = Attribution(eid=1, tile=1, cx=0.0, cy=0.0, frame=0, ts=0.0,
                     logits={'my_river': -1200.0, 'right_river': -1300.0,
                             'top_river': float('-inf'),
                             'left_river': -1500.0})
    pr = at.probs()
    assert abs(sum(pr.values()) - 1.0) < 1e-6
    assert at.entropy() > 0
    # 全 -inf → 均匀分布, 不崩溃
    at2 = Attribution(eid=2, tile=2, cx=0.0, cy=0.0, frame=0, ts=0.0,
                      logits={p: float('-inf') for p in at.logits})
    pr2 = at2.probs()
    assert abs(sum(pr2.values()) - 1.0) < 1e-6
    assert at2.entropy() > 0


def test_vanished_tile_removed_from_claimed_ids():
    from src.mahjong_ai.pipeline.events import TileVanished

    d = _decoder()
    d.add(_make('right_river', 1, 0))
    d.add(TileVanished(eid=99, tile=1, river='right_river',
                       appeared_eid=1, frame=1, ts=0.1))
    assert 1 in d.claimed_ids()


def test_meld_player_prior_overrides_spatial():
    """MeldFormed.player(事件推断)压过空间证据 — 副露区框错也不归错。"""
    d = _decoder()
    # 空间上远离 left_river 的副露, 但推断出副露者是 left_river
    d.add(MeldFormed(eid=60, tiles=(8, 8, 8), cx=1150.0, cy=550.0,
                     bbox=(1100.0, 500.0, 1200.0, 600.0),
                     frame=0, ts=0.0, player='left_river'))
    assert d.meld_player(60) == 'left_river'
    assert d.melds()[0][0] == 'left_river'


def test_meld_without_player_still_spatial():
    # _decoder() 的 meld_ev 未播种(退化: 空间得分全同 → 不归属),
    # 空间回退需播副露种子(与 test_meld_soft_probs_distribution 一致)
    ev = EvidenceModel()
    ev.reseed(_SEED, 1280, 800)
    meld_ev = EvidenceModel()
    meld_ev.reseed(_SEED, 1280, 800)
    d = WindowDecoder(ev, meld_ev, k=4, beam=16)
    d.add(MeldFormed(eid=61, tiles=(8, 8, 8), cx=1150.0, cy=550.0,
                     bbox=(1100.0, 500.0, 1200.0, 600.0),
                     frame=0, ts=0.0))
    assert d.meld_player(61) == 'right_river'  # 空间最近
