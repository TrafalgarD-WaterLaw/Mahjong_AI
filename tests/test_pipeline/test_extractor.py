"""事件提取器单测 — 合成 TileDet 帧序列 → 事件(不带归属)。"""

from src.mahjong_ai.pipeline.events import (
    FlowerShown,
    HandChanged,
    MeldFormed,
    TileAppeared,
    TileClaimed,
)
from src.mahjong_ai.pipeline.extractor import EventExtractor
from src.mahjong_cv.detections import TileDet


def _mk(tile, x1, y1, x2, y2, conf=0.8, tid=None) -> TileDet:
    return TileDet(tile=tile, x1=x1, y1=y1, x2=x2, y2=y2, conf=conf,
                   track_id=tid)


def _hand_frame(n=13, x0=100.0, tid_base=100) -> list[TileDet]:
    """底行横排手牌: 宽 40 高 70, 间距 40(与真实比例一致)。"""
    return [_mk(i % 30, x0 + i * 40, 700.0, x0 + i * 40 + 40, 770.0,
                tid=tid_base + i) for i in range(n)]


def _feed(ext: EventExtractor, frames: list[list[TileDet]]) -> list:
    out = []
    for i, dets in enumerate(frames):
        out += ext.process(dets, frame=i, ts=i * 0.1)
    return out


def test_hand_changed_draw_and_discard():
    ext = EventExtractor()
    frames = [_hand_frame(13)] * 3          # 去抖稳定 13
    # 摸到的第 14 张必须是手牌里没有的 tile(手牌位置按 tile 键存储,
    # 重复 tile 会互相覆盖位置) — 用 13, 不在 0-12 里
    frames += [_hand_frame(13) + [_mk(13, 620.0, 700.0, 660.0, 770.0,
                                      tid=999)]] * 4   # 摸到 14(majority 3 + 迟滞 2 帧)
    events = _feed(ext, frames)
    changed = [e for e in events if isinstance(e, HandChanged)]
    assert changed and changed[0].n_old == 0
    assert changed[-1].n_old == 13 and changed[-1].n_new == 14
    # 位置序: 第 14 张在 x 620-660, 位于 tile 12(x 580-620)之后
    assert ext.my_hand() == list(range(13)) + [13]


def test_appeared_emitted_once_when_stable():
    ext = EventExtractor()
    frames = [_hand_frame(13)] * 4        # 去抖+迟滞稳定 → 门开(基线在 f3, 不含 tile)
    frames += [_hand_frame(13)
               + [_mk(7, 700.0, 400.0, 715.0, 415.0, tid=1)]] * 2
    events = _feed(ext, frames)
    appeared = [e for e in events if isinstance(e, TileAppeared)]
    assert len(appeared) == 1
    assert appeared[0].tile == 7


def test_appeared_gated_before_hand_stable():
    """发牌动画期(手牌 <13)新轨迹不产出事件。"""
    ext = EventExtractor()
    frames = [_hand_frame(6)] * 2
    frames += [frames[0] + [_mk(7, 700.0, 400.0, 715.0, 415.0, tid=1)]] * 2
    events = _feed(ext, frames)
    assert not [e for e in events if isinstance(e, TileAppeared)]


def test_flower_shown_deduped():
    ext = EventExtractor()
    dets = _hand_frame(13) + [_mk(34, 900.0, 750.0, 940.0, 820.0, tid=77)]
    events = _feed(ext, [dets, dets])
    flowers = [e for e in events if isinstance(e, FlowerShown)]
    assert len(flowers) == 1


def test_visual_meld_confirmed_and_claim_paired():
    """视觉通道: 合法 3 张组持续 ≥5 帧 → 兜底确认副露 + 最近同类配对。"""
    ext = EventExtractor()
    ext.set_regions(
        {'my_river': (600.0, 300.0, 800.0, 450.0),
         'right_river': (900.0, 300.0, 1200.0, 450.0)},
        {'my_river': (800.0, 600.0, 1000.0, 700.0)})
    base = _hand_frame(13)
    # 手牌先去抖稳定(门开), 再打出一张 8(tid 500)落进 right_river
    f1 = base + [_mk(8, 950.0, 400.0, 965.0, 415.0, tid=500)]
    # 碰组: 3 张紧贴同牌, 与手牌远离(独立簇)
    meld = [_mk(8, 700.0, 700.0, 740.0, 770.0, tid=501),
            _mk(8, 740.0, 700.0, 780.0, 770.0, tid=502),
            _mk(8, 780.0, 700.0, 820.0, 770.0, tid=503)]
    frames = [base] * 3 + [f1] * 2 + [base + meld] * 5   # 视觉组持续 5 帧
    events = _feed(ext, frames)
    melds = [e for e in events if isinstance(e, MeldFormed)]
    claims = [e for e in events if isinstance(e, TileClaimed)]
    assert melds and melds[0].tiles == (8, 8, 8)
    assert claims and claims[0].claimed is not None  # 最近同类打出配对


def test_meld_claimed_none_when_no_prior_discard():
    """碰组无已打出轨迹(三张全新 tid)→ TileClaimed.claimed is None。"""
    ext = EventExtractor()
    base = _hand_frame(13)
    meld = [_mk(8, 1150.0, 520.0, 1190.0, 590.0, tid=501),
            _mk(8, 1190.0, 520.0, 1230.0, 590.0, tid=502),
            _mk(8, 1230.0, 520.0, 1270.0, 590.0, tid=503)]
    events = _feed(ext, [base, base, base] + [base + meld] * 5)
    meld_events = [e for e in events if isinstance(e, MeldFormed)]
    claims = [e for e in events if isinstance(e, TileClaimed)]
    assert meld_events, '碰组 5 帧应确认副露'
    assert claims and claims[0].claimed is None


def test_hand_changed_requires_hold():
    """手牌数变化需持续 2 帧才提交 — 1 帧闪烁不产生 HandChanged(P1)。"""
    ext = EventExtractor()
    frames = [_hand_frame(13)] * 3
    # 1 帧虚 14(手牌簇吞 1 张河牌)后回落
    frames += [_hand_frame(13)
               + [_mk(13, 620.0, 700.0, 660.0, 770.0, tid=999)]]
    frames += [_hand_frame(13)] * 3
    events = _feed(ext, frames)
    changed = [e for e in events if isinstance(e, HandChanged)]
    assert not [c for c in changed if c.n_new == 14], '1 帧闪烁不应提交 14'
    assert ext.my_hand() == list(range(13))


def test_new_game_signal_after_empty_table():
    """空桌(手牌 0 持续 ≥30 帧)+ 发牌到 13 → consume_new_game 触发一次(P0)。"""
    ext = EventExtractor()
    _feed(ext, [_hand_frame(13)] * 4)          # 上一局进行中
    assert ext.consume_new_game() is False
    _feed(ext, [[]] * 30)                      # 结算/空桌(实测 80+ 帧)
    _feed(ext, [_hand_frame(4)] * 2)           # 发牌中
    assert ext.consume_new_game() is False     # 还没到 13
    _feed(ext, [_hand_frame(13)] * 4)          # 发牌完成
    assert ext.consume_new_game() is True      # 触发一次
    assert ext.consume_new_game() is False     # 已消费


def test_no_new_game_on_brief_collapse():
    """短暂塌 0(20 帧 ≈2.5s, 中局检测抽风实测)再恢复 → 不触发新局。"""
    ext = EventExtractor()
    _feed(ext, [_hand_frame(13)] * 4)
    _feed(ext, [[]] * 20)                      # 中局抽风: 20 帧 < 30
    _feed(ext, [_hand_frame(13)] * 4)
    assert ext.consume_new_game() is False


def test_gate_stays_open_during_hand_collapse():
    """中局手牌塌 0(数秒)期间, 别家打出照常发事件。

    真实对局回归: 手牌全检测不到时四家事件全停 — 手牌在场门禁
    是逐帧判断, 手牌簇塌到 <4 就把所有玩家的事件关了。改为
    闩锁: 进过局就一直算"局中", 只有空桌信号(手牌 0 持续 ≥5 帧)
    才解锁。
    """
    ext = EventExtractor()
    ext.set_regions({'my_river': (600.0, 400.0, 900.0, 600.0)})
    _feed(ext, [_hand_frame(13)] * 4)          # 进局: 门闩上
    tile = _mk(7, 700.0, 450.0, 715.0, 465.0, tid=1)
    # 手牌塌 0(3 帧, <5 不算空桌)— 期间别家打出照发
    events = _feed(ext, [[tile], [tile], [tile]])
    appeared = [e for e in events if isinstance(e, TileAppeared)]
    assert len(appeared) == 1, f'手牌塌陷期间应照发事件: {appeared}'


def test_no_events_without_hand_present():
    """大厅/进局前: 手牌簇不足 4 张 → 区域内稳定轨迹也不发事件。

    真实对局回归: 大厅 UI 误检的牌(基线之后才出现)被发成事件,
    开局后冻结进某家牌河。手牌在场是"对局进行中"的可靠信号。
    """
    ext = EventExtractor()
    ext.set_regions({'my_river': (600.0, 400.0, 900.0, 600.0)})
    tile = _mk(7, 700.0, 450.0, 715.0, 465.0, tid=1)
    # 空帧(大厅) → 误检牌中途出现并保持稳定 → 无手牌不发事件
    events = _feed(ext, [[], [], [tile], [tile], [tile]])
    assert not [e for e in events if isinstance(e, TileAppeared)]
    # 手牌出现后照常发(基线在手牌出现帧跑, 新轨迹不受影响;
    # 留 1 帧余量让门闩先于 tile 打开)
    ext2 = EventExtractor()
    ext2.set_regions({'my_river': (600.0, 400.0, 900.0, 600.0)})
    events2 = _feed(ext2, [_hand_frame(6), _hand_frame(6), _hand_frame(6),
                           _hand_frame(6) + [tile],
                           _hand_frame(6) + [tile]])
    appeared = [e for e in events2 if isinstance(e, TileAppeared)]
    assert len(appeared) == 1


def test_invalid_visual_group_rejected():
    """南南北类假副露: 组合非法 → 永不确认。"""
    ext = EventExtractor()
    base = _hand_frame(13)
    bogus = [_mk(28, 700.0, 700.0, 740.0, 770.0, tid=601),
             _mk(30, 740.0, 700.0, 780.0, 770.0, tid=602),
             _mk(30, 780.0, 700.0, 820.0, 770.0, tid=603)]
    frames = [base] * 3 + [base + bogus] * 8
    events = _feed(ext, frames)
    assert not [e for e in events if isinstance(e, MeldFormed)]


def test_two_tile_group_not_confirmed():
    """2 张组(动画中/手牌碎片)不确认 — 旧门槛 ≥2 张的回归修复。"""
    ext = EventExtractor()
    base = _hand_frame(13)
    pair = [_mk(7, 700.0, 700.0, 740.0, 770.0, tid=701),
            _mk(8, 740.0, 700.0, 780.0, 770.0, tid=702)]
    frames = [base] * 3 + [base + pair] * 8
    events = _feed(ext, frames)
    assert not [e for e in events if isinstance(e, MeldFormed)]


def test_flower_excluded_from_melds_and_hand():
    """花牌(34)不混入副露组/手牌(真实对局: "pong 花" — 花亮在手牌
    角落, 被区域副露检测并进组)。"""
    ext = EventExtractor()
    ext.set_regions(None, {'my_river': (680.0, 600.0, 940.0, 700.0)})
    base = _hand_frame(13)
    # 副露区: 3 张真副露 + 1 张花
    meld = [_mk(8, 700.0, 700.0, 740.0, 770.0, tid=501),
            _mk(8, 740.0, 700.0, 780.0, 770.0, tid=502),
            _mk(8, 780.0, 700.0, 820.0, 770.0, tid=503),
            _mk(34, 820.0, 700.0, 860.0, 770.0, tid=504)]
    events = _feed(ext, [base + meld] * 5)
    meld_events = [e for e in events if isinstance(e, MeldFormed)]
    assert meld_events, '真副露组应确认'
    assert all(34 not in m.tiles for m in meld_events), \
        f'花牌不得混入副露: {[m.tiles for m in meld_events]}'
    assert 34 not in ext.my_hand()


def test_region_gate_filters_outside_tiles():
    """区域门禁: 手牌数振荡不再卡事件 — 区域内稳定新轨迹发事件,
    区域外(对手手牌/桌面残留)不发。手牌 6 张(<13)也照发。"""
    ext = EventExtractor()
    ext.set_regions({'my_river': (600.0, 400.0, 900.0, 600.0)})
    frames = [_hand_frame(13)] * 3
    inside = _mk(7, 700.0, 450.0, 715.0, 465.0, tid=1)
    outside = _mk(8, 300.0, 100.0, 315.0, 115.0, tid=2)
    frames += [_hand_frame(6) + [inside, outside]] * 2
    events = _feed(ext, frames)
    appeared = [e for e in events if isinstance(e, TileAppeared)]
    assert len(appeared) == 1, f'只有区域内轨迹发事件: {appeared}'
    assert appeared[0].tile == 7


def test_moving_tile_settles_then_emits():
    """飞行中的牌: 首现位置在空中 → 落定后连续 2 帧位移小即发事件。

    真实对局回归: 区域门禁常开后, 打出动画中的牌首现于空中,
    锚定首现位置的旧逻辑判其"不稳定"而丢弃 — 整局大量打出丢失。
    稳定性必须锚定最近位置(连续 2 帧位移 < 0.5×框宽 = 已落定)。
    """
    ext = EventExtractor()
    ext.set_regions({'my_river': (600.0, 400.0, 900.0, 600.0)})
    frames = [_hand_frame(13)] * 3
    # tid 1: 空中(400,300) → 落定(700,450) → 保持(700,450) → 发事件
    flying = _mk(7, 400.0, 300.0, 415.0, 315.0, tid=1)
    settled = _mk(7, 700.0, 450.0, 715.0, 465.0, tid=1)
    frames += [_hand_frame(13) + [flying],
               _hand_frame(13) + [settled],
               _hand_frame(13) + [settled]]
    events = _feed(ext, frames)
    appeared = [e for e in events if isinstance(e, TileAppeared)]
    assert len(appeared) == 1, f'落定后应发事件: {appeared}'
    assert appeared[0].tile == 7


def test_baseline_swallows_existing_boxes():
    """门开瞬间已存在的非手牌框(别家手牌)不发事件; 新轨迹才发。"""
    ext = EventExtractor()
    base = _hand_frame(13)
    opp_hand = [_mk(i % 30, 1300.0 + i * 40, 100.0,
                    1340.0 + i * 40, 170.0, tid=300 + i)
                for i in range(13)]
    frames = [base + opp_hand] * 5     # 门开(帧 2)基线吞掉别家手牌
    frames += [base + opp_hand
               + [_mk(7, 700.0, 400.0, 715.0, 415.0, tid=1)]] * 2
    events = _feed(ext, frames)
    appeared = [e for e in events if isinstance(e, TileAppeared)]
    assert len(appeared) == 1  # 只有新轨迹; 别家手牌轨迹不发事件
    assert appeared[0].tile == 7


def test_hand_band_rejects_river_row():
    """手牌带锚定: 手牌塌陷后, 带外的横排牌河不能被当成手牌。

    真实对局回归: 局中手牌检测塌陷(0 帧)时, 横排牌河行满足
    (横排 + 尺寸锚 + cy 最大)被聚类判成手牌 — 显示全是牌河牌。
    带 = 手牌行最后已知 y 范围, 带外聚类一律拒绝(宁可空, 不可错)。
    """
    ext = EventExtractor()
    _feed(ext, [_hand_frame(13)] * 3)      # 建立手牌带 (700, 770)
    river_row = [_mk(i, 500.0 + i * 40, 400.0, 540.0 + i * 40, 470.0,
                     tid=200 + i) for i in range(5)]
    _feed(ext, [river_row] * 3)
    assert ext.my_hand() == []


def test_hand_band_rejects_meld_row():
    """副露行(y 620-675)与手牌带(690-780)相邻但不相交 — 拒绝。

    真实对局回归: 副露后手牌塌陷, 副露行比牌河更靠下, 聚类优先
    选它当手牌(settle.png 复现: hand cluster = 副露 4 张)。
    """
    ext = EventExtractor()
    _feed(ext, [_hand_frame(13)] * 3)
    meld_row = [_mk(8, 700.0 + i * 40, 620.0, 740.0 + i * 40, 675.0,
                    tid=300 + i) for i in range(3)]
    _feed(ext, [meld_row] * 3)
    assert ext.my_hand() == []


def test_reset_hand_band_reanchors():
    """窗口移动/画面变化: 重锚后新位置的手牌行恢复识别。"""
    ext = EventExtractor()
    _feed(ext, [_hand_frame(13)] * 3)
    moved = [_mk(i % 30, 100.0 + i * 40, 640.0, 100.0 + i * 40 + 40, 710.0,
                 tid=400 + i) for i in range(13)]
    _feed(ext, [moved] * 3)
    assert ext.my_hand() == []           # 带外 → 拒绝
    ext.reset_hand_band()                # 重锚
    _feed(ext, [moved] * 3)
    assert len(ext.my_hand()) == 13      # 新带建立, 恢复识别


def test_band_dets_diagnostic():
    """诊断: 带内探针(±50px)返回手牌区域所有检测(不限 conf)。"""
    ext = EventExtractor()
    _feed(ext, [_hand_frame(13)] * 3)
    assert len(ext.band_dets()) == 13
    assert all(d[3] == 0.8 for d in ext.band_dets())
