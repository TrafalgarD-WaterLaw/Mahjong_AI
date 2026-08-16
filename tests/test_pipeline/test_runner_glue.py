"""管道运行器胶水单测 — GameView → Advice 转换。"""

from pathlib import Path

import pytest

from scripts.run_assistant import build_advice, load_seed_regions
from src.mahjong_ai.pipeline.events import PLAYERS
from src.mahjong_ai.pipeline.state import GameView, MeldView, PlayerView
from src.mahjong_ai.strategy import StrategyEngine
from src.mahjong_engine import get as get_rules


def _engine():
    rules = get_rules('huanyu')
    return StrategyEngine(rules), rules


def _view() -> GameView:
    players = {p: PlayerView(player=p)
               for p in ('my_river', 'right_river', 'top_river', 'left_river')}
    players['my_river'].river = [5, 8]
    players['my_river'].visible_river = [5]
    players['right_river'].melds = [MeldView(kind='pong', tiles=(8, 8, 8))]
    # project() 的输出形状: 四家键齐全
    return GameView(my_hand=[1, 2, 3], players=players, unfrozen=[],
                    provisional={p: [] for p in PLAYERS})


def test_build_advice_maps_visible_river_and_melds():
    engine, rules = _engine()
    view = _view()
    advice, _boxes, provisional = build_advice(view, engine, rules)
    snap = advice.snapshot
    assert snap.my_hand == [1, 2, 3]
    assert snap.players['my_river'].river == [5]   # 显示用 visible
    assert snap.players['right_river'].melds[0].tile == 8
    assert set(provisional) == set(PLAYERS)  # 四家键齐全(可为空表)


def test_build_advice_passes_provisional_through():
    """待定牌(未冻结 MAP)原样透传 — 客户端半透明即时显示用。"""
    engine, rules = _engine()
    view = _view()
    view.provisional['my_river'] = [7]
    view.provisional['right_river'] = [9]
    _advice, _boxes, provisional = build_advice(view, engine, rules)
    assert provisional['my_river'] == [7]
    assert provisional['right_river'] == [9]


def test_build_advice_waiting_when_13():
    """13 张听牌(4 面子 + 单骑)→ waiting 非空, discard 为 None。"""
    engine, rules = _engine()
    view = GameView(
        my_hand=[0, 1, 2, 3, 4, 5, 9, 10, 11, 18, 19, 20, 27],
        players={p: PlayerView(player=p)
                 for p in ('my_river', 'right_river', 'top_river',
                           'left_river')},
        unfrozen=[])
    advice, _boxes, _prov = build_advice(view, engine, rules)
    assert advice.waiting is not None
    assert advice.discard is None


def test_load_seed_regions_remaps_meld_keys():
    """C1: 配置键 *_meld → *_river 重映射; 键 ∈ PLAYERS, 值均为 4 元组。"""
    if not Path('config/river_regions.json').exists():
        pytest.skip('config/river_regions.json 不存在')
    rivers, melds = load_seed_regions()
    assert set(rivers) <= set(PLAYERS)
    assert set(melds) <= set(PLAYERS)  # 重映射后全部落回 PLAYERS 名
    assert all(len(v) == 4 for v in rivers.values())
    assert all(len(v) == 4 for v in melds.values())


def test_build_advice_discard_when_14():
    engine, rules = _engine()
    view = GameView(
        my_hand=[0, 0, 0, 1, 2, 3, 9, 10, 11, 18, 19, 20, 27, 31],
        players={p: PlayerView(player=p)
                 for p in ('my_river', 'right_river', 'top_river',
                           'left_river')},
        unfrozen=[])
    advice, _boxes, _prov = build_advice(view, engine, rules)
    assert advice.discard is not None
    assert advice.discard.tile in view.my_hand


def test_melded_tenpai_shows_waiting_not_discard():
    """副露局听牌(M3): 手牌 10 + 副露展开 = 13 已听 → waiting, 不推荐打牌。"""
    engine, rules = _engine()
    view = GameView(
        my_hand=[0, 0, 0, 1, 2, 3, 9, 10, 11, 27],  # 10 张
        players={'my_river': PlayerView(
            player='my_river',
            melds=[MeldView(kind='chi', tiles=(6, 7, 8))]),
            **{p: PlayerView(player=p)
               for p in ('right_river', 'top_river', 'left_river')}},
        unfrozen=[])
    advice, _boxes, _prov = build_advice(view, engine, rules)
    assert advice.waiting is not None, '听牌应显示等待牌而非推荐'
    assert advice.discard is None


def test_should_reseed_first_time_with_none_area():
    """area=None(无黑边)首次也必须播种 — 真实对局 bug 回归。"""
    from scripts.run_assistant import _should_reseed
    assert _should_reseed(False, None, None) is True
    assert _should_reseed(True, None, None) is False
    assert _should_reseed(True, (0, 0, 100, 100), None) is True
    assert _should_reseed(True, (0, 0, 100, 100), (0, 0, 100, 100)) is False


def test_build_advice_resolves_opponent_meld_content():
    """对手副露内容未知 + 可用张数唯一解 → 显示确定组合(吃 6-7-8)。

    我手牌 1 张 8 + 右家河打出 2 张 8 → 池中 8 只剩 1, 碰不可能;
    顺子 (6,7,8) 是唯一 → 显示真实牌面, 不再是被碰牌+暗块。
    """
    engine, rules = _engine()
    players = {p: PlayerView(player=p)
               for p in ('my_river', 'right_river', 'top_river',
                         'left_river')}
    players['right_river'].river = [8, 8]
    players['right_river'].melds = [MeldView(kind='pong', tiles=(8,))]
    view = GameView(
        my_hand=[8, 0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14],
        players=players, unfrozen=[])
    advice, _boxes, _prov = build_advice(view, engine, rules)
    meld = advice.snapshot.players['right_river'].melds[0]
    assert meld.tiles == (6, 7, 8)
    assert meld.kind == 'chi'


def test_build_advice_ambiguous_meld_stays_unknown():
    """碰与吃都可行(歧义) → 保持被碰牌+暗块, 不猜。"""
    engine, rules = _engine()
    players = {p: PlayerView(player=p)
               for p in ('my_river', 'right_river', 'top_river',
                         'left_river')}
    players['right_river'].river = [8]
    players['right_river'].melds = [MeldView(kind='pong', tiles=(8,))]
    view = GameView(
        my_hand=[0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15],
        players=players, unfrozen=[])
    advice, _boxes, _prov = build_advice(view, engine, rules)
    meld = advice.snapshot.players['right_river'].melds[0]
    assert meld.tiles == (8,)       # 歧义: 不猜
    assert meld.kind == 'pong'


def test_build_advice_reason_mentions_ukeire():
    """新推荐的理由文本含进张枚数(纯牌效算法已接入)。"""
    engine, rules = _engine()
    view = GameView(
        my_hand=[0, 0, 0, 1, 1, 1, 2, 2, 2, 27, 30, 33, 5, 6],
        players={p: PlayerView(player=p)
                 for p in ('my_river', 'right_river', 'top_river',
                           'left_river')},
        unfrozen=[])
    advice, _boxes, _prov = build_advice(view, engine, rules)
    assert advice.discard is not None
    assert '进张' in advice.discard.reason or '听牌' in advice.discard.reason
