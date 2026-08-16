"""副露推断 — 消失 + 台阶/行动权 + 视觉填充, 不依赖看对三张牌。"""

from src.mahjong_ai.pipeline.events import MeldFormed, TileClaimed, TileVanished
from src.mahjong_ai.pipeline.hand_tracker import HandTracker
from src.mahjong_ai.pipeline.meld_inferrer import (
    MeldInferrer,
    VisualMeld,
    meld_kind,
    meld_valid,
)


def _eid() -> list[int]:
    buf = [100]

    def nxt() -> int:
        buf[0] += 1
        return buf[0]
    return nxt


def _vanish(tile=8, eid=3, ts=10.0) -> TileVanished:
    return TileVanished(eid=eid, tile=tile, river='right_river',
                        appeared_eid=5, frame=50, ts=ts)


def test_meld_valid_rules():
    assert meld_valid((8, 8, 8))            # 碰
    assert meld_valid((6, 7, 8))            # 吃
    assert meld_valid((0, 1, 2))            # 一万二万三万
    assert meld_valid((8, 8, 8, 8))         # 杠
    assert not meld_valid((8, 8))           # 2 张组(动画中)
    assert not meld_valid((27, 28, 29))     # 字牌不吃
    assert not meld_valid((8, 9, 10))       # 跨花色(8=九万, 9=一饼)
    assert not meld_valid((7, 8, 8))        # 非同牌非连号
    assert meld_kind((8, 8, 8, 8)) == 'kong'
    assert meld_kind((8, 8, 8)) == 'pong'
    assert meld_kind((7, 8, 9)) == 'chi'


def test_my_meld_from_step_and_delta():
    inf = MeldInferrer(_eid())
    tr = HandTracker()
    # 手牌 13 → 11, 掉的两张 = 双 8(副露 8,8)
    hand13 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 10, 11]
    hand11 = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11]
    t = 0.0
    for _ in range(5):
        t += 0.1
        tr.tick(hand13, t)
    v = _vanish(ts=t + 0.5)
    t += 1.0
    tr.tick(hand11, t)                      # 手牌 13→11(副露 8,8)
    for _ in range(4):
        t += 0.1
        tr.tick(hand11, t)
    # 台阶(掉 2) + 消失 → 我的副露, 内容 = 8 + 手牌差(8,8)
    steps = []
    for _ in range(30):
        t += 0.1
        s = tr.tick(hand11, t)
        if s:
            steps.append(s)
    events = inf.tick([v], steps, [], [], tr, frame=60, ts=t)
    melds = [e for e in events if isinstance(e, MeldFormed)]
    assert len(melds) == 1
    assert melds[0].tiles == (8, 8, 8)
    assert melds[0].player == 'my_river'
    claims = [e for e in events if isinstance(e, TileClaimed)]
    assert claims and claims[0].claimed == v.appeared_eid


def test_delta_wrong_length_content_not_adopted():
    """手牌差长度 1(慢弃/漏检污染) → 内容不采用, 按 (X,) 发射。"""
    inf = MeldInferrer(_eid())
    tr = HandTracker()
    hand13 = list(range(13))
    t = 0.0
    for _ in range(5):
        t += 0.1
        tr.tick(hand13, t)
    v = _vanish(ts=t + 0.5)                 # v.ts ≈ 1.0
    t += 1.0
    tr.tick(list(range(11)), t)             # 13→11 台阶(掉 2)
    t += 0.1
    tr.tick(list(range(11)), t)             # 只 2 帧, 不落快照
    steps = []
    for _ in range(30):                     # 污染: 变 12 张并稳定
        t += 0.1
        s = tr.tick(list(range(11)) + [12], t)
        if s:
            steps.append(s)
    events = inf.tick([v], steps, [], [], tr, frame=60, ts=t)
    melds = [e for e in events if isinstance(e, MeldFormed)]
    assert len(melds) == 1
    assert melds[0].tiles == (8,)           # 差长 1 → 内容未知, 按被碰牌发射


def test_drop_one_step_never_pairs_vanish():
    """摸打 ±1 台阶不是副露证据(真实对局: 8→7 配消失 17 → 假副露
    (17,20,21) 归属我)。drop-1 的台阶不能把消失候选判给我的副露。"""
    inf = MeldInferrer(_eid())
    tr = HandTracker()
    hand13 = list(range(13))
    t = 0.0
    for _ in range(5):
        t += 0.1
        tr.tick(hand13, t)
    v = _vanish(tile=17, ts=t + 0.5)
    t += 1.0
    tr.tick(list(range(12)), t)             # 13→12(drop 1, 摸打)
    steps = []
    for _ in range(30):
        t += 0.1
        s = tr.tick(list(range(12)), t)
        if s:
            steps.append(s)
    events = inf.tick([v], steps, [], [], tr, frame=60, ts=t)
    assert not [e for e in events if isinstance(e, MeldFormed)]
    events2 = inf.tick([], [], [], [], tr, frame=200, ts=t + 20.0)
    assert not [e for e in events2 if isinstance(e, MeldFormed)]


def test_hand_delta_completion_rejects_illegal_chi():
    """手牌差补全合法性校验(真实对局: 消失 17 + 被污染手牌差(20,21)
    → 假吃 (17,20,21)); 不合法 → 超宽限按未知 (17,) 发射。"""
    inf = MeldInferrer(_eid())
    tr = HandTracker()
    hand13 = list(range(11)) + [20, 21]
    hand11 = list(range(11))
    t = 0.0
    for _ in range(5):
        t += 0.1
        tr.tick(hand13, t)
    v = _vanish(tile=17, ts=t + 0.5)
    t += 1.0
    tr.tick(hand11, t)                      # 13→11(drop 2)
    steps = []
    for _ in range(30):
        t += 0.1
        s = tr.tick(hand11, t)
        if s:
            steps.append(s)
    # 台阶在窗内 → melder=my_river; 手牌差 (20,21) + 17 不合法 → 不采用
    events = inf.tick([v], steps, [], [], tr, frame=60, ts=t)
    events += inf.tick([], [], [], [], tr, frame=100, ts=t + 3.0)
    melds = [e for e in events if isinstance(e, MeldFormed)]
    assert len(melds) == 1
    assert melds[0].tiles == (17,)


def test_opponent_meld_from_first_discard():
    inf = MeldInferrer(_eid())
    tr = HandTracker()
    v = _vanish(ts=10.0)
    # 消失后第一家出牌 = right_river(行动权在副露者)
    discards = [('my_river', 3, 7.5, 30),      # 窗口外(碰前)
                ('right_river', 5, 11.0, 31),  # 第一家 → 副露者
                ('top_river', 6, 13.0, 32)]
    events = inf.tick([v], [], discards, [], tr, frame=60, ts=14.0)
    melds = [e for e in events if isinstance(e, MeldFormed)]
    assert len(melds) == 1
    assert melds[0].player == 'right_river'
    assert melds[0].tiles == (8,)          # 内容未知 → 只有被碰牌


def test_claimed_tile_itself_not_melder():
    """被碰那张的打出事件不能把丢弃者当成副露者。"""
    inf = MeldInferrer(_eid())
    tr = HandTracker()
    v = _vanish(ts=10.0)                    # appeared_eid=5
    discards = [('top_river', 8, 8.5, 5),   # 被碰牌自己的事件
                ('right_river', 5, 11.0, 31)]
    events = inf.tick([v], [], discards, [], tr, frame=60, ts=14.0)
    melds = [e for e in events if isinstance(e, MeldFormed)]
    assert len(melds) == 1
    assert melds[0].player == 'right_river'


def test_visual_fills_opponent_content():
    """视觉合法组(含被碰牌)在候选期内 → 填充对手副露内容。"""
    inf = MeldInferrer(_eid())
    tr = HandTracker()
    v = _vanish(tile=7, ts=10.0)
    g = VisualMeld(player='right_river', tiles=(6, 7, 8),
                   cx=950.0, cy=400.0, bbox=(900.0, 360.0, 1000.0, 440.0))
    discards = [('right_river', 5, 11.0, 31)]
    events = inf.tick([v], [], discards, [g], tr, frame=60, ts=14.0)
    melds = [e for e in events if isinstance(e, MeldFormed)]
    assert melds[0].tiles == (6, 7, 8)      # 吃(视觉填充)
    assert melds[0].player == 'right_river'


def test_timeout_drops_candidate():
    """超时(5s)+宽限(3s)无证据 → 丢弃, 不发假事件。"""
    inf = MeldInferrer(_eid())
    tr = HandTracker()
    v = _vanish(ts=10.0)
    events = inf.tick([v], [], [], [], tr, frame=100, ts=10.5)
    assert events == []
    events = inf.tick([], [], [], [], tr, frame=200, ts=20.0)
    assert events == []


def test_visual_fallback_without_vanish():
    """消失信号漏掉: 合法视觉组兜底确认, 被碰牌用最近同类打出配对。"""
    inf = MeldInferrer(_eid())
    tr = HandTracker()
    g = VisualMeld(player='left_river', tiles=(8, 8, 8),
                   cx=500.0, cy=400.0, bbox=(450.0, 360.0, 550.0, 440.0))
    discards = [('top_river', 8, 9.0, 41)]  # 最近同类打出
    events = inf.tick([], [], discards, [g], tr, frame=60, ts=10.0)
    melds = [e for e in events if isinstance(e, MeldFormed)]
    claims = [e for e in events if isinstance(e, TileClaimed)]
    assert len(melds) == 1
    assert melds[0].player == 'left_river'
    assert claims[0].claimed == 41


def _hand_with_meld(dup: int) -> tuple[list[int], list[int]]:
    """含两张 dup 的 13 张手牌与去掉它们后的 11 张手牌。"""
    hand13 = list(range(12)) + [dup]
    hand11 = [i for i in range(12) if i != dup]
    return hand13, hand11


def _steps_for(tr, hand11, t0, n=40) -> list:
    """保持 hand11 至台阶发出, 返回 (steps, 当前时刻)。"""
    steps, t = [], t0
    for _ in range(n):
        t += 0.1
        s = tr.tick(hand11, t)
        if s:
            steps.append(s)
    return steps, t


def test_fast_pong_step_only():
    """快碰: 被碰牌从未进牌河(无消失) — 台阶 + 手牌差独立确认。"""
    inf = MeldInferrer(_eid())
    tr = HandTracker()
    hand13, hand11 = _hand_with_meld(8)
    t = 0.0
    for _ in range(5):
        t += 0.1
        tr.tick(hand13, t)
    steps, t = _steps_for(tr, hand11, t + 0.5)
    assert steps and steps[-1].drop == 2
    discards = [('top_river', 8, t - 1.0, 55)]
    events = inf.tick([], steps, discards, [], tr, frame=80, ts=t)
    melds = [e for e in events if isinstance(e, MeldFormed)]
    claims = [e for e in events if isinstance(e, TileClaimed)]
    assert len(melds) == 1
    assert melds[0].tiles == (8, 8, 8)
    assert melds[0].player == 'my_river'
    assert claims[0].claimed == 55


def test_fast_chi_step_only():
    """快吃: 手牌差 (6,7) → 顺子 6-7-8(被碰牌由最近打出配对)。"""
    inf = MeldInferrer(_eid())
    tr = HandTracker()
    hand13 = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13]
    hand11 = [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13]
    t = 0.0
    for _ in range(5):
        t += 0.1
        tr.tick(hand13, t)
    steps, t = _steps_for(tr, hand11, t + 0.5)
    discards = [('top_river', 8, t - 1.0, 66)]
    events = inf.tick([], steps, discards, [], tr, frame=80, ts=t)
    melds = [e for e in events if isinstance(e, MeldFormed)]
    claims = [e for e in events if isinstance(e, TileClaimed)]
    assert len(melds) == 1
    assert melds[0].tiles == (6, 7, 8)
    assert melds[0].player == 'my_river'
    assert claims[0].claimed == 66


def test_concealed_kong_step_only():
    """暗杠: 无被碰牌 — 台阶 -3 + 手牌差 3 张同牌 → 杠。"""
    inf = MeldInferrer(_eid())
    tr = HandTracker()
    hand13 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 9, 10]
    hand10 = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]
    t = 0.0
    for _ in range(5):
        t += 0.1
        tr.tick(hand13, t)
    steps, t = _steps_for(tr, hand10, t + 0.5)
    events = inf.tick([], steps, [], [], tr, frame=80, ts=t)
    melds = [e for e in events if isinstance(e, MeldFormed)]
    claims = [e for e in events if isinstance(e, TileClaimed)]
    assert len(melds) == 1
    assert melds[0].tiles == (8, 8, 8, 8)
    assert claims[0].claimed is None  # 暗杠无被碰牌


def test_step_only_invalid_delta_dropped():
    """手牌差不成副露(检测污染) → 不确认, 不发假事件。"""
    inf = MeldInferrer(_eid())
    tr = HandTracker()
    hand13 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    hand11 = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11]  # 差 = (8, 12) 跨花色
    t = 0.0
    for _ in range(5):
        t += 0.1
        tr.tick(hand13, t)
    steps, t = _steps_for(tr, hand11, t + 0.5)
    events = inf.tick([], steps, [], [], tr, frame=80, ts=t)
    assert not [e for e in events if isinstance(e, MeldFormed)]


def test_step_only_skips_when_vanish_candidate_open():
    """消失候选在等时, 台阶不另起副露(防双事件)。"""
    inf = MeldInferrer(_eid())
    tr = HandTracker()
    hand13, hand11 = _hand_with_meld(8)
    t = 0.0
    for _ in range(5):
        t += 0.1
        tr.tick(hand13, t)
    steps, t = _steps_for(tr, hand11, t + 0.5)
    v = _vanish(tile=8, eid=3, ts=t + 3.0)   # 消失晚到(台阶窗口外)
    events = inf.tick([v], steps, [], [], tr, frame=80, ts=t + 3.0)
    assert not [e for e in events if isinstance(e, MeldFormed)]


def test_vanish_after_step_only_not_duplicated():
    """台阶已确认后, 同张牌的消失晚到 → 不重复发副露。"""
    inf = MeldInferrer(_eid())
    tr = HandTracker()
    hand13, hand11 = _hand_with_meld(8)
    t = 0.0
    for _ in range(5):
        t += 0.1
        tr.tick(hand13, t)
    steps, t = _steps_for(tr, hand11, t + 0.5)
    discards = [('top_river', 8, t - 1.0, 55)]
    events = inf.tick([], steps, discards, [], tr, frame=80, ts=t)
    assert len([e for e in events if isinstance(e, MeldFormed)]) == 1
    v = _vanish(tile=8, eid=3, ts=t + 2.0)
    events2 = inf.tick([v], [], [], [], tr, frame=90, ts=t + 2.0)
    assert not [e for e in events2 if isinstance(e, MeldFormed)]


def test_fallback_then_late_vanish_not_duplicated():
    """视觉兜底先发(杠 27×4), 消失信号晚到 → 不重复发。

    真实对局回归: 杠东风后副露显示 5 张东风 — 兜底发的 4 张 +
    晚到消失信号又发的 1 张。
    """
    inf = MeldInferrer(_eid())
    tr = HandTracker()
    g = VisualMeld(player='my_river', tiles=(27, 27, 27, 27),
                   cx=1097.0, cy=651.0,
                   bbox=(1012.0, 623.0, 1190.0, 676.0))
    events = inf.tick([], [], [], [g], tr, frame=60, ts=41.0)
    assert len([e for e in events if isinstance(e, MeldFormed)]) == 1
    v = _vanish(tile=27, eid=3, ts=51.0)
    # 杠后我的出牌(行动权窗口内) → 无去重时会把消失候选配成"我的副露"
    discards = [('my_river', 9, 52.0, 80)]
    events2 = inf.tick([v], [], discards, [], tr, frame=70, ts=53.0)
    assert not [e for e in events2 if isinstance(e, MeldFormed)]


def test_step_only_after_fallback_not_duplicated():
    """视觉兜底已发后, 台阶晚到 → 不重复发。"""
    inf = MeldInferrer(_eid())
    tr = HandTracker()
    g = VisualMeld(player='my_river', tiles=(27, 27, 27, 27),
                   cx=1097.0, cy=651.0,
                   bbox=(1012.0, 623.0, 1190.0, 676.0))
    inf.tick([], [], [], [g], tr, frame=60, ts=41.0)
    hand13 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 27, 27, 27]
    hand10 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    t = 40.0
    for _ in range(5):
        t += 0.1
        tr.tick(hand13, t)
    steps, t = _steps_for(tr, hand10, t + 0.5)
    events = inf.tick([], steps, [], [], tr, frame=80, ts=t)
    assert not [e for e in events if isinstance(e, MeldFormed)]


def test_adjacent_chi_not_blocked_by_recent_meld():
    """别家相近顺子不误伤: 已有 (5,6,7), 我的 (6,7,8) 照发。"""
    inf = MeldInferrer(_eid())
    tr = HandTracker()
    g = VisualMeld(player='top_river', tiles=(5, 6, 7),
                   cx=500.0, cy=130.0, bbox=(450.0, 100.0, 550.0, 160.0))
    inf.tick([], [], [], [g], tr, frame=60, ts=30.0)
    hand13 = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13]
    hand11 = [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13]
    t = 30.0
    for _ in range(5):
        t += 0.1
        tr.tick(hand13, t)
    steps, t = _steps_for(tr, hand11, t + 0.5)
    discards = [('top_river', 8, t - 1.0, 77)]
    events = inf.tick([], steps, discards, [], tr, frame=80, ts=t)
    melds = [e for e in events if isinstance(e, MeldFormed)]
    assert len(melds) == 1
    assert melds[0].tiles == (6, 7, 8)



def test_concealed_kong_delta_four():
    """暗杠: 13→10(净 -3, 差 4 张同牌)也确认杠。"""
    inf = MeldInferrer(_eid())
    tr = HandTracker()
    hand13 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 27, 27, 27, 27]
    hand10 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    t = 0.0
    for _ in range(5):
        t += 0.1
        tr.tick(hand13, t)
    steps, t = _steps_for(tr, hand10, t + 0.5)
    events = inf.tick([], steps, [], [], tr, frame=80, ts=t)
    melds = [e for e in events if isinstance(e, MeldFormed)]
    assert len(melds) == 1
    assert melds[0].tiles == (27, 27, 27, 27)


def test_visual_fallback_not_duplicated_after_vanish_meld():
    """视觉兜底晚到(消失路径已发过同一副露, 49 秒后) → 不重复发。

    真实对局回归: 左家碰 9 先由消失路径确认(内容 (9,)), 49 秒后
    视觉组 (9,9,9) 才确认 — 重复发导致观察器粒子缩两次(10→7),
    听牌恒 0(副露是永久场景特征, 去重不能只看 10 秒窗口)。
    """
    inf = MeldInferrer(_eid())
    tr = HandTracker()
    v = _vanish(tile=9, eid=3, ts=10.0)
    discards = [('left_river', 5, 11.0, 31)]  # 左家副露后出牌 → melder=left
    events = inf.tick([v], [], discards, [], tr, frame=60, ts=12.0)
    assert len([e for e in events if isinstance(e, MeldFormed)]) == 1
    g = VisualMeld(player='left_river', tiles=(9, 9, 9),
                   cx=258.0, cy=459.0, bbox=(226.0, 411.0, 285.0, 503.0))
    events2 = inf.tick([], [], [], [g], tr, frame=400, ts=117.0)
    assert not [e for e in events2 if isinstance(e, MeldFormed)]


def test_visual_fallback_new_meld_still_emitted():
    """同一玩家稍后的不同副露(碰 8)照常发 — 永久去重不误伤。"""
    inf = MeldInferrer(_eid())
    tr = HandTracker()
    v = _vanish(tile=9, eid=3, ts=10.0)
    discards = [('left_river', 5, 11.0, 31)]
    inf.tick([v], [], discards, [], tr, frame=60, ts=12.0)
    g = VisualMeld(player='left_river', tiles=(8, 8, 8),
                   cx=258.0, cy=459.0, bbox=(226.0, 411.0, 285.0, 503.0))
    events = inf.tick([], [], [], [g], tr, frame=400, ts=117.0)
    melds = [e for e in events if isinstance(e, MeldFormed)]
    assert len(melds) == 1
    assert melds[0].tiles == (8, 8, 8)
