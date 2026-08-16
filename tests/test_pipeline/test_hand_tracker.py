"""手牌账本 + 台阶签名 — 塌陷兜底与副露内容推断。"""

from src.mahjong_ai.pipeline.hand_tracker import HandTracker


def _stable(tracker: HandTracker, hand: list[int], t0: float,
            ticks: int = 5, dt: float = 0.1) -> float:
    """喂一个稳定纪元(计数连续不变 ticks 帧), 返回结束时刻。"""
    t = t0
    for _ in range(ticks):
        t += dt
        tracker.tick(hand, t)
    return t


def test_stable_epoch_builds_snapshot_and_fallback_none():
    tr = HandTracker()
    t = _stable(tr, list(range(13)), 0.0)
    assert tr.fallback_hand(t) is None  # 健康: 不需要兜底


def test_step_signature_on_drop_two():
    tr = HandTracker()
    t = _stable(tr, list(range(13)), 0.0)
    out = None
    for _ in range(50):  # 手牌 13→11(副露), 保持 11
        t += 0.1
        out = tr.tick(list(range(11)), t) or out
    assert out is not None
    assert out.drop == 2


def test_oscillation_no_step():
    """抽风振荡(13→7→4→10…)不构成台阶。"""
    tr = HandTracker()
    _stable(tr, list(range(13)), 0.0)
    seq = [7, 4, 10, 5, 0, 3, 2, 5, 4, 4, 4, 3, 0]
    t = 5.0
    steps = []
    for n in seq:
        t += 0.15
        s = tr.tick(list(range(max(n, 1))), t)
        if s:
            steps.append(s)
    assert steps == []


def test_fallback_hand_deductions():
    """塌陷兜底: 最后已知 − 我的打出 − 副露扣除, 未知槽 = None。"""
    tr = HandTracker()
    t = _stable(tr, list(range(13)), 0.0)
    # 塌陷后我打出 5, 副露扣除(内容未知, drop=2)
    tr.on_my_discard(5, t + 0.5)
    tr.on_my_meld([], 2)
    fb = tr.fallback_hand(t + 3.0)
    assert fb is not None
    assert 5 not in fb
    known = [x for x in fb if x is not None]
    assert len(known) == 10        # 13 − 1 − 2
    assert fb.count(None) == 0     # 未知槽 = 稳定计数 − 已知
    assert len(fb) == 10


def test_hand_delta_gives_meld_tiles():
    """副露内容 = 前后稳定手牌差, 剔除窗口内打出的牌。"""
    tr = HandTracker()
    hand13 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 10, 11]
    t = _stable(tr, hand13, 0.0)
    # 持续稳定期快照刷新(每 2s 重记)
    t = _stable(tr, hand13, t, ticks=20)
    vanish_ts = t + 0.5
    # 副露: 手牌 13 → 11(打出 8,8 进副露) → 稳定
    t = _stable(tr, [x for x in hand13 if x != 8], vanish_ts + 0.3)
    delta = tr.hand_delta(vanish_ts)
    assert delta == [8, 8]


def test_hand_delta_none_when_no_snapshot():
    tr = HandTracker()
    assert tr.hand_delta(1.0) is None
