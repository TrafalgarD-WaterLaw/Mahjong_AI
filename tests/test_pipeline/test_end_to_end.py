"""端到端回归 — 合成对局(含漏检/抖动/碰截断)解码准确率与熵诊断。"""

from tests.test_pipeline.synth import make_decoder, synth_game


def test_end_to_end_accuracy_and_entropy():
    events = synth_game(seed=7, miss_every=9)
    dec = make_decoder(k=3)   # 小窗口: 多冻结、重解频繁, 更严苛
    for _truth, ev in events:
        dec.add(ev)
    # 冻结事件的 MAP 归属 vs 真值
    frozen = dec.frozen()
    truth = {ev.eid: p for p, ev in events}
    correct = sum(1 for a in frozen
                  if truth.get(a.eid) == a.player)
    assert len(frozen) >= 8, f'冻结数过少: {len(frozen)}'
    acc = correct / len(frozen)
    assert acc >= 0.85, f'归属准确率 {acc:.0%} < 85%'
    # 自诊断: 错误归属的熵应高于正确归属(证据不足 ≠ 假装确定)
    wrong = [a for a in frozen if truth.get(a.eid) != a.player]
    right = [a for a in frozen if truth.get(a.eid) == a.player]
    if wrong:
        assert (sum(a.entropy() for a in wrong) / len(wrong)
                >= sum(a.entropy() for a in right) / len(right))
