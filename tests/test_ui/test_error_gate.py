"""paint 兜底节流 — 同一异常 5 秒内只记一次日志(不依赖 Qt)。"""

from src.mahjong_ui.client import _ErrorGate


def test_error_gate_throttles(monkeypatch):
    gate = _ErrorGate()
    calls = []
    now = [0.0]

    monkeypatch.setattr('src.mahjong_ui.client.time.monotonic',
                        lambda: now[0])
    monkeypatch.setattr('src.mahjong_ui.client.traceback.print_exception',
                        lambda exc: calls.append(exc))
    gate.log(ValueError('a'))
    gate.log(ValueError('b'))   # 0 秒后 → 节流
    assert len(calls) == 1 and str(calls[0]) == 'a'
    now[0] = 5.0                # 满 5 秒 → 放行
    gate.log(ValueError('c'))
    assert len(calls) == 2 and str(calls[1]) == 'c'


def test_error_gate_new_instance_independent(monkeypatch):
    """A 记过日志不影响 B 的节流(状态实例独立, 不复用类级时间戳)。"""
    a, b = _ErrorGate(), _ErrorGate()
    calls = []
    now = [0.0]

    monkeypatch.setattr('src.mahjong_ui.client.time.monotonic',
                        lambda: now[0])
    monkeypatch.setattr('src.mahjong_ui.client.traceback.print_exception',
                        lambda exc: calls.append(exc))
    a.log(ValueError('a'))
    b.log(ValueError('b'))   # b 不受 a 的节流影响
    assert len(calls) == 2
