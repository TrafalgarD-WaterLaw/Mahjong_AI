"""客户端回退单测 — 事件流管道路径(无归属框)下牌河显示读快照投影。"""

import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from src.mahjong_ai.session import Advice  # noqa: E402
from src.mahjong_ai.state.snapshot import GameSnapshot, PlayerState  # noqa: E402
from src.mahjong_ui.client import _InfoPanel  # noqa: E402


@pytest.fixture(scope='session', autouse=True)
def _qapp() -> QApplication:
    """_InfoPanel 创建 QWidget — 需要 QApplication 实例(离屏平台)。"""
    return QApplication.instance() or QApplication([])


def test_visible_tiles_fallback_to_snapshot_river():
    snap = GameSnapshot(
        my_hand=[1, 2, 3],
        players={'my_river': PlayerState(river=[5, 8]),
                 'right_river': PlayerState(river=[]),
                 'top_river': PlayerState(river=[]),
                 'left_river': PlayerState(river=[])})
    panel = _InfoPanel()
    panel.set_advice(Advice(snapshot=snap), boxes={'hand': []})
    assert panel._visible_tiles('my_river') == [5, 8]


def test_display_hand_holds_through_collapse():
    """手牌塌 0 的数秒内显示冻结在最后一次非空手牌(抽风宽限)。"""
    panel = _InfoPanel()
    panel.set_advice(Advice(snapshot=GameSnapshot(
        my_hand=[1, 2, 3],
        players={p: PlayerState()
                 for p in ('my_river', 'right_river', 'top_river',
                           'left_river')})))
    assert panel._display_hand() == [1, 2, 3]
    panel.set_advice(Advice(snapshot=GameSnapshot(
        my_hand=[],
        players={p: PlayerState()
                 for p in ('my_river', 'right_river', 'top_river',
                           'left_river')})))  # 塌 0
    assert panel._display_hand() == [1, 2, 3]  # 宽限期内冻结显示


def test_empty_list_boxes_still_win():
    """旧管道空列表(river 键存在但无框) → 显示为空, 不回退到历史。"""
    snap = GameSnapshot(
        my_hand=[], players={'my_river': PlayerState(river=[5, 8]),
                             'right_river': PlayerState(river=[]),
                             'top_river': PlayerState(river=[]),
                             'left_river': PlayerState(river=[])})
    panel = _InfoPanel()
    panel.set_advice(Advice(snapshot=snap), boxes={'my_river': []})
    assert panel._visible_tiles('my_river') == []


def test_boxes_still_win_when_present():
    """旧路径不变: 有归属框时仍以框为准。"""
    from src.mahjong_cv.detections import TileDet

    snap = GameSnapshot(
        my_hand=[], players={p: PlayerState()
                             for p in ('my_river', 'right_river',
                                       'top_river', 'left_river')})
    panel = _InfoPanel()
    panel.set_advice(Advice(snapshot=snap),
                     boxes={'my_river': [TileDet(tile=3, x1=0, y1=0,
                                                 x2=10, y2=10, conf=0.9)]})
    assert panel._visible_tiles('my_river') == [3]
