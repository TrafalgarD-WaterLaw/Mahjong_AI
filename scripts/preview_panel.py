"""客户端面板离屏渲染预览 — 多尺寸 PNG(排版验收: 任意尺寸不挤不溢)。

用法:
    QT_QPA_PLATFORM=offscreen python scripts/preview_panel.py
输出: data/preview/panel_<W>x<H>.png(每尺寸两张: 推荐分支 + 听牌分支)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from src.mahjong_ai.efficiency.discard_selector import (  # noqa: E402
    DiscardRecommendation,
)
from src.mahjong_ai.session import Advice, InferenceResult  # noqa: E402
from src.mahjong_ai.state.snapshot import (  # noqa: E402
    TURN_MY,
    GameSnapshot,
    Meld,
    PlayerState,
)
from src.mahjong_engine.judges.tenpai_judge import WaitingTile  # noqa: E402

#: 预览尺寸(覆盖三档 + 常用尺寸 + 极值)
_SIZES = [(380, 480), (400, 900), (420, 600), (430, 600), (560, 620),
          (700, 700), (1000, 800)]


class _Det:
    """最小检测框对象(_boxes 用: 只有 tile 被牌河显示消费)。"""

    def __init__(self, tile: int) -> None:
        self.tile = tile


def _fake_advice() -> Advice:
    """合成一帧内容齐全的建议(四家牌河/副露/听牌/放铳/待定牌)。"""
    players = {
        'my_river': PlayerState(
            hand_count=13,
            river=[1, 5, 9, 17, 22, 28, 30, 31, 2, 8, 33, 27],
            melds=[Meld(kind='pong', tile=20, tiles=(20, 20, 20))]),
        'right_river': PlayerState(
            hand_count=10, river=[3, 6, 10, 18, 26, 29, 32, 0, 4, 15],
            melds=[Meld(kind='pong', tile=7, tiles=(7, 7, 7)),
                   Meld(kind='kong', tile=33, tiles=(33, 33, 33, 33))]),
        'top_river': PlayerState(
            hand_count=13,
            river=[11, 12, 13, 14, 15, 16, 21, 23, 24, 25, 19, 7, 8, 9]),
        'left_river': PlayerState(
            hand_count=11,
            river=[2, 17, 26, 29, 30, 1, 10, 19, 28, 0, 9],
            melds=[Meld(kind='chi', tile=4, tiles=(3, 4, 5))]),
    }
    snap = GameSnapshot(
        my_hand=[1, 2, 4, 5, 9, 9, 9, 18, 19, 20, 27, 27, 27, 30],
        turn=TURN_MY, players=players)
    return Advice(
        snapshot=snap,
        discard=DiscardRecommendation(
            tile=30, reason='打出 南 后 1 向听, 有效进张 28 枚(副露 1 组)',
            shanten_before=1, shanten_after=1, acceptance=0),
        waiting=None,
        inference=InferenceResult(
            tenpai_probs={'right_river': 0.23, 'top_river': 0.61,
                          'left_river': 0.08},
            waiting={'right_river': [7, 27], 'top_river': [22, 3, 26],
                     'left_river': [11]},
            discard_risk={0: 0.12, 1: 0.03, 3: 0.18, 8: 0.07, 17: 0.05,
                          22: 0.02, 26: 0.11, 29: 0.0},
        ),
    )


def _fake_advice_tenpai() -> Advice:
    """听牌分支(中央"你已听牌"): 与 _fake_advice 同快照, 无推荐。"""
    adv = _fake_advice()
    adv.discard = None
    adv.waiting = [WaitingTile(tile=4), WaitingTile(tile=27)]
    return adv


def main() -> None:
    from PySide6.QtGui import QImage  # noqa: PLC0415
    from PySide6.QtWidgets import QApplication  # noqa: PLC0415

    from src.mahjong_ui.client import _InfoPanel  # noqa: PLC0415

    app = QApplication.instance() or QApplication([])
    panel = _InfoPanel()
    provisional = {'my_river': [30], 'right_river': [7]}
    boxes = {'right_river': [_Det(7)], 'my_river': [_Det(30)]}
    out = Path('data/preview')
    out.mkdir(parents=True, exist_ok=True)
    for w, h in _SIZES:
        for suffix, adv in (('discard', _fake_advice()),
                            ('tenpai', _fake_advice_tenpai())):
            panel.set_advice(adv, boxes=boxes, provisional=provisional)
            panel._widget.resize(w, h)  # noqa: SLF001
            img = QImage(w, h, QImage.Format.Format_RGB32)
            img.fill(0)
            panel._widget.render(img)  # noqa: SLF001 — 目标 = QPaintDevice
            path = out / f'panel_{w}x{h}_{suffix}.png'
            img.save(str(path))
            print(f'saved {path}')
    print('done')


if __name__ == '__main__':
    main()
