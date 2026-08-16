"""检测框悬浮层 — 覆盖游戏窗口的透明层, 实时画归属色细框。

- 全区域透明 + 点击穿透(不挡游戏操作)
- 只画细框(2px)不画标签 — 标签黑底遮挡牌面, 被 mss 截进
  下一帧截图会污染检测; 细线影响可忽略
- 分组颜色(与客户端牌河区/图例一致):
  手牌=绿  副露=蓝  我=黄  右家(下家)=红  上家(对家)=青  左家(上家)=橙
"""

from __future__ import annotations

from collections.abc import Callable

from src.mahjong_cv.detections import TileDet

#: 分组颜色(BGR 风格元组, QColor 用 RGB — 传入时翻转)
_COLORS = {
    'hand': (0, 220, 0),         # 绿
    'meld': (255, 0, 255),       # 洋红
    'my_river': (0, 215, 255),   # 黄
    'right_river': (0, 0, 255),  # 红
    'top_river': (255, 200, 0),  # 青
    'left_river': (0, 165, 255), # 橙
}


class DetOverlay:
    """无边框置顶透明框层, 覆盖 rect_fn 返回的窗口区域。"""

    def __init__(
        self, rect_fn: Callable[[], tuple[int, int, int, int] | None]
    ) -> None:
        from PySide6.QtCore import Qt, QTimer  # noqa: PLC0415
        from PySide6.QtWidgets import QWidget  # noqa: PLC0415

        self._rect_fn = rect_fn
        self._dets: dict[str, list[TileDet]] = {}

        self._widget = QWidget()
        self._widget.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput  # 点击穿透
        )
        self._widget.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._widget.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._widget.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        setattr(self._widget, 'paintEvent', self._paint_event)  # noqa: B010

        self._timer = QTimer()
        self._timer.timeout.connect(self._refresh)
        self._timer.start(100)

    def set_dets(self, dets: dict[str, list[TileDet]]) -> None:
        """更新检测框(分组: hand/meld/各家牌河; 截图坐标 = 窗口内坐标)。"""
        self._dets = dets
        if not getattr(self, '_debugged', False):
            rect = self._rect_fn()
            all_dets = [d for v in dets.values() for d in v]
            if all_dets:
                xs = [d.x1 for d in all_dets] + [d.x2 for d in all_dets]
                ys = [d.y1 for d in all_dets] + [d.y2 for d in all_dets]
                print(f'[DETS] 窗口rect={rect} 截图框 x∈[{min(xs):.0f},'
                      f'{max(xs):.0f}] y∈[{min(ys):.0f},{max(ys):.0f}] '
                      f'widgetDPR={self._widget.devicePixelRatio()} '
                      f'截图{rect[2]}x{rect[3]}')
            else:
                print(f'[DETS] 窗口rect={rect} 检测框为空')
            self._debugged = True
        self._widget.update()

    def _paint_event(self, _event: object) -> None:
        from PySide6.QtGui import QColor, QPainter, QPen  # noqa: PLC0415

        painter = QPainter(self._widget)
        dpr = getattr(self, '_dpr', 1.0) or 1.0
        if dpr != 1.0:
            painter.scale(1.0 / dpr, 1.0 / dpr)  # 物理框坐标 → 逻辑绘制
        # 只画细框 + 归属色, 不画标签 — 标签黑底会遮挡牌面,
        # 被 mss 截进下一帧截图污染检测; 细线(2px)影响可忽略
        for group, boxes in self._dets.items():
            b, g, r = _COLORS.get(group, (128, 128, 128))
            painter.setPen(QPen(QColor(r, g, b), 2))
            for d in boxes:
                painter.drawRect(int(d.x1), int(d.y1),
                                 int(d.x2 - d.x1), int(d.y2 - d.y1))
        painter.end()

    def _refresh(self) -> None:
        rect = self._rect_fn()
        if rect is None:
            self._widget.hide()
            return
        left, top, width, height = rect
        # 窗口状态异常(最小化/句柄失效时 ClientToScreen 返回垃圾坐标):
        # 跳过定位, 避免 Qt 钳制警告与错位
        if (width <= 0 or height <= 0 or left < -10000 or top < -10000
                or left > 100000 or top > 100000):
            self._widget.hide()
            return
        # DPI 换算: mss/Win32 是物理像素, Qt widget 是逻辑像素
        # (High-DPI scaling 下 devicePixelRatio = 系统缩放比, 如 1.5)。
        # 不换算会把悬浮窗按逻辑坐标放置 → 整体漂移向右下角。
        dpr = self._widget.devicePixelRatio() or 1.0
        if dpr <= 0:
            dpr = 1.0
        self._dpr = dpr
        self._widget.setGeometry(
            round(left / dpr), round(top / dpr),
            round(width / dpr), round(height / dpr))
        self._widget.show()
