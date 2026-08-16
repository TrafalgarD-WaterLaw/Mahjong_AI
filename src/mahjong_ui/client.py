"""麻将AI助手客户端 — 牌桌俯视信息面板(三档自适应布局)。

面板 = 一张微型牌桌: 四家牌河按真实方位排布(上=对家, 左=上家,
右=下家, 下=我), 中央推荐区, 底部我的手牌。每张牌是标准牌面块
(数字/字 + 花色色: 万红/饼蓝/条绿/风黑/箭红), 牌河随打出实时
增长 — 结构即信息: 位置对应玩家, 不需要标签解释(窄档除外: 纵排
布局失去方位映射, 改用标签)。

布局: src/mahjong_ui/layout.py 的 plan_layout 纯函数规划三档
(窄 <420 纵排 / 中 420-700 牌桌 / 宽 ≥700 放大全展开), paint 只
消费 LayoutPlan; 空间不足按降级链隐藏次要元素(听牌组合 → 图例 →
副露标签 → 我的牌河 → 牌河 +N → 理由缩行), 绝不溢出裁剪(设计:
docs/superpowers/specs/2026-08-16-client-redesign-design.md)。

视觉: 深绿毡布桌面 + 象牙白牌面, 概率危险用朱红/橙 — 全部来自
麻将本身的材质, 不引入无关装饰。

健壮性: paint 全量兜底(_ErrorGate — 同一异常 5 秒只记一次日志,
异常时画"显示异常"提示), 绘制 bug 不再每帧刷屏。
"""

from __future__ import annotations

import time
import traceback

from src.mahjong_ai.session import Advice
from src.mahjong_core.tile import tile_display
from src.mahjong_ui.layout import BAND_H, band_fit, fit_count, plan_layout

# ---- 颜色(麻将材质) ----
_FELT = (30, 59, 47)         # 毡布深绿(背景)
_FELT_LINE = (58, 92, 76)    # 毡布分区线
_CARD = (36, 66, 52)         # 分区卡片底
_BAND = (24, 46, 36)         # 副露带底
_IVORY = (242, 231, 207)     # 牌面象牙白
_IVORY_DARK = (218, 205, 176)  # 牌面暗部(白板/花)
_WAN = (200, 64, 42)         # 万/箭 朱红
_BING = (62, 107, 140)       # 饼 蓝
_TIAO = (76, 138, 76)        # 条 绿
_WIND = (43, 43, 38)         # 风 黑
_TEXT = (232, 223, 200)      # 面板文字米白
_DIM = (150, 158, 143)       # 弱化文字
_WARN = (217, 131, 36)       # 警示橙
_BADGE_DIM = (110, 118, 108)  # 听牌徽章弱档底色(比 _DIM 亮, 保证白字可读)

# ---- 布局常量(与 layout.py 契约一致) ----
#: 窗口最小尺寸(与 layout.MIN_W/MIN_H 一致, Qt 钳制)
_MIN_W, _MIN_H = 360, 480
#: 副露亮牌块尺寸(真实牌块可视化: 碰=三张同牌, 吃=顺子三张)
_MELD_TILE_W, _MELD_TILE_H = 16, 22
#: 窄档对手条: 方位标签宽 / 牌河 mini 块
_ROW_LABEL_W = 18
_MINI_W, _MINI_H = 14, 20
#: 手牌显示保持秒数: 检测抽风手牌塌 0 的数秒内, 显示冻结在
#: 最后一次非空手牌(不闪烁消失) — 真实对局: 结算/抽风期手牌
#: "突然全检测不到"的显示层宽限
_HAND_HOLD_SECONDS = 3.0
#: 听牌组合显示门槛: 低于此概率时 top3 等待牌只是少数听牌粒子的
#: 噪声(开局 2% 先验 = 8 粒子, "还没出牌就有听牌组合" — 用户实测)
_WAITING_MIN_PROB = 0.10

#: 检测框颜色(与悬浮框一致, BGR) — 图例用
_BOX_COLORS = {
    'hand': (0, 220, 0),         # 绿
    'meld': (255, 0, 255),       # 洋红
    'my_river': (0, 215, 255),   # 黄
    'right_river': (0, 0, 255),  # 红
    'top_river': (255, 200, 0),  # 青
    'left_river': (0, 165, 255), # 橙
}


class _ErrorGate:
    """paint 兜底节流: 同一异常 5 秒内只记一次日志(避免每帧刷屏)。

    上次 AttributeError(adv.action)每帧抛一次把日志刷满 — 兜底必须
    节流, 否则"保护"本身变成刷屏源。
    """

    _INTERVAL = 5.0

    def __init__(self) -> None:
        self._last = -float('inf')

    def log(self, exc: BaseException) -> None:
        now = time.monotonic()
        if now - self._last >= self._INTERVAL:
            traceback.print_exception(exc)
            self._last = now


def _tile_glyph(tile: int) -> tuple[str, str, tuple[int, int, int]]:
    """牌面内容: (主体字, 花色小字, 花色色)。

    标准麻将牌面: 数字/字为主体, 数牌下方带花色小字(万/饼/条) —
    颜色只是辅助, 不靠颜色也能认牌。
    """
    if 0 <= tile <= 8:
        return str(tile + 1), '万', _WAN
    if 9 <= tile <= 17:
        return str(tile - 8), '饼', _BING
    if 18 <= tile <= 26:
        return str(tile - 17), '条', _TIAO
    if 27 <= tile <= 30:
        return '东南西北'[tile - 27], '', _WIND
    if 31 <= tile <= 33:
        return '中发白'[tile - 31], '', _WAN
    return '花', '', _IVORY_DARK


class MahjongClient:
    """独立信息窗口(牌桌面板)。"""

    def __init__(self, title: str = '麻将AI助手') -> None:
        from PySide6.QtWidgets import QMainWindow  # noqa: PLC0415

        self._window = QMainWindow()
        self._window.setWindowTitle(title)
        self._window.resize(430, 600)  # 紧凑默认: 塞进游戏窗口旁边
        self._window.setMinimumSize(_MIN_W, _MIN_H)  # 布局数学的下限
        self._panel = _InfoPanel()
        self._window.setCentralWidget(self._panel.widget())
        self._window.show()

    def update(self, advice: Advice, boxes: dict | None = None,
               provisional: dict[str, list[int]] | None = None) -> None:
        """每帧驱动: 更新建议面板(含归属检测框; 待定牌另行传入)。"""
        self._panel.set_advice(advice, boxes, provisional)

    def set_status(self, text: str) -> None:
        """显示状态(如模型加载中); 下次 update 自动替换。"""
        self._panel.set_status(text)


class _InfoPanel:
    """牌桌面板: 直接渲染 Advice(布局由 layout.plan_layout 规划)。"""

    def __init__(self) -> None:
        from PySide6.QtWidgets import QWidget  # noqa: PLC0415

        self._advice: Advice | None = None
        self._status: str | None = None
        self._boxes: dict = {}                  # 归属检测框(tracker 提供)
        self._provisional: dict[str, list[int]] = {}  # 待定牌(未冻结 MAP)
        self._held_hand: list[int] = []         # 手牌显示保持(抽风宽限)
        self._held_hand_ts = 0.0
        self._gate = _ErrorGate()               # paint 兜底节流
        self._widget = QWidget()
        self._widget.setStyleSheet(
            f'background-color: rgb{_FELT};')
        setattr(self._widget, 'paintEvent', self._paint)  # noqa: B010

    def widget(self):
        return self._widget

    def set_advice(self, advice: Advice, boxes: dict | None = None,
                   provisional: dict[str, list[int]] | None = None) -> None:
        self._advice = advice
        if boxes is not None:
            self._boxes = boxes
        if provisional is not None:
            self._provisional = provisional
        self._status = None
        self._widget.update()

    def set_status(self, text: str) -> None:
        self._advice = None
        self._status = text
        self._widget.update()

    def _display_hand(self) -> list[int | None]:
        """手牌显示(带保持 + 塌陷兜底): 检测抽风手牌塌 0 的数秒内
        冻结在最后一次非空手牌; 超过宽限后用账本兜底(未知槽 = None
        画暗牌位) — 不闪烁消失, 也不显示牌河误判的假手牌。"""
        if self._advice is None:
            return []
        hand = self._advice.snapshot.my_hand
        if hand:
            self._held_hand = hand
            self._held_hand_ts = time.time()
            return hand
        if time.time() - self._held_hand_ts < _HAND_HOLD_SECONDS:
            return self._held_hand
        fb = self._advice.snapshot.my_hand_fallback
        return fb or []

    def _visible_tiles(self, player: str) -> list[int]:
        """牌河显示用: 当前物理可见的牌。

        优先 tracker 归属框(旧管道); 该玩家键不存在时回退快照投影
        (事件流管道: visible_river 已排除被碰走的牌)。用 `is not None`
        而非真值判断 — 旧管道恒写入四家 river 键(可能为空列表)。
        """
        dets = self._boxes.get(player)
        if dets is not None:
            return [d.tile for d in dets]
        if self._advice is not None:
            return list(self._advice.snapshot.players[player].river)
        return []

    # ---- 绘制 ----

    def _paint(self, _event: object) -> None:
        """绘制入口: 全量兜底 — 任何绘制异常只画"显示异常", 不刷屏。"""
        from PySide6.QtGui import QColor, QPainter  # noqa: PLC0415

        painter = QPainter(self._widget)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        try:
            self._paint_inner(painter)
        except Exception as exc:  # noqa: BLE001 — 兜底: 异常不杀 UI 不刷屏
            self._gate.log(exc)
            painter.fillRect(painter.window(), QColor(*_FELT))
            painter.setPen(QColor(*_DIM))
            painter.drawText(painter.window(), 0x84, '显示异常, 详见日志')
        painter.end()

    def _paint_inner(self, painter) -> None:
        from PySide6.QtGui import QColor, QFont  # noqa: PLC0415

        painter.fillRect(painter.window(), QColor(*_FELT))
        if self._status is not None:
            painter.setFont(QFont('Microsoft YaHei', 16,
                                  QFont.Weight.Bold))
            painter.setPen(QColor(*_TEXT))
            painter.drawText(painter.window(), 0x84, self._status)
            return
        if self._advice is None:
            return
        w = painter.window().width()
        h = painter.window().height()
        plan = plan_layout(w, h)
        if plan.tier == 'narrow':
            self._paint_narrow(painter, plan)
        else:
            self._paint_table(painter, plan)

    def _advice_data(self):
        """一帧数据解包: (advice, snapshot, 听牌概率, 听牌组合)。"""
        adv = self._advice
        snap = adv.snapshot
        inf = adv.inference
        probs = inf.tenpai_probs if inf is not None else {}
        waiting = inf.waiting if inf is not None else {}
        return adv, snap, probs, waiting

    def _paint_narrow(self, painter, plan) -> None:
        """窄档: 对手条×3 → 推荐 → 我的区。"""
        adv, snap, probs, waiting = self._advice_data()
        rows = (('opp_right', '右', 'right_river'),
                ('opp_top', '上', 'top_river'),
                ('opp_left', '左', 'left_river'))
        for rect_key, label, player_key in rows:
            self._draw_opp_row(painter, plan.rects[rect_key], label,
                               player_key, probs.get(player_key, 0.0))
        self._draw_advice_zone(painter, plan.rects['advice'], adv, snap,
                               plan.fonts['title'])
        self._draw_my_zone(painter, plan)

    def _paint_table(self, painter, plan) -> None:
        """中/宽档: 图例(条件) → 对家/左右家 → 推荐 → 我的区。"""
        adv, snap, probs, waiting = self._advice_data()
        if 'legend' in plan.rects:
            self._draw_legend(painter, plan.rects['legend'])
        self._draw_river_zone(painter, plan.rects['top'], 'top_river',
                              horizontal=True,
                              show_waiting=plan.show_waiting,
                              waiting=waiting.get('top_river', []))
        self._draw_river_zone(painter, plan.rects['left'], 'left_river',
                              horizontal=False,
                              show_waiting=plan.show_waiting,
                              waiting=waiting.get('left_river', []))
        self._draw_river_zone(painter, plan.rects['right'], 'right_river',
                              horizontal=False,
                              show_waiting=plan.show_waiting,
                              waiting=waiting.get('right_river', []))
        self._draw_advice_zone(painter, plan.rects['advice'], adv, snap,
                               plan.fonts['title'])
        self._draw_my_zone(painter, plan)

    # ---- 组件 ----

    def _draw_tile(self, painter, x: int, y: int, w: int, h: int,
                   tile: int, font_size: int) -> None:
        """标准牌面块: 象牙白圆角 + 主体字 + 花色小字(数牌)。

        布局: 主体字居中偏上(60% 高), 花色小字在下沿 — 与真实牌面一致,
        认牌不依赖颜色。字牌(风/箭/花)只有主体字。
        """
        from PySide6.QtGui import QColor, QFont  # noqa: PLC0415

        painter.setPen(QColor(20, 24, 20))
        painter.setBrush(QColor(*_IVORY))
        painter.drawRoundedRect(x, y, w, h, 4, 4)
        glyph, suit, color = _tile_glyph(tile)
        painter.setPen(QColor(*color))
        painter.setFont(QFont('Microsoft YaHei', font_size,
                              QFont.Weight.Bold))
        if suit:
            # 主体字(上 60%) + 花色小字(下 25%)
            painter.drawText(x, y, w, int(h * 0.62), 0x84, glyph)
            painter.setFont(QFont('Microsoft YaHei', max(7, font_size - 3),
                                  QFont.Weight.Normal))
            painter.drawText(x, y + int(h * 0.58), w, int(h * 0.38), 0x84,
                             suit)
        else:
            painter.drawText(x, y, w, h, 0x84, glyph)

    def _draw_elided(self, painter, x: int, y: int, w: int, h: int,
                     text: str) -> None:
        """预算矩形内画文字, 超宽省略号(降级链 6: 理由缩行不溢出)。"""
        from PySide6.QtCore import Qt  # noqa: PLC0415

        fm = painter.fontMetrics()
        elided = fm.elidedText(text, Qt.TextElideMode.ElideRight, w)
        painter.drawText(x, y, w, h, 0x84 | 0x80, elided)

    def _draw_prob_badge(self, painter, right_x: int, y: int,
                         prob: float) -> int:
        """听牌概率徽章(右对齐, 返回宽度): 弱 <30% / 橙 30-60% / 红 ≥60%。

        圆角 pill 替代角落裸文字 — 参与布局计算, 不再固定偏移。
        """
        from PySide6.QtCore import Qt  # noqa: PLC0415
        from PySide6.QtGui import QColor, QFont  # noqa: PLC0415

        text = f'听{prob:.0%}'
        painter.setFont(QFont('Microsoft YaHei', 8, QFont.Weight.Bold))
        fm = painter.fontMetrics()
        bw = fm.horizontalAdvance(text) + 12
        x = right_x - bw
        if prob >= 0.6:
            color = _WAN
        elif prob >= 0.3:
            color = _WARN
        else:
            color = _BADGE_DIM
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(*color))
        painter.drawRoundedRect(x, y, bw, 16, 8, 8)
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(x, y + 1, bw, 16, 0x84, text)
        return bw

    def _draw_legend(self, painter, rect) -> None:
        """颜色图例: 悬浮框/预览框颜色 ↔ 归属(中档 W≥560 / 宽档)。"""
        from PySide6.QtGui import QColor, QFont  # noqa: PLC0415

        legend = [('hand', '手牌'), ('meld', '副露'),
                  ('my_river', '我'), ('right_river', '右'),
                  ('top_river', '上'), ('left_river', '左')]
        painter.setFont(QFont('Microsoft YaHei', 8))
        x, y, w, h = rect
        cx = x
        for key, name in legend:
            b, g, r = _BOX_COLORS[key]
            painter.fillRect(cx, y + 2, 10, 10, QColor(r, g, b))
            painter.setPen(QColor(*_TEXT))
            painter.drawText(cx + 13, y + 10, name)
            cx += 48

    def _draw_opp_row(self, painter, rect, label: str, player_key: str,
                      prob: float) -> None:
        """窄档对手条: 标签 + 副露牌块 + 牌河 mini 块 + 余数 + 听牌徽章。"""
        from PySide6.QtGui import QColor, QFont  # noqa: PLC0415

        x, y, w, h = rect
        painter.setPen(QColor(*_FELT_LINE))
        painter.setBrush(QColor(*_CARD))
        painter.drawRoundedRect(x, y, w, h, 5, 5)
        painter.setFont(QFont('Microsoft YaHei', 9, QFont.Weight.Bold))
        painter.setPen(QColor(*_TEXT))
        painter.drawText(x + 4, y, _ROW_LABEL_W, h, 0x84 | 0x80, label)
        # 听牌徽章(右对齐)先画 — 副露/牌河容量都以它为右界
        badge_w = self._draw_prob_badge(painter, x + w - 4,
                                        y + (h - 16) // 2, prob)
        badge_left = x + w - 4 - badge_w - 4
        # 副露牌块(标签右侧, band_fit 预算: 放不下的 +N)
        player = self._advice.snapshot.players[player_key]
        cx = x + 4 + _ROW_LABEL_W
        ty = y + (h - _MELD_TILE_H) // 2
        meld_tiles: list[int | None] = []
        for m in player.melds:
            expected = 4 if m.kind == 'kong' else 3
            mtiles = list(m.tiles) if m.tiles else [m.tile] * expected
            meld_tiles.extend(t for t in mtiles[:expected])
            meld_tiles.extend(None for _ in range(
                max(0, expected - len(mtiles))))
        mfit, mrest = band_fit(badge_left - cx, _MELD_TILE_W, 2,
                               len(meld_tiles))
        for t in meld_tiles[:mfit]:
            if t is None:
                painter.setPen(QColor(60, 64, 60))
                painter.setBrush(QColor(40, 44, 40))
                painter.drawRoundedRect(cx, ty, _MELD_TILE_W,
                                        _MELD_TILE_H, 3, 3)
            else:
                self._draw_tile(painter, cx, ty, _MELD_TILE_W,
                                _MELD_TILE_H, t, 8)
            cx += _MELD_TILE_W + 2
        if mrest:
            painter.setFont(QFont('Microsoft YaHei', 8,
                                  QFont.Weight.Bold))
            painter.setPen(QColor(*_DIM))
            painter.drawText(cx, ty, 24, _MELD_TILE_H, 0x84 | 0x80,
                             f'+{mrest}')
            cx += 24
        # 牌河 mini 块填充中间空间
        river = self._visible_tiles(player_key)
        prov = self._provisional.get(player_key, [])
        total = len(river) + len(prov)
        space = badge_left - cx
        fit, rest = fit_count(space, _MINI_W, 2, total)
        my = y + (h - _MINI_H) // 2
        for i, t in enumerate(river[:fit]):
            self._draw_tile(painter, cx + i * (_MINI_W + 2), my,
                            _MINI_W, _MINI_H, t, 7)
        if prov and fit > len(river):
            painter.setOpacity(0.55)
            for i, t in enumerate(prov[:fit - len(river)],
                                  start=len(river)):
                self._draw_tile(painter, cx + i * (_MINI_W + 2), my,
                                _MINI_W, _MINI_H, t, 7)
            painter.setOpacity(1.0)
        if rest:
            painter.setFont(QFont('Microsoft YaHei', 8,
                                  QFont.Weight.Bold))
            painter.setPen(QColor(*_DIM))
            painter.drawText(cx + fit * (_MINI_W + 2), my, 30, _MINI_H,
                             0x84 | 0x80, f'+{rest}')

    def _draw_advice_zone(self, painter, rect, adv, snap,
                          title_font: int) -> None:
        """推荐区(三档共用): 大牌块 + 标题 + 理由(预算矩形内省略号)。"""
        from PySide6.QtGui import QColor, QFont, QPen  # noqa: PLC0415

        x, y, w, h = rect
        painter.setPen(QPen(QColor(*_FELT_LINE), 1))
        painter.setBrush(QColor(34, 63, 50))
        painter.drawRoundedRect(x, y, w, h, 6, 6)
        cx = x + w // 2
        if adv.discard is not None:
            # 推荐打出: 大牌块(自适应) + 标题 + 理由
            tw = min(56, max(34, w // 4), h - 46)
            th = int(tw * 1.4)
            self._draw_tile(painter, cx - tw // 2, y + 10, tw, th,
                            adv.discard.tile, tw // 2)
            painter.setFont(QFont('Microsoft YaHei', title_font,
                                  QFont.Weight.Bold))
            painter.setPen(QColor(*_WARN))
            painter.drawText(x, y + th + 18, w, 20, 0x84, '打出这张')
            painter.setFont(QFont('Microsoft YaHei', max(8, title_font - 2)))
            painter.setPen(QColor(*_TEXT))
            self._draw_elided(painter, x + 6, y + th + 40, w - 12,
                              h - th - 48, adv.discard.reason)
        elif adv.waiting:
            tiles = ' '.join(tile_display(wt.tile) for wt in adv.waiting)
            painter.setFont(QFont('Microsoft YaHei', title_font + 2,
                                  QFont.Weight.Bold))
            painter.setPen(QColor(*_TIAO))
            painter.drawText(x, y + 26, w, 26, 0x84, '你已听牌')
            painter.setFont(QFont('Microsoft YaHei', title_font))
            painter.setPen(QColor(*_TEXT))
            self._draw_elided(painter, x + 6, y + 60, w - 12, 30,
                              f'听 {tiles}')
        else:
            painter.setFont(QFont('Microsoft YaHei', title_font))
            painter.setPen(QColor(*_DIM))
            n = len(snap.my_hand)
            painter.drawText(x, y + h // 2 - 10, w, 22, 0x84,
                             f'等待局面变化…(手牌 {n} 张)')

    def _draw_my_zone(self, painter, plan) -> None:
        """我的区: 手牌(必保) → 放铳风险行(必保) → 副露带 → 牌河。"""
        from PySide6.QtGui import QColor  # noqa: PLC0415

        hand = self._display_hand()
        r = plan.rects['my_hand']
        x, y, w, h = r
        n = max(1, len(hand) or 1)
        gap = 3
        tw = min(int(h * 0.72), max(14, (w - gap * (n - 1)) // n))
        th = int(tw * 1.4)
        for i, t in enumerate(hand):
            if (i + 1) * (tw + gap) - gap > w:
                break
            tx = x + i * (tw + gap)
            ty = y + (h - th) // 2
            if t is None:
                # 未知槽: 暗牌位(兜底手牌 — 摸进的新牌检测不到)
                painter.setPen(QColor(60, 64, 60))
                painter.setBrush(QColor(40, 44, 40))
                painter.drawRoundedRect(tx, ty, tw, th, 4, 4)
                continue
            self._draw_tile(painter, tx, ty, tw, th, t, max(8, tw // 2))
        self._draw_risk_row(painter, plan.rects['my_risk'],
                            self._advice.inference)
        mr = plan.rects['my_meld']
        self._draw_meld_band(painter, mr[0], mr[1], mr[2], mr[3],
                             self._advice.snapshot.players['my_river'].melds,
                             label=True)
        rr = plan.rects.get('my_river')
        if rr is not None:
            self._draw_my_river(painter, rr)

    def _draw_risk_row(self, painter, rect, inf) -> None:
        """放铳风险行(必保): 候选牌小块 + 概率。

        排序确定性(-风险, 牌) — 同值不逐帧换位; ~0 概率过滤
        (粒子抖动 0↔0.001 会闪); 行位固定不随内容出现/消失跳动。
        """
        from PySide6.QtGui import QColor, QFont  # noqa: PLC0415

        if inf is None:
            return
        x, y, w, h = rect
        painter.setFont(QFont('Microsoft YaHei', 9, QFont.Weight.Bold))
        painter.setPen(QColor(*_DIM))
        painter.drawText(x, y, 30, h, 0x84 | 0x80, '放铳')
        rx = x + 34
        rt = min(20, h - 14)
        rw = min(18, rt)
        shown = sorted(((t, r) for t, r in inf.discard_risk.items()
                        if r >= 0.01),
                       key=lambda tr: (-tr[1], tr[0]))
        for t, r in shown[:8]:
            if rx + rw > x + w:
                break
            color = _WAN if r >= 0.15 else (_WARN if r >= 0.05 else _DIM)
            self._draw_tile(painter, rx, y, rw, rt, t, max(8, rw // 2))
            painter.setPen(QColor(*color))
            painter.setFont(QFont('Microsoft YaHei', 7,
                                  QFont.Weight.Bold))
            painter.drawText(rx - 4, y + rt, rw + 8, 12, 0x84, f'{r:.0%}')
            rx += rw + 5

    def _draw_meld_band(self, painter, x: int, y: int, max_w: int,
                        h: int, melds, label: bool) -> int:
        """副露带(内容自适宽): 深色底 + 分隔线 + 亮牌块(暗牌位补足)。

        返回实际使用宽度。牌块经 band_fit 预算, 放不下的以 +N 呈现 —
        左右家窄区修复: 牌块曾无视带宽压到听牌徽章上(用户实测)。
        """
        from PySide6.QtGui import QColor, QFont, QPen  # noqa: PLC0415

        if not melds:
            return 0
        tiles: list[int | None] = []
        for m in melds:
            expected = 4 if m.kind == 'kong' else 3
            mtiles = list(m.tiles) if m.tiles else [m.tile] * expected
            tiles.extend(t for t in mtiles[:expected])
            # 内容未知(事件推断只知被碰牌): 暗牌位补足张数 —
            # 真实对局: 对面副露只显示一张一万
            tiles.extend(None for _ in range(
                max(0, expected - len(mtiles))))
        mx = x + 7
        if label and max_w > 150:  # 降级链 3: 窄区省略文字让位给牌块
            mx += 24
        fit, rest = band_fit(max_w - (mx - x), _MELD_TILE_W, 2,
                             len(tiles))
        used = mx - x + fit * (_MELD_TILE_W + 2) + (24 if rest else 0)
        painter.fillRect(x, y, used, h, QColor(*_BAND))
        painter.setPen(QPen(QColor(*_FELT_LINE), 1))
        painter.drawLine(x, y + h - 1, x + used, y + h - 1)
        if label and max_w > 150:
            painter.setFont(QFont('Microsoft YaHei', 8))
            painter.setPen(QColor(*_DIM))
            painter.drawText(x + 7, y + 18, '副露')
        ty = y + (h - _MELD_TILE_H) // 2
        for t in tiles[:fit]:
            if t is None:
                painter.setPen(QColor(60, 64, 60))
                painter.setBrush(QColor(40, 44, 40))
                painter.drawRoundedRect(mx, ty, _MELD_TILE_W,
                                        _MELD_TILE_H, 3, 3)
            else:
                self._draw_tile(painter, mx, ty, _MELD_TILE_W,
                                _MELD_TILE_H, t, 8)
            mx += _MELD_TILE_W + 2
        if rest:
            painter.setFont(QFont('Microsoft YaHei', 8,
                                  QFont.Weight.Bold))
            painter.setPen(QColor(*_DIM))
            painter.drawText(mx, ty, 24, _MELD_TILE_H, 0x84 | 0x80,
                             f'+{rest}')
        return used

    def _draw_my_river(self, painter, rect) -> None:
        """我的牌河(1 行): 放得下几张画几张 + 余数徽章(降级链 4/5)。"""
        from PySide6.QtGui import QColor, QFont  # noqa: PLC0415

        river = self._visible_tiles('my_river')
        prov = self._provisional.get('my_river', [])
        total = len(river) + len(prov)
        if not total:
            return
        x, y, w, h = rect
        tw = min(18, int(h * 0.72))
        th = int(tw * 1.4)
        gap = 2
        fit, rest = fit_count(w, tw, gap, total)
        for i, t in enumerate(river[:fit]):
            self._draw_tile(painter, x + i * (tw + gap), y, tw, th, t,
                            max(7, tw // 2))
        if prov and fit > len(river):
            painter.setOpacity(0.55)
            for i, t in enumerate(prov[:fit - len(river)],
                                  start=len(river)):
                self._draw_tile(painter, x + i * (tw + gap), y, tw, th, t,
                                max(7, tw // 2))
            painter.setOpacity(1.0)
        if rest:
            painter.setFont(QFont('Microsoft YaHei', 8,
                                  QFont.Weight.Bold))
            painter.setPen(QColor(*_DIM))
            painter.drawText(x + fit * (tw + gap), y, 30, th, 0x84 | 0x80,
                             f'+{rest}')

    def _draw_river_zone(self, painter, rect, player_key: str,
                         horizontal: bool, show_waiting: bool,
                         waiting: list[int] | None) -> None:
        """一家牌河区: 卡片 + 听牌徽章 + 听牌组合 + 副露带 + 牌面流。

        听牌徽章右对齐恒显; 听牌组合放得下才画(降级链 1); 副露带
        避让徽章(降级链 3 窄区省略标签); 牌河经容量预算画 +N(降级链 5)。
        """
        from PySide6.QtGui import QColor, QFont, QPen  # noqa: PLC0415

        x, y, w, h = rect
        snap = self._advice.snapshot
        player = snap.players[player_key]
        inf = self._advice.inference
        prob = inf.tenpai_probs.get(player_key, 0.0) if inf is not None \
            else 0.0
        # 卡片底 + 危险描边(概率越高越危险: 橙→红)
        border = QColor(*_FELT_LINE)
        if prob >= 0.6:
            border = QColor(*_WAN)
        elif prob >= 0.3:
            border = QColor(*_WARN)
        painter.setPen(QPen(border, 2))
        painter.setBrush(QColor(*_CARD))
        painter.drawRoundedRect(x, y, w, h, 6, 6)
        # 副露带(顶行): 有副露时占满整行(左右家窄竖条里徽章+牌块
        # 物理放不下同一行 — 两个都永不降级 → 上下堆叠, 互不遮挡)
        inner_top = y + 28
        if player.melds:
            self._draw_meld_band(painter, x + 3, y + 4, w - 6, BAND_H,
                                 player.melds, label=True)
            badge_y = y + 4 + BAND_H + 2
            inner_top = badge_y + 18 + 2
        else:
            badge_y = y + 6
        # 听牌徽章(右上, 恒显) — 副露/听牌组合都以徽章为右界
        badge_w = self._draw_prob_badge(painter, x + w - 6, badge_y, prob)
        badge_left = x + w - 6 - badge_w - 4
        # 听牌组合(徽章左侧, 放得下才画; 概率过低时是粒子噪声不画)
        if show_waiting and waiting and prob >= _WAITING_MIN_PROB:
            ww, wh = 16, 22
            n_avail = (badge_left - (x + 6) - 6) // (ww + 2)
            drawn = waiting[:max(0, n_avail)]
            wx = badge_left - len(drawn) * (ww + 2)
            wy = badge_y - 3 if player.melds else y + 6
            for wt in drawn:
                self._draw_tile(painter, wx, wy, ww, wh, wt, 8)
                wx += ww + 2
        # 牌面流(尺寸自适应 + 容量预算)
        gap = 2
        inner_x, inner_y = x + 6, inner_top
        inner_w, inner_h = w - 12, y + h - 6 - inner_top
        river = self._visible_tiles(player_key)
        prov = self._provisional.get(player_key, [])
        total = len(river) + len(prov)
        if horizontal:
            tw = max(16, min(26, (inner_w + gap) // 8 - gap))
            th = min(30, int(tw * 1.4))
            cols = max(1, (inner_w + gap) // (tw + gap))
            rows = max(1, (inner_h + gap) // (th + gap))
        else:  # 竖排: 从上到下, 满了换列
            th = max(18, min(30, (inner_h + gap) // 8 - gap))
            tw = min(22, int(th * 0.72))
            rows = max(1, (inner_h + gap) // (th + gap))
            cols = max(1, (inner_w + gap) // (tw + gap))
        cap = cols * rows
        fit = min(total, cap)
        for i, t in enumerate(river[:fit]):
            if horizontal:
                row, col = divmod(i, cols)
            else:
                col, row = divmod(i, rows)
            self._draw_tile(painter, inner_x + col * (tw + gap),
                            inner_y + row * (th + gap), tw, th, t,
                            max(8, tw // 2))
        # 待定牌(半透明)接在已定牌之后
        if prov and fit > len(river):
            painter.setOpacity(0.55)
            for i, t in enumerate(prov[:fit - len(river)],
                                  start=len(river)):
                if horizontal:
                    row, col = divmod(i, cols)
                else:
                    col, row = divmod(i, rows)
                self._draw_tile(painter, inner_x + col * (tw + gap),
                                inner_y + row * (th + gap), tw, th, t,
                                max(8, tw // 2))
            painter.setOpacity(1.0)
        rest = total - fit
        if rest:
            last_i = fit - 1
            if horizontal:
                row, col = divmod(last_i, cols)
            else:
                col, row = divmod(last_i, rows)
            painter.setFont(QFont('Microsoft YaHei', 8,
                                  QFont.Weight.Bold))
            painter.setPen(QColor(*_DIM))
            painter.drawText(inner_x + (col + 1) * (tw + gap),
                             inner_y + row * (th + gap) + 4, 30, th,
                             0x84 | 0x80, f'+{rest}')
