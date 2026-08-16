"""客户端布局规划 — 纯函数(不依赖 Qt/PySide6, 可单测)。

三档布局 + 降级链(设计: docs/superpowers/specs/2026-08-16-client-redesign-design.md):
  窄档 W<420: 纵排(对手条×3 带标签 + 推荐 + 手牌 + 放铳 + 副露 + 牌河)
  中档 420≤W<700: 微型牌桌(对家横条/左右竖条/中央推荐/底部我的区)
  宽档 W≥700: 牌桌放大, 全信息展开
paint 只消费 LayoutPlan, 不自算布局; plan_layout 对任意输入不抛
异常(入口钳制到 Qt 最小尺寸之上 — 与 client.py setMinimumSize 一致)。
"""

from __future__ import annotations

from dataclasses import dataclass

#: 窗口最小尺寸(Qt 钳制 + 纯函数入口兜底一致)
MIN_W, MIN_H = 360, 480
#: 分档阈值(按窗口宽度)
NARROW_MAX = 420      # W < 420 → 窄档
WIDE_MIN = 700        # W >= 700 → 宽档
#: 副露带高(与 client._MELD_BAND_H 语义一致)
BAND_H = 28
#: 窄档固定结构
_ROW_H = 30           # 对手条行高
_ROW_GAP = 4
_ADVICE_NARROW_H = 110
#: 图例(W >= 560 才画 — 降级链 2)
_LEGEND_MIN_W = 560
_LEGEND_H = 14

Rect = tuple[int, int, int, int]  # (x, y, w, h)


@dataclass(frozen=True)
class LayoutPlan:
    """一帧布局规划(纯数据, painter 只读)。"""

    tier: str                      # 'narrow' | 'medium' | 'wide'
    rects: dict[str, Rect]
    show_waiting: bool             # 听牌组合小块(降级链 1)
    fonts: dict[str, int]          # {'base', 'small', 'title'}


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def fit_count(space: int, tile_w: int, gap: int, n: int) -> tuple[int, int]:
    """放得下几张画几张: (可见张数, 余数)。余数由调用方画 +N 徽章。

    降级链 5: 牌河明细永不裁剪溢出 — 画不下的张数以 +N 呈现。
    """
    if tile_w <= 0 or space <= 0 or n <= 0:
        return 0, n
    per = tile_w + gap
    fit = min(n, max(1, (space + gap) // per))
    return fit, n - fit


def plan_layout(w: int, h: int) -> LayoutPlan:
    """任意 (w, h) → 布局规划(不抛异常)。"""
    w, h = max(MIN_W, w), max(MIN_H, h)
    if w < NARROW_MAX:
        return _plan_narrow(w, h)
    tier = 'wide' if w >= WIDE_MIN else 'medium'
    return _plan_table(w, h, tier)


#: +N 余数徽章预留宽度(副露带容量预算用)
_REST_BADGE_W = 26


def band_fit(avail: int, tile_w: int, gap: int, n: int) -> tuple[int, int]:
    """副露带容量: (画几张, 余数) — 有截断时给 +N 徽章让位。

    与 fit_count 的区别: 放不下时先预留 +N 徽章宽度(26px)再算
    张数 — 保证"+N"永远完整画在带内(副露牌块永不降级, 放不下
    的以 +N 呈现; 但 +N 本身不能溢出 — 左右家窄区实测副露牌块
    压到听牌徽章上的根因修复)。全部放得下时不做预留。
    """
    fit, rest = fit_count(avail, tile_w, gap, n)
    if rest == 0:
        return fit, 0
    fit = fit_count(avail - _REST_BADGE_W, tile_w, gap, n)[0]
    return fit, n - fit


def _plan_narrow(w: int, h: int) -> LayoutPlan:
    """窄档: 纵排 — 对手条(带标签) → 推荐 → 手牌 → 放铳 → 副露 → 牌河。"""
    m = 4
    rects: dict[str, Rect] = {}
    y = m
    for key in ('opp_right', 'opp_top', 'opp_left'):
        rects[key] = (m, y, w - 2 * m, _ROW_H)
        y += _ROW_H + _ROW_GAP
    y += 4
    rects['advice'] = (m, y, w - 2 * m, _ADVICE_NARROW_H)
    y += _ADVICE_NARROW_H + 4
    hand_h = _clamp(int(h * 0.12), 40, 64)
    rects['my_hand'] = (m, y, w - 2 * m, hand_h)
    y += hand_h + 4
    rects['my_risk'] = (m, y, w - 2 * m, 34)
    y += 34 + 4
    rects['my_meld'] = (m, y, w - 2 * m, BAND_H)
    y += BAND_H + 4
    if h - y >= 60:  # 降级链 4: 剩余高度足够才显示我的牌河
        rects['my_river'] = (m, y, w - 2 * m, h - y - m)
    return LayoutPlan(tier='narrow', rects=rects, show_waiting=False,
                      fonts={'base': 9, 'small': 8, 'title': 10})


def _plan_table(w: int, h: int, tier: str) -> LayoutPlan:
    """中/宽档: 微型牌桌 — 对家横条/左右竖条/中央推荐/我的区(含子区)。"""
    wide = tier == 'wide'
    m = _clamp(w // 30, 8 if wide else 6, 20 if wide else 16)
    top_h = _clamp(int(h * (0.16 if wide else 0.14)),
                   40 if wide else 34, 90 if wide else 64)
    side_w = _clamp(int(w * (0.23 if wide else 0.21)),
                    140 if wide else 84, 230 if wide else 150)
    mid_h = _clamp(int(h * (0.40 if wide else 0.38)),
                   200 if wide else 160, 340 if wide else 260)
    rects: dict[str, Rect] = {}
    top_y = m
    if not wide and w < _LEGEND_MIN_W:
        pass  # 降级链 2: 中档窄段不画图例, 对家横条从顶部开始
    else:
        rects['legend'] = (m, 2, w - 2 * m, _LEGEND_H)
        top_y = 2 + _LEGEND_H + 2
    rects['top'] = (m, top_y, w - 2 * m, top_h)
    mid_y = top_y + top_h + m
    rects['left'] = (m, mid_y, side_w, mid_h)
    rects['right'] = (w - m - side_w, mid_y, side_w, mid_h)
    ax = m + side_w + 6
    rects['advice'] = (ax, mid_y + 4, w - 2 * (m + side_w + 6), mid_h - 8)
    my_y = mid_y + mid_h + m
    hand_h = _clamp(int(h * 0.12), 40, 64)
    y = my_y
    rects['my_hand'] = (m, y, w - 2 * m, hand_h)
    y += hand_h + 4
    rects['my_risk'] = (m, y, w - 2 * m, 34)
    y += 34 + 4
    rects['my_meld'] = (m, y, w - 2 * m, BAND_H)
    y += BAND_H + 4
    rects['my_river'] = (m, y, w - 2 * m, max(1, h - y - m))
    fonts = ({'base': 12, 'small': 10, 'title': 14} if wide
             else {'base': 10, 'small': 8, 'title': 11})
    return LayoutPlan(tier=tier, rects=rects, show_waiting=True, fonts=fonts)


__all__ = ['BAND_H', 'LayoutPlan', 'MIN_H', 'MIN_W', 'NARROW_MAX',
           'WIDE_MIN', 'band_fit', 'fit_count', 'plan_layout']
