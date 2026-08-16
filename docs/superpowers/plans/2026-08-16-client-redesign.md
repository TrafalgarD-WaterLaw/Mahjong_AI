# 客户端排版样式重设计 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 客户端任意窗口尺寸都不挤、不溢出 — 三档布局 + 降级链, paint 兜底, 删除死代码, 接口不变。

**Architecture:** 布局数学抽成纯模块 `src/mahjong_ui/layout.py`(`plan_layout(w, h) -> LayoutPlan`, 无 Qt 依赖可单测), `client.py` 的 paint 只消费 LayoutPlan; 尺寸扫描测试证明"任意尺寸不越界、不重叠、必保元素恒有正尺寸"。

**Tech Stack:** Python 3.12, PySide6(保持函数内延迟导入), pytest。

## Global Constraints

- 无 git: 项目明确不使用 git(用户决定)— 所有 commit 步骤跳过
- 数值以 spec `docs/superpowers/specs/2026-08-16-client-redesign-design.md` 为准: 档阈值 W<420 窄 / 420≤W<700 中 / W≥700 宽; 窗口最小 360×480; 听牌徽章阈值 30%/60%; 图例仅 W≥560; 字号窄 8-9/中 8-11/宽 10-14
- 接口不变: `MahjongClient.update(advice, boxes, provisional)`、`set_status` 签名不动; `run_assistant.py` 零改动
- 现有 342 测试保持全绿
- 代码风格与现有文件一致: 中文注释、`from __future__ import annotations`、PySide6 函数内延迟导入

---

### Task 1: layout.py 布局规划纯模块

**Files:**
- Create: `src/mahjong_ui/layout.py`
- Create: `tests/test_ui/__init__.py`(空文件)
- Create: `tests/test_ui/test_layout.py`

**Interfaces:**
- Produces: `fit_count(space: int, tile_w: int, gap: int, n: int) -> tuple[int, int]`(可见张数, 余数); `LayoutPlan`(frozen dataclass: `tier: str`, `rects: dict[str, tuple[int, int, int, int]]`, `show_waiting: bool`, `fonts: dict[str, int]`); `plan_layout(w: int, h: int) -> LayoutPlan`; 常量 `MIN_W, MIN_H, NARROW_MAX, WIDE_MIN, BAND_H`
- rects 键契约(Task 3 消费): 窄档 `opp_right/opp_top/opp_left/advice/my_hand/my_risk/my_meld/my_river(条件)`; 中/宽档 `top/left/right/advice/my_hand/my_risk/my_meld/my_river/legend(条件)`; 所有 rect 为 `(x, y, w, h)`

- [ ] **Step 1: 写失败测试**

```python
"""布局规划纯函数单测 — 三档/降级链/任意尺寸不变量。"""

from src.mahjong_ui.layout import (
    MIN_H, MIN_W, LayoutPlan, fit_count, plan_layout,
)


def _inside(rect, w, h) -> bool:
    x, y, rw, rh = rect
    return rw > 0 and rh > 0 and x >= 0 and y >= 0 \
        and x + rw <= w and y + rh <= h


def _overlap(a, b) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def test_fit_count_exact_fit():
    assert fit_count(86, 20, 2, 4) == (4, 0)   # 4×20+3×2=86


def test_fit_count_partial_and_rest():
    assert fit_count(50, 20, 2, 5) == (2, 3)


def test_fit_count_zero_guards():
    assert fit_count(0, 20, 2, 5) == (0, 5)
    assert fit_count(100, 0, 2, 5) == (0, 5)
    assert fit_count(100, 20, 2, 0) == (0, 0)


def test_tier_thresholds():
    assert plan_layout(100, 100).tier == 'narrow'   # 入口钳制后仍窄
    assert plan_layout(419, 480).tier == 'narrow'
    assert plan_layout(420, 480).tier == 'medium'
    assert plan_layout(699, 480).tier == 'medium'
    assert plan_layout(700, 480).tier == 'wide'
    assert plan_layout(2000, 1200).tier == 'wide'


def test_narrow_structure():
    plan = plan_layout(380, 480)
    r = plan.rects
    assert set(r) >= {'opp_right', 'opp_top', 'opp_left', 'advice',
                      'my_hand', 'my_risk', 'my_meld'}
    # 对手条按序纵排, 其后推荐/手牌/放铳/副露依次
    assert r['opp_right'][1] < r['opp_top'][1] < r['opp_left'][1] \
        < r['advice'][1] < r['my_hand'][1] < r['my_risk'][1] < r['my_meld'][1]
    assert plan.show_waiting is False  # 窄档降级链 1: 听牌组合省略
    assert 'legend' not in r           # 窄档无图例(对手条有标签)
    assert 'my_river' in r             # 合法高度下窄档余量恒 ≥60
    assert plan.fonts == {'base': 9, 'small': 8, 'title': 10}


def test_medium_structure_and_legend_flag():
    plan = plan_layout(430, 600)
    r = plan.rects
    assert set(r) >= {'top', 'left', 'right', 'advice', 'my_hand',
                      'my_risk', 'my_meld', 'my_river'}
    assert 'legend' not in r  # W < 560 不画图例(降级链 2)
    assert plan.show_waiting is True
    assert plan.fonts == {'base': 10, 'small': 8, 'title': 11}
    assert 'legend' in plan_layout(560, 600).rects


def test_wide_all_shown():
    plan = plan_layout(900, 700)
    assert 'legend' in plan.rects
    assert plan.show_waiting is True
    assert plan.fonts == {'base': 12, 'small': 10, 'title': 14}


def test_must_keep_positive_at_min_size():
    plan = plan_layout(MIN_W, MIN_H)
    for k in ('my_hand', 'my_risk', 'advice'):
        x, y, w, h = plan.rects[k]
        assert w > 0 and h > 0, k
    assert plan.rects['opp_right'][3] > 0


def test_sweep_no_overflow_no_overlap():
    """任意尺寸: 所有区域不越界、两两不重叠(核心不变量)。"""
    for w in range(360, 2001, 80):
        for h in range(480, 1201, 80):
            plan = plan_layout(w, h)
            rects = list(plan.rects.values())
            for rect in rects:
                assert _inside(rect, w, h), f'{w}x{h}: {rect} 越界'
            for i, a in enumerate(rects):
                for b in rects[i + 1:]:
                    assert not _overlap(a, b), f'{w}x{h}: {a} 与 {b} 重叠'
```

- [ ] **Step 2: 运行确认失败**

Run: `PYTHONIOENCODING=utf-8 python -m pytest tests/test_ui/ -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'src.mahjong_ui.layout'`(测试文件先创建 `tests/test_ui/__init__.py` 空文件再跑)

- [ ] **Step 3: 写实现**

```python
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
           'WIDE_MIN', 'fit_count', 'plan_layout']
```

- [ ] **Step 4: 运行确认通过**

Run: `PYTHONIOENCODING=utf-8 python -m pytest tests/test_ui/ -q`
Expected: PASS(9 个测试)

- [ ] **Step 5: 全量回归**

Run: `PYTHONIOENCODING=utf-8 python -m pytest tests/ -q`
Expected: 342 + 9 = 351 passed

---

### Task 2: paint 兜底节流 _ErrorGate

**Files:**
- Create: `tests/test_ui/test_error_gate.py`
- Modify: `src/mahjong_ui/client.py`(新增 `_ErrorGate` 类 — 位置: 模块常量之后)

**Interfaces:**
- Produces: `class _ErrorGate`:`__init__()`、`log(exc: BaseException) -> None` — 同一异常 5 秒内只记一次日志(Task 3 的 `_paint` 兜底使用)

- [ ] **Step 1: 写失败测试**

```python
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


def test_error_gate_new_instance_independent():
    a, b = _ErrorGate(), _ErrorGate()
    assert a._last != b._last  # noqa: SLF001 — 状态独立(不复用类级时间戳)
```

- [ ] **Step 2: 运行确认失败**

Run: `PYTHONIOENCODING=utf-8 python -m pytest tests/test_ui/test_error_gate.py -q`
Expected: FAIL, `ImportError: cannot import name '_ErrorGate'`

- [ ] **Step 3: 写实现**(client.py 顶部 import 区与常量区之间插入)

`client.py` 顶部现有 import 为:

```python
from __future__ import annotations

from src.mahjong_ai.session import Advice
from src.mahjong_core.tile import tile_display
```

改为:

```python
from __future__ import annotations

import time
import traceback

from src.mahjong_ai.session import Advice
from src.mahjong_core.tile import tile_display
```

并在 `_HAND_HOLD_SECONDS = 3.0` 常量之后追加:

```python
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
```

- [ ] **Step 4: 运行确认通过**

Run: `PYTHONIOENCODING=utf-8 python -m pytest tests/test_ui/test_error_gate.py -q`
Expected: PASS(2 个测试)

- [ ] **Step 5: 全量回归**

Run: `PYTHONIOENCODING=utf-8 python -m pytest tests/ -q`
Expected: 353 passed

---

### Task 3: client.py 消费 LayoutPlan 重写 + 预览脚本

**Files:**
- Modify: `src/mahjong_ui/client.py`(全量重写 — 完整新文件内容见下)
- Create: `scripts/preview_panel.py`(离屏渲染多尺寸 PNG, 排版验收)

**Interfaces:**
- Consumes(Task 1): `plan_layout`, `fit_count`, `BAND_H`, `LayoutPlan`(rects 键契约见 Task 1)
- Consumes(Task 2): `_ErrorGate`
- 保持对外接口: `MahjongClient(title).update(advice, boxes, provisional)`、`set_status(text)`; 移除 `set_preview`

- [ ] **Step 1: 重写 client.py(完整内容)**

```python
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
from src.mahjong_ui.layout import BAND_H, fit_count, plan_layout

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
#: 最后一次非空手牌(不闪烁消失)
_HAND_HOLD_SECONDS = 3.0

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
        # 副露牌块(标签右侧)
        player = self._advice.snapshot.players[player_key]
        cx = x + 4 + _ROW_LABEL_W
        ty = y + (h - _MELD_TILE_H) // 2
        for m in player.melds:
            expected = 4 if m.kind == 'kong' else 3
            mtiles = list(m.tiles) if m.tiles else [m.tile] * expected
            for t in mtiles[:expected]:
                self._draw_tile(painter, cx, ty, _MELD_TILE_W,
                                _MELD_TILE_H, t, 8)
                cx += _MELD_TILE_W + 2
            cx += 4
        # 听牌徽章(右对齐) → 牌河 mini 块填充中间空间
        badge_w = self._draw_prob_badge(painter, x + w - 4,
                                        y + (h - 16) // 2, prob)
        river = self._visible_tiles(player_key)
        prov = self._provisional.get(player_key, [])
        total = len(river) + len(prov)
        space = (x + w - 4 - badge_w) - 4 - cx
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
        self._draw_meld_band(painter, plan.rects['my_meld'],
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

    def _draw_meld_band(self, painter, rect, melds, label: bool) -> None:
        """副露带(同规格): 深色底 + 分隔线 + 亮牌块(暗牌位补足张数)。"""
        from PySide6.QtGui import QColor, QFont, QPen  # noqa: PLC0415

        if not melds:
            return
        x, y, w, h = rect
        painter.fillRect(x, y, w, h, QColor(*_BAND))
        painter.setPen(QPen(QColor(*_FELT_LINE), 1))
        painter.drawLine(x, y + h - 1, x + w, y + h - 1)
        mx = x + 7
        if label and w > 150:  # 降级链 3: 窄区省略文字让位给牌块
            painter.setFont(QFont('Microsoft YaHei', 8))
            painter.setPen(QColor(*_DIM))
            painter.drawText(mx, y + 18, '副露')
            mx += 24
        ty = y + (h - _MELD_TILE_H) // 2
        for m in melds:
            expected = 4 if m.kind == 'kong' else 3
            mtiles = list(m.tiles) if m.tiles else [m.tile] * expected
            for t in mtiles[:expected]:
                self._draw_tile(painter, mx, ty, _MELD_TILE_W,
                                _MELD_TILE_H, t, 8)
                mx += _MELD_TILE_W + 2
            # 内容未知(事件推断只知被碰牌): 暗牌位补足张数 —
            # 真实对局: 对面副露只显示一张一万
            for _ in range(max(0, expected - len(mtiles))):
                painter.setPen(QColor(60, 64, 60))
                painter.setBrush(QColor(40, 44, 40))
                painter.drawRoundedRect(mx, ty, _MELD_TILE_W,
                                        _MELD_TILE_H, 3, 3)
                mx += _MELD_TILE_W + 2
            mx += 6

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
        # 听牌徽章(右上, 恒显) → 听牌组合(徽章左侧, 放得下才画)
        badge_w = self._draw_prob_badge(painter, x + w - 6, y + 6, prob)
        left_bound = x + w - 6 - badge_w - 4
        if show_waiting and waiting:
            ww, wh = 16, 22
            n_avail = (left_bound - (x + 6) - 6) // (ww + 2)
            drawn = waiting[:max(0, n_avail)]
            wx = left_bound - len(drawn) * (ww + 2)
            for wt in drawn:
                self._draw_tile(painter, wx, y + 6, ww, wh, wt, 8)
                wx += ww + 2
            left_bound = left_bound - len(drawn) * (ww + 2) - 4
        # 副露带(顶行, 避让徽章/组合)
        inner_top = y + 28
        if player.melds:
            band = (x + 3, y + 4, max(1, left_bound - (x + 3)),
                    BAND_H)
            self._draw_meld_band(painter, band, player.melds, label=True)
            inner_top = y + 4 + BAND_H + 4
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
```

- [ ] **Step 2: 写预览脚本**

```python
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
    from PySide6.QtGui import QImage, QPainter  # noqa: PLC0415
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
            painter = QPainter(img)
            panel._widget.render(painter)  # noqa: SLF001
            painter.end()
            path = out / f'panel_{w}x{h}_{suffix}.png'
            img.save(str(path))
            print(f'saved {path}')
    print('done')


if __name__ == '__main__':
    main()
```

- [ ] **Step 3: 全量回归**

Run: `PYTHONIOENCODING=utf-8 python -m pytest tests/ -q`
Expected: 353 passed(client.py 无单测依赖, 现有测试不受影响)

- [ ] **Step 4: 渲染预览**

Run: `QT_QPA_PLATFORM=offscreen PYTHONIOENCODING=utf-8 python scripts/preview_panel.py`
Expected: `data/preview/` 下 14 张 PNG, 无异常输出

- [ ] **Step 5: 视觉验收(读 PNG 检查)**

用 Read 工具逐张查看 PNG, 检查清单:
- 窄档(380×480): 三条对手条完整(标签/牌块/徽章), 推荐/手牌/放铳/副露/牌河各行不重叠
- 中档(430×600 / 560×620): 牌桌四区 + 中央推荐无挤压; 560 起有图例
- 宽档(700×700 / 1000×800): 全信息展开, 字号放大
- 全部: 无元素出界/重叠/被裁剪; 必保信息(手牌/推荐/放铳/对手听牌/牌河)可见
- 如发现视觉问题: 修正 client.py 对应绘制参数后重跑 Step 4

---

## Self-Review

**1. Spec coverage:**
- §1 三档布局 → Task 1(`_plan_narrow`/`_plan_table`)+ Task 3(`_paint_narrow`/`_paint_table`)✓
- §2 降级链 6 条 → 1 听牌组合(`show_waiting`/`_draw_river_zone` 放得下才画)、2 图例(`_LEGEND_MIN_W`)、3 副露标签(`label and w > 150`)、4 我的牌河(窄档条件 rect + 1 行)、5 牌河 +N(`fit_count` 全部牌面流)、6 理由省略号(`_draw_elided`)✓
- §3 视觉样式 → 徽章 `_draw_prob_badge`、卡片化 `_CARD`、字号分档 `plan.fonts`、色板保留 ✓
- §4 健壮性 → paint 兜底 + `_ErrorGate`(Task 2/3)、布局纯函数(Task 1)、删死代码(set_preview/_draw_preview 不在新文件)、接口不变、最小尺寸 ✓
- §5 测试 → 扫尺寸不变量/必保正尺寸/tier 阈值/降级单调(Task 1 测试)+ 预览视觉验收(Task 3 Step 5)✓

**2. Placeholder scan:** 无 TBD/TODO; 每个代码步骤含完整代码 ✓

**3. Type consistency:** `LayoutPlan.rects` 键(opp_*/top/left/right/advice/my_hand/my_risk/my_meld/my_river/legend)在 Task 1 契约、Task 1 测试、Task 3 消费三处一致; `fit_count(space, tile_w, gap, n) -> (fit, rest)` 签名一致; `_ErrorGate.log(exc)` 一致; `plan.fonts` 键 base/small/title 一致 ✓
