# P1 线上闭环 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在腾讯欢乐麻将 PC 客户端(推倒胡模式)上跑通「截屏 → 识别 → 状态 → 建议 → HUD」实时闭环,实现设计文档 P1 里程碑:打一局全程自动提示。

**Architecture:** 五层可替换管线中的前四层(采集/视觉/状态/决策)+ 展示层 HUD。线上识别采用本项目训练的 YOLO26 屏幕牌检测模型(class id = 牌编码 0-33,与实物模型无关);训练数据由「模板匹配教师」自动打标生成(屏幕牌是固定 2D 图标,模板匹配作为零人工标注的教师)。决策层全部复用现有 mahjong_engine / mahjong_ai 代码,新增欢乐推倒胡规则集(允许吃)。

**Tech Stack:** Python 3.11、ultralytics(YOLO26)、mss(截屏)、ctypes(Win32)、OpenCV、PySide6(HUD)、pytest/mypy/ruff。

## Global Constraints

- **无 git**(用户明确不要),所有任务**没有 commit 步骤**;每任务最后一步是 ruff 检查
- 基线已修复:`src/mahjong_cv/scene_pipeline.py:13` 缩进 bug 已修,pytest 103 全绿
- 质量门禁:每任务 `pytest <本任务测试文件>` 通过 + `ruff check <改动文件>`;最终任务(Task 11)跑全量 `pytest` + `mypy src` + `ruff check src tests scripts`
- 导入模式:测试/脚本一律 `from src.mahjong_* import ...`(pytest 已配置 pythonpath=src;脚本顶部 `sys.path.insert(0, 项目根)`,参考 `scripts/demo.py`)
- **第三方库存根策略**:cv2/ultralytics/mss/PySide6 无 py.typed,`mypy src` 对它们报 `import-not-found`。解决方式(而非 `# type: ignore` 注释,避免 unused-ignore 双向不匹配):在 pyproject.toml 加 `[[tool.mypy.overrides]]` `ignore_missing_imports = true`(见 Task 3 Step 3),**新代码不再写第三方的 type: ignore 注释**
- 屏幕模型约定:class id = 牌编码 0-33(万 0-8、饼 9-17、条 18-26、东 27 南 28 西 29 北 30、中 31 发 32 白 33),**不使用** `yolo_mapping.py`(那是实物模型的 42 类映射)
- 坐标约定:检测框以截图像素计;屏幕区域用归一化坐标(0-1);模板匹配教师置信度阈值默认 0.80,状态机手牌阈值 0.55
- P1 范围:无副露局给出牌建议;检测到副露时 HUD 显示降级提示(「检测到副露,出牌建议暂不可用」),完整副露支持在 P2
- 路径约定:`data/templates/`(34 张模板)、`data/screen_dataset/`(数据集)、`data/models/screen/mahjong_screen_detector/weights/best.pt`(训练产物)
- 每任务 Interfaces 块中的签名是本计划内唯一权威定义,后续任务必须与之一致

---

### Task 1: 欢乐推倒胡规则集(允许吃牌)

**Files:**
- Create: `src/mahjong_engine/rules/huanyu_rules.py`
- Modify: `src/mahjong_engine/rules/registry.py`(注册)
- Modify: `src/mahjong_engine/__init__.py`(导出)
- Test: `tests/test_engine/test_huanyu_rules.py`

**Interfaces:**
- Consumes: `IRuleSet`(`src/mahjong_engine/rules/interface.py`)、`ShaanxiRules`、`can_chi`(`src/mahjong_engine/judges/action_judge.py`)
- Produces: `HuanyuRules`(name() == "huanyu"),注册进 registry 后 `get("huanyu")` 可用

- [ ] **Step 1: 写失败测试**

```python
"""欢乐推倒胡规则测试。"""
from src.mahjong_core import Hand
from src.mahjong_core.tile import B4, B5, B6, DONG, T7, T8, T9, W1, W2, W3, W4
from src.mahjong_engine import get, list_names
from src.mahjong_engine.rules.huanyu_rules import HuanyuRules


class TestHuanyuRules:
    def setup_method(self):
        self.rules = HuanyuRules()

    def test_name(self):
        assert self.rules.name() == 'huanyu'

    def test_registered(self):
        assert 'huanyu' in list_names()
        assert get('huanyu').name() == 'huanyu'

    def test_can_chi_as_tail(self):
        h = Hand([W1, W2])
        combos = self.rules.can_chi(h, W3)
        assert [W1, W2, W3] in combos

    def test_can_chi_as_middle(self):
        h = Hand([W1, W3])
        combos = self.rules.can_chi(h, W2)
        assert [W1, W2, W3] in combos

    def test_can_chi_not_allowed_for_honor(self):
        h = Hand([W1, W2])
        assert self.rules.can_chi(h, DONG) == []

    def test_shaanxi_rules_inherited(self):
        # 136张/七对/十三幺与陕西一致
        assert self.rules.get_tile_set().total_tiles == 136
        h = Hand([W1, W2, W3, B4, B5, B6, T7, T8, T9, W2, W3, W4, W4])
        assert self.rules.is_winning_hand(h, new_tile=W4).can_win is True
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_engine/test_huanyu_rules.py -v`
Expected: FAIL,`ModuleNotFoundError: No module named 'src.mahjong_engine.rules.huanyu_rules'`

- [ ] **Step 3: 实现**

`src/mahjong_engine/rules/huanyu_rules.py`:

```python
"""欢乐推倒胡规则实现(腾讯欢乐麻将「推倒胡」模式)。"""

from src.mahjong_core import Hand
from src.mahjong_engine.judges.action_judge import can_chi
from src.mahjong_engine.rules.shaanxi_rules import ShaanxiRules


class HuanyuRules(ShaanxiRules):
    """欢乐推倒胡。

    与陕西推倒胡的唯一区别: 允许吃牌(上家打出时手牌可组成顺子)。
    其余规则(136张/碰/杠/七对/十三幺/无万能牌)完全继承。
    """

    def name(self) -> str:
        return 'huanyu'

    def can_chi(self, hand: Hand, tile: int) -> list[list[int]]:
        return can_chi(list(hand), tile)
```

`src/mahjong_engine/rules/registry.py` 修改(在 import 区追加,并在文件末尾 `register(ShaanxiRules())` 后追加一行):

```python
from src.mahjong_engine.rules.huanyu_rules import HuanyuRules
# ...
register(HuanyuRules())
```

`src/mahjong_engine/__init__.py` 修改:

```python
from src.mahjong_engine.rules.huanyu_rules import HuanyuRules
# __all__ 增加 'HuanyuRules'
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_engine/test_huanyu_rules.py -v`
Expected: 6 passed(测试代码共 6 个测试方法)

- [ ] **Step 5: ruff 检查**

Run: `ruff check src/mahjong_engine/rules/huanyu_rules.py src/mahjong_engine/rules/registry.py src/mahjong_engine/__init__.py tests/test_engine/test_huanyu_rules.py`
Expected: All checks passed!

---

### Task 2: 屏幕布局常量 ScreenLayout

**Files:**
- Create: `src/mahjong_cv/screen_layout.py`
- Test: `tests/test_cv/test_screen_layout.py`

**Interfaces:**
- Consumes: 无(纯几何)
- Produces: `ScreenRegion(x1, y1, x2, y2)`(frozen dataclass,`.contains(nx, ny) -> bool`)、`ScreenLayout`(`.region_of(nx, ny) -> str`)、`DEFAULT_LAYOUT`

- [ ] **Step 1: 写失败测试**

```python
"""屏幕布局区域判定的单元测试。"""
from src.mahjong_cv.screen_layout import DEFAULT_LAYOUT, ScreenRegion


class TestScreenRegion:
    def test_contains_inside(self):
        r = ScreenRegion(0.1, 0.2, 0.5, 0.6)
        assert r.contains(0.3, 0.4) is True

    def test_contains_on_boundary(self):
        r = ScreenRegion(0.1, 0.2, 0.5, 0.6)
        assert r.contains(0.1, 0.2) is True
        assert r.contains(0.5, 0.6) is True

    def test_contains_outside(self):
        r = ScreenRegion(0.1, 0.2, 0.5, 0.6)
        assert r.contains(0.9, 0.9) is False


class TestScreenLayout:
    def test_region_of_my_hand(self):
        assert DEFAULT_LAYOUT.region_of(0.5, 0.95) == 'my_hand'

    def test_region_of_table(self):
        assert DEFAULT_LAYOUT.region_of(0.5, 0.5) == 'table'

    def test_region_of_unknown(self):
        assert DEFAULT_LAYOUT.region_of(0.99, 0.02) == 'unknown'

    def test_region_of_precedence_my_meld_before_table(self):
        # 副露区在手牌上方, 不与 table 重叠: (0.7, 0.8) 属于 my_meld
        assert DEFAULT_LAYOUT.region_of(0.7, 0.8) == 'my_meld'
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_cv/test_screen_layout.py -v`
Expected: FAIL,`ModuleNotFoundError: No module named 'src.mahjong_cv.screen_layout'`

- [ ] **Step 3: 实现**

`src/mahjong_cv/screen_layout.py`:

```python
"""屏幕布局 — 欢乐麻将窗口各区域(归一化坐标, 0-1)。

区域为占位默认值, 首次真机运行时按实际窗口布局调整 DEFAULT_LAYOUT
(见 Task 5 数据采集手册的校准步骤)。
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ScreenRegion:
    """归一化矩形区域 (0-1 坐标)。"""

    x1: float
    y1: float
    x2: float
    y2: float

    def contains(self, nx: float, ny: float) -> bool:
        """归一化坐标是否落在区域内(含边界)。"""
        return self.x1 <= nx <= self.x2 and self.y1 <= ny <= self.y2


@dataclass(frozen=True)
class ScreenLayout:
    """欢乐麻将窗口区域定义。region_of 按优先级: 手牌 > 副露 > 桌面 > 牌河。"""

    my_hand: ScreenRegion    # 我的手牌区(底部中央)
    my_meld: ScreenRegion    # 我的副露区(手牌上方)
    table: ScreenRegion      # 桌面中央(牌河/各家打出的牌)
    discard: ScreenRegion    # 我打出牌的区域(桌面右侧, 竖排)

    def region_of(self, nx: float, ny: float) -> str:
        """归一化坐标 → 区域名: 'my_hand' | 'my_meld' | 'table' | 'discards' | 'unknown'。"""
        if self.my_hand.contains(nx, ny):
            return 'my_hand'
        if self.my_meld.contains(nx, ny):
            return 'my_meld'
        if self.table.contains(nx, ny):
            return 'table'
        if self.discard.contains(nx, ny):
            return 'discards'
        return 'unknown'


DEFAULT_LAYOUT = ScreenLayout(
    my_hand=ScreenRegion(0.15, 0.86, 0.85, 1.00),
    my_meld=ScreenRegion(0.55, 0.70, 0.95, 0.86),
    table=ScreenRegion(0.25, 0.30, 0.75, 0.70),
    discard=ScreenRegion(0.05, 0.60, 0.20, 0.90),
)
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_cv/test_screen_layout.py -v`
Expected: 7 passed

- [ ] **Step 5: ruff 检查**

Run: `ruff check src/mahjong_cv/screen_layout.py tests/test_cv/test_screen_layout.py`
Expected: All checks passed!

---

### Task 3: 捕获层 Win32Capture(欢乐麻将窗口截屏)

**Files:**
- Create: `src/mahjong_cv/capture/__init__.py`
- Create: `src/mahjong_cv/capture/win32.py`
- Modify: `environment.yml`(pip 段加 `mss`)
- Modify: `pyproject.toml`(mypy overrides 段加第三方库 ignore_missing_imports)
- Test: `tests/test_cv/test_capture.py`

**Interfaces:**
- Consumes: `mss`(pip 依赖)
- Produces: `find_window(title_substring) -> int | None`、`client_rect(hwnd) -> tuple[int, int, int, int]`(left, top, width, height)、`Win32Capture(title_substring="欢乐麻将")` 含 `.hwnd -> int | None`、`.client_rect() -> tuple[int,int,int,int] | None`、`.capture() -> np.ndarray | None`(BGR)

- [ ] **Step 1: 写失败测试**

```python
"""捕获层单元测试(纯函数与无窗口场景, 不依赖真实游戏窗口)。"""
import numpy as np

from src.mahjong_cv.capture.win32 import Win32Capture, rect_to_monitor


class TestRectToMonitor:
    def test_basic(self):
        assert rect_to_monitor((10, 20, 800, 600)) == {
            'left': 10, 'top': 20, 'width': 800, 'height': 600,
        }


class TestWin32Capture:
    def test_no_window_returns_none(self):
        cap = Win32Capture('__不存在的窗口标题__')
        assert cap.client_rect() is None
        assert cap.capture() is None

    def test_dpi_aware_flag_set(self):
        # 确保 DPI 感知已设置(防止截屏坐标虚拟化)
        from src.mahjong_cv.capture import win32
        assert win32._DPI_AWARE is True
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_cv/test_capture.py -v`
Expected: FAIL,`ModuleNotFoundError: No module named 'src.mahjong_cv.capture.win32'`

- [ ] **Step 3: 实现**

`src/mahjong_cv/capture/__init__.py`:

```python
"""采集层 — 欢乐麻将窗口捕获。"""
from src.mahjong_cv.capture.win32 import Win32Capture

__all__ = ['Win32Capture']
```

`src/mahjong_cv/capture/win32.py`:

```python
"""欢乐麻将 PC 客户端窗口捕获(Win32 + mss)。

Win32 调用全部用 ctypes(无第三方依赖); 截屏用 mss(高速)。
DPI: 启动时设置 Per-Monitor DPI Aware, 保证坐标与截屏像素一致。
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from typing import Any

import numpy as np

user32 = ctypes.windll.user32

_DPI_AWARE = False


def _set_dpi_aware() -> bool:
    """设置为 Per-Monitor DPI Aware。失败返回 False(坐标可能被虚拟化)。"""
    global _DPI_AWARE
    if _DPI_AWARE:
        return True
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except OSError:
        try:
            user32.SetProcessDPIAware()
        except OSError:
            _DPI_AWARE = False
            return False
    _DPI_AWARE = True
    return True


_set_dpi_aware()


class _RECT(ctypes.Structure):
    _fields_ = [
        ('left', wintypes.LONG),
        ('top', wintypes.LONG),
        ('right', wintypes.LONG),
        ('bottom', wintypes.LONG),
    ]


def find_window(title_substring: str) -> int | None:
    """按标题子串查找可见顶层窗口, 返回 hwnd; 找不到返回 None。"""

    results: list[int] = []

    def _callback(hwnd: int, _lparam: int) -> bool:
        length = user32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        if title_substring in buf.value and user32.IsWindowVisible(hwnd):
            results.append(hwnd)
        return True

    _ProcType = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user32.EnumWindows(_ProcType(_callback), 0)
    return results[0] if results else None


def client_rect(hwnd: int) -> tuple[int, int, int, int]:
    """返回窗口客户区屏幕坐标 (left, top, width, height)。"""
    rect = _RECT()
    user32.GetClientRect(hwnd, ctypes.byref(rect))
    pt = wintypes.POINT(rect.left, rect.top)
    user32.ClientToScreen(hwnd, ctypes.byref(pt))
    return pt.x, pt.y, rect.right - rect.left, rect.bottom - rect.top


def rect_to_monitor(rect: tuple[int, int, int, int]) -> dict[str, int]:
    """客户区矩形 → mss monitor 字典。"""
    left, top, width, height = rect
    return {'left': left, 'top': top, 'width': width, 'height': height}


class Win32Capture:
    """欢乐麻将窗口截屏。轮询节奏(10-30fps)由调用方控制。"""

    def __init__(self, title_substring: str = '欢乐麻将') -> None:
        self._title = title_substring
        self._hwnd: int | None = None
        self._sct: Any = None

    @property
    def hwnd(self) -> int | None:
        if self._hwnd is None:
            self._hwnd = find_window(self._title)
        return self._hwnd

    def client_rect(self) -> tuple[int, int, int, int] | None:
        hwnd = self.hwnd
        if hwnd is None:
            return None
        return client_rect(hwnd)

    def capture(self) -> np.ndarray | None:
        """截取客户区, 返回 BGR ndarray; 窗口不存在或尺寸异常返回 None。"""
        hwnd = self.hwnd
        if hwnd is None:
            return None
        rect = client_rect(hwnd)
        if rect[2] <= 0 or rect[3] <= 0:
            return None
        if self._sct is None:
            import mss  # pyproject 已配 ignore_missing_imports

            self._sct = mss.mss()
        # shot.bgra = 4字节/像素 BGRA(mss 的 .rgb 是 3字节真RGB, 不能 reshape(..., 4))
        shot = self._sct.grab(rect_to_monitor(rect))
        arr = np.frombuffer(shot.bgra, dtype=np.uint8).reshape(
            shot.height, shot.width, 4
        )
        return arr[:, :, :3].copy()
```

`environment.yml` pip 段追加:

```yaml
    - mss
```

`pyproject.toml` mypy overrides 段(即 `[[tool.mypy.overrides]]` 列表)追加:

```toml
[[tool.mypy.overrides]]
module = ["cv2.*", "mss.*", "ultralytics.*", "PySide6.*"]
ignore_missing_imports = true
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_cv/test_capture.py -v`
Expected: 3 passed

- [ ] **Step 5: ruff 检查**

Run: `ruff check src/mahjong_cv/capture tests/test_cv/test_capture.py`
Expected: All checks passed!

- [ ] **Step 6: 手动冒烟(可选)**

Run: `python -c "import sys; sys.path.insert(0,'.'); from src.mahjong_cv.capture.win32 import Win32Capture; c=Win32Capture(); print(c.client_rect())"`
Expected: 欢乐麻将未打开时打印 None;打开后打印客户区坐标。

---

### Task 4: 模板匹配教师(数据标注)

**Files:**
- Create: `src/mahjong_cv/detections.py`
- Create: `src/mahjong_cv/template/__init__.py`
- Create: `src/mahjong_cv/template/registry.py`
- Create: `src/mahjong_cv/template/matcher.py`
- Test: `tests/test_cv/test_template_matcher.py`(test_cv 已有 `__init__.py`, 无需新建)

**Interfaces:**
- Consumes: `tile_display` / `tile_from_str`(`src/mahjong_core/tile.py`)
- Produces: `TileDet(tile, x1, y1, x2, y2, conf)`(frozen dataclass)、`TemplateRegistry(template_dir)` 含 `.templates -> dict[int, np.ndarray]`(灰度 float32)、`TemplateMatcher(registry)` 含 `.match(frame: np.ndarray, conf_threshold=0.80) -> list[TileDet]`、`nms(dets, iou_threshold=0.3) -> list[TileDet]`
- Note: `TileDet` 由屏幕视觉(Task 6)、状态机(Task 7)、会话(Task 8)共同使用

- [ ] **Step 1: 写失败测试**

```python
"""模板匹配教师单元测试(合成图像, 不依赖真实截图)。"""
import tempfile
from pathlib import Path

import cv2
import numpy as np

from src.mahjong_core.tile import tile_display
from src.mahjong_cv.detections import TileDet
from src.mahjong_cv.template.matcher import TemplateMatcher, nms
from src.mahjong_cv.template.registry import TemplateRegistry


def make_tile_image(tile: int, size: int = 32) -> np.ndarray:
    """合成牌图: 灰底 + 白框 + 数字(数字=tile, 使不同牌可区分)。"""
    img = np.full((size, size, 3), 60, dtype=np.uint8)
    cv2.rectangle(img, (4, 4), (size - 5, size - 5), (200, 200, 200), -1)
    cv2.putText(img, str(tile), (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return img


class TestNms:
    def test_overlapping_keeps_best(self):
        a = TileDet(1, 0, 0, 100, 100, 0.9)
        b = TileDet(1, 5, 5, 105, 105, 0.6)
        assert nms([a, b]) == [a]

    def test_separate_boxes_both_kept(self):
        a = TileDet(1, 0, 0, 50, 50, 0.9)
        b = TileDet(1, 100, 0, 150, 50, 0.9)
        assert len(nms([a, b])) == 2


class TestTemplateMatcher:
    def test_matches_synthetic_frame(self):
        tiles = [0, 9, 18]  # 一万 一饼 一条
        with tempfile.TemporaryDirectory() as td:
            for t in range(34):  # 完整34张模板(实现要求齐全)
                # 中文文件名: cv2.imwrite 在中文 Windows 写不出, 用 imencode + write_bytes
                ok, buf = cv2.imencode('.png', make_tile_image(t))
                assert ok
                (Path(td) / f'{tile_display(t)}.png').write_bytes(buf.tobytes())
            registry = TemplateRegistry(td)
            assert set(registry.templates) == set(range(34))

            canvas = np.full((200, 400, 3), 30, dtype=np.uint8)
            for i, t in enumerate(tiles):
                canvas[20:52, 40 + i * 120:72 + i * 120] = make_tile_image(t)

            dets = TemplateMatcher(registry).match(canvas, conf_threshold=0.5)
            assert sorted(d.tile for d in dets) == sorted(tiles)

    def test_incomplete_registry_raises(self):
        with tempfile.TemporaryDirectory() as td:
            for t in (0, 1):
                ok, buf = cv2.imencode('.png', make_tile_image(t))
                assert ok
                (Path(td) / f'{tile_display(t)}.png').write_bytes(buf.tobytes())
            try:
                TemplateRegistry(td).templates  # noqa: B018  (触发懒加载以断言抛错)
                raise AssertionError('应抛出 ValueError')
            except ValueError as e:
                assert '模板不完整' in str(e)
```

注:`test_matches_synthetic_frame` 中模板目录写满 34 张(测试用合成图),与实现的「模板必须齐全」约定一致;若你实现为允许部分模板,可把断言放宽为 `set(registry.templates) >= {0, 9, 18}`,但推荐保持严格 34 张。

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_cv/test_template_matcher.py -v`
Expected: FAIL,`ModuleNotFoundError: No module named 'src.mahjong_cv.template'`

- [ ] **Step 3: 实现**

`src/mahjong_cv/detections.py`:

```python
"""检测结果数据类 — 屏幕视觉/模板匹配/状态机共用。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TileDet:
    """一张牌的检测结果。

    tile: 牌编码 0-33(与 mahjong_core.tile 一致)
    x1/y1/x2/y2: 像素坐标框
    conf: 置信度 0-1
    """

    tile: int
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
```

`src/mahjong_cv/template/__init__.py`:

```python
"""模板匹配教师 — 为数据管线自动打标(非实时)。"""
```

`src/mahjong_cv/template/registry.py`:

```python
"""模板注册表 — 从目录加载 34 种牌的模板图片。

模板文件命名: 中文牌名 + .png, 如 '一万.png' '东.png'(tile_display 名称)。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.mahjong_core.tile import tile_display, tile_from_str


class TemplateRegistry:
    """按文件名加载模板, 归一化为灰度 float32。缺模板时启动即报错。"""

    def __init__(self, template_dir: str) -> None:
        self._dir = Path(template_dir)
        self._templates: dict[int, np.ndarray] | None = None

    @property
    def templates(self) -> dict[int, np.ndarray]:
        if self._templates is None:
            self._templates = self._load()
        return self._templates

    def _load(self) -> dict[int, np.ndarray]:
        result: dict[int, np.ndarray] = {}
        for path in sorted(self._dir.glob('*.png')):
            tile = tile_from_str(path.stem)
            import cv2  # 无类型存根, 函数内 import

            # Windows 中文路径: cv2.imread 无法读中文文件名(ACP=936 编码问题),
            # 用 np.fromfile + imdecode 变通(中文Windows 实测必须)
            data = np.fromfile(str(path), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f'无法读取模板: {path}')
            result[tile] = img.astype(np.float32) / 255.0
        if len(result) != 34:
            missing = [tile_display(t) for t in range(34) if t not in result]
            raise ValueError(f'模板不完整, 需要34张, 缺少: {missing}')
        return result
```

`src/mahjong_cv/template/matcher.py`:

```python
"""模板匹配教师 — 用 34 张模板在截图中定位并识别所有牌。

逐类 matchTemplate(TM_CCOEFF_NORMED), 阈值过滤后 NMS。
仅用于数据标注管线(约 0.5s/帧), 非实时推理(实时用 YOLO, 见 screen_vision)。
"""

from __future__ import annotations

import numpy as np

from src.mahjong_cv.detections import TileDet
from src.mahjong_cv.template.registry import TemplateRegistry


def _iou(a: TileDet, b: TileDet) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
    return inter / union if union > 0 else 0.0


def nms(dets: list[TileDet], iou_threshold: float = 0.3) -> list[TileDet]:
    """按置信度降序, 与已保留框 IoU 超阈值则丢弃。"""
    kept: list[TileDet] = []
    for det in sorted(dets, key=lambda d: d.conf, reverse=True):
        if all(_iou(det, k) < iou_threshold for k in kept):
            kept.append(det)
    return kept


class TemplateMatcher:
    """模板匹配识别截图中全部牌。"""

    def __init__(self, registry: TemplateRegistry) -> None:
        self._registry = registry

    def match(
        self, frame: np.ndarray, conf_threshold: float = 0.80
    ) -> list[TileDet]:
        """返回 NMS 后的检测列表。frame 可为 BGR 或灰度。"""
        import cv2  # 无类型存根, 函数内 import

        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        gray = gray.astype(np.float32) / 255.0
        h, w = gray.shape

        candidates: list[TileDet] = []
        for tile, tpl in self._registry.templates.items():
            th, tw = tpl.shape
            if th > h or tw > w:
                continue
            result = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
            ys, xs = np.where(result >= conf_threshold)
            for y, x in zip(ys, xs, strict=True):
                candidates.append(
                    TileDet(
                        tile=tile,
                        x1=float(x),
                        y1=float(y),
                        x2=float(x + tw),
                        y2=float(y + th),
                        conf=float(result[y, x]),
                    )
                )
        return nms(candidates)
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_cv/test_template_matcher.py -v`
Expected: 4 passed

- [ ] **Step 5: ruff 检查**

Run: `ruff check src/mahjong_cv/detections.py src/mahjong_cv/template tests/test_cv/test_template_matcher.py`
Expected: All checks passed!

- [ ] **Step 6: 生成真实模板(手动, 一次性)**

1. 启动欢乐麻将进入牌局,**保持窗口固定尺寸**(建议最大化;模板按该尺寸截取,后续采集/推理期间不要改窗口大小),用任意截图工具截一张清晰的牌局图 `shot.png`
2. 用图像编辑器(画图/PS)从 `shot.png` 的**手牌区**依次裁剪 34 种牌的方形小图(每张约 60-90px),命名为 `一万.png`、`一饼.png`、`一条.png`、`东.png`、`南.png`、`西.png`、`北.png`、`中.png`、`发.png`、`白.png`(其余按 `tile_display` 名称),放入 `data/templates/`
3. 快速验证:`python scripts/capture_dataset.py --frames 1 --preview` 能检测出手牌区全部牌(此步依赖 Task 5 的脚本,可在 Task 5 完成后回头执行)

---

### Task 5: 数据管线脚本(采集/拆分/训练)

**Files:**
- Create: `scripts/capture_dataset.py`
- Create: `scripts/build_dataset_yaml.py`
- Create: `scripts/train_screen_yolo.py`
- Create: `tests/test_scripts/__init__.py`(空文件, 与既有 test_ai 等目录模式一致)
- Test: `tests/test_scripts/test_dataset_builder.py`(save_sample 与 split_and_write_yaml 纯函数)

**Interfaces:**
- Consumes: `Win32Capture`(Task 3)、`TemplateMatcher`/`TemplateRegistry`(Task 4)、`tile_display`(core)
- Produces: `save_sample(sample_dir: Path, frame: np.ndarray, dets: list[TileDet], index: int) -> None`(写 YOLO 格式 img + txt)、`split_and_write_yaml(dataset_root: Path, val_ratio=0.2) -> Path`(返回 data.yaml 路径)
- 数据布局(Ultralytics 约定): `data/screen_dataset/{images,labels}/{train,val}`、`data/screen_dataset/data.yaml`

- [ ] **Step 1: 写失败测试**

```python
"""数据管线纯函数的单元测试(合成数据)。"""
import tempfile
from pathlib import Path

import cv2
import numpy as np

from scripts.build_dataset_yaml import split_and_write_yaml
from scripts.capture_dataset import save_sample
from src.mahjong_core.tile import tile_display
from src.mahjong_cv.detections import TileDet


def make_dets() -> list[TileDet]:
    return [
        TileDet(0, 10, 20, 42, 52, 0.95),   # 一万
        TileDet(9, 100, 200, 132, 232, 0.9),  # 一饼
    ]


class TestSaveSample:
    def test_writes_image_and_label(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            frame = np.zeros((300, 400, 3), dtype=np.uint8)
            save_sample(out, frame, make_dets(), 7)

            img = out / 'images' / 'frame_000007.jpg'
            lbl = out / 'labels' / 'frame_000007.txt'
            assert img.exists() and lbl.exists()

            text = lbl.read_text(encoding='utf-8').strip().split('\n')
            assert len(text) == 2
            cls, cx, cy, bw, bh = text[0].split()
            assert cls == '0'
            assert 0 <= float(cx) <= 1 and 0 <= float(bw) <= 1


class TestSplitAndWriteYaml:
    def test_split_and_yaml(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / 'images').mkdir(parents=True, exist_ok=True)
            (root / 'labels').mkdir(parents=True, exist_ok=True)
            for i in range(6):
                cv2.imwrite(str(root / 'images' / f'f{i}.jpg'), np.zeros((10, 10, 3), np.uint8))
                (root / 'labels' / f'f{i}.txt').write_text('0 0.5 0.5 0.1 0.1\n', encoding='utf-8')

            yaml_path = split_and_write_yaml(root, val_ratio=0.2)

            train_imgs = list((root / 'images' / 'train').glob('*.jpg'))
            val_imgs = list((root / 'images' / 'val').glob('*.jpg'))
            assert len(train_imgs) == 5 and len(val_imgs) == 1

            content = yaml_path.read_text(encoding='utf-8')
            assert 'nc: 34' in content
            assert tile_display(0) in content  # names 含 '一万'
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_scripts/test_dataset_builder.py -v`
Expected: FAIL,`ModuleNotFoundError: No module named 'scripts.capture_dataset'`
(若 `scripts` 无 `__init__.py` 报导入错误,先按 Step 3 创建脚本即可;pytest 的 rootdir 路径已含项目根;测试共 2 个测试方法,Step 4 预期 2 passed)

- [ ] **Step 3: 实现**

`scripts/capture_dataset.py`:

```python
"""采集屏幕数据集: 截屏 → 模板匹配自动打标 → YOLO 格式保存。

用法:
    python scripts/capture_dataset.py --frames 200 --interval 1.0
    # 开着欢乐麻将, 覆盖: 起手/中盘/听牌/副露/结算 不同阶段
    # 每保存一帧后切换一下局面(打牌/等待别家), 避免大量重复帧
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mahjong_cv.capture.win32 import Win32Capture
from src.mahjong_cv.detections import TileDet
from src.mahjong_cv.template.matcher import TemplateMatcher
from src.mahjong_cv.template.registry import TemplateRegistry

TEMPLATE_DIR = 'data/templates'


def save_sample(
    sample_dir: Path, frame: np.ndarray, dets: list[TileDet], index: int
) -> None:
    """保存一帧 YOLO 格式样本: images/<name>.jpg + labels/<name>.txt。

    标签行: <class_id> <cx> <cy> <w> <h>(归一化)。class_id = 牌编码 0-33。
    """
    h, w = frame.shape[:2]
    img_dir = sample_dir / 'images'
    lbl_dir = sample_dir / 'labels'
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    name = f'frame_{index:06d}'
    cv2.imwrite(str(img_dir / f'{name}.jpg'), frame)

    lines = []
    for d in dets:
        cx = (d.x1 + d.x2) / 2 / w
        cy = (d.y1 + d.y2) / 2 / h
        bw = (d.x2 - d.x1) / w
        bh = (d.y2 - d.y1) / h
        lines.append(f'{d.tile} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}')
    (lbl_dir / f'{name}.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description='采集欢乐麻将屏幕牌数据集(自动打标)')
    parser.add_argument('--out', default='data/screen_dataset')
    parser.add_argument('--frames', type=int, default=200)
    parser.add_argument('--interval', type=float, default=1.0, help='帧间隔秒')
    parser.add_argument('--title', default='欢乐麻将')
    parser.add_argument('--templates', default=TEMPLATE_DIR)
    parser.add_argument('--min-tiles', type=int, default=10, help='每帧至少检测到的牌数(过滤空帧)')
    parser.add_argument('--preview', action='store_true', help='显示标注预览')
    args = parser.parse_args()

    cap = Win32Capture(args.title)
    matcher = TemplateMatcher(TemplateRegistry(args.templates))
    out = Path(args.out)

    if cap.client_rect() is None:
        print(f"未找到窗口: '{args.title}', 请先启动欢乐麻将")
        sys.exit(1)

    saved = 0
    while saved < args.frames:
        frame = cap.capture()
        if frame is None:
            time.sleep(1.0)
            continue
        dets = matcher.match(frame)
        if len(dets) < args.min_tiles:
            continue
        save_sample(out, frame, dets, saved)
        saved += 1
        print(f'[{saved}/{args.frames}] 检测到 {len(dets)} 张牌', flush=True)
        if args.preview:
            for d in dets:
                cv2.rectangle(
                    frame, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)),
                    (0, 255, 0), 2,
                )
            cv2.imshow('preview', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        time.sleep(args.interval)

    print(f'完成, 共保存 {saved} 帧到 {out}')


if __name__ == '__main__':
    main()
```

`scripts/build_dataset_yaml.py`:

```python
"""把采集好的数据集拆分为 train/val 并生成 data.yaml。

用法:
    python scripts/build_dataset_yaml.py --dataset data/screen_dataset
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mahjong_core.tile import tile_display


def split_and_write_yaml(dataset_root: Path, val_ratio: float = 0.2) -> Path:
    """按排序取末尾 val_ratio 帧为 val, 其余为 train; 写 data.yaml。"""
    images = sorted((dataset_root / 'images').glob('*.jpg'))
    if not images:
        raise ValueError(f'数据集为空: {dataset_root / "images"}')

    n_val = max(1, int(len(images) * val_ratio))
    for subset in ('train', 'val'):
        (dataset_root / 'images' / subset).mkdir(parents=True, exist_ok=True)
        (dataset_root / 'labels' / subset).mkdir(parents=True, exist_ok=True)

    for i, img in enumerate(images):
        subset = 'val' if i >= len(images) - n_val else 'train'
        label = dataset_root / 'labels' / (img.stem + '.txt')
        if not label.exists():
            raise ValueError(f'缺少标签: {label}')
        img.rename(dataset_root / 'images' / subset / img.name)
        label.rename(dataset_root / 'labels' / subset / label.name)

    names = [tile_display(t) for t in range(34)]
    yaml_path = dataset_root / 'data.yaml'
    yaml_path.write_text(
        f'path: {dataset_root.resolve().as_posix()}\n'
        'train: images/train\n'
        'val: images/val\n'
        'nc: 34\n'
        f'names: {json.dumps(names, ensure_ascii=False)}\n',
        encoding='utf-8',
    )
    return yaml_path


def main() -> None:
    parser = argparse.ArgumentParser(description='数据集拆分并生成 data.yaml')
    parser.add_argument('--dataset', default='data/screen_dataset')
    parser.add_argument('--val-ratio', type=float, default=0.2)
    args = parser.parse_args()
    yaml_path = split_and_write_yaml(Path(args.dataset), args.val_ratio)
    print(f'已生成: {yaml_path}')


if __name__ == '__main__':
    main()
```

`scripts/train_screen_yolo.py`:

```python
"""训练屏幕牌检测模型(基于 yolov8n.pt 微调)。

用法:
    python scripts/train_screen_yolo.py --data data/screen_dataset/data.yaml
    # 产出: data/models/screen/mahjong_screen_detector/weights/best.pt
    # 如需换基座: --model yolo26n.pt(同数据集 A/B 对比后选优)
"""

import argparse

from ultralytics import YOLO  # noqa: PLC0415  (脚本层允许顶层导入)


def main() -> None:
    parser = argparse.ArgumentParser(description='训练欢乐麻将屏幕牌检测模型')
    parser.add_argument('--data', default='data/screen_dataset/data.yaml')
    parser.add_argument('--model', default='yolov8n.pt')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--imgsz', type=int, default=1280, help='屏幕牌较小, 用1280保小目标精度')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--name', default='mahjong_screen_detector')
    args = parser.parse_args()

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        project='data/models/screen',
        exist_ok=True,
        patience=20,
        device=0,
    )
    print(f'训练完成, 最佳模型: {results.save_dir}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_scripts/test_dataset_builder.py -v`
Expected: 2 passed(测试代码共 2 个测试方法)

- [ ] **Step 5: ruff 检查**

Run: `ruff check scripts/capture_dataset.py scripts/build_dataset_yaml.py scripts/train_screen_yolo.py tests/test_scripts/test_dataset_builder.py`
Expected: All checks passed!

- [ ] **Step 6: 数据采集 + 训练(手动, 关键路径)**

1. 打开欢乐麻将(推倒胡模式)进入牌局,**保持窗口与 Task 4 截模板时相同的尺寸**
2. 运行:`python scripts/capture_dataset.py --frames 2000 --interval 1.0 --preview`
   - 目标:**总帧数 ≥ 5000**(分 3-4 次跑,每次换不同牌局阶段:起手/中盘/听牌/副露/结算)
   - 期间正常打牌,每帧间隔 1 秒足够变化
   - `--preview` 观察绿色框是否准确框住每张牌;框歪/漏检 → 回 Task 4 Step 6 补模板
3. 校验数据集:`python scripts/build_dataset_yaml.py --dataset data/screen_dataset`
4. 训练:`python scripts/train_screen_yolo.py --data data/screen_dataset/data.yaml`
   - 目标:mAP@0.5 ≥ 0.95(2D 图标任务);若不足,先加数据(重复步骤 2)再训
5. 记录训练结果路径,Task 10 的 `MODEL` 常量指向 `data/models/screen/mahjong_screen_detector/weights/best.pt`

---

### Task 6: 屏幕视觉 ScreenVision(YOLO 推理封装)

**Files:**
- Create: `src/mahjong_cv/screen_vision.py`
- Test: `tests/test_cv/test_screen_vision.py`

**Interfaces:**
- Consumes: `TileDet`(Task 4)、ultralytics
- Produces: `ScreenVision(model_path)` 含 `.process(frame: np.ndarray, conf_threshold=0.5, detector=None) -> list[TileDet]`(detector 可注入, 测试用)

- [ ] **Step 1: 写失败测试**

```python
"""ScreenVision 单元测试(注入假检测器, 不加载真实模型)。"""
from types import SimpleNamespace

import numpy as np
import pytest

from src.mahjong_cv.screen_vision import ScreenVision


class FakeBoxes:
    def __init__(self, cls, conf, xyxy):
        self.cls = np.array(cls, dtype=np.float32).reshape(-1, 1)
        self.conf = np.array(conf, dtype=np.float32).reshape(-1, 1)
        self.xyxy = np.array(xyxy, dtype=np.float32)

    def __iter__(self):
        """模拟 ultralytics Boxes 迭代: 逐框产出 (cls, conf, xyxy) 行视图。"""
        for i in range(self.cls.shape[0]):
            yield SimpleNamespace(
                cls=self.cls[i : i + 1],
                conf=self.conf[i : i + 1],
                xyxy=self.xyxy[i : i + 1],
            )


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeDetector:
    def __call__(self, frame, conf=0.5, verbose=False):
        return [
            FakeResult(FakeBoxes(
                cls=[0, 9, 40],          # 一万 / 一饼 / 越界class(应跳过)
                conf=[0.9, 0.8, 0.7],
                xyxy=[[10, 20, 42, 52], [100, 200, 132, 232], [0, 0, 10, 10]],
            ))
        ]


class TestScreenVision:
    def test_maps_class_id_to_tile(self):
        dets = ScreenVision('unused.pt').process(
            np.zeros((300, 400, 3), dtype=np.uint8),
            detector=FakeDetector(),
        )
        assert [d.tile for d in dets] == [0, 9]

    def test_box_coordinates_passed_through(self):
        dets = ScreenVision('unused.pt').process(
            np.zeros((300, 400, 3), dtype=np.uint8),
            detector=FakeDetector(),
        )
        assert dets[0].x1 == 10 and dets[0].y2 == 52
        assert dets[0].conf == pytest.approx(0.9)
```

注: `FakeBoxes.__iter__` 用 SimpleNamespace 产出单行视图, 与 ultralytics 的 `for box in result.boxes` 迭代模式一致; `conf` 断言用 `pytest.approx`(float32 的 0.9 存储为 0.8999999..., 精确相等必挂)。

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_cv/test_screen_vision.py -v`
Expected: FAIL,`ModuleNotFoundError: No module named 'src.mahjong_cv.screen_vision'`

- [ ] **Step 3: 实现**

`src/mahjong_cv/screen_vision.py`:

```python
"""屏幕视觉 — 欢乐麻将截图牌检测。

屏幕模型约定: class id = 牌编码 0-33(与 yolo_mapping 的实物 42 类无关)。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.mahjong_cv.detections import TileDet


class ScreenVision:
    """YOLO 屏幕牌检测封装。"""

    def __init__(self, model_path: str) -> None:
        self._model_path = model_path
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from ultralytics import YOLO  # pyproject 已配 ignore_missing_imports

            self._model = YOLO(self._model_path)
        return self._model

    def process(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.5,
        detector: Any | None = None,
    ) -> list[TileDet]:
        """检测一帧全部牌。detector 用于测试注入, 默认 self._model。"""
        model = detector if detector is not None else self._load()
        results = model(frame, conf=conf_threshold, verbose=False)

        dets: list[TileDet] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                if not 0 <= cls_id <= 33:
                    continue  # 越界 class 防御
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].tolist()
                dets.append(
                    TileDet(
                        tile=cls_id,
                        x1=xyxy[0],
                        y1=xyxy[1],
                        x2=xyxy[2],
                        y2=xyxy[3],
                        conf=conf,
                    )
                )
        return dets
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_cv/test_screen_vision.py -v`
Expected: 2 passed(测试代码共 2 个测试方法)

- [ ] **Step 5: ruff 检查**

Run: `ruff check src/mahjong_cv/screen_vision.py tests/test_cv/test_screen_vision.py`
Expected: All checks passed!

---

### Task 7: 状态机 GameStateTracker

**Files:**
- Create: `src/mahjong_ai/state/__init__.py`
- Create: `src/mahjong_ai/state/tracker.py`
- Create: `tests/test_state/__init__.py`(空文件)
- Test: `tests/test_state/test_tracker.py`

**Interfaces:**
- Consumes: `TileDet`(Task 4)、`ScreenLayout`/`DEFAULT_LAYOUT`(Task 2)
- Produces: `DiscardEvent(tile, frame_count)`、`GameSnapshot(my_hand, my_melds, turn, discard_event, meld_detected)`、`GameStateTracker(layout=DEFAULT_LAYOUT, window=5, majority=3)` 含 `.update(dets: list[TileDet], frame_size: tuple[int, int]) -> GameSnapshot`
- 常量:`TURN_MY = "MY_TURN"`、`TURN_WAITING = "WAITING"`、`TURN_UNKNOWN = "UNKNOWN"`;`HAND_CONF = 0.55`

- [ ] **Step 1: 写失败测试**

```python
"""状态机单元测试 — 时序去抖/回合推断/出牌事件。"""
from src.mahjong_ai.state.tracker import (
    TURN_MY,
    TURN_WAITING,
    GameStateTracker,
)
from src.mahjong_cv.detections import TileDet
from src.mahjong_cv.screen_layout import ScreenLayout, ScreenRegion

# 简化布局: 手牌区(下)/桌面区(中), 便于构造测试
TEST_LAYOUT = ScreenLayout(
    my_hand=ScreenRegion(0.0, 0.8, 1.0, 1.0),
    my_meld=ScreenRegion(0.0, 0.7, 1.0, 0.8),
    table=ScreenRegion(0.0, 0.2, 1.0, 0.6),
    discard=ScreenRegion(0.0, 0.6, 1.0, 0.7),
)

# FW 必须 ≥ 520: 13 张手牌 x 坐标 0..480, 归一化后需全部落在手牌区内
FW, FH = 600, 300


def hand_det(tile: int, x: int, conf: float = 0.9) -> TileDet:
    """手牌区检测框(底部 y=290)。"""
    return TileDet(tile, float(x), 290.0, float(x + 30), 299.0, conf)


def table_det(tile: int, x: int) -> TileDet:
    """桌面区检测框(中部 y=150)。"""
    return TileDet(tile, float(x), 150.0, float(x + 30), 179.0, 0.9)


def mk_tracker(window: int = 3, majority: int = 2) -> GameStateTracker:
    return GameStateTracker(layout=TEST_LAYOUT, window=window, majority=majority)


class TestHandDebounce:
    def test_hand_stabilizes_after_majority_frames(self):
        tr = mk_tracker()
        hand13 = [hand_det(t, x * 40) for x, t in enumerate(range(13))]
        for _ in range(3):  # 连续3帧一致 → 去抖提交
            snap = tr.update(hand13, (FW, FH))
        assert snap.my_hand == list(range(13))
        assert snap.turn == TURN_WAITING

    def test_stray_detection_never_enters_hand(self):
        tr = mk_tracker()
        base = [hand_det(t, x * 40) for x, t in enumerate(range(13))]
        frames = [base + [hand_det(20, 300, conf=0.3)] for _ in range(5)]
        # 低置信度帧 + 只在1帧出现的杂讯
        frames[2] = base + [hand_det(21, 300, conf=0.9)]
        for f in frames:
            snap = tr.update(f, (FW, FH))
        assert 20 not in snap.my_hand and 21 not in snap.my_hand
        assert len(snap.my_hand) == 13

    def test_fourteen_tiles_means_my_turn(self):
        tr = mk_tracker()
        hand14 = [hand_det(t, x * 40) for x, t in enumerate(range(14))]
        for _ in range(3):
            snap = tr.update(hand14, (FW, FH))
        assert snap.turn == TURN_MY
        assert len(snap.my_hand) == 14

    def test_pair_hand_preserves_multiplicity(self):
        # 真实手牌常含对子: 14张但只有13种 → 去抖必须保留重复张数
        tr = mk_tracker()
        hand14 = [hand_det(0, 0), hand_det(0, 40)]  # 对子
        hand14 += [hand_det(t, (t + 1) * 40) for t in range(1, 13)]
        for _ in range(3):
            snap = tr.update(hand14, (FW, FH))
        assert len(snap.my_hand) == 14
        assert snap.my_hand.count(0) == 2
        assert snap.turn == TURN_MY

    def test_hand_capped_at_14(self):
        tr = mk_tracker()
        dets = [hand_det(t, (t % 14) * 40) for t in range(16)]  # 16张重复
        for _ in range(3):
            snap = tr.update(dets, (FW, FH))
        assert len(snap.my_hand) <= 14


class TestDiscardEvent:
    def test_new_table_tile_fires_event_once(self):
        tr = mk_tracker()
        hand13 = [hand_det(t, x * 40) for x, t in enumerate(range(13))]
        for _ in range(2):
            tr.update(hand13, (FW, FH))  # 先稳定13张

        snap = tr.update(hand13 + [table_det(30, 100)], (FW, FH))
        assert snap.discard_event is not None
        assert snap.discard_event.tile == 30

        for _ in range(4):
            snap = tr.update(hand13 + [table_det(30, 100)], (FW, FH))
        assert snap.discard_event is None  # 事件3帧后过期

    def test_first_frame_no_event(self):
        tr = mk_tracker()
        snap = tr.update([table_det(30, 100)], (FW, FH))
        assert snap.discard_event is None

    def test_no_event_when_hand_is_14(self):
        tr = mk_tracker()
        hand14 = [hand_det(t, x * 40) for x, t in enumerate(range(14))]
        for _ in range(2):
            tr.update(hand14, (FW, FH))
        snap = tr.update(hand14 + [table_det(30, 100)], (FW, FH))
        assert snap.discard_event is None


class TestMeldRegion:
    def test_meld_detected_flag(self):
        tr = mk_tracker()
        hand13 = [hand_det(t, x * 40) for x, t in enumerate(range(13))]
        meld = TileDet(5, 200.0, 215.0, 230.0, 224.0, 0.9)  # my_meld 区
        for _ in range(2):
            snap = tr.update(hand13 + [meld], (FW, FH))
        assert snap.meld_detected is True
        assert 5 in snap.my_melds
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_state/test_tracker.py -v`
Expected: FAIL,`ModuleNotFoundError: No module named 'src.mahjong_ai.state'`

- [ ] **Step 3: 实现**

`src/mahjong_ai/state/__init__.py`:

```python
"""牌局状态层 — 帧序列去抖与状态跟踪。"""
from src.mahjong_ai.state.tracker import (
    TURN_MY,
    TURN_UNKNOWN,
    TURN_WAITING,
    DiscardEvent,
    GameSnapshot,
    GameStateTracker,
)

__all__ = [
    'TURN_MY', 'TURN_WAITING', 'TURN_UNKNOWN',
    'DiscardEvent', 'GameSnapshot', 'GameStateTracker',
]
```

`src/mahjong_ai/state/tracker.py`:

```python
"""状态机 — 检测帧序列 → 去抖后的牌局快照。

去抖: 手牌区每帧的牌集合进入滑动窗口, 出现 ≥majority 次才提交,
抑制 YOLO 偶发漏检/误检造成的抖动。P1 范围: 无副露局完整支持,
检测到副露时置 meld_detected 供上层降级(完整副露支持在 P2)。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from src.mahjong_cv.detections import TileDet
from src.mahjong_cv.screen_layout import DEFAULT_LAYOUT, ScreenLayout

TURN_MY = 'MY_TURN'
TURN_WAITING = 'WAITING'
TURN_UNKNOWN = 'UNKNOWN'

HAND_CONF = 0.55
DEBOUNCE_WINDOW = 5
DEBOUNCE_MAJORITY = 3


@dataclass
class DiscardEvent:
    """检测到桌面中央出现新牌(别家打出)。"""

    tile: int
    frame_count: int = 0


@dataclass
class GameSnapshot:
    """去抖后的牌局状态。"""

    my_hand: list[int] = field(default_factory=list)
    my_melds: list[int] = field(default_factory=list)
    turn: str = TURN_UNKNOWN
    discard_event: DiscardEvent | None = None
    meld_detected: bool = False


class GameStateTracker:
    """对检测帧做区域归类 + 时序去抖。"""

    def __init__(
        self,
        layout: ScreenLayout = DEFAULT_LAYOUT,
        window: int = DEBOUNCE_WINDOW,
        majority: int = DEBOUNCE_MAJORITY,
    ) -> None:
        self._layout = layout
        self._window = window
        self._majority = majority
        self._hand_history: list[Counter[int]] = []
        self._prev_table: set[int] | None = None
        self._last_event: DiscardEvent | None = None
        self._last_event_age = 0

    def _region_of_det(self, d: TileDet, frame_w: int, frame_h: int) -> str:
        cx = (d.x1 + d.x2) / 2 / frame_w
        cy = (d.y1 + d.y2) / 2 / frame_h
        return self._layout.region_of(cx, cy)

    def _hand_region_counts(
        self, dets: list[TileDet], frame_w: int, frame_h: int
    ) -> Counter[int]:
        hand: Counter[int] = Counter()
        for d in dets:
            if d.conf >= HAND_CONF and self._region_of_det(d, frame_w, frame_h) == 'my_hand':
                hand[d.tile] += 1
        return hand

    def _debounced_hand(self) -> list[int]:
        """滑动窗口内出现 ≥majority 次的牌型进入手牌, 多张数取窗口内最大。

        多张数必须保留: 真实手牌几乎都含对子/刻子, 若按牌型去抖会折叠
        重复牌, 14 张手牌只能去抖出 ≤13 种 → TURN_MY 永不触发
        (Task 8 审查发现的缺陷, 已修复)。
        """
        counts: Counter[int] = Counter()
        for c in self._hand_history:
            counts.update(c)
        result: list[int] = []
        for t, n in counts.items():
            if n >= self._majority:
                mult = max(c[t] for c in self._hand_history)
                result.extend([t] * mult)
        result.sort(key=lambda t: (-counts[t], t))
        return result[:14]

    def update(self, dets: list[TileDet], frame_size: tuple[int, int]) -> GameSnapshot:
        """输入一帧检测结果, 返回去抖后的快照。"""
        frame_w, frame_h = frame_size

        self._hand_history.append(self._hand_region_counts(dets, frame_w, frame_h))
        if len(self._hand_history) > self._window:
            self._hand_history.pop(0)
        my_hand = self._debounced_hand()

        table = {
            d.tile
            for d in dets
            if d.conf >= HAND_CONF and self._region_of_det(d, frame_w, frame_h) == 'table'
        }
        new_tiles: set[int] = set()
        if self._prev_table is not None:
            new_tiles = table - self._prev_table
        else:
            self._prev_table = set()
        self._prev_table = table

        if new_tiles and len(my_hand) == 13:
            self._last_event = DiscardEvent(tile=sorted(new_tiles)[0], frame_count=0)
            self._last_event_age = 0
        elif self._last_event is not None:
            self._last_event_age += 1
            if self._last_event is not None:
                self._last_event.frame_count = self._last_event_age  # 事件存活帧数
            if self._last_event_age > 3:
                self._last_event = None

        meld_tiles = [
            d.tile
            for d in dets
            if d.conf >= HAND_CONF and self._region_of_det(d, frame_w, frame_h) == 'my_meld'
        ]

        if len(my_hand) == 14:
            turn = TURN_MY
        elif len(my_hand) == 13:
            turn = TURN_WAITING
        else:
            turn = TURN_UNKNOWN

        return GameSnapshot(
            my_hand=my_hand,
            my_melds=sorted(set(meld_tiles)),
            turn=turn,
            discard_event=self._last_event,
            meld_detected=bool(meld_tiles),
        )
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_state/test_tracker.py -v`
Expected: 8 passed

- [ ] **Step 5: ruff 检查**

Run: `ruff check src/mahjong_ai/state tests/test_state/test_tracker.py`
Expected: All checks passed!

---

### Task 8: 决策会话 AssistantSession

**Files:**
- Create: `src/mahjong_ai/session.py`
- Test: `tests/test_ai/test_session.py`

**Interfaces:**
- Consumes: `GameStateTracker`(Task 7)、`StrategyEngine`(`src/mahjong_ai/strategy.py`)、`advise_action`(`src/mahjong_ai/action_advisor.py`)、`get_waiting_tiles`(engine)、`IRuleSet`
- Produces: `Advice(snapshot, discard, waiting, action)`、`AssistantSession(rules: IRuleSet)` 含 `.tick(dets: list[TileDet], frame_size: tuple[int, int]) -> Advice`、`.tracker` 属性
- 语义: 14张+自己回合 → `discard` 出牌推荐; 13张 → `waiting` 听牌列表; 出牌事件+13张 → `action` 碰/杠/胡/过建议; 检测到副露 → 仅 snapshot(降级)

- [ ] **Step 1: 写失败测试**

```python
"""决策会话单元测试 — 用合成检测序列驱动完整决策链路。"""
from src.mahjong_ai.session import AssistantSession
from src.mahjong_ai.state.tracker import GameStateTracker, TURN_MY
from src.mahjong_cv.detections import TileDet
from src.mahjong_cv.screen_layout import ScreenLayout, ScreenRegion
from src.mahjong_core.tile import B4, B5, B6, DONG, T7, T8, T9, W1, W2, W3, W4
from src.mahjong_engine.rules.huanyu_rules import HuanyuRules

TEST_LAYOUT = ScreenLayout(
    my_hand=ScreenRegion(0.0, 0.8, 1.0, 1.0),
    my_meld=ScreenRegion(0.0, 0.7, 1.0, 0.8),
    table=ScreenRegion(0.0, 0.2, 1.0, 0.6),
    discard=ScreenRegion(0.0, 0.6, 1.0, 0.7),
)

FW, FH = 400, 300


def hand_det(tile: int, x: int) -> TileDet:
    return TileDet(tile, float(x), 290.0, float(x + 30), 299.0, 0.9)


def table_det(tile: int, x: int) -> TileDet:
    return TileDet(tile, float(x), 150.0, float(x + 30), 179.0, 0.9)


class TestAssistantSession:
    def setup_method(self):
        self.session = AssistantSession(HuanyuRules())
        self.session.tracker = self._tracker()  # 用测试布局

    @staticmethod
    def _tracker() -> GameStateTracker:
        return GameStateTracker(layout=TEST_LAYOUT, window=3, majority=2)

    def _feed(self, dets: list[TileDet], n: int = 3) -> None:
        for _ in range(n):
            self.session.tick(dets, (FW, FH))

    def test_14_tiles_recommends_discard(self):
        hand14 = [hand_det(t, i * 40) for i, t in enumerate(
            [W1, W2, W3, W4, W1, W2, W3, W4, B4, B5, B6, T7, T8, T9]
        )]
        self._feed(hand14)
        advice = self.session.tick(hand14, (FW, FH))
        assert advice.discard is not None
        assert advice.waiting is None
        assert advice.snapshot.turn == TURN_MY

    def test_13_tiles_shows_waiting(self):
        hand13 = [hand_det(t, i * 40) for i, t in enumerate(
            [W1, W2, W3, B4, B5, B6, T7, T8, T9, DONG, DONG, DONG, W4]
        )]
        self._feed(hand13)
        advice = self.session.tick(hand13, (FW, FH))
        assert advice.waiting is not None and len(advice.waiting) >= 1
        assert advice.discard is None

    def test_discard_event_win_advice(self):
        hand13 = [hand_det(t, i * 40) for i, t in enumerate(
            [W1, W2, W3, B4, B5, B6, T7, T8, T9, DONG, DONG, DONG, W4]
        )]
        self._feed(hand13)
        advice = self.session.tick(hand13 + [table_det(W4, 100)], (FW, FH))
        assert advice.action is not None
        assert advice.action.action == 'win'

    def test_discard_event_pass_advice(self):
        hand13 = [hand_det(t, i * 40) for i, t in enumerate(
            [W1, W2, W3, B4, B5, B6, T7, T8, T9, DONG, DONG, DONG, W4]
        )]
        self._feed(hand13)
        advice = self.session.tick(hand13 + [table_det(20, 100)], (FW, FH))
        assert advice.action is not None
        assert advice.action.action == 'pass'

    def test_meld_detected_degrades_advice(self):
        hand13 = [hand_det(t, i * 40) for i, t in enumerate(range(13))]
        meld = TileDet(5, 200.0, 215.0, 230.0, 224.0, 0.9)
        self._feed(hand13 + [meld])
        advice = self.session.tick(hand13 + [meld], (FW, FH))
        assert advice.snapshot.meld_detected is True
        assert advice.discard is None and advice.waiting is None
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_ai/test_session.py -v`
Expected: FAIL,`ModuleNotFoundError: No module named 'src.mahjong_ai.session'`

- [ ] **Step 3: 实现**

`src/mahjong_ai/session.py`:

```python
"""决策会话 — 状态快照 → AI 建议(规则引擎 + 出牌推荐 + 操作建议)。

P1 范围: 无副露局完整建议; 副露时仅输出快照(HUD 显示降级提示)。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.mahjong_ai.action_advisor import ActionAdvice, advise_action
from src.mahjong_ai.efficiency.discard_selector import DiscardRecommendation
from src.mahjong_ai.state.tracker import GameSnapshot, GameStateTracker
from src.mahjong_ai.strategy import StrategyEngine
from src.mahjong_core import Hand
from src.mahjong_cv.detections import TileDet
from src.mahjong_engine.judges.tenpai_judge import WaitingTile
from src.mahjong_engine.rules.interface import IRuleSet


@dataclass
class Advice:
    """一帧的完整建议(供 HUD 渲染)。"""

    snapshot: GameSnapshot
    discard: DiscardRecommendation | None = None
    waiting: list[WaitingTile] | None = None
    action: ActionAdvice | None = None


class AssistantSession:
    """把视觉帧转化为可展示的 AI 建议。"""

    def __init__(self, rules: IRuleSet) -> None:
        self._rules = rules
        self._engine = StrategyEngine(rules)
        self._tracker = GameStateTracker()

    @property
    def tracker(self) -> GameStateTracker:
        return self._tracker

    @tracker.setter
    def tracker(self, tracker: GameStateTracker) -> None:
        self._tracker = tracker

    def tick(self, dets: list[TileDet], frame_size: tuple[int, int]) -> Advice:
        """处理一帧检测结果, 返回当前建议。"""
        snapshot = self._tracker.update(dets, frame_size)
        advice = Advice(snapshot=snapshot)

        if snapshot.meld_detected:
            return advice  # 副露降级(完整支持在 P2)

        hand = Hand(snapshot.my_hand)
        n = len(hand)

        if snapshot.discard_event is not None and n == 13:
            advice.action = advise_action(hand, snapshot.discard_event.tile, self._rules)

        if n == 14 and snapshot.turn == TURN_MY:
            advice.discard = self._engine.recommend_discard(hand)
        elif n == 13:
            waiting = self._rules.get_waiting_tiles(hand)
            advice.waiting = waiting if waiting else None

        return advice
```

注意: `TURN_MY` 需在文件顶部导入:

```python
from src.mahjong_ai.state.tracker import (
    TURN_MY,
    GameSnapshot,
    GameStateTracker,
)
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_ai/test_session.py -v`
Expected: 5 passed

- [ ] **Step 5: ruff 检查**

Run: `ruff check src/mahjong_ai/session.py tests/test_ai/test_session.py`
Expected: All checks passed!

---

### Task 9: HUD 悬浮窗(展示层)

**Files:**
- Create: `src/mahjong_ui/format.py`
- Create: `src/mahjong_ui/hud.py`
- Create: `tests/test_ui/__init__.py`(空文件)
- Modify: `environment.yml`(pip 段加 `pyside6`)
- Test: `tests/test_ui/test_format.py`

**Interfaces:**
- Consumes: `Advice`(Task 8)、PySide6
- Produces: `advice_to_lines(advice: Advice) -> list[str]`(纯函数)、`HudOverlay(rect_fn: Callable[[], tuple[int,int,int,int] | None])` 含 `.set_advice(advice: Advice) -> None`

- [ ] **Step 1: 写失败测试**

```python
"""HUD 文本格式化纯函数测试。"""
from src.mahjong_ai.session import Advice
from src.mahjong_ai.state.tracker import GameSnapshot
from src.mahjong_core.tile import W1, W2, W3
from src.mahjong_ui.format import advice_to_lines


def snap(hand: list[int]) -> GameSnapshot:
    return GameSnapshot(my_hand=hand)


class TestAdviceToLines:
    def test_empty_hand(self):
        lines = advice_to_lines(Advice(snapshot=snap([])))
        assert '未检测到手牌' in lines[0]

    def test_discard_line(self):
        from src.mahjong_ai.efficiency.discard_selector import DiscardRecommendation

        advice = Advice(
            snapshot=snap([W1, W2, W3]),
            discard=DiscardRecommendation(
                tile=W1, reason='打出 一万 后听牌', shanten_before=2, shanten_after=1,
            ),
        )
        lines = advice_to_lines(advice)
        assert '推荐打出: 一万' in lines[1]
        assert '2→1' in lines[1]

    def test_waiting_line(self):
        from src.mahjong_engine.judges.tenpai_judge import WaitingTile

        advice = Advice(
            snapshot=snap([W1, W2, W3]),
            waiting=[WaitingTile(tile=W1), WaitingTile(tile=W2)],
        )
        lines = advice_to_lines(advice)
        assert '听牌: 一万 二万' in lines[1]

    def test_action_line(self):
        from src.mahjong_ai.action_advisor import ActionAdvice

        advice = Advice(
            snapshot=snap([W1, W2, W3]),
            action=ActionAdvice(action='win', tile=W1, reason='可以胡牌 (standard)'),
        )
        lines = advice_to_lines(advice)
        assert 'WIN: 一万' in lines[1]

    def test_meld_degrade_notice(self):
        advice = Advice(snapshot=snap([W1]), )
        advice.snapshot.meld_detected = True
        lines = advice_to_lines(advice)
        assert '副露' in lines[-1]

    def test_waiting_for_change(self):
        advice = Advice(snapshot=snap([W1, W2, W3]))
        lines = advice_to_lines(advice)
        assert '等待局面变化' in lines[1]
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_ui/test_format.py -v`
Expected: FAIL,`ModuleNotFoundError: No module named 'src.mahjong_ui.format'`

- [ ] **Step 3: 实现**

`src/mahjong_ui/format.py`:

```python
"""HUD 文本格式化 — 纯函数, 可单测。"""

from __future__ import annotations

from src.mahjong_ai.session import Advice
from src.mahjong_core.tile import tile_display


def advice_to_lines(advice: Advice) -> list[str]:
    """把建议转成 HUD 显示的行列表。"""
    if not advice.snapshot.my_hand:
        return ['未检测到手牌...']

    lines = ['手牌: ' + ' '.join(tile_display(t) for t in advice.snapshot.my_hand)]

    if advice.discard is not None:
        lines.append(
            f'推荐打出: {tile_display(advice.discard.tile)} '
            f'(向听 {advice.discard.shanten_before}→{advice.discard.shanten_after})'
        )
    elif advice.action is not None and advice.action.action != 'pass':
        # 胡/杠/碰提示优先于听牌行: 能胡必听牌, 若后显示会被遮蔽(最终审查 C2)
        lines.append(
            f'{advice.action.action.upper()}: {tile_display(advice.action.tile)} '
            f'— {advice.action.reason}'
        )
    elif advice.waiting is not None and advice.waiting:
        wt = ' '.join(tile_display(w.tile) for w in advice.waiting)
        lines.append(f'听牌: {wt}')
    elif advice.action is not None:
        lines.append(
            f'{advice.action.action.upper()}: {tile_display(advice.action.tile)} '
            f'— {advice.action.reason}'
        )
    else:
        lines.append('等待局面变化...')

    if advice.snapshot.meld_detected:
        lines.append('(检测到副露, 出牌建议暂不可用)')
    return lines
```

`src/mahjong_ui/hud.py`:

```python
"""悬浮窗 HUD — 无边框置顶半透明, 叠在游戏窗口上方。"""

from __future__ import annotations

from collections.abc import Callable

from src.mahjong_ai.session import Advice
from src.mahjong_ui.format import advice_to_lines


class HudOverlay:
    """无边框置顶悬浮窗, 每 100ms 重新吸附游戏窗口。

    绘制通过给 QWidget 实例赋绑定方法 paintEvent 实现(Qt 以
    paintEvent(event) 调用, 绑定方法已绑定 self, 签名恰好匹配);
    PySide6 在运行时 import(无类型存根, 保持 mypy strict)。
    """

    def __init__(
        self, rect_fn: Callable[[], tuple[int, int, int, int] | None]
    ) -> None:
        from PySide6.QtCore import Qt, QTimer  # noqa: PLC0415
        from PySide6.QtWidgets import QWidget  # noqa: PLC0415

        self._rect_fn = rect_fn
        self._lines: list[str] = ['启动中...']

        self._widget = QWidget()
        self._widget.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self._widget.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._widget.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._widget.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        # PySide6 6.x 自带 py.typed 存根: 直接赋值报 mypy method-assign,
        # setattr 又触发 ruff B010——这是真实的二选一, noqa 必要
        setattr(self._widget, 'paintEvent', self._paint_event)  # noqa: B010

        self._timer = QTimer()
        self._timer.timeout.connect(self._refresh)
        self._timer.start(100)

    def set_advice(self, advice: Advice) -> None:
        self._lines = advice_to_lines(advice)
        self._widget.update()

    def _paint_event(self, _event: object) -> None:
        from PySide6.QtGui import QColor, QFont, QPainter  # noqa: PLC0415

        painter = QPainter(self._widget)
        painter.fillRect(painter.window(), QColor(0, 0, 0, 120))
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont('Microsoft YaHei', 16, QFont.Weight.Bold))
        for i, line in enumerate(self._lines):
            painter.drawText(20, 40 + i * 32, line)
        painter.end()

    def _refresh(self) -> None:
        rect = self._rect_fn()
        if rect is None:
            self._widget.hide()
            return
        left, top, width, height = rect
        self._widget.setGeometry(left, top, width, height)
        self._widget.show()
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_ui/test_format.py -v`
Expected: 6 passed

- [ ] **Step 5: ruff 检查**

Run: `ruff check src/mahjong_ui tests/test_ui/test_format.py`
Expected: All checks passed!

- [ ] **Step 6: HUD 布局冒烟(手动)**

跳过,由 Task 10 Step 6 端到端验收覆盖(HUD 首次展示时检查: 悬浮层无边框、半透明、不挡鼠标、吸附游戏窗口)。

---

### Task 10: 主程序 run_assistant.py + 端到端验收

**Files:**
- Create: `scripts/run_assistant.py`
- Test: `tests/test_scripts/test_run_assistant.py`(纯函数: annotate)

**Interfaces:**
- Consumes: `Win32Capture`(Task 3)、`ScreenVision`(Task 6)、`AssistantSession`(Task 8)、`HudOverlay`(Task 9)、`get_rules`(engine)、`advice_to_lines`(Task 9)
- Produces: CLI 入口 `python scripts/run_assistant.py [--model ...] [--rules huanyu] [--image shot.png] [--debug-dir dbg]`

- [ ] **Step 1: 写失败测试**

```python
"""run_assistant 纯函数测试。"""
import numpy as np

from scripts.run_assistant import annotate
from src.mahjong_cv.detections import TileDet


class TestAnnotate:
    def test_draws_boxes(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        out = annotate(frame, [TileDet(0, 10, 20, 40, 50, 0.9)])
        # 框线处应出现绿色像素
        assert out[20:22, 10:12].sum() > 0
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_scripts/test_run_assistant.py -v`
Expected: FAIL,`ModuleNotFoundError: No module named 'scripts.run_assistant'`

- [ ] **Step 3: 实现**

`scripts/run_assistant.py`:

```python
"""欢乐麻将实时 AI 辅助: 截屏 → 识别 → 状态 → 建议 → HUD。

用法:
    python scripts/run_assistant.py                        # 实时模式(需欢乐麻将已开)
    python scripts/run_assistant.py --image shot.png       # 单帧调试
    python scripts/run_assistant.py --debug-dir dbg        # 实时+保存标注帧
    python scripts/run_assistant.py --model <best.pt>      # 指定屏幕模型
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mahjong_ai.session import AssistantSession
from src.mahjong_cv.capture.win32 import Win32Capture
from src.mahjong_cv.detections import TileDet
from src.mahjong_cv.screen_vision import ScreenVision
from src.mahjong_engine import get as get_rules

MODEL = 'data/models/screen/mahjong_screen_detector/weights/best.pt'


def annotate(frame: np.ndarray, dets: list[TileDet]) -> np.ndarray:
    """在帧上画绿色检测框(调试用)。"""
    for d in dets:
        cv2.rectangle(
            frame, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)),
            (0, 255, 0), 2,
        )
    return frame


def _print_advice(advice) -> None:
    from src.mahjong_ui.format import advice_to_lines

    for line in advice_to_lines(advice):
        print(line)


def single_frame(image: str, model: str, rules_name: str) -> None:
    """单帧调试: 处理一张截图, 打印建议。"""
    frame = cv2.imread(image)
    if frame is None:
        print(f'无法读取图片: {image}')
        sys.exit(1)
    vision = ScreenVision(model)
    dets = vision.process(frame)
    session = AssistantSession(get_rules(rules_name))
    # tracker 去抖需要 ≥3 帧一致才提交手牌, 单帧模式下喂 3 次同一帧
    # (最终审查 C1: 只 tick 一次永远得空手牌)
    for _ in range(3):
        advice = session.tick(dets, frame.shape[:2][::-1])
    print(f'检测到 {len(dets)} 张牌')
    _print_advice(advice)


def main() -> None:
    parser = argparse.ArgumentParser(description='欢乐麻将实时 AI 辅助')
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--rules', default='huanyu')
    parser.add_argument('--image', help='单帧调试: 处理一张截图后退出')
    parser.add_argument('--debug-dir', help='实时模式: 周期性保存标注帧')
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f'模型不存在: {args.model}')
        print('请先完成数据采集与训练(见 README「欢乐麻将实操」), 再运行本程序')
        sys.exit(1)

    if args.image:
        single_frame(args.image, args.model, args.rules)
        return

    from PySide6.QtCore import QTimer  # noqa: PLC0415
    from PySide6.QtWidgets import QApplication  # noqa: PLC0415

    from src.mahjong_ui.hud import HudOverlay

    app = QApplication(sys.argv)

    cap = Win32Capture('欢乐麻将')
    vision = ScreenVision(args.model)
    session = AssistantSession(get_rules(args.rules))
    hud = HudOverlay(cap.client_rect)

    def tick() -> None:
        frame = cap.capture()
        if frame is None:
            return
        dets = vision.process(frame)
        advice = session.tick(dets, frame.shape[:2][::-1])
        hud.set_advice(advice)
        if args.debug_dir:
            out = Path(args.debug_dir)
            out.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / 'annotated.jpg'), annotate(frame, dets))

    timer = QTimer()
    timer.timeout.connect(tick)
    timer.start(66)  # ~15fps
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_scripts/test_run_assistant.py -v`
Expected: 1 passed

- [ ] **Step 5: ruff 检查**

Run: `ruff check scripts/run_assistant.py tests/test_scripts/test_run_assistant.py`
Expected: All checks passed!

- [ ] **Step 6: 端到端验收(手动, P1 里程碑)**

1. 单帧自检:`python scripts/run_assistant.py --image <一张欢乐麻将截图>`
   Expected: 打印检测牌数 + 手牌行 + 推荐/听牌行
2. 实时:`python scripts/run_assistant.py --debug-dir dbg`
   - 打开欢乐麻将进一局, HUD 悬浮层覆盖在游戏窗口上
   - 我的回合(14张) → 显示「推荐打出」; 等摸牌(13张) → 显示「听牌」
   - `dbg/annotated.jpg` 每帧更新, 绿色框覆盖所有牌
3. 若手牌区识别偏(框不在牌上): 调整 `src/mahjong_cv/screen_layout.py` 的 `DEFAULT_LAYOUT` 各区域数值, 直到手牌/桌面区分正确
4. **里程碑达成**: 完整打完一局欢乐麻将推倒胡, 全程自动提示无崩溃

---

### Task 11: 全量质量门禁 + README

**Files:**
- Modify: `README.md`
- (无新代码)

- [ ] **Step 1: 全量测试**

Run: `python -m pytest -q`
Expected: 全部通过(原 103 + 新增约 35 ≈ 138 passed)

- [ ] **Step 2: mypy strict**

Run: `mypy src`
Expected: Success, no issues found
注意: `src/mahjong_cv/pipeline.py:27` 与 `scene_pipeline.py` 存在**既有 mypy 错误**(pipeline.py 的 `# type: ignore[attr-defined]` 与实际错误码 import-not-found 不匹配;scene_pipeline.py 缺注解/None 可调用/元组注解)。Task 3 已加 pyproject override 后, `pipeline.py:27` 的 ignore 注释变成 unused——按下面顺序修复:
1. `pipeline.py:27`: 删除该行 `# type: ignore[attr-defined]` 注释(overrides 已覆盖)
2. `scene_pipeline.py`: `_load` 补 `-> None` 返回注解并处理 None;`process` 内 `self._scene`/`self._tile` 判 None;`in_region` 参数 `region: tuple[float, float, float, float]`;`region_of` 内联变量加注解消除 no-any-return(修复后重跑 mypy 直到该文件干净)
3. 其余文件若 mypy 报错,同样修复后重跑

- [ ] **Step 3: ruff 全量**

Run: `ruff check src tests scripts`
Expected: All checks passed!

- [ ] **Step 4: 若有失败, 修复对应任务模块后重跑前三步直到全绿**

- [ ] **Step 5: 依赖协调(与全量门禁同批处理)**

- OpenCV 来源协调: `environment.yml` 的 conda 段有 `opencv`, 若本机 cv2 来自 pip 的 `opencv-python`, 确认二者来源统一(实测中文 Windows 需 cv2 能处理中文路径——已由 imdecode 变通兜底, 版本固定仅防御性); 在 `environment.yml` pip 段固定 `opencv-python==4.10.0.84` 或删除 conda opencv 二选一, 保证环境可复现
- Python 版本: 本机实测为 3.12.3 而 environment.yml 声明 3.11——若保留 3.12 运行, 把 environment.yml 的 python 声明改为 3.12(或明确 3.11 环境重建); 在 README 注明最低版本要求

- [ ] **Step 6: README 增加「欢乐麻将实操」章节**

在 `README.md` 功能概要后追加:

```markdown
## 欢乐麻将实操(线上闭环)

1. 训练数据采集: `python scripts/capture_dataset.py --frames 2000 --preview`
2. 数据集拆分: `python scripts/build_dataset_yaml.py`
3. 模型训练: `python scripts/train_screen_yolo.py`
4. 实时辅助: `python scripts/run_assistant.py`(需欢乐麻将 PC 客户端已开启)

数据与模型约定见 `docs/superpowers/specs/2026-08-05-mahjong-assistant-v2-design.md` §3。
```

- [ ] **Step 6: 收尾确认**

确认 P1 里程碑达成(见 Task 10 Step 6),并将状态更新到设计文档(阶段路线图 P1 打勾)。P2 开始前,用 brainstorming → writing-plans 流程产出 P2 计划。
