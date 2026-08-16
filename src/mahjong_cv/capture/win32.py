"""欢乐麻将 PC 客户端窗口捕获(Win32 + mss)。

Win32 调用全部用 ctypes(无第三方依赖); 截屏用 mss(高速)。
DPI: 首次截屏时按需设置 Per-Monitor DPI Aware(与 Qt 默认 V2 一致),
不在模块导入时抢先设置, 避免与 Qt 启动时的 DPI 设置冲突(ACCESS_DENIED 警告)。
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from typing import Any

import numpy as np

user32 = ctypes.windll.user32

_DPI_AWARE = False

#: DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2(Win10 1703+, 与 Qt 默认一致)
_DCA_V2 = ctypes.c_void_p(-4)


def _set_dpi_aware() -> bool:
    """设置 Per-Monitor DPI Aware(V2)。失败返回 False(坐标可能被虚拟化)。

    若进程已被其他组件(如 Qt)设为 DPI aware, 本调用会失败——
    这正是期望行为: 说明坐标空间已正确, 静默跳过即可。
    """
    global _DPI_AWARE
    if _DPI_AWARE:
        return True
    try:
        if user32.SetProcessDpiAwarenessContext(_DCA_V2):
            _DPI_AWARE = True
            return True
    except OSError:
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except OSError:
        try:
            user32.SetProcessDPIAware()
        except OSError:
            return False
    _DPI_AWARE = True
    return True


class _RECT(ctypes.Structure):
    _fields_ = [
        ('left', wintypes.LONG),
        ('top', wintypes.LONG),
        ('right', wintypes.LONG),
        ('bottom', wintypes.LONG),
    ]


def list_window_titles() -> list[str]:
    """列出所有可见顶层窗口标题(诊断用)。"""
    titles: list[str] = []

    def _callback(hwnd: int, _lparam: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        titles.append(buf.value)
        return True

    _ProcType = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user32.EnumWindows(_ProcType(_callback), 0)
    return titles


#: 欢乐麻将窗口标题候选: 正常中文 + 编码乱码形态
#: 「欢乐麻将全集」在 GBK 码页下显示为「娆箰楹诲皢鍏ㄩ泦」, 确定性乱码
DEFAULT_TITLE_CANDIDATES = ('欢乐麻将', '娆箰楹诲', '鍏ㄩ泦')


def find_window_candidates(title_substrings: tuple[str, ...]) -> int | None:
    """按候选标题子串依次查找, 返回第一个命中的 hwnd; 全部未命中返回 None。"""
    for t in title_substrings:
        hwnd = find_window(t)
        if hwnd is not None:
            return hwnd
    return None


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
    """欢乐麻将窗口截屏。轮询节奏(10-30fps)由调用方控制。

    candidates: 标题候选(子串匹配), 依次尝试; 缺省只查 title_substring。
    游戏窗口标题可能是编码乱码(如「娆箰楹诲皢鍏ㄩ泦」), 用候选兜底。
    """

    def __init__(
        self,
        title_substring: str = '欢乐麻将',
        candidates: tuple[str, ...] | None = None,
    ) -> None:
        self._candidates = candidates or (title_substring,)
        self._hwnd: int | None = None
        self._sct: Any = None

    @property
    def hwnd(self) -> int | None:
        if self._hwnd is None:
            self._hwnd = find_window_candidates(self._candidates)
        return self._hwnd

    def client_rect(self) -> tuple[int, int, int, int] | None:
        hwnd = self.hwnd
        if hwnd is None:
            return None
        return client_rect(hwnd)

    def capture(self) -> np.ndarray | None:
        """截取客户区, 返回 BGR ndarray; 失败返回 None(调用方跳过该帧)。

        BitBlt 会瞬时失败(窗口最小化/切换瞬间/图形资源竞争) —
        单帧失败不应杀死主循环, 返回 None 由调用方跳过。
        """
        _set_dpi_aware()  # 按需设置(与 Qt 冲突时静默失败, 坐标已正确)
        hwnd = self.hwnd
        if hwnd is None:
            return None
        rect = client_rect(hwnd)
        if rect[2] <= 0 or rect[3] <= 0:
            return None
        if self._sct is None:
            import mss  # pyproject 已配 ignore_missing_imports

            self._sct = mss.mss()
        try:
            # shot.bgra = 4字节/像素 BGRA(mss 的 .rgb 是 3字节真RGB, 不能 reshape(..., 4))
            shot = self._sct.grab(rect_to_monitor(rect))
        except Exception:  # noqa: BLE001 — BitBlt 瞬时失败: 跳过本帧
            return None
        arr = np.frombuffer(shot.bgra, dtype=np.uint8).reshape(
            shot.height, shot.width, 4
        )
        return arr[:, :, :3].copy()
