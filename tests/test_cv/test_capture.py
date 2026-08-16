"""捕获层单元测试(纯函数与无窗口场景, 不依赖真实游戏窗口)。"""
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
        # DPI 感知按需设置(导入时不抢先, 避免与 Qt 冲突)
        from src.mahjong_cv.capture import win32

        assert win32._set_dpi_aware() is True
        assert win32._DPI_AWARE is True
