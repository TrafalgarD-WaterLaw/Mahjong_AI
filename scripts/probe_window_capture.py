"""窗口捕获探针 — 实测 BitBlt / PrintWindow 对游戏窗口的捕获效果。

背景: mss 屏幕抓取对遮挡窗口无能为力; BitBlt 从窗口 DC 拷贝理论上
不受遮挡影响, 但 DirectX/硬件加速渲染的窗口内容不在 GDI DC 里 →
可能黑屏。本脚本输出两张对比图, 人工确认哪张有效:
    data/probe_bitblt.png       BitBlt(直接拷贝窗口 DC)
    data/probe_printwindow.png  PrintWindow(PW_RENDERFULLCONTENT, 强制 DWM 渲染)

用法:
    python scripts/probe_window_capture.py [--title 欢乐麻将]
"""

import argparse
import ctypes
import sys
from ctypes import wintypes
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mahjong_cv.capture.win32 import (  # noqa: E402
    DEFAULT_TITLE_CANDIDATES,
    find_window_candidates,
)

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32

_PW_RENDERFULLCONTENT = 2


class _BITMAPINFO(ctypes.Structure):
    _fields_ = [
        ('biSize', wintypes.DWORD), ('biWidth', wintypes.LONG),
        ('biHeight', wintypes.LONG), ('biPlanes', wintypes.WORD),
        ('biBitCount', wintypes.WORD), ('biCompression', wintypes.DWORD),
        ('biSizeImage', wintypes.DWORD), ('biXPelsPerMeter', wintypes.LONG),
        ('biYPelsPerMeter', wintypes.LONG), ('biClrUsed', wintypes.DWORD),
        ('biClrImportant', wintypes.DWORD),
    ]


def capture_with(method: str, hwnd: int) -> np.ndarray | None:
    """用指定方法捕获窗口客户区 → BGR ndarray。"""
    rect = wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(rect))
    w, h = rect.right, rect.bottom
    if w <= 0 or h <= 0:
        return None

    hwnd_dc = user32.GetWindowDC(hwnd)
    if not hwnd_dc:
        return None
    try:
        mem_dc = gdi32.CreateCompatibleDC(hwnd_dc)
        bmp = gdi32.CreateCompatibleBitmap(hwnd_dc, w, h)
        gdi32.SelectObject(mem_dc, bmp)

        if method == 'bitblt':
            ok = gdi32.BitBlt(mem_dc, 0, 0, w, h, hwnd_dc, 0, 0, 0x00CC0020)
        else:  # printwindow
            ok = user32.PrintWindow(hwnd, mem_dc, _PW_RENDERFULLCONTENT)

        info = _BITMAPINFO()
        info.biSize = ctypes.sizeof(_BITMAPINFO)
        info.biWidth = w
        info.biHeight = -h  # 负 = 自顶向下
        info.biPlanes = 1
        info.biBitCount = 32
        buf = ctypes.create_string_buffer(w * h * 4)
        gdi32.GetDIBits(hwnd_dc, bmp, 0, h, buf, ctypes.byref(info),
                        wintypes.DWORD(0))
        img = np.frombuffer(buf.raw, dtype=np.uint8).reshape(h, w, 4)
        gdi32.DeleteObject(bmp)
        gdi32.DeleteDC(mem_dc)
        return cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGRA2BGR) if ok else None
    finally:
        user32.ReleaseDC(hwnd, hwnd_dc)


def main() -> None:
    parser = argparse.ArgumentParser(description='窗口捕获方式探针')
    parser.add_argument('--title', default='欢乐麻将', help='窗口标题(子串匹配)')
    args = parser.parse_args()

    candidates = DEFAULT_TITLE_CANDIDATES if args.title == '欢乐麻将' \
        else (args.title,)
    hwnd = find_window_candidates(candidates)
    if hwnd is None:
        print(f'未找到窗口: {args.title}')
        sys.exit(1)
    print(f'窗口句柄: {hwnd}')

    out = Path('data')
    out.mkdir(parents=True, exist_ok=True)
    for method, name in (('bitblt', 'probe_bitblt.png'),
                         ('printwindow', 'probe_printwindow.png')):
        img = capture_with(method, hwnd)
        if img is None:
            print(f'{method}: 捕获失败')
            continue
        mean = float(img.mean())
        cv2.imwrite(str(out / name), img)
        print(f'{method}: 已保存 {out / name}  平均亮度={mean:.1f}'
              f'(黑屏≈0-15, 正常画面≈80+)')


if __name__ == '__main__':
    main()
