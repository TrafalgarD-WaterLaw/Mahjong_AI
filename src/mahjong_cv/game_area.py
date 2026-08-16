"""游戏画面区域检测 — 自动去黑边, 区域坐标相对画面自适应。

窗口非等比缩放时游戏客户端通常"等比画面 + 黑边填充" —
框选的牌河区域若相对整个客户区归一化会错位。检测实际画面
(四边向内收缩到非黑色边界), 区域相对画面区域映射, 窗口
大小/比例随意变化都跟随。
"""

from __future__ import annotations

import cv2
import numpy as np

#: 黑边判定阈值(灰度最大值低于此值视为黑边)
_BLACK_MAX = 12


def detect_game_area(frame: np.ndarray) -> tuple[int, int, int, int] | None:
    """检测画面区域 (x1, y1, x2, y2); 无黑边返回 None(全帧即画面)。

    黑边 = 整行/整列的灰度最大值 < _BLACK_MAX。最多收缩 1/3 宽高
    (防止画面本身有大片黑误判)。
    """
    if frame.size == 0:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    top, bottom = 0, h - 1
    while top < h // 3 and int(gray[top, :].max()) < _BLACK_MAX:
        top += 1
    while bottom > h * 2 // 3 and int(gray[bottom, :].max()) < _BLACK_MAX:
        bottom -= 1
    left, right = 0, w - 1
    while left < w // 3 and int(gray[:, left].max()) < _BLACK_MAX:
        left += 1
    while right > w * 2 // 3 and int(gray[:, right].max()) < _BLACK_MAX:
        right -= 1
    if top == 0 and bottom == h - 1 and left == 0 and right == w - 1:
        return None  # 无黑边: 全帧即画面
    return (left, top, right + 1, bottom + 1)


def map_regions_to_frame(
    regions: dict[str, tuple[float, float, float, float]],
    area: tuple[int, int, int, int] | None,
    frame_w: int,
    frame_h: int,
) -> dict[str, tuple[float, float, float, float]]:
    """框选区域(相对画面归一化) → 当前帧像素坐标(含黑边偏移)。

    regions 保存时相对"框选帧的画面区域"; 运行时画面区域变化
    (黑边出现/窗口比例变)后按当前画面区域重映射 — 跟随界面大小。
    """
    if area is None:
        ax1, ay1, ax2, ay2 = 0, 0, frame_w, frame_h
    else:
        ax1, ay1, ax2, ay2 = area
    aw, ah = max(1, ax2 - ax1), max(1, ay2 - ay1)
    out: dict[str, tuple[float, float, float, float]] = {}
    for p, (rx1, ry1, rx2, ry2) in regions.items():
        out[p] = (ax1 + rx1 * aw, ay1 + ry1 * ah,
                  ax1 + rx2 * aw, ay1 + ry2 * ah)
    return out
