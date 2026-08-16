"""PIL 中文文本叠加 — cv2.putText 不支持中文, 用 PIL 渲染后合回。

训练链工具(label_review/pick_anchors/pick_regions/visualize_fp)的
提示文字渲染共用此模块。
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

#: 中文字体(Windows 微软雅黑; 训练链工具仅在本机运行)
_FONT_PATH = 'msyh.ttc'
_FONT_SIZE = 18


def render_text_overlay(
    frame: cv2.Mat,
    items: list[tuple[int, int, str, tuple[int, int, int],
                      tuple[int, int, int]]],
) -> cv2.Mat:
    """在 cv2 图像(BGR)上叠加中文文本。

    items: [(x, y, 文本, 字色 RGB, 底色 RGB)] — 底色画实心矩形,
    文本居中于矩形; 空列表原样返回。
    """
    if not items:
        return frame
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(_FONT_PATH, _FONT_SIZE)
    for x, y, text, color, bg in items:
        box = draw.textbbox((0, 0), text, font=font)
        w, h = box[2] - box[0], box[3] - box[1]
        draw.rectangle((x - 2, y - 2, x + w + 4, y + h + 4), fill=bg)
        draw.text((x, y - box[1]), text, font=font, fill=color)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
