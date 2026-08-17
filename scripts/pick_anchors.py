"""在屏幕上点选四家牌河中心 — 鼠标点击生成归属锚点配置。

用法:
    python scripts/pick_anchors.py --image shot.png  # 在截图上选点
    python scripts/pick_anchors.py --live            # 截取当前游戏窗口选点

交互:
    左键 按顺序点选 4 个点: ①我的牌河 ②右家牌河 ③上家牌河 ④左家牌河
    — 锚点横平竖直 + 十字交点(上下连线与
    左右连线的交点)为圆心等距化。
    R    撤销上一个点
    S    保存到 config/river_anchors.json
    Esc  取消(不保存)

保存的坐标为归一化(相对选点帧尺寸), 运行时乘回当前帧尺寸 —
窗口缩放/分辨率变化自动适配。之后 run_assistant
加载该配置做最近归属。
"""

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mahjong_ai.state.snapshot import RIVER_NAMES  # noqa: E402

DEFAULT_OUT = Path('config/river_anchors.json')

#: 选点顺序与提示色(与客户端牌河区一致)
_ORDER = [
    ('my_river', '我的牌河', (0, 215, 255)),
    ('right_river', '右家牌河', (0, 0, 255)),
    ('top_river', '上家牌河', (255, 200, 0)),
    ('left_river', '左家牌河', (0, 165, 255)),
]


def _render_overlay(frame: cv2.Mat, picked: list[tuple[int, int]]) -> cv2.Mat:
    """画点 + 中文提示(cv2.putText 不支持中文, 用 PIL 渲染)。

    颜色: _ORDER 中是 BGR, PIL 需要 RGB — 翻转后传入。
    """
    from src.mahjong_ui.pil_text import render_text_overlay  # noqa: PLC0415

    vis = frame.copy()
    for i, ((_name, _label, color), (px, py)) in enumerate(zip(_ORDER, picked)):
        cv2.circle(vis, (px, py), 7, color, 2)
        cv2.putText(vis, str(i + 1), (px - 9, py - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    items: list = [
        (10, 8, '左键依次点选: ①我的牌河 → ②右家牌河 → ③上家牌河 → ④左家牌河',
         (255, 255, 255), (0, 0, 0)),
        (10, 40, f'R=撤销  S=保存  Esc=取消   当前: 第 {len(picked) + 1}/4 个',
         (255, 255, 255), (0, 0, 0)),
    ]
    for i, ((_name, label, color), (px, py)) in enumerate(zip(_ORDER, picked)):
        items.append((px + 12, py + 2, f'{i + 1}.{label}',
                      (color[2], color[1], color[0]), (0, 0, 0)))
    return render_text_overlay(vis, items)


def pick(frame: cv2.Mat) -> dict | None:
    """交互选点, 返回 {player: (x, y)} 像素坐标; 取消返回 None。"""
    window = 'pick_anchors'  # ASCII 标题(FindWindow 匹配/编码兼容)
    # 初始 = 图像原尺寸, 放大保持等比(WINDOW_KEEPRATIO 防拉伸变形)
    cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window, frame.shape[1], frame.shape[0])
    picked: list[tuple[int, int]] = []
    done = [False]

    def on_mouse(event: int, x: int, y: int, _flags: int, _param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN or done[0]:
            return
        if len(picked) >= 4:
            return
        picked.append((x, y))
        render()

    def render() -> None:
        cv2.imshow(window, _render_overlay(frame, picked))

    cv2.setMouseCallback(window, on_mouse)
    render()
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('r') and picked:
            picked.pop()
            render()
        elif key == ord('s') and len(picked) == 4:
            cv2.destroyWindow(window)
            return {name: (x, y) for (name, _, _), (x, y)
                    in zip(_ORDER, picked)}
        elif key == 27:  # Esc
            cv2.destroyWindow(window)
            return None


def main() -> None:
    parser = argparse.ArgumentParser(description='点选四家牌河中心')
    parser.add_argument('--image', default=None, help='截图路径')
    parser.add_argument('--live', action='store_true',
                        help='截取当前游戏窗口(需欢乐麻将已开)')
    parser.add_argument('--title', default='欢乐麻将',
                        help='游戏窗口标题(子串匹配)')
    parser.add_argument('--out', default=str(DEFAULT_OUT))
    args = parser.parse_args()

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f'无法读取: {args.image}')
            sys.exit(1)
    elif args.live:
        from src.mahjong_cv.capture.win32 import (  # noqa: PLC0415
            DEFAULT_TITLE_CANDIDATES,
            Win32Capture,
            list_window_titles,
        )

        candidates = DEFAULT_TITLE_CANDIDATES \
            if args.title == '欢乐麻将' else (args.title,)
        cap = Win32Capture(args.title, candidates)
        if cap.client_rect() is None:
            print('[窗口] 未找到游戏窗口, 当前可见窗口:')
            for t in list_window_titles():
                print(f'   - {t}')
            print('请先启动欢乐麻将, 或使用 --title 指定标题')
            sys.exit(1)
        frame = cap.capture()
    else:
        parser.error('需要 --image 或 --live')

    pixels = pick(frame)
    if pixels is None:
        print('已取消, 未保存')
        return

    fh, fw = frame.shape[:2]
    anchors = {name: (x / fw, y / fh) for name, (x, y) in pixels.items()}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {'frame_size': [fw, fh], 'anchors': anchors}, indent=2),
        encoding='utf-8')
    print(f'已保存 {out}:')
    for name, (x, y) in anchors.items():
        px, py = pixels[name]
        print(f'  {name}: 像素({px}, {py}) → 归一化({x:.3f}, {y:.3f})')
    print('提示: 锚点可手动微调至横平竖直')


if __name__ == '__main__':
    main()
