"""在屏幕上框选四家牌河区域 + 四家副露区域 — 每家 4 次点击。

牌河/副露都是固定 UI 区域(欢乐麻将 PC 布局不变): 框住后
落点在区域内 → 直接归该家(权威); 区域外(空隙/交错) → 时间兜底。
比点锚点更准: 区域天然覆盖牌河扩张/换行与副露亮牌位置。

用法:
    python scripts/pick_regions.py --image shot.png  # 在截图上框选
    python scripts/pick_regions.py --live            # 截取当前游戏窗口框选

交互:
    左键 依次框选 8 个区域(每家 2 个: 先牌河后副露):
      ①我的牌河 ②我的副露 ③右家牌河 ④右家副露
      ⑤上家牌河 ⑥上家副露 ⑦左家牌河 ⑧左家副露
    每个区域点 2 次(左上角 → 右下角), 画矩形预览
    R    撤销上一个点
    S    保存到 config/river_regions.json
    Esc  取消(不保存)

保存为相对画面区域的归一化坐标(自动去黑边), 运行时按当前
画面区域重映射 — 窗口大小/比例随意变化都跟随。
"""

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_OUT = Path('config/river_regions.json')

#: 框选顺序与颜色(与客户端牌河区一致; 副露用洋红)
_ORDER = [
    ('my_river', '我的牌河', (0, 215, 255)),
    ('my_meld', '我的副露', (255, 0, 255)),
    ('right_river', '右家牌河', (0, 0, 255)),
    ('right_meld', '右家副露', (255, 0, 255)),
    ('top_river', '上家牌河', (255, 200, 0)),
    ('top_meld', '上家副露', (255, 0, 255)),
    ('left_river', '左家牌河', (0, 165, 255)),
    ('left_meld', '左家副露', (255, 0, 255)),
]


def _render(frame: cv2.Mat, picked: list[tuple[int, int]]) -> cv2.Mat:
    """画矩形(每 2 点一个) + 中文提示(PIL 渲染)。"""
    from src.mahjong_ui.pil_text import render_text_overlay  # noqa: PLC0415

    vis = frame.copy()
    for i in range(0, len(picked), 2):
        name, label, color = _ORDER[i // 2]
        if i + 1 < len(picked):
            p1, p2 = picked[i], picked[i + 1]
            cv2.rectangle(vis, p1, p2, color, 2)
            cx, cy = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
            cv2.circle(vis, (cx, cy), 4, color, -1)
        else:
            p1 = picked[i]
            cv2.circle(vis, p1, 4, color, 2)
    # 未完成的当前区域画十字线
    if len(picked) % 2 == 1:
        p = picked[-1]
        cv2.line(vis, (0, p[1]), (vis.shape[1], p[1]), (180, 180, 180), 1)
        cv2.line(vis, (p[0], 0), (p[0], vis.shape[0]), (180, 180, 180), 1)
    items = [
        (10, 8, '框选 8 个区域(每家 2 个: 牌河 → 副露): '
                '①我牌河 ②我副露 ③右牌河 ④右副露 ⑤上牌河 ⑥上副露 '
                '⑦左牌河 ⑧左副露',
         (255, 255, 255), (0, 0, 0)),
        (10, 40, f'R=撤销  S=保存  Esc=取消   当前: 第 {len(picked) // 2 + 1}/8 区'
                 f'{" 第 1 点" if len(picked) % 2 == 0 else " 第 2 点"}',
         (255, 255, 255), (0, 0, 0)),
    ]
    for i in range(len(picked) // 2):
        name, label, color = _ORDER[i]
        p1, p2 = picked[i * 2], picked[i * 2 + 1]
        items.append(((p1[0] + p2[0]) // 2, min(p1[1], p2[1]) - 18,
                      f'{i + 1}.{label}', (color[2], color[1], color[0]),
                      (0, 0, 0)))
    return render_text_overlay(vis, items)


def pick(frame: cv2.Mat) -> dict | None:
    """交互框选, 返回 {player: (x1, y1, x2, y2)} 像素矩形。"""
    window = 'pick_regions'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window, frame.shape[1], frame.shape[0])
    picked: list[tuple[int, int]] = []

    def on_mouse(event: int, x: int, y: int, _flags: int, _param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if len(picked) >= 16:
            return
        picked.append((x, y))
        cv2.imshow(window, _render(frame, picked))

    cv2.setMouseCallback(window, on_mouse)
    cv2.imshow(window, _render(frame, picked))
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('r') and picked:
            picked.pop()
            cv2.imshow(window, _render(frame, picked))
        elif key == ord('s') and len(picked) == 16:
            cv2.destroyWindow(window)
            return {name: (min(picked[i][0], picked[i + 1][0]),
                           min(picked[i][1], picked[i + 1][1]),
                           max(picked[i][0], picked[i + 1][0]),
                           max(picked[i][1], picked[i + 1][1]))
                    for i, (name, _l, _c) in zip(range(0, 16, 2), _ORDER)}
        elif key == 27:  # Esc
            cv2.destroyWindow(window)
            return None


def main() -> None:
    parser = argparse.ArgumentParser(description='框选四家牌河区域')
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

    regions = pick(frame)
    if regions is None:
        print('已取消, 未保存')
        return

    fh, fw = frame.shape[:2]
    # 区域相对"画面区域"(自动去黑边)保存 — 窗口大小/比例变化后,
    # 运行时按当前画面区域重映射, 区域跟随界面自动调整
    from src.mahjong_cv.game_area import detect_game_area  # noqa: PLC0415

    area = detect_game_area(frame)
    if area is None:
        ax1, ay1, aw, ah = 0, 0, fw, fh
    else:
        ax1, ay1, ax2, ay2 = area
        aw, ah = ax2 - ax1, ay2 - ay1
    norm = {name: [(r[0] - ax1) / aw, (r[1] - ay1) / ah,
                   (r[2] - ax1) / aw, (r[3] - ay1) / ah]
            for name, r in regions.items()}
    river_norm = {k: v for k, v in norm.items() if k.endswith('_river')}
    meld_norm = {k: v for k, v in norm.items() if k.endswith('_meld')}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {'frame_size': [fw, fh], 'regions': river_norm,
         'meld_regions': meld_norm}, indent=2),
        encoding='utf-8')
    print(f'已保存 {out}(区域相对画面区域):')
    for name, (x1, y1, x2, y2) in norm.items():
        print(f'  {name}: ({x1:.3f},{y1:.3f}) → ({x2:.3f},{y2:.3f})')


if __name__ == '__main__':
    main()
