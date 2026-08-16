"""聚类归属可视化 — 一帧检测框按 手牌/副露/四家牌河 着色。

颜色: 手牌=绿  副露=蓝  我的牌河=黄  右家=红  上家=青  左家=橙  零散=灰
每簇标注归属名; 输出统计到终端。

用法:
    python scripts/visualize_cluster.py <截图> [--out data/cluster_viz/<name>.jpg]
    python scripts/visualize_cluster.py --auto   # 自动选 test_real 中牌河框最多的一张
"""

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mahjong_ai.state.snapshot import (  # noqa: E402
    RIVER_NAMES,
    GameStateTracker,
)
from src.mahjong_cv.det_cluster import cluster_dets  # noqa: E402
from src.mahjong_cv.screen_vision import ScreenVision  # noqa: E402

_COLORS = {
    'hand': (0, 200, 0),          # 绿
    'meld': (255, 0, 0),          # 蓝
    'my_river': (0, 215, 255),    # 黄
    'right_river': (0, 0, 255),   # 红
    'top_river': (255, 200, 0),   # 青
    'left_river': (0, 165, 255),  # 橙
    'other': (128, 128, 128),     # 灰
}
_NAMES = {'my_river': '我', 'right_river': '右', 'top_river': '上',
          'left_river': '左'}


def visualize(image: str, model: str, out: str) -> None:
    vision = ScreenVision(model)
    frame = cv2.imread(image)
    if frame is None:
        print(f'无法读取: {image}')
        sys.exit(1)
    dets = vision.process(frame)
    fh, fw = frame.shape[:2]

    hand, melds, rivers = cluster_dets(dets)

    # 锚点 = 用户点选配置(pick_anchors.py 生成), 必须存在
    import json  # noqa: PLC0415
    from pathlib import Path as _Path  # noqa: PLC0415

    cfg = _Path('config/river_anchors.json')
    if not cfg.exists():
        print('[锚点] 未找到 config/river_anchors.json — 请先运行 '
              'scripts/pick_anchors.py 点选四家牌河位置')
        sys.exit(1)
    data = json.loads(cfg.read_text(encoding='utf-8'))
    anchors = {name: tuple(v) for name, v in data['anchors'].items()}
    tracker = GameStateTracker(anchors=anchors)
    assigned = tracker._assign_river_clusters(  # noqa: SLF001
        rivers, fw, fh)

    def draw(d, color: tuple[int, int, int]) -> None:
        cv2.rectangle(frame, (int(d.x1), int(d.y1)),
                      (int(d.x2), int(d.y2)), color, 2)

    for d in hand:
        draw(d, _COLORS['hand'])
    for m in melds:
        for d in m:
            draw(d, _COLORS['meld'])
    for p in RIVER_NAMES:
        for d in assigned.get(p, []):
            draw(d, _COLORS[p])

    # 归属中心点(诊断: 点选锚点)
    centers = {p: (x * fw, y * fh) for p, (x, y) in anchors.items()}
    for p in RIVER_NAMES:
        px, py = int(centers[p][0]), int(centers[p][1])
        cv2.circle(frame, (px, py), 7, _COLORS[p], 2)
        cv2.putText(frame, _NAMES[p], (px + 10, py + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, _COLORS[p], 2)

    # 图例 + 每簇归属标注
    cv2.putText(frame, f'手牌{len(hand)} 副露{len(melds)}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, _COLORS['hand'], 2)
    for i, (p, color) in enumerate((('我', _COLORS['my_river']),
                                    ('右', _COLORS['right_river']),
                                    ('上', _COLORS['top_river']),
                                    ('左', _COLORS['left_river']))):
        cv2.putText(frame, p, (10, 60 + i * 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2)
    # 簇中心标注归属
    for p in RIVER_NAMES:
        cluster = assigned.get(p, [])
        if not cluster:
            continue
        cx = sum((d.x1 + d.x2) / 2 for d in cluster) / len(cluster)
        cy = sum((d.y1 + d.y2) / 2 for d in cluster) / len(cluster)
        cv2.putText(frame, _NAMES[p], (int(cx) - 10, int(cy) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, _COLORS[p], 2)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), frame)
    print(f'检测 {len(dets)} 框 → 手牌 {len(hand)} 副露 {len(melds)} 组')
    for p in RIVER_NAMES:
        print(f'  {_NAMES[p]}家牌河: {len(assigned.get(p, []))} 张')
    print(f'→ 保存 {out_path}')


def main() -> None:
    parser = argparse.ArgumentParser(description='聚类归属可视化')
    parser.add_argument('image', nargs='?', default=None,
                        help='截图路径; --auto 时自动选')
    parser.add_argument('--model',
                        default='data/models/screen/mahjong_screen_detector/weights/best.pt')
    parser.add_argument('--out', default=None, help='输出路径')
    parser.add_argument('--auto', action='store_true',
                        help='自动选 test_real 中牌河框最多的一张')
    args = parser.parse_args()

    if args.auto:
        from src.mahjong_cv.det_cluster import cluster_dets  # noqa: PLC0415

        vision = ScreenVision(args.model)
        best_img, best_n = None, -1
        for p in sorted((Path('data/test_real/images')).glob('*.jpg')):
            frame = cv2.imread(str(p))
            if frame is None:
                continue
            dets = vision.process(frame)
            _h, _m, rivers = cluster_dets(dets)
            n = sum(len(c) for c in rivers)
            if n > best_n:
                best_n, best_img = n, p
        if best_img is None:
            print('test_real 无图')
            sys.exit(1)
        image = str(best_img)
        print(f'自动选择: {Path(image).name} (牌河 {best_n} 张)')
        out = args.out or f"data/cluster_viz/{Path(image).stem}.jpg"
    else:
        if not args.image:
            parser.error('需要 image 或 --auto')
        image = args.image
        out = args.out or f"data/cluster_viz/{Path(image).stem}.jpg"
    visualize(image, args.model, out)


if __name__ == '__main__':
    main()
