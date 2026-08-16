"""对 config/river_anchors.json 做横平竖直 + 十字交点等距化。

处理(归一化坐标):
1. 横平竖直: 上家与我的牌河 x 对齐(竖直同线), 左家与右家牌河 y 对齐
   (水平同线) — 取两点平均, 各自位移最小。
2. 中间点 = 上下连线与左右连线的交点(十字交叉处)。
3. 等距化: 四点各自保持原方向(中间点 → 该点), 沿径向调整到统一
   半径 R = 四点平均距离 — "四个点到中间点的距离相同"。
   微调: 远的往中间拉, 近的往外放。

用法:
    python scripts/normalize_anchors.py            # 处理并写回
    python scripts/normalize_anchors.py --dry-run  # 只打印不写
"""

import argparse
import json
import sys
from pathlib import Path

DEFAULT_CFG = Path('config/river_anchors.json')


def normalize(anchors: dict) -> dict:
    """横平竖直 → 交点 → 等距化, 返回 (新锚点, 中间点)。"""
    out = {k: list(v) for k, v in anchors.items()}

    # 1. 横平竖直: 上下 x 对齐, 左右 y 对齐(各取平均, 位移最小化)
    mx = (out['top_river'][0] + out['my_river'][0]) / 2
    out['top_river'][0] = mx
    out['my_river'][0] = mx
    my_row = (out['left_river'][1] + out['right_river'][1]) / 2
    out['left_river'][1] = my_row
    out['right_river'][1] = my_row

    # 2. 中间点 = 上下连线与左右连线的交点
    center = (mx, my_row)

    # 3. 等距化: 保持方向, 半径 = 四点平均距离
    dists = {}
    for name in anchors:
        dx, dy = out[name][0] - center[0], out[name][1] - center[1]
        dists[name] = (dx * dx + dy * dy) ** 0.5
    r = sum(dists.values()) / len(dists)
    for name in anchors:
        d = dists[name]
        if d < 1e-6:
            continue  # 点与圆心重合: 不动
        out[name][0] = center[0] + (out[name][0] - center[0]) * r / d
        out[name][1] = center[1] + (out[name][1] - center[1]) * r / d
    return {k: tuple(v) for k, v in out.items()}, center


def main() -> None:
    parser = argparse.ArgumentParser(description='锚点横平竖直 + 距离均衡')
    parser.add_argument('--cfg', default=str(DEFAULT_CFG))
    parser.add_argument('--dry-run', action='store_true', help='只打印不写回')
    args = parser.parse_args()

    cfg = Path(args.cfg)
    if not cfg.exists():
        print(f'配置不存在: {cfg}')
        sys.exit(1)
    data = json.loads(cfg.read_text(encoding='utf-8'))
    anchors = {k: tuple(v) for k, v in data['anchors'].items()}
    new, center = normalize(anchors)

    cx, cy = center
    r = sum(((nx - cx) ** 2 + (ny - cy) ** 2) ** 0.5
            for nx, ny in new.values()) / len(new)
    print(f'中间点(上下连线×左右连线交点) ({cx:.3f},{cy:.3f})  半径 {r:.3f}')
    print(f'{"锚点":<12} {"处理前":<22} {"处理后":<22} 移动量')
    for name in ('my_river', 'right_river', 'top_river', 'left_river'):
        ox, oy = anchors[name]
        nx, ny = new[name]
        mv = ((nx - ox) ** 2 + (ny - oy) ** 2) ** 0.5
        print(f'{name:<12} ({ox:.3f},{oy:.3f})      '
              f'({nx:.3f},{ny:.3f})      {mv:.3f}')

    if not args.dry_run:
        data['anchors'] = {k: list(v) for k, v in new.items()}
        cfg.write_text(json.dumps(data, indent=2), encoding='utf-8')
        print(f'已写回 {cfg}')


if __name__ == '__main__':
    main()
