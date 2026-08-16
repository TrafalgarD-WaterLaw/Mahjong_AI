"""FP 构成分析 — 把误检拆成四类, 决定下一步(去重后处理 / 补类别数据 / 采负样本)。

复用 scripts/analyze_threshold 的预测收集与贪心匹配逻辑(同一份匹配、
同一份聚类分组), 但对每个匹配失败的预测框(FP)归因:

  重复框(dup):  存在 IoU≥0.5 且类别同、组同的 GT — 该 GT 已被更高 conf
                的预测占用(同一张牌被检两遍, 多出的框无 GT 可配)
  类别错(class):存在 IoU≥0.5 但类别不同的 GT(检到了牌但认错)
  组错(group):  只存在 IoU≥0.5 且类别同、组不同的 GT(归属错, hand↔meld 重叠区)
  纯误检(noise):与任何 GT 都 IoU<0.5(背景/文字/牌间隙被当成牌)

行动指引:
  - 重复框多 → 后处理按簇内重叠去重即可, 不用动模型
  - 类别错多 → 看混淆对表(预测→GT), 补对应类别样本
  - 纯误检多 → 看位置分布(组 × 3x3 网格), 采集该区域负样本

另附: 纯误检与 GT 的最大 IoU 分布(区分「完全无关」与「框偏了差点匹配」,
后者往往是 GT 漏标或框漂移)。

用法:
    python scripts/analyze_fp.py [--model <best.pt>] [--test data/test_real]
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))          # scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # 项目根

from analyze_threshold import _group_map, _img_size, iou  # noqa: E402
from src.mahjong_cv.detections import TileDet  # noqa: E402
from src.mahjong_core.tile import tile_display  # noqa: E402

FP_CLASSES = ('dup', 'class', 'group', 'noise')
_FP_NAMES = {'dup': '重复框', 'class': '类别错', 'group': '组错', 'noise': '纯误检'}
GROUPS = ('hand', 'river', 'other')


def classify_fp(
    box: np.ndarray, tile: int, group: str,
    gt: list[tuple[int, np.ndarray, str]],
) -> tuple[str, int | None]:
    """分类一个 FP 框。返回 (类别, 关联 GT 类别) — 类别错时带预测→GT 对。

    与 GT 的 IoU≥0.5 才算「有关」; 匹配条件(同类同组)之外的 GT:
      类别不同 → class(哪怕该 GT 已被占用, 本质是认错)
      类别同、组不同 → group
      类别同、组同 → 必然已被更高 conf 的预测占用 → dup
      无任何 IoU≥0.5 的 GT → noise
    """
    any_gt = False
    for gt_tile, gt_box, gt_group in gt:
        if iou(box, gt_box) < 0.5:
            continue
        any_gt = True
        if gt_tile != tile:
            return 'class', gt_tile
        if gt_group != group:
            return 'group', gt_tile
    return ('dup', tile) if any_gt else ('noise', None)


def collect_fp(
    model: object, test_dir: Path,
) -> tuple[dict[str, Counter[str]], dict[str, list[tuple[float, float]]],
           list[float], Counter[tuple[str, str]], dict[str, int]]:
    """逐图预测 + 贪心匹配(同 analyze_threshold), FP 分类统计。

    返回 (stats, noise_pos, noise_iou, conf_pairs, tp_count):
      stats[组][FP类]     每聚类组的四类 FP 计数
      noise_pos[组]       纯误检框中心(归一化) — 位置分布用
      noise_iou           纯误检与任一 GT 的最大 IoU
      conf_pairs          类别错的 (预测名, GT名) 计数
      tp_count[组]        正确匹配数(对照用)
    """
    stats: dict[str, Counter[str]] = {g: Counter() for g in GROUPS}
    noise_pos: dict[str, list[tuple[float, float]]] = {g: [] for g in GROUPS}
    noise_iou: list[float] = []
    conf_pairs: Counter[tuple[str, str]] = Counter()
    tp_count: dict[str, int] = {g: 0 for g in GROUPS}

    for img_path in sorted((test_dir / 'images').glob('*.jpg')):
        label_path = test_dir / 'labels' / (img_path.stem + '.txt')
        if not label_path.exists():
            continue
        h, w = _img_size(img_path)
        gt_boxes: list[TileDet] = []
        for line in label_path.read_text(encoding='utf-8').splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            tile = int(parts[0])
            cx, cy, bw, bh = (float(v) for v in parts[1:])
            gt_boxes.append(TileDet(tile,
                                    (cx - bw / 2) * w, (cy - bh / 2) * h,
                                    (cx + bw / 2) * w, (cy + bh / 2) * h,
                                    1.0))
        gt_groups = _group_map(gt_boxes)
        gt: list[tuple[int, np.ndarray, str]] = []
        for b in gt_boxes:
            gt.append((b.tile, np.array([b.x1, b.y1, b.x2, b.y2]),
                       gt_groups[id(b)]))

        frame = cv2.imread(str(img_path))
        # ScreenVision: 含 ROI 牌面分类精修(与实时推理/标定同路径)
        pred_boxes = model.process(frame, conf_threshold=0.01)
        pred_groups = _group_map(pred_boxes)
        preds: list[tuple[float, int, np.ndarray, str]] = []
        for b in pred_boxes:
            preds.append((b.conf, b.tile, np.array([b.x1, b.y1, b.x2, b.y2]),
                          pred_groups[id(b)]))

        # 贪心匹配: 置信度降序, IoU≥0.5、同类、同组(与 analyze_threshold 一致)
        used = [False] * len(gt)
        for conf, tile, box, group in sorted(preds, key=lambda p: -p[0]):
            best_i, best_score = -1, 0.5
            for i, (gt_tile, gt_box, gt_group) in enumerate(gt):
                if used[i] or gt_tile != tile or gt_group != group:
                    continue
                s = iou(box, gt_box)
                if s > best_score:
                    best_i, best_score = i, s
            if best_i >= 0:
                used[best_i] = True
                tp_count[group] += 1
                continue
            # FP → 归因
            fp_class, gt_tile = classify_fp(box, tile, group, gt)
            stats[group][fp_class] += 1
            if fp_class == 'noise':
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                noise_pos[group].append((cx / w, cy / h))
                noise_iou.append(max(iou(box, gb) for _, gb, _ in gt))
            elif fp_class == 'class' and gt_tile is not None:
                conf_pairs[(tile_display(tile), tile_display(gt_tile))] += 1
    return stats, noise_pos, noise_iou, conf_pairs, tp_count


def _grid(pos: list[tuple[float, float]]) -> list[list[int]]:
    """归一化中心 → 3x3 网格(行=上中下, 列=左中右)。"""
    g = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for nx, ny in pos:
        g[min(2, int(ny * 3))][min(2, int(nx * 3))] += 1
    return g


def main() -> None:
    parser = argparse.ArgumentParser(description='FP 构成分析(重复/类别/组/噪声)')
    parser.add_argument('--model',
                        default='data/models/screen/mahjong_screen_detector/weights/best.pt')
    parser.add_argument('--test', default='data/test_real_review')
    args = parser.parse_args()

    from src.mahjong_cv.screen_vision import ScreenVision  # noqa: PLC0415

    model = ScreenVision(args.model)  # 检测 + ROI 分类精修
    stats, noise_pos, noise_iou, conf_pairs, tp_count = collect_fp(
        model, Path(args.test))

    print(f'FP 构成(按聚类组, TP 对照):')
    for g in GROUPS:
        c = stats[g]
        total = sum(c.values())
        if not total and not tp_count[g]:
            print(f'\n[{g}] 无样本')
            continue
        print(f'\n[{g}] TP={tp_count[g]}, FP={total}:')
        for k in FP_CLASSES:
            n = c[k]
            if n:
                print(f'  {_FP_NAMES[k]}: {n} ({n / total:.1%})')

    if conf_pairs:
        print(f'\n类别错混淆对 top 10(预测 → GT):')
        for (pred, g), n in conf_pairs.most_common(10):
            print(f'  {pred} → {g}: {n}')

    n_lo = sum(1 for v in noise_iou if v < 0.3)
    n_mid = sum(1 for v in noise_iou if 0.3 <= v < 0.5)
    print(f'\n纯误检与 GT 最大 IoU: <0.3(完全无关): {n_lo}, '
          f'0.3-0.5(框偏/漏标嫌疑): {n_mid}')
    for g in GROUPS:
        if noise_pos[g]:
            print(f'  纯误检位置 [{g}] 3x3 网格(上中下×左中右):')
            for row in _grid(noise_pos[g]):
                print('    ' + '  '.join(f'{v:>3}' for v in row))

    # 行动建议
    print()
    for g in GROUPS:
        c = stats[g]
        total = sum(c.values())
        if not total:
            continue
        hints = []
        if c['dup'] / total > 0.5:
            hints.append('重复框过半 → 先做簇内重叠去重, 不动模型')
        if c['class'] / total > 0.3:
            hints.append('类别错占比高 → 按混淆对补数据')
        if c['noise'] / total > 0.4:
            hints.append('纯误检占比高 → 看位置网格, 采该区域负样本')
        if hints:
            print(f'[{g}] {"; ".join(hints)}')


if __name__ == '__main__':
    main()
