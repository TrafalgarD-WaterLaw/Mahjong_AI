"""在真实测试集上标定伪标注的置信度阈值。

思路: 模型在测试集(人工标签)上逐框预测, 与 GT 匹配(IoU≥0.5、同类且
同聚类组)区分 TP/FP, 统计不同 conf 阈值下的 precision/recall, 推荐
「precision 达标且 recall 尽量高」的操作点 — 半监督伪标注用它过滤可信标签。

分组(det_cluster.cluster_dets, 与 pseudo_label 同口径, 不依赖布局):
    GT 与预测各自空间聚类 → hand / river 簇 / other, 匹配要求同组。

用法:
    python scripts/analyze_threshold.py [--model <best.pt>] [--test data/test_real]
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mahjong_cv.det_cluster import cluster_dets
from src.mahjong_cv.detections import TileDet


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """a/b: (x1, y1, x2, y2) 像素坐标。"""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = ((a[2] - a[0]) * (a[3] - a[1])
             + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


def _group_map(boxes: list[TileDet]) -> dict[int, str]:
    """聚类分组: id(检测框) → 'hand' | 'river' | 'other'。

    副露(明牌堆)与牌河同类, 归 'river' 组。
    """
    hand, melds, rivers = cluster_dets(boxes, conf_min=0.0)
    groups: dict[int, str] = {id(d): 'hand' for d in hand}
    for c in rivers:
        for d in c:
            groups[id(d)] = 'river'
    for m in melds:
        for d in m:
            groups[id(d)] = 'river'
    return groups


def collect_predictions(
    model: Any, test_dir: Path,
) -> tuple[dict[str, list[float]], dict[str, list[float]], dict[str, int]]:
    """对测试集逐图预测并匹配 GT。

    每帧 GT 与预测各自聚类分组(hand/river/other), 匹配要求同组。
    返回 (TP confs, FP confs, GT 总数), 均按组 (hand/river/other) 分桶。
    """
    tp: dict[str, list[float]] = {'hand': [], 'river': [], 'other': []}
    fp: dict[str, list[float]] = {'hand': [], 'river': [], 'other': []}
    gt_total: dict[str, int] = {'hand': 0, 'river': 0, 'other': 0}

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
            gt_total[gt_groups[id(b)]] += 1

        frame = cv2.imread(str(img_path))
        # ScreenVision: 含 ROI 牌面分类精修(与实时推理同路径)
        pred_boxes = model.process(frame, conf_threshold=0.01)
        used = [False] * len(gt)
        pred_groups = _group_map(pred_boxes)
        preds: list[tuple[float, int, np.ndarray, str]] = []
        for b in pred_boxes:
            preds.append((b.conf, b.tile, np.array([b.x1, b.y1, b.x2, b.y2]),
                          pred_groups[id(b)]))

        # 贪心匹配: 置信度降序, IoU≥0.5、同类、同组
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
                tp[group].append(conf)
            else:
                fp[group].append(conf)
    return tp, fp, gt_total


def _img_size(img_path: Path) -> tuple[int, int]:
    import cv2

    img = cv2.imread(str(img_path))
    if img is None:
        return 1, 1
    h, w = img.shape[:2]
    return h, w


def main() -> None:
    parser = argparse.ArgumentParser(description='伪标注置信度阈值标定(分区域)')
    parser.add_argument('--model', default='data/models/screen/mahjong_screen_detector/weights/best.pt')
    parser.add_argument('--test', default='data/test_real_review')
    parser.add_argument('--min-precision', type=float, default=0.95,
                        help='期望的 precision 下限(手牌/其他区)')
    parser.add_argument('--min-precision-river', type=float, default=0.90,
                        help='牌河区 precision 下限(容忍噪声换取样本量)')
    args = parser.parse_args()

    from src.mahjong_cv.screen_vision import ScreenVision  # noqa: PLC0415

    model = ScreenVision(args.model)  # 检测 + ROI 分类精修
    tp, fp, gt_total = collect_predictions(model, Path(args.test))
    print(f'GT 分布: {gt_total}')

    for group, min_prec in (('hand', args.min_precision),
                            ('river', args.min_precision_river),
                            ('other', args.min_precision)):
        tp_c, fp_c = tp[group], fp[group]
        gt_n = gt_total[group]
        if not tp_c and not fp_c:
            print(f'\n[{group}] 无样本')
            continue
        best_t, best_recall = 0.0, -1.0
        print(f'\n[{group}] GT={gt_n}, TP={len(tp_c)}, FP={len(fp_c)}:')
        for t in np.arange(0.3, 0.96, 0.05):
            t = round(float(t), 2)
            n_tp = sum(1 for c in tp_c if c >= t)
            n_fp = sum(1 for c in fp_c if c >= t)
            precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) else 0.0
            recall = n_tp / gt_n if gt_n else 0.0
            flag = '  ← 推荐' if (precision >= min_prec and recall > best_recall) else ''
            if flag:
                best_t, best_recall = t, recall
            print(f'  conf≥{t:.2f}: precision={precision:.3f}  recall={recall:.3f}'
                  f'  样本={n_tp + n_fp}{flag}')
        if best_t > 0:
            print(f'  → 推荐阈值: {best_t:.2f} (precision≥{min_prec:.0%}, '
                  f'recall={best_recall:.3f})')
        else:
            print(f'  → 无阈值达到 precision≥{min_prec:.0%}')


if __name__ == '__main__':
    main()
