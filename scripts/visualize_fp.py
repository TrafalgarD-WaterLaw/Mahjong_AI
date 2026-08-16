"""FP 可视化 — 把误检框画到测试图上, 看误检的是什么(UI/牌/漏标)。

分类与 analyze_fp 同口径(复用其收集逻辑):
  纯误检(红框): 与任何 GT 都 IoU<0.5 — 背景/文字/牌间隙被当成牌
  类别错(黄框): 与 GT IoU≥0.5 但类别不同 — 检到了牌但认错
  组错(蓝框):   与 GT IoU≥0.5 类别同但聚类组不同 — 归属错
  正确匹配(绿框): TP

输出 data/fp_vis/<图名>.jpg(仅含 FP 的图), 终端打印每张图的 FP 摘要。

用法:
    python scripts/visualize_fp.py [--model <best.pt>] [--test data/test_real]
                                   [--out data/fp_vis]
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
from analyze_fp import classify_fp  # noqa: E402
from src.mahjong_ui.pil_text import render_text_overlay  # noqa: E402
from src.mahjong_cv.detections import TileDet  # noqa: E402
from src.mahjong_core.tile import tile_display  # noqa: E402

_COLORS = {'tp': (0, 200, 0), 'noise': (0, 0, 255),
           'class': (0, 255, 255), 'group': (255, 0, 0)}
_NAMES = {'noise': '纯误检', 'class': '类别错', 'group': '组错'}


def collect_vis(
    model: object, test_dir: Path,
) -> tuple[dict[str, Counter[str]], list[tuple[Path, np.ndarray]],
           list[tuple[np.ndarray, str, int, int]]]:
    """逐图预测 + 贪心匹配(同 analyze_threshold), 收集 FP 分类与可视化图。

    返回 (stats, images, noise_crops): images 每项 = (原图路径, 画好框的图);
    noise_crops 每项 = (纯误检裁剪图, 位置标签, 宽, 高) — 拼贴用。
    """
    stats: dict[str, Counter[str]] = {'hand': Counter(), 'river': Counter(),
                                      'other': Counter()}
    images: list[tuple[Path, np.ndarray]] = []
    noise_crops: list[tuple[np.ndarray, str, int, int]] = []

    for img_path in sorted((test_dir / 'images').glob('*.jpg')):
        label_path = test_dir / 'labels' / (img_path.stem + '.txt')
        if not label_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
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

        results = model(str(img_path), conf=0.01, verbose=False)
        pred_boxes: list[TileDet] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                pred_boxes.append(TileDet(int(box.cls[0].item()), x1, y1, x2, y2,
                                          float(box.conf[0].item())))
        pred_groups = _group_map(pred_boxes)
        preds: list[tuple[float, int, np.ndarray, str]] = []
        for b in pred_boxes:
            preds.append((b.conf, b.tile, np.array([b.x1, b.y1, b.x2, b.y2]),
                          pred_groups[id(b)]))

        used = [False] * len(gt)
        bad: list[tuple[str, int, np.ndarray]] = []  # (类, tile, box)
        n_tp = 0
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
                n_tp += 1
            else:
                fp_class, _gt_tile = classify_fp(box, tile, group, gt)
                stats[group][fp_class] += 1
                bad.append((fp_class, tile, box))
                if fp_class == 'noise':
                    x1, y1, x2, y2 = (int(v) for v in box)
                    crop = img[max(0, y1):y2, max(0, x1):x2]
                    if crop.size:
                        noise_crops.append(
                            (crop, f'{x1},{y1}', x2 - x1, y2 - y1))
        if bad:
            overlays: list[tuple[int, int, str, tuple[int, int, int],
                                 tuple[int, int, int]]] = []
            for fp_class, tile, box in bad:
                x1, y1, x2, y2 = (int(v) for v in box)
                cv2.rectangle(img, (x1, y1), (x2, y2), _COLORS[fp_class], 2)
                if fp_class == 'class':
                    text = f'{_NAMES[fp_class]} {tile_display(tile)}'
                else:
                    text = _NAMES[fp_class]
                overlays.append((x1, max(0, y1 - 22), text, (0, 0, 0),
                                 _COLORS[fp_class]))
            overlays.append((10, 10, f'{img_path.name}: 绿=TP 红=纯误检 '
                                      f'黄=类别错 蓝=组错', (255, 255, 255),
                             (25, 25, 25)))
            images.append((img_path, render_text_overlay(img, overlays)))
    return stats, images, noise_crops


def _collage(crops: list[tuple[np.ndarray, str, int, int]],
             cell: int = 96, cols: int = 12) -> np.ndarray:
    """纯误检裁剪图拼贴(保持纵横比, 底部标坐标与尺寸)。"""
    if not crops:
        return np.zeros((cell, cols * cell, 3), dtype=np.uint8)
    rows = (len(crops) + cols - 1) // cols
    canvas = np.full((rows * (cell + 14), cols * cell, 3), 35, dtype=np.uint8)
    for i, (crop, label, bw, bh) in enumerate(crops):
        r, c = divmod(i, cols)
        scale = min((cell - 6) / crop.shape[1], (cell - 6) / crop.shape[0])
        th = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)),
                               max(1, int(crop.shape[0] * scale))))
        ox = c * cell + (cell - th.shape[1]) // 2
        oy = r * (cell + 14) + 3
        canvas[oy:oy + th.shape[0], ox:ox + th.shape[1]] = th
        cv2.putText(canvas, f'{bw}x{bh}', (c * cell + 2, oy + cell - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description='FP 可视化(看误检内容)')
    parser.add_argument('--model',
                        default='data/models/screen/mahjong_screen_detector/weights/best.pt')
    parser.add_argument('--test', default='data/test_real')
    parser.add_argument('--out', default='data/fp_vis')
    args = parser.parse_args()

    from ultralytics import YOLO  # type: ignore[attr-defined]  # noqa: PLC0415

    model = YOLO(args.model)
    stats, images, noise_crops = collect_vis(model, Path(args.test))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for img_path, vis in images:
        cv2.imwrite(str(out / img_path.name), vis)
    print(f'FP 汇总:')
    for g in ('hand', 'river', 'other'):
        if stats[g]:
            print(f'  [{g}] ' + ', '.join(
                f'{_NAMES[k]} {n}' for k, n in stats[g].items()))
    print(f'可视化 {len(images)} 张 -> {out}/')

    if noise_crops:
        ws = [w for _, _, w, _ in noise_crops]
        hs = [h for _, _, _, h in noise_crops]
        n_slim = sum(1 for w, h in zip(ws, hs)
                     if w > 0 and h > 0 and w / h < 0.55)
        print(f'\n纯误检框尺寸: 平均 {sum(ws) / len(ws):.0f}x'
              f'{sum(hs) / len(hs):.0f}(宽x高)  细长条(w/h<0.55): {n_slim} '
              f'({n_slim / len(noise_crops):.0%})  '
              f'牌尺寸占比: {sum(1 for w, h in zip(ws, hs) if 0.55 <= w / h <= 1.2 and 20 <= w <= 60) / len(noise_crops):.0%}')
        collage = _collage(noise_crops)
        cv2.imwrite(str(out / 'noise_collage.jpg'), collage)
        print(f'纯误检拼贴 {len(noise_crops)} 个 -> {out}/noise_collage.jpg')


if __name__ == '__main__':
    main()
