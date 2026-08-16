"""模型预测合并进数据集 label — 同一个位置的框以新预测为准。

对 data/screen_dataset 全部图片用当前 best.pt(imgsz=1280)重新预测:
- 与旧 label 同一位置(IoU ≥ 阈值)的框 → 用预测结果(位置+类别以新为准)
- 旧 label 中模型没检出的框 → 保留(人工标注兜底)
- 模型检出的新框 → 追加(补漏标)

用法:
    python scripts/remerge_labels.py                # 全量(train+val)
    python scripts/remerge_labels.py --split train  # 只处理 train
    python scripts/remerge_labels.py --conf 0.3     # 预测置信度下限
    python scripts/remerge_labels.py --dry-run      # 只统计不写回
"""

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mahjong_cv.screen_vision import ScreenVision  # noqa: E402

DATASET = Path('data/screen_dataset')
MODEL = 'data/models/screen/mahjong_screen_detector/weights/best.pt'
#: 预测置信度下限(标定开闸线: 牌河 precision 95%@0.30)
_CONF_MIN = 0.3
#: 同位置判定: 预测框与旧框 IoU ≥ 此值 → 视为同一张牌, 以新为准
_IOU_MATCH = 0.3


def read_txt(path: Path) -> list[list[float]]:
    """读取 YOLO label: 每行 [class, cx, cy, w, h](归一化)。"""
    boxes = []
    for line in path.read_text(encoding='utf-8').splitlines():
        parts = line.split()
        if len(parts) >= 5:
            boxes.append([float(parts[0]), *map(float, parts[1:5])])
    return boxes


def write_txt(path: Path, boxes: list[list[float]]) -> None:
    path.write_text('\n'.join(f'{int(b[0])} {b[1]:.6f} {b[2]:.6f} '
                              f'{b[3]:.6f} {b[4]:.6f}' for b in boxes)
                    + '\n', encoding='utf-8')


def iou(a: list[float], b: list[float]) -> float:
    """归一化坐标 [cx, cy, w, h] 的 IoU。"""
    ax1, ay1 = a[1] - a[3] / 2, a[2] - a[4] / 2
    ax2, ay2 = a[1] + a[3] / 2, a[2] + a[4] / 2
    bx1, by1 = b[1] - b[3] / 2, b[2] - b[4] / 2
    bx2, by2 = b[1] + b[3] / 2, b[2] + b[4] / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = a[3] * a[4]
    area_b = b[3] * b[4]
    return inter / (area_a + area_b - inter + 1e-9)


def merge(old: list[list[float]], pred: list[list[float]]
          ) -> tuple[list[list[float]], int, int]:
    """合并: 同位置以新预测为准, 旧框保留, 新框追加。

    返回 (合并结果, 覆盖数, 追加数)。
    """
    kept_old = [True] * len(old)
    merged: list[list[float]] = []
    added = 0
    for p in pred:
        matched = next((i for i, o in enumerate(old)
                        if kept_old[i] and iou(o, p) >= _IOU_MATCH), None)
        if matched is not None:
            kept_old[matched] = False  # 同位置: 以新预测为准
        else:
            added += 1
        merged.append(p)
    replaced = sum(1 for k in kept_old if not k)
    merged.extend(o for o, keep in zip(old, kept_old) if keep)
    return merged, replaced, added


def main() -> None:
    parser = argparse.ArgumentParser(description='模型预测合并进数据集 label')
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--split', nargs='+', default=['train', 'val'],
                        choices=['train', 'val'])
    parser.add_argument('--conf', type=float, default=_CONF_MIN)
    parser.add_argument('--dry-run', action='store_true', help='只统计不写回')
    args = parser.parse_args()

    vision = ScreenVision(args.model)
    total_img = total_replaced = total_added = total_kept = 0
    for split in args.split:
        img_dir = DATASET / 'images' / split
        label_dir = DATASET / 'labels' / split
        label_dir.mkdir(parents=True, exist_ok=True)
        images = sorted(list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')))
        for i, img in enumerate(images):
            frame = cv2.imread(str(img))
            if frame is None:
                continue
            dets = vision.process(frame, imgsz=1280)
            fh, fw = frame.shape[:2]
            pred = [[float(d.tile),
                     (d.x1 + d.x2) / 2 / fw, (d.y1 + d.y2) / 2 / fh,
                     (d.x2 - d.x1) / fw, (d.y2 - d.y1) / fh]
                    for d in dets if d.conf >= args.conf]
            label_path = label_dir / f'{img.stem}.txt'
            old = read_txt(label_path) if label_path.exists() else []
            merged, replaced, added = merge(old, pred)
            kept = len(merged) - replaced - added
            total_img += 1
            total_replaced += replaced
            total_added += added
            total_kept += kept
            if not args.dry_run:
                write_txt(label_path, merged)
            if (i + 1) % 50 == 0 or i == len(images) - 1:
                print(f'[{split}] {i + 1}/{len(images)} '
                      f'(覆盖{replaced} 追加{added} 保留{kept})')
    print(f'\n完成 {total_img} 张: 覆盖(同位置新为准) {total_replaced}, '
          f'追加(补漏标) {total_added}, 保留(人工兜底) {total_kept}')
    if args.dry_run:
        print('(--dry-run 未写回)')


if __name__ == '__main__':
    main()
