"""补拍截图: best 模型预测 → 直接合并进 screen_dataset(train)。

用法:
    python scripts/ingest_new_shots.py data/pseudo_raw/images
    # 图片复制到 screen_dataset/images/train/, 预测 label 写 labels/train/
    # 默认 conf ≥ 0.3(标定开闸线); --dry-run 只统计; --min-boxes 过滤空帧
"""

import argparse
import shutil
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mahjong_cv.screen_vision import ScreenVision  # noqa: E402

DATASET = Path('data/screen_dataset')
MODEL = 'data/models/screen/mahjong_screen_detector/weights/best.pt'
#: 预测置信度下限(标定开闸线)
_CONF_MIN = 0.3
#: 整帧保留框数下限(全空/噪声帧丢弃)
_MIN_BOXES = 5


def main() -> None:
    parser = argparse.ArgumentParser(description='补拍截图合并进数据集')
    parser.add_argument('images', help='补拍截图目录')
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--conf', type=float, default=_CONF_MIN)
    parser.add_argument('--min-boxes', type=int, default=_MIN_BOXES)
    parser.add_argument('--dry-run', action='store_true', help='只统计不写入')
    args = parser.parse_args()

    src = Path(args.images)
    files = sorted(src.glob('*.jpg')) + sorted(src.glob('*.png'))
    if not files:
        print(f'没有找到图片: {src}')
        sys.exit(1)

    img_dir = DATASET / 'images' / 'train'
    label_dir = DATASET / 'labels' / 'train'
    img_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    existing = {p.stem for p in img_dir.glob('*.jpg')}

    vision = ScreenVision(args.model)
    saved = total_boxes = skipped_empty = skipped_dup = 0
    for f in files:
        frame = cv2.imread(str(f))
        if frame is None:
            continue
        dets = vision.process(frame, conf_threshold=args.conf, imgsz=1280)
        if len(dets) < args.min_boxes:
            skipped_empty += 1
            continue
        fh, fw = frame.shape[:2]
        stem = f.stem
        if stem in existing:  # 与训练集重名 → 加后缀
            stem = f'{stem}_ingest'
            n = 1
            while stem in existing:
                stem = f'{f.stem}_ingest_{n}'
                n += 1
        if not args.dry_run:
            shutil.copy2(str(f), str(img_dir / f'{stem}.jpg'))
            lines = [f'{d.tile} {(d.x1 + d.x2) / 2 / fw:.6f} '
                     f'{(d.y1 + d.y2) / 2 / fh:.6f} '
                     f'{(d.x2 - d.x1) / fw:.6f} {(d.y2 - d.y1) / fh:.6f}'
                     for d in dets]
            (label_dir / f'{stem}.txt').write_text(
                '\n'.join(lines) + '\n', encoding='utf-8')
        total_boxes += len(dets)
        saved += 1
        if saved % 20 == 0:
            print(f'[{saved}/{len(files)}] 已处理 {saved} 帧', flush=True)

    print(f'\n完成: 合并 {saved} 帧, 共 {total_boxes} 个框 '
          f'(空帧丢弃 {skipped_empty}, 重名跳过 {skipped_dup})')
    print(f'→ {img_dir} / {label_dir}')
    if args.dry_run:
        print('(--dry-run 未写入)')
    else:
        print('下一步: 服务器重训 train_screen_yolo.py(数据集已更新)')


if __name__ == '__main__':
    main()
