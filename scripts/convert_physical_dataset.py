"""实物麻将数据集转换: YOLO 类序 → 牌编码类序(0-41)。

源数据集(Mahjong.v83i.yolov8)的标签类序为 YOLO 顺序
(1B,1C,1D,1F,1S...), 与 best.pt 一致; 转换后类序 = 牌编码
(0-41: 万饼条风箭花), 与屏幕数据集/合成数据/训练管线完全统一。

用法:
    python scripts/convert_physical_dataset.py [--verify]      # 先校验类序
    python scripts/convert_physical_dataset.py                  # 转换(默认)
    # --src  源数据集目录(默认 F:\\mahjong_data\\Mahjong.v83i.yolov8)
    # --out  输出目录(默认 F:\\mahjong_data\\Mahjong.v83i.tile_encoding)
"""

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mahjong_core.tile import TOTAL_TILES, tile_display
from src.mahjong_cv.yolo_mapping import yolo_to_tile

DEFAULT_SRC = r'F:\mahjong_data\Mahjong.v83i.yolov8'
DEFAULT_OUT = r'F:\mahjong_data\Mahjong.v83i.tile_encoding'


def remap_label_line(line: str) -> str | None:
    """把 YOLO 类序的一行标签转成牌编码类序; 非法返回 None。"""
    parts = line.split()
    if len(parts) != 5:
        return None
    try:
        cls_id = int(parts[0])
        tile = yolo_to_tile(cls_id)
    except (KeyError, ValueError):
        return None
    return f'{tile} {parts[1]} {parts[2]} {parts[3]} {parts[4]}'


def verify_order(src: Path, model_path: str, n: int = 6) -> None:
    """用 best.pt 在验证集上预测, 与原标签比对, 校验类序一致性。"""
    from ultralytics import YOLO  # noqa: PLC0415

    model = YOLO(model_path)
    valid_imgs = sorted((src / 'valid' / 'images').glob('*.jpg'))
    if not valid_imgs:
        valid_imgs = sorted((src / 'valid' / 'images').glob('*.png'))
    total = 0
    match = 0
    for img_path in valid_imgs[:n]:
        label_path = src / 'valid' / 'labels' / (img_path.stem + '.txt')
        if not label_path.exists():
            continue
        gt = [int(line.split()[0]) for line in
              label_path.read_text(encoding='utf-8').splitlines() if line.strip()]
        results = model(str(img_path), conf=0.25, verbose=False)
        preds = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                preds.append(int(box.cls[0].item()))
        # 标签与预测都按 yolo_mapping 转成牌编码后逐框比对(按数量对齐取前 N)
        gt_tiles = sorted(yolo_to_tile(c) for c in gt)
        pred_tiles = sorted(yolo_to_tile(c) for c in preds)
        k = min(len(gt_tiles), len(pred_tiles))
        total += k
        match += sum(1 for a, b in zip(gt_tiles[:k], pred_tiles[:k], strict=False)
                     if a == b)
        print(f'  {img_path.name}: GT {len(gt_tiles)} 个, 预测 {len(pred_tiles)} 个, '
              f'前 {k} 个匹配 {sum(1 for a, b in zip(gt_tiles[:k], pred_tiles[:k], strict=False) if a == b)}/{k}')
    if total == 0:
        print('验证集为空, 跳过校验')
        return
    rate = match / total
    print(f'类序校验: {match}/{total} 匹配 ({rate:.0%})')
    if rate < 0.6:
        print('⚠️ 匹配率低: 当前标签顺序与 best.pt 不一致, 转换前需人工确认')
        sys.exit(1)
    print('✅ 类序一致(当前标签 = YOLO 顺序), 可以安全转换')


def convert(src: Path, out: Path) -> None:
    """按 yolo_mapping 转换全部标签并复制图片, 写出统一 data.yaml。"""
    dist: Counter[int] = Counter()
    skipped = 0
    t0 = time.time()
    for split in ('train', 'valid', 'test'):
        img_dir = src / split / 'images'
        lbl_dir = src / split / 'labels'
        out_img = out / split / 'images'
        out_lbl = out / split / 'labels'
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)

        images = sorted(img_dir.glob('*.jpg')) + sorted(img_dir.glob('*.png'))
        for i, img_path in enumerate(images, start=1):
            label_path = lbl_dir / (img_path.stem + '.txt')
            if not label_path.exists():
                skipped += 1
                continue
            lines_out: list[str] = []
            for line in label_path.read_text(encoding='utf-8').splitlines():
                if not line.strip():
                    continue
                new = remap_label_line(line)
                if new is None:
                    skipped += 1
                    continue
                lines_out.append(new)
                dist[int(new.split()[0])] += 1
            if not lines_out:
                skipped += 1
                continue
            (out_lbl / (img_path.stem + '.txt')).write_text(
                '\n'.join(lines_out) + '\n', encoding='utf-8')
            # 图片复制(jpg/png 原样)
            with open(img_path, 'rb') as fsrc, open(out_img / img_path.name, 'wb') as fdst:
                import shutil

                shutil.copyfileobj(fsrc, fdst, length=1 << 20)
            if i % 1000 == 0:
                print(f'  {split}: {i}/{len(images)} ({time.time()-t0:.0f}s)', flush=True)
        print(f'{split}: 完成 {len(images)} 张')

    names = [tile_display(t) for t in range(TOTAL_TILES)]
    (out / 'data.yaml').write_text(
        f'path: {out.resolve().as_posix()}\n'
        'train: train/images\n'
        'val: valid/images\n'
        'test: test/images\n'
        'nc: {TOTAL_TILES}\n'
        f'names: {names}\n',
        encoding='utf-8',
    )
    covered = len(dist)
    print(f'\n转换完成 -> {out}')
    print(f'覆盖类数: {covered}/42, 实例总数: {sum(dist.values())}, 跳过: {skipped}')
    if covered < 42:
        missing = [tile_display(t) for t in range(TOTAL_TILES) if dist[t] == 0]
        print(f'缺失类: {missing}')
    else:
        low = sorted(dist.items(), key=lambda kv: kv[1])[:5]
        print(f'实例最少的 5 类: {[(tile_display(t), c) for t, c in low]}')


def main() -> None:
    parser = argparse.ArgumentParser(description='实物麻将数据集标签转换')
    parser.add_argument('--src', default=DEFAULT_SRC)
    parser.add_argument('--out', default=DEFAULT_OUT)
    parser.add_argument('--verify', action='store_true',
                        help='先用 best.pt 校验类序(建议先跑)')
    parser.add_argument('--model', default='data/models/mahjong_tile_detector/weights/best.pt')
    parser.add_argument('--verify-n', type=int, default=6, help='校验用图片数')
    args = parser.parse_args()

    src = Path(args.src)
    if not (src / 'data.yaml').exists():
        print(f'源数据集不存在: {src}')
        sys.exit(1)

    if args.verify:
        print('类序校验(用 best.pt 在验证集上比对)...')
        verify_order(src, args.model, args.verify_n)

    convert(src, Path(args.out))


if __name__ == '__main__':
    main()
