"""构建牌面 ROI 分类数据集 — 检测框裁剪放大, 训练独立分类头。

两级识别: 检测器(yolov8m)找框 → 分类器(yolov8-cls)精修牌面类别。
分类器专门学"放大后的牌面"(训练域 = 推理域, 解决小框牌面难分辨
的类别混淆: 七万/九万、五条/七条、南/北…)。

位置抖动: 每个框生成 原样 + N-1 张偏移变体(随机平移/缩放) —
模拟检测框不精确, 让分类器对"偏移的 crop"鲁棒
(实测: 分类器对 GT crop 98.3%, 对检测框 crop 仅 ~88% — 偏移是主因)。

背景类(id=35): 随机非牌区域 crop + 检测器误检框 crop — 支持
"检测阈值降到 0.15 找回漏检 + 分类器滤误检"(低阈值下误检框暴增,
没有背景类就无法区分"低 conf 真牌"和"假框")。

数据源:
  train = screen_dataset 全部框 crop(类别 = label) + 背景类
  val   = test_real 人工 GT 框 crop(真值准, 无抖动) + 背景类

用法:
    python scripts/build_roi_dataset.py                 # 构建(默认)
    python scripts/train_roi_cls.py                     # 训练分类器
"""

import random
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATASET = Path('data/screen_dataset')
TEST = Path('data/test_real')
OUT = Path('data/roi_dataset')
#: crop 输出尺寸(放大: 15px 牌 → 96px)
ROI_SIZE = 96
#: crop 外扩像素(带一点背景, 分类器见过上下文更稳)
PAD = 4
#: 抖动偏移上限(px): 变体在 [x-PAD-J, x+PAD+J] 内随机平移
JITTER_PX = 5
#: 每框生成变体数(1 = 只有原样; 2 = 原样 + 1 张偏移变体)
EXTRA = 2
#: 背景类 id(35 = 非牌区域/误检框 — 低阈值检测的误检过滤)
BG_CLASS = 35
#: 每张图随机背景 crop 数(负样本 — 3 时分类器对误检框过滤不足:
#: 背景 3798 vs 牌类 74648 严重不平衡, 纯误检逃过滤; 提到 8)
BG_PER_IMAGE = 8


def crop_backgrounds(frame, gt_boxes: list[tuple[int, int, int, int]],
                     out_dir: Path, n: int, stem: str,
                     rng) -> int:
    """随机非牌区域 crop → 背景类(35)。避开所有 GT 框。"""
    fh, fw = frame.shape[:2]
    cls_dir = out_dir / str(BG_CLASS)
    cls_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for k in range(n):
        for _ in range(20):  # 最多 20 次尝试
            w = int(fw * rng.uniform(0.02, 0.08))
            x = rng.randint(0, max(1, fw - w))
            y = rng.randint(0, max(1, fh - w))
            if any(x + w > bx1 - 6 and x < bx2 + 6 and y + w > by1 - 6
                   and y < by2 + 6 for (bx1, by1, bx2, by2) in gt_boxes):
                continue  # 与 GT 框重叠 → 重试
            crop = frame[y:y + w, x:x + w]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, (ROI_SIZE, ROI_SIZE))
            cv2.imwrite(str(cls_dir / f'{stem}_bg{k}.jpg'), crop)
            saved += 1
            break
    return saved


def crop_boxes(frame, label_path: Path, out_dir: Path,
               jitter_px: int, extra: int) -> int:
    """从一帧裁剪所有标注框(含抖动变体) → out_dir/<cls>/<stem>_<i>.jpg。"""
    fh, fw = frame.shape[:2]
    if not label_path.exists():
        return 0
    n = 0
    for line in label_path.read_text(encoding='utf-8').splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls, cx, cy, w, h = int(parts[0]), *map(float, parts[1:5])
        x1, y1 = int((cx - w / 2) * fw), int((cy - h / 2) * fh)
        x2, y2 = int((cx + w / 2) * fw), int((cy + h / 2) * fh)
        cls_dir = out_dir / str(cls)
        cls_dir.mkdir(parents=True, exist_ok=True)

        def save_crop(px1: int, py1: int, px2: int, py2: int,
                      scale: float) -> None:
            nonlocal n
            px1, py1 = max(0, px1), max(0, py1)
            px2, py2 = min(fw, px2), min(fh, py2)
            crop = frame[py1:py2, px1:px2]
            if crop.size == 0:
                return
            crop = cv2.resize(crop, (ROI_SIZE, ROI_SIZE),
                              interpolation=cv2.INTER_LINEAR)
            if scale != 1.0:  # 缩放变体
                crop = cv2.resize(
                    crop, (int(ROI_SIZE * scale), int(ROI_SIZE * scale)),
                    interpolation=cv2.INTER_LINEAR)
                crop = cv2.resize(crop, (ROI_SIZE, ROI_SIZE),
                                  interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(cls_dir / f'{label_path.stem}_{n}.jpg'), crop)
            n += 1

        # 原样
        save_crop(x1 - PAD, y1 - PAD, x2 + PAD, y2 + PAD, 1.0)
        # 抖动变体: 随机平移 + 随机缩放(模拟检测框偏移)
        for _ in range(extra - 1):
            dx = random.randint(-jitter_px, jitter_px)
            dy = random.randint(-jitter_px, jitter_px)
            scale = random.uniform(0.9, 1.1)
            save_crop(x1 - PAD + dx, y1 - PAD + dy,
                      x2 + PAD + dx, y2 + PAD + dy, scale)
    return n


def main() -> None:
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(description='构建 ROI 分类数据集')
    parser.add_argument('--jitter', type=int, default=JITTER_PX,
                        help='抖动偏移像素(默认 5; 0 = 关闭抖动)')
    parser.add_argument('--extra', type=int, default=EXTRA,
                        help='每框变体数(默认 2: 原样 + 抖动变体)')
    args = parser.parse_args()

    # train: screen_dataset 全量(含抖动变体 + 背景类)
    train_out = OUT / 'images' / 'train'
    total = total_bg = 0
    rng = random.Random(42)
    for split in ('train', 'val'):
        img_dir = DATASET / 'images' / split
        label_dir = DATASET / 'labels' / split
        for p in sorted(img_dir.glob('*.jpg')):
            frame = cv2.imread(str(p))
            if frame is None:
                continue
            label_path = label_dir / f'{p.stem}.txt'
            total += crop_boxes(frame, label_path, train_out,
                                args.jitter, args.extra)
            # 背景类: 随机非牌区域
            fh, fw = frame.shape[:2]
            gts = []
            if label_path.exists():
                for line in label_path.read_text(
                        encoding='utf-8').splitlines():
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    cx, cy, w, h = map(float, parts[1:5])
                    gts.append((int((cx - w / 2) * fw),
                                int((cy - h / 2) * fh),
                                int((cx + w / 2) * fw),
                                int((cy + h / 2) * fh)))
            total_bg += crop_backgrounds(frame, gts, train_out,
                                         BG_PER_IMAGE, p.stem, rng)
    print(f'train crop(含抖动): {total} 张 + 背景 {total_bg} 张')

    # val: test_real 人工 GT(真值准, 无抖动 — 验证用干净样本) + 背景类
    val_out = OUT / 'images' / 'val'
    val_n = val_bg = 0
    for p in sorted(TEST.glob('images/*.jpg')):
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        label_path = TEST / 'labels' / f'{p.stem}.txt'
        val_n += crop_boxes(frame, label_path, val_out, 0, 1)
        fh, fw = frame.shape[:2]
        gts = []
        if label_path.exists():
            for line in label_path.read_text(encoding='utf-8').splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cx, cy, w, h = map(float, parts[1:5])
                gts.append((int((cx - w / 2) * fw),
                            int((cy - h / 2) * fh),
                            int((cx + w / 2) * fw),
                            int((cy + h / 2) * fh)))
        val_bg += crop_backgrounds(frame, gts, val_out,
                                   BG_PER_IMAGE, p.stem, rng)
    print(f'val crop(test_real 人工, 无抖动): {val_n} 张 + 背景 {val_bg} 张')
    print(f'→ {OUT}(ImageFolder 结构, 类别 0-34 = 牌, 35 = 背景)')


if __name__ == '__main__':
    main()
