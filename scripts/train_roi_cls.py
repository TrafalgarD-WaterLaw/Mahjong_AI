"""训练牌面 ROI 分类器(yolov8-cls) — 检测框 crop 后的类别精修。

用法:
    python scripts/train_roi_cls.py                     # 训练(默认)
    # 输出: data/models/roi_cls/weights/best.pt(自动同步固定路径)
    # 验证: test_real GT crop 的类别准确率(对比检测器全图 68.7%)

训练配置(2026-08-14 调优, 36 类 = 35 牌 + 背景):
    cache='ram': 11.9 万张小图读进内存 — 磁盘读取(0.2MB/s)是
        之前 4.3 小时训练的主因, 缓存后 15-25 分钟
    batch=512: 96×96 小图显存富余, 大 batch 提升吞吐
    workers=0: Windows 下 DataLoader 多进程 spawn 会 pickle 失败
        (OSError 22 / pickle truncated) — cache='ram' 后单进程不慢
    epochs=40 + patience=15: 数据量翻倍后收敛更快(之前 25 早停)
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FIXED = Path('data/models/roi_cls')


def main() -> None:
    parser = argparse.ArgumentParser(description='训练牌面 ROI 分类器')
    parser.add_argument('--model', default='yolov8n-cls.pt',
                        help='分类基座(ultralytics 自动下载)')
    parser.add_argument('--data', default='data/roi_dataset/images')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--imgsz', type=int, default=96)
    parser.add_argument('--batch', type=int, default=512)
    args = parser.parse_args()

    from ultralytics import YOLO  # type: ignore[attr-defined]  # noqa: PLC0415

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        cache='ram',       # 数据集全读内存(磁盘慢是主瓶颈)
        workers=0,         # Windows spawn 多进程 pickle 失败 → 单进程
        name='roi_cls',
        project=str(Path('data/models').resolve()),
        patience=15,
        device=0,
    )
    best = Path(results.save_dir) / 'weights' / 'best.pt'
    FIXED.mkdir(parents=True, exist_ok=True)
    (FIXED / 'weights').mkdir(parents=True, exist_ok=True)
    shutil.copy(str(best), str(FIXED / 'weights' / 'best.pt'))
    print(f'已同步: {FIXED / "weights" / "best.pt"}')


if __name__ == '__main__':
    main()
