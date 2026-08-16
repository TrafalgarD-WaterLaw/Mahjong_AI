"""半监督伪标注: 屏幕模型预测 → 聚类归属 + 置信度过滤 → 可信标签入库。

自训练循环的一环:
    截新图 → 伪标注(本脚本)→ 混入训练集 → 重训 → 标定新阈值 → 循环

归属(det_cluster.cluster_dets, 不依赖布局文件):
    空间聚类: 手牌 = 最靠底单行带; 牌河 = 紧凑的一堆(每簇一家);
    孤立单框 = 其他区。

阈值(analyze_threshold.py 在真实测试集标定):
    手牌区 conf≥0.70 (precision 99.6%)    → 伪标注主力
    其他区   conf≥0.85 (precision 96%)
    牌河区   默认禁用! 真实牌河检测 precision 仅 0.19(瞎猜),
           收牌河伪标签会把垃圾灌进训练集。等模型牌河精度上 85%
           再用 --conf-river 显式放开。

用法:
    python scripts/pseudo_label.py <原始截图目录> [--conf 0.7] [--out data/pseudo_out]
    # 输出到独立暂存区(data/pseudo_out), 不混入数据集;
    # 审核后再用 build_dataset_yaml --ingest 合并入库。
    # 已处理过的源文件自动跳过(输出名带源文件名); --redo 强制重标
"""

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.capture_dataset import next_index, save_sample
from src.mahjong_core.tile import TOTAL_TILES, tile_display
from src.mahjong_cv.det_cluster import cluster_dets
from src.mahjong_cv.detections import TileDet

DEFAULT_CONF = 0.70    # 手牌区/默认: precision 99.6%
OTHER_CONF = 0.85      # 其他区: precision 96%


def pseudo_exists(out: Path, source_stem: str) -> bool:
    """该源文件是否已有伪标注输出(pseudo_*_<源文件名>.jpg)。"""
    return bool(list((out / 'images').glob(f'**/pseudo_*_{source_stem}.jpg')))


def process_image(
    model: Any, img_path: Path, conf: float, conf_river: float | None,
) -> tuple[list[TileDet], Counter[str]]:
    """单图预测并聚类归属 + 阈值过滤, 返回 (可信检测, 区域计数)。

    归属: cluster_dets 空间聚类(手牌/牌河簇/零散, 不依赖布局);
    阈值: 手牌 conf / 牌河 conf_river(默认禁用) / 其他 OTHER_CONF。
    """
    results = model(str(img_path), conf=min(conf, conf_river or conf),
                    verbose=False)
    preds: list[TileDet] = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            tile = int(box.cls[0].item())
            if not 0 <= tile < TOTAL_TILES:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            preds.append(TileDet(tile, x1, y1, x2, y2,
                                 float(box.conf[0].item())))
    if not preds:
        return [], Counter()

    hand, _melds, rivers = cluster_dets(preds)
    dets: list[TileDet] = []
    region_count: Counter[str] = Counter()
    for d in hand:
        if d.conf >= conf:
            dets.append(d)
            region_count['hand'] += 1
    for cluster in rivers:  # 每簇=一家牌河, 簇内共享牌河阈值
        for d in cluster:
            if conf_river is not None and d.conf >= conf_river:
                dets.append(d)
                region_count['river'] += 1
    return dets, region_count


def main() -> None:
    parser = argparse.ArgumentParser(description='半监督伪标注(区域感知阈值过滤)')
    parser.add_argument('images', help='原始截图目录')
    parser.add_argument('--model', default='data/models/screen/mahjong_screen_detector/weights/best.pt')
    parser.add_argument('--conf', type=float, default=DEFAULT_CONF,
                        help='手牌区可信阈值(默认 0.70, precision 99.6%)')
    parser.add_argument('--conf-river', type=float, default=None,
                        help='牌河区阈值(默认禁用; 模型牌河 precision≥85% 后再放开)')
    parser.add_argument('--out', default='data/pseudo_out',
                        help='伪标注暂存区(默认 data/pseudo_out; 审核后 --ingest 合并)')
    parser.add_argument('--min-boxes', type=int, default=1,
                        help='每帧至少保留的框数, 否则丢弃该帧')
    parser.add_argument('--redo', action='store_true', help='重新标注已处理文件')
    args = parser.parse_args()

    from ultralytics import YOLO  # type: ignore[attr-defined]  # noqa: PLC0415

    model = YOLO(args.model)
    out = Path(args.out)
    if args.conf_river is None:
        print('[牌河] 伪标注禁用(真实牌河检测 precision 仅 0.19); '
              '模型提升后再用 --conf-river 放开')

    src = Path(args.images)
    files = sorted(src.glob('*.jpg')) + sorted(src.glob('*.png'))
    if not files:
        print(f'没有找到图片: {args.images}')
        sys.exit(1)
    pending = files if args.redo else [
        f for f in files if not pseudo_exists(out, f.stem)
    ]
    skipped = len(files) - len(pending)
    if skipped:
        print(f'跳过 {skipped} 张已伪标注的图片')

    index = next_index(out / 'images', 'pseudo')
    saved = 0
    total_boxes = 0
    class_dist: Counter[int] = Counter()
    region_total: Counter[str] = Counter()

    for f in pending:
        dets, region_count = process_image(model, f, args.conf, args.conf_river)
        if len(dets) < args.min_boxes:
            continue
        frame = cv2.imread(str(f))
        if frame is None:
            continue
        save_sample(out, frame, dets, index, name=f'pseudo_{index:06d}_{f.stem}')
        total_boxes += len(dets)
        region_total.update(region_count)
        for d in dets:
            class_dist[d.tile] += 1
        index += 1
        saved += 1
        if saved % 20 == 0:
            print(f'[{saved}/{len(pending)}] 已伪标注 {saved} 帧', flush=True)

    print(f'\n完成: 伪标注 {saved} 帧, 共 {total_boxes} 个框 -> {out}')
    print(f'区域分布: {dict(region_total)}')
    print(f'类覆盖: {len(class_dist)}/{TOTAL_TILES}')
    low = sorted(class_dist.items(), key=lambda kv: kv[1])[:3]
    print('最少类:', [(tile_display(t), n) for t, n in low])


if __name__ == '__main__':
    main()
