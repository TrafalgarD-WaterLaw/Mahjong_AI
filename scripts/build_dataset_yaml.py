"""把采集好的数据集拆分为 train/val 并生成 data.yaml。

用法:
    python scripts/build_dataset_yaml.py --dataset data/screen_dataset

拆分策略: 按来源前缀(如 frame_/review_/syn_)分层随机抽取 val,
保证验证集包含各来源的代表(否则按文件名排序取末尾会把真实数据
全部排除在验证集外, mAP 虚高)。
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mahjong_core.tile import TOTAL_TILES, tile_display


def _source_prefix(stem: str) -> str:
    """文件名来源前缀: 去掉数字部分('frame_000123'->'frame', 'f0'->'f')。"""
    m = re.match(r'[^0-9]*', stem)
    return (m.group(0) if m else stem).rstrip('_') or stem


def _unsplit(dataset_root: Path) -> None:
    """把已拆分的 train/val 子目录移回根目录(幂等重跑支持)。

    根目录已有同名文件时(如暂存区合并进来的新审核版)以根目录为准,
    丢弃子目录旧版, 避免重跑时 FileExistsError。
    """
    for subset in ('train', 'val'):
        for p in sorted((dataset_root / 'images' / subset).glob('*')):
            dst = dataset_root / 'images' / p.name
            if dst.exists():
                p.unlink()
            else:
                p.rename(dst)
        for p in sorted((dataset_root / 'labels' / subset).glob('*')):
            dst = dataset_root / 'labels' / p.name
            if dst.exists():
                p.unlink()
            else:
                p.rename(dst)


def _source_key(stem: str) -> str:
    """最末源: 最后一个 shot_ 开头、含全部数字段的完整标识。

    新格式带日期批次(shot_20260808_000042)与旧格式(shot_000042)是
    不同源; review/pseudo 内嵌的源名须整体取, 不能只取一段数字。
    没有 shot_ 段则取最后一段。
    """
    shots = re.findall(r'(shot_(?:\d+_)*\d+)', stem)
    return shots[-1] if shots else stem.rsplit('_', 1)[-1]


def _ingest(dataset_root: Path, ingest_dir: Path | None) -> int:
    """把外部暂存区(伪标注/审核输出)的样本移入数据集根目录。

    暂存区结构: <ingest>/images/*.jpg + <ingest>/labels/*.txt(与数据集同构)。
    目标重名时跳过并警告。返回移入的样本数。
    """
    if ingest_dir is None or not (ingest_dir / 'images').exists():
        return 0
    n = 0
    for img in sorted((ingest_dir / 'images').glob('*.jpg')):
        lbl = ingest_dir / 'labels' / (img.stem + '.txt')
        if not lbl.exists():
            print(f'跳过缺标签: {img.name}')
            continue
        dst_img = dataset_root / 'images' / img.name
        if dst_img.exists():
            print(f'跳过重名(已存在): {img.name}')
            continue
        img.rename(dst_img)
        lbl.rename(dataset_root / 'labels' / lbl.name)
        n += 1
    return n


def _dedup_keep_review(dataset_root: Path) -> int:
    """同源去重: 同一最末源同时有 review 与 pseudo 时, 删除伪标注版。

    审核是人工把关的标签, 伪标注是自动充量 — 同图必须以后者为准;
    重复样本会让模型对同一画面学两遍。返回删除的图数(图+标签同步)。
    """
    images = list((dataset_root / 'images').glob('*.jpg'))
    by_src: dict[str, list[Path]] = {}
    for img in images:
        by_src.setdefault(_source_key(img.stem), []).append(img)

    removed = 0
    for _src, imgs in by_src.items():
        if len(imgs) < 2:
            continue
        has_review = any(i.stem.startswith('review_') for i in imgs)
        if not has_review:
            continue
        for img in imgs:
            if img.stem.startswith('pseudo_'):
                (dataset_root / 'labels' / (img.stem + '.txt')).unlink(missing_ok=True)
                img.unlink()
                removed += 1
    return removed


def split_and_write_yaml(
    dataset_root: Path, val_ratio: float = 0.2, seed: int = 42,
    ingest_dir: Path | None = None,
) -> Path:
    """合并暂存区 → 同源去重 → 分层随机拆分; 写 data.yaml。"""
    if ingest_dir is not None:
        n = _ingest(dataset_root, ingest_dir)
        if n:
            print(f'合并暂存区: 移入 {n} 张 -> {dataset_root}')
    _unsplit(dataset_root)
    removed = _dedup_keep_review(dataset_root)
    if removed:
        print(f'同源去重: 删除 {removed} 张伪标注(已有审核版)')
    images = sorted((dataset_root / 'images').glob('*.jpg'))
    if not images:
        raise ValueError(f'数据集为空: {dataset_root / "images"}')

    # 按来源前缀分层
    by_source: dict[str, list[Path]] = {}
    for img in images:
        by_source.setdefault(_source_prefix(img.stem), []).append(img)

    rng = random.Random(seed)
    val_set: set[Path] = set()
    for _src, imgs in by_source.items():
        rng.shuffle(imgs)
        n_val = max(1, int(len(imgs) * val_ratio))
        val_set.update(imgs[:n_val])

    for subset in ('train', 'val'):
        (dataset_root / 'images' / subset).mkdir(parents=True, exist_ok=True)
        (dataset_root / 'labels' / subset).mkdir(parents=True, exist_ok=True)

    for img in images:
        subset = 'val' if img in val_set else 'train'
        label = dataset_root / 'labels' / (img.stem + '.txt')
        if not label.exists():
            raise ValueError(f'缺少标签: {label}')
        img.rename(dataset_root / 'images' / subset / img.name)
        label.rename(dataset_root / 'labels' / subset / label.name)

    names = [tile_display(t) for t in range(TOTAL_TILES)]
    yaml_path = dataset_root / 'data.yaml'
    yaml_path.write_text(
        f'path: {dataset_root.resolve().as_posix()}\n'
        'train: images/train\n'
        'val: images/val\n'
        f'nc: {TOTAL_TILES}\n'
        f'names: {json.dumps(names, ensure_ascii=False)}\n',
        encoding='utf-8',
    )
    return yaml_path


def write_test_yaml(dataset_root: Path) -> Path:
    """测试集 yaml: 不拆分, 全部作为 val(真实留出集评测用)。"""
    images = sorted((dataset_root / 'images').glob('*.jpg'))
    if not images:
        raise ValueError(f'数据集为空: {dataset_root / "images"}')
    names = [tile_display(t) for t in range(TOTAL_TILES)]
    yaml_path = dataset_root / 'data.yaml'
    yaml_path.write_text(
        f'path: {dataset_root.resolve().as_posix()}\n'
        # ultralytics 要求 train 键存在(纯评测集, 指向同一批图即可)
        'train: images\n'
        'val: images\n'
        f'nc: {TOTAL_TILES}\n'
        f'names: {json.dumps(names, ensure_ascii=False)}\n',
        encoding='utf-8',
    )
    return yaml_path


def main() -> None:
    parser = argparse.ArgumentParser(description='数据集拆分并生成 data.yaml')
    parser.add_argument('--dataset', default='data/screen_dataset')
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ingest', default=None,
                        help='先合并外部暂存区(如 data/pseudo_out)再拆分')
    parser.add_argument('--no-split', action='store_true',
                        help='不拆分, 全部作 val(真实留出测试集)')
    args = parser.parse_args()
    if args.no_split:
        yaml_path = write_test_yaml(Path(args.dataset))
    else:
        ingest = Path(args.ingest) if args.ingest else None
        yaml_path = split_and_write_yaml(Path(args.dataset), args.val_ratio,
                                         args.seed, ingest)
    print(f'已生成: {yaml_path}')


if __name__ == '__main__':
    main()
