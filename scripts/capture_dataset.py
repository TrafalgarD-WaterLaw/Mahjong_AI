"""采集屏幕原始截图(供人工审核打标, label_review.py 用)。

用法:
    python scripts/capture_dataset.py --raw --frames 100 --interval 2.0
    # --raw: 只截帧不匹配(模板匹配自动打标已废弃, 由 YOLO 实时检测取代)

命名规范(数据文件名, 全项目统一):
    {prefix}_{序号:06d}                无源数据(shot 原始截帧 / syn 合成)
    {prefix}_{序号:06d}_{源文件名}      派生数据(pseudo 伪标注带 shot 源名,
                                       review 审核带 pseudo 或 shot 源名)
    - 流水号一律 6 位(0-pad), 各前缀独立从 0 续号
    - 内嵌源名 = 源文件完整文件名(断点续审按它匹配, 改名须同步引用)
    - 前缀供分层拆分(_source_prefix)与断点续审识别来源
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mahjong_cv.capture.win32 import DEFAULT_TITLE_CANDIDATES, Win32Capture
from src.mahjong_cv.detections import TileDet

SEQ_FILE = Path('data/.seq.json')


def next_index(img_dir: Path, prefix: str,
               seq_file: Path | None = None) -> int:
    """返回下一个可用序号: 全局计数器(只增不减, 永不回收)。

    计数器存 data/.seq.json(可注入 seq_file 隔离测试);
    删除旧数据/换目录都不会重名 — 命名永不回收。
    首次使用某前缀时扫描 img_dir 现有文件初始化(向后兼容历史数据)。
    编号取「前缀后最后一个数字段」: 新格式 shot_20260808_000001 的
    seq 是 000001(日期段在前); 旧格式 shot_000194 只有一个数字段。
    """
    seq_file = seq_file or SEQ_FILE
    seq: dict[str, int] = {}
    if seq_file.exists():
        seq = json.loads(seq_file.read_text(encoding='utf-8'))
    cur = seq.get(prefix, -1)
    if cur < 0:
        # 首次: 扫描现有文件取最大编号(兼容已存在的数据)
        max_i = -1
        for p in img_dir.glob(f'{prefix}_*.jpg'):
            # 取前缀后最后一个数字段(兼容 review_00005_shot_00042 这类多段名)
            parts = p.stem.split('_')[1:]
            for part in reversed(parts):
                if part.isdigit():
                    max_i = max(max_i, int(part))
                    break
        cur = max_i
    seq[prefix] = cur + 1
    seq_file.parent.mkdir(parents=True, exist_ok=True)
    seq_file.write_text(json.dumps(seq), encoding='utf-8')
    return cur + 1


def save_sample(
    sample_dir: Path, frame: np.ndarray, dets: list[TileDet], index: int,
    name: str | None = None,
) -> None:
    """保存一帧 YOLO 格式样本: images/<name>.jpg + labels/<name>.txt。

    标签行: <class_id> <cx> <cy> <w> <h>(归一化)。class_id = 牌编码 0-41。
    name: 自定义文件名(不含扩展名); 缺省 frame_<index:06d>。
    """
    h, w = frame.shape[:2]
    img_dir = sample_dir / 'images'
    lbl_dir = sample_dir / 'labels'
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    name = name or f'frame_{index:06d}'
    cv2.imwrite(str(img_dir / f'{name}.jpg'), frame)

    lines = []
    for d in dets:
        cx = (d.x1 + d.x2) / 2 / w
        cy = (d.y1 + d.y2) / 2 / h
        bw = (d.x2 - d.x1) / w
        bh = (d.y2 - d.y1) / h
        lines.append(f'{d.tile} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}')
    (lbl_dir / f'{name}.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def raw_capture(cap: Win32Capture, out: Path, frames: int, interval: float) -> None:
    """--raw 模式: 只截帧保存图片, 不做匹配打标(供人工审核)。

    命名用截屏时刻的微秒时间戳(shot_20260807225430123456):
    天然唯一 — 不需要计数器/批次管理, 删除旧数据、换目录都不会重名。
    """
    img_dir = out / 'images'
    img_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    while saved < frames:
        frame = cap.capture()
        if frame is None:
            time.sleep(1.0)
            continue
        # datetime.strftime 支持 %f(微秒), time.strftime 不支持
        name = f"shot_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        cv2.imwrite(str(img_dir / f'{name}.jpg'), frame)
        saved += 1
        if saved % 10 == 0 or saved == frames:
            print(f'[{saved}/{frames}] 已保存 {saved} 帧', flush=True)
        time.sleep(interval)
    print(f'完成: {saved} 帧原始截图 -> {img_dir}')


def main() -> None:
    parser = argparse.ArgumentParser(description='采集欢乐麻将屏幕原始截图(供人工审核)')
    parser.add_argument('--out', default='data/pseudo_raw')
    parser.add_argument('--frames', type=int, default=200)
    parser.add_argument('--interval', type=float, default=1.0, help='帧间隔秒')
    parser.add_argument('--title', default='欢乐麻将')
    parser.add_argument('--raw', action='store_true',
                        help='只截帧不匹配(供 label_review 人工审核)')
    args = parser.parse_args()

    if not args.raw:
        print('模板匹配自动打标已废弃(由 YOLO 实时检测取代), 请用 --raw 截帧')
        sys.exit(1)

    candidates = DEFAULT_TITLE_CANDIDATES if args.title == '欢乐麻将' else (args.title,)
    cap = Win32Capture(args.title, candidates)
    if cap.client_rect() is None:
        print(f"未找到窗口(候选: {candidates}), 请先启动欢乐麻将")
        sys.exit(1)

    raw_capture(cap, Path(args.out), args.frames, args.interval)


if __name__ == '__main__':
    main()
