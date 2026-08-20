"""识别管线离线性能基准 — 真实游戏截图逐帧跑识别, 分段计时。

用 data/pseudo_raw/images/ 的连续截图(真实对局采集)离线复现
ScreenVision.track 全流程, 输出 检测+跟踪 / 解析 / ROI 精修 三段耗时,
支持消融(imgsz、开/关 ROI 分类器)定位瓶颈。

用法:
    uv run python scripts/bench_recognize.py
    uv run python scripts/bench_recognize.py --imgsz 640
    uv run python scripts/bench_recognize.py --no-roi
    uv run python scripts/bench_recognize.py --frames 60 --skip 5
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.mahjong_cv.screen_vision import ScreenVision  # noqa: E402

MODEL = 'data/models/screen/mahjong_screen_detector/weights/best.pt'
DEFAULT_SHOTS = 'data/pseudo_raw/images'


def load_frames(directory: str, frames: int, skip: int) -> list[np.ndarray]:
    """按文件名序取连续截图(skip 步长) — 近似对局帧流。"""
    files = sorted(Path(directory).glob('*.jpg'))
    if not files:
        raise SystemExit(f'{directory} 下没有截图')
    selected = files[:: max(1, skip)][:frames]
    out = []
    for f in selected:
        img = cv2.imread(str(f))
        if img is not None:
            out.append(img)
    return out


def percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    i = min(len(sorted_vals) - 1, int(p * len(sorted_vals)))
    return sorted_vals[i]


def stat_row(name: str, vals: list[float]) -> str:
    s = sorted(vals)
    mean = statistics.fmean(s) if s else 0.0
    return (f'  {name:<14} {mean * 1000:>8.1f}ms '
            f'{percentile(s, 0.5) * 1000:>7.1f}ms '
            f'{percentile(s, 0.95) * 1000:>7.1f}ms '
            f'{percentile(s, 0.99) * 1000:>7.1f}ms '
            f'{s[-1] * 1000 if s else 0.0:>8.1f}ms')


def main() -> None:
    parser = argparse.ArgumentParser(description='识别管线离线性能基准')
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--shots', default=DEFAULT_SHOTS)
    parser.add_argument('--frames', type=int, default=40)
    parser.add_argument('--skip', type=int, default=1,
                        help='截图步长(1=逐帧, 5=每 5 张取 1 模拟抽帧)')
    parser.add_argument('--imgsz', type=int, default=None,
                        help='推理分辨率(默认模型训练尺寸)')
    parser.add_argument('--conf', type=float, default=0.15)
    parser.add_argument('--no-roi', action='store_true',
                        help='跳过 ROI 分类器(量化其成本)')
    parser.add_argument('--no-track', action='store_true',
                        help='纯检测不做跟踪(量化跟踪开销)')
    args = parser.parse_args()

    print(f'加载模型: {args.model}')
    roi = None if args.no_roi else ScreenVision.DEFAULT_ROI
    pipe = ScreenVision(args.model, roi_model_path=roi)
    frames = load_frames(args.shots, args.frames, args.skip)
    print(f'截图: {len(frames)} 帧, 尺寸 {frames[0].shape[1]}x{frames[0].shape[0]}')
    print(f'消融: imgsz={args.imgsz or "默认"} conf={args.conf} '
          f'ROI={"关" if args.no_roi else "开"} '
          f'track={"关" if args.no_track else "开"}')

    # 预热: 首帧跑一次(模型加载/GPU 初始化不进入统计)
    t0 = time.perf_counter()
    pipe.track(frames[0], conf_threshold=args.conf, imgsz=args.imgsz)
    warm = time.perf_counter() - t0
    print(f'预热(首帧): {warm * 1000:.0f}ms — 冷启动开销单独报告\n')

    total: list[float] = []
    detect: list[float] = []
    parse: list[float] = []
    roi_t: list[float] = []
    n_dets: list[int] = []
    for _, frame in enumerate(frames[1:]):
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        if args.no_track:
            dets = pipe.process(frame, conf_threshold=args.conf,
                                imgsz=args.imgsz)
            timings = {'detect': time.perf_counter() - t0, 'parse': 0.0,
                       'roi': 0.0}
        else:
            dets = pipe.track(frame, conf_threshold=args.conf,
                              imgsz=args.imgsz, timings=timings)
        dt = time.perf_counter() - t0
        total.append(dt)
        detect.append(timings.get('detect', 0.0))
        parse.append(timings.get('parse', 0.0))
        roi_t.append(timings.get('roi', 0.0))
        n_dets.append(len(dets))

    print('阶段            均值       P50      P95      P99     最大')
    print(stat_row('detect+track', detect))
    print(stat_row('parse', parse))
    print(stat_row('roi_refine', roi_t))
    print(stat_row('total', total))
    print(f'  框数/帧       均值 {statistics.fmean(n_dets):.0f}  '
          f'最大 {max(n_dets)}')
    print(f'\n  有效帧率: {1.0 / statistics.fmean(total):.1f} fps'
          f'(均值耗时换算, 不含冷启动)')
    if args.imgsz is not None and not args.no_roi and not args.no_track:
        print(f'  ROI 占比: {100 * sum(roi_t) / sum(total):.0f}%  '
              f'(检测+跟踪占比 {100 * sum(detect) / sum(total):.0f}%)')


if __name__ == '__main__':
    main()
