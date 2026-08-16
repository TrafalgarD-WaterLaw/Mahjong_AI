"""训练屏幕牌检测模型(基于 yolo26m.pt 微调, 2026-08-14 切换基座)。

用法:
    python scripts/train_screen_yolo.py --data data/screen_dataset/data.yaml
    # 每次训练输出到带时间戳的新目录(不覆盖历史权重):
    #   data/models/screen/mahjong_screen_detector_<YYYYmmdd_HHMMSS>/
    # 训练后 best/last.pt 自动同步到固定路径(下游工具默认读取位置):
    #   data/models/screen/mahjong_screen_detector/weights/best.pt
    # 接着上次模型续训: --model data/models/screen/mahjong_screen_detector/weights/best.pt
    # 中断恢复: --resume(从固定路径 last.pt 续训, 继续写原时间戳目录)
    #   数据/超参沿用中断时的配置; 想恢复到指定训练用 --model <目录>/weights/last.pt --resume
    # 换基座: --model yolo26n.pt / yolo26l.pt(默认 m, 与历史 yolov8m 同级别对比)
"""

import argparse
import shutil
import time
from pathlib import Path

import torch
from ultralytics import YOLO  # type: ignore[attr-defined]  # noqa: PLC0415  (脚本层允许顶层导入)

FIXED_DIR = Path('data/models/screen/mahjong_screen_detector')


def main() -> None:
    parser = argparse.ArgumentParser(description='训练欢乐麻将屏幕牌检测模型')
    parser.add_argument('--data', default='data/screen_dataset/data.yaml')
    parser.add_argument('--model', default='yolo26m.pt',
                        help='基座(默认 yolo26m, ultralytics 自动下载)')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--patience', type=int, default=15,
                        help='早停耐心: 验证指标连续 N 轮无提升即停(默认40, 防平台期误停)')
    parser.add_argument('--imgsz', type=int, default=1280,
                        help='屏幕牌较小, 用1280保小目标精度')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--scale', default='0.5',
                        help='缩放增强: 单值如 0.5 = (1-0.5, 1+0.5) 默认; '
                             '或双值区间如 0.75,1.5')
    parser.add_argument('--fliplr', type=float, default=0.0,
                        help='水平翻转概率(牌面字符翻转=镜像假牌, 默认0)')
    parser.add_argument('--name', default='mahjong_screen_detector')
    parser.add_argument('--resume', action='store_true',
                        help='中断恢复: 从 last.pt 续训(默认固定路径的 last.pt)')
    args = parser.parse_args()

    if args.resume:
        resume_pt = Path(args.model) if Path(args.model).is_file() \
            else FIXED_DIR / 'weights' / 'last.pt'
        print(f'从 {resume_pt} 续训(数据/超参沿用中断时的配置)…')
        # 仅中断的训练可 resume: 训练完成时 last.pt 的 epoch 被置 -1,
        # 不保留 optimizer 状态; 此时 resume 会被 ultralytics 静默降级为
        # 默认参数的新训练(coco8.yaml!)— 必须在这里拦截报错
        ckpt = torch.load(str(resume_pt), map_location='cpu', weights_only=False)
        epoch = ckpt.get('epoch')
        if not isinstance(epoch, int) or epoch < 0 \
                or 'optimizer_state_dict' not in ckpt:
            print(f'错误: {resume_pt} 不是可恢复的 checkpoint'
                  f'(epoch={epoch}, 训练可能已完成)。')
            print('训练完成后的续训请用「热启动」(加载权重重新训练, '
                  '不带 --resume):')
            print(f'  uv run python scripts/train_screen_yolo.py '
                  f'--model {resume_pt} --data data/screen_dataset/data.yaml')
            raise SystemExit(1)
        # resume 的 epochs/patience 取自 ckpt 内 train_args(不是 args.yaml):
        # 原训练已跑到上限时 resume 会立即结束, 必须先改写 ckpt 内的上限
        ta = ckpt.get('train_args')
        if isinstance(ta, dict):
            old_ep = ta.get('epochs')
            ta['epochs'] = args.epochs
            ta['patience'] = args.patience
            print(f'续训上限: epochs {old_ep} → {args.epochs}, '
                  f'patience → {args.patience}')
        torch.save(ckpt, str(resume_pt))
        model = YOLO(str(resume_pt))
        results = model.train(resume=True)
        print(f'续训完成: {results.save_dir}')
        _sync_fixed(results.save_dir)
        return

    # 每次训练一个带时间戳的新目录, 历史权重全部保留
    run_name = f'{args.name}_{time.strftime("%Y%m%d_%H%M%S")}'
    project = Path('data/models/screen').resolve()
    print(f'本次输出目录: {project / run_name}(训练中权重即写此处)')
    model = YOLO(args.model)
    scale_arg = tuple(map(float, args.scale.split(',')))
    # 单值(如 0.5)直接传 float: ultralytics 内部转 (1-0.5, 1+0.5);
    # 双值('0.75,1.5')传区间 tuple
    scale = scale_arg[0] if len(scale_arg) == 1 else scale_arg
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=run_name,
        project=str(project),  # 绝对路径, 避免被拼到 runs/detect 下
        exist_ok=False,
        patience=args.patience,
        device=0,
        scale=scale,
        fliplr=args.fliplr,
    )
    print(f'训练完成, 本次模型目录: {results.save_dir}')
    _sync_fixed(results.save_dir)


def _sync_fixed(save_dir: Path) -> None:
    """同步 best/last 到固定路径(下游默认读取), 历史权重保留在时间戳目录。"""
    fixed = FIXED_DIR / 'weights'
    fixed.mkdir(parents=True, exist_ok=True)
    for name in ('best.pt', 'last.pt'):
        src = save_dir / 'weights' / name
        if src.exists():
            shutil.copy(str(src), str(fixed / name))
    print(f'已同步固定路径(下游默认): {fixed}/best.pt')


if __name__ == '__main__':
    main()
