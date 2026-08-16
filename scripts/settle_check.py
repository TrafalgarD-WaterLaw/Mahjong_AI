"""结算核对 — 对局结束亮牌 vs 系统推断(设计文档 §4 验证闭环)。

流程:
    1. 对局中用 run_assistant --log-dir 记录推断快照(每 10 帧一条 jsonl)
    2. 结算界面截图(欢乐麻将胡牌后亮出胡牌家手牌)
    3. 本脚本: 交互框选亮牌区域(或 --region) → 屏幕模型识别牌 →
       与日志终局推断对比 → 核对报告 + 累计 stats

核对指标:
    - 终局听牌概率 vs 实际是否听牌(校准: 概率分桶 → 实际听牌比例)
    - waiting top3 是否包含实际听牌(命中率)
    - 推断概率 vs 牌局阶段(累计出牌事件数)曲线 — 早期推断差/后期准

用法:
    python scripts/settle_check.py settle.png --log game_xxx.jsonl --player right_river
    python scripts/settle_check.py settle.png --region 300,200,800,500 --log game_xxx.jsonl --player my_river
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mahjong_ai.session import OPPONENT_PLAYERS  # noqa: E402
from src.mahjong_core import Hand  # noqa: E402
from src.mahjong_core.tile import tile_display  # noqa: E402
from src.mahjong_engine.rules.interface import IRuleSet  # noqa: E402

#: 概率分桶(校准统计)
_BUCKETS = ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01))


def bucket_of(prob: float) -> str:
    """概率 → 桶名('0-20%' 等, 边界归下界: 0.2 → '20%-40%')。"""
    for lo, hi in _BUCKETS:
        if lo <= prob < hi:
            return f'{lo:.0%}-{hi:.0%}' if hi < 1.01 else f'{lo:.0%}-100%'
    return f'{_BUCKETS[-1][0]:.0%}-100%'


def verify_settle(
    predicted_waiting: list[int],
    predicted_prob: float,
    actual_hand: list[int],
    rules: IRuleSet,
) -> dict:
    """单局核对(纯函数): 终局推断 vs 实际亮牌手牌。

    predicted_waiting: 系统推断的等待牌(降序, 通常 top3);
    actual_hand: 结算亮出的胡牌家手牌(13 张);
    返回: 实际听牌集合 / top3 命中 / 校准桶等核对结果。
    """
    if len(actual_hand) != 13:
        raise ValueError(f'亮牌手牌应为 13 张, 实际 {len(actual_hand)} 张'
                         '(胡牌那张在牌河/别处, 只框手牌区)')
    actual_waiting = {w.tile
                      for w in rules.get_waiting_tiles(Hand(actual_hand))}
    return {
        'pred_prob': round(float(predicted_prob), 3),
        'pred_waiting': list(predicted_waiting),
        'actual_tenpai': bool(actual_waiting),
        'actual_waiting': sorted(actual_waiting),
        'top3_hit': bool(set(predicted_waiting) & actual_waiting),
        'bucket': bucket_of(predicted_prob),
    }


def tenpai_curve(player: str, records: list[dict]) -> list[tuple[int, float]]:
    """该家的听牌概率 vs 累计出牌事件数(牌局阶段曲线, 纯函数)。

    records: 日志记录列表(按时间序), 返回 [(事件数, 听牌概率)]。
    """
    curve: list[tuple[int, float]] = []
    for rec in records:
        prob = rec.get('tenpai', {}).get(player, 0.0)
        curve.append((int(rec.get('events_total', 0)), float(prob)))
    return curve


def update_stats(stats: dict, result: dict) -> dict:
    """多局累计(纯函数): 校准桶 + top3 命中 + 平均听牌概率。"""
    s = dict(stats)
    s['games'] = int(s.get('games', 0)) + 1
    s['top3_hit_n'] = int(s.get('top3_hit_n', 0)) + int(result['top3_hit'])
    s['tenpai_n'] = int(s.get('tenpai_n', 0)) + int(result['actual_tenpai'])
    probs = list(s.get('probs', []))
    probs.append(result['pred_prob'])
    s['probs'] = probs
    buckets = dict(s.get('buckets', {}))
    b = result['bucket']
    entry = dict(buckets.get(b, {'n': 0, 'tenpai': 0}))
    entry['n'] += 1
    entry['tenpai'] += int(result['actual_tenpai'])
    buckets[b] = entry
    s['buckets'] = buckets
    return s


def report(result: dict, curve: list[tuple[int, float]],
           player: str, rules: IRuleSet) -> list[str]:
    """核对报告(纯函数 → 行列表)。"""
    lines = [
        f'=== 结算核对: {player} ===',
        '实际听牌: ' + (' '.join(tile_display(t) for t in result['actual_waiting'])
                        if result['actual_tenpai'] else '无(未听牌)'),
        f"终局推断: 听牌概率 {result['pred_prob']:.0%}, waiting ["
        + ' '.join(tile_display(t) for t in result['pred_waiting']) + ']',
        f"waiting top3 命中: {'✅' if result['top3_hit'] else '❌'}",
    ]
    if curve:
        pts = ' → '.join(f'{n}事件 {p:.0%}' for n, p in curve)
        lines.append(f'阶段曲线: {pts}')
    return lines


def detect_region_tiles(
    model: object, frame: np.ndarray, region: tuple[int, int, int, int],
) -> list[tuple[int, float, float]]:
    """区域内的牌检测: 返回 [(tile, 中心x, 中心y)] 按 (y, x) 排序。

    用框中心判断归属(框 jitter 略出区域也算区域内), 排序按 (y, x)
    手牌单行时即从左到右。
    """
    x1, y1, x2, y2 = region
    results = model(frame, conf=0.5, verbose=False)
    tiles: list[tuple[int, float, float]] = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            bx1, by1, bx2, by2 = box.xyxy[0].tolist()
            cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                tiles.append((int(box.cls[0].item()), cx, cy))
    tiles.sort(key=lambda t: (t[2], t[1]))
    return tiles


def main() -> None:
    parser = argparse.ArgumentParser(description='结算亮牌 vs 推断核对')
    parser.add_argument('image', help='结算界面截图')
    parser.add_argument('--model',
                        default='data/models/screen/mahjong_screen_detector/weights/best.pt')
    parser.add_argument('--log', required=True, help='run_assistant 对局日志 jsonl')
    parser.add_argument('--player', required=True, choices=OPPONENT_PLAYERS,
                        help='亮牌是哪家(核对对象)')
    parser.add_argument('--region', default=None,
                        help='亮牌区域 x1,y1,x2,y2(缺省交互框选)')
    parser.add_argument('--stats', default='data/settle_stats.json',
                        help='多局累计统计文件')
    parser.add_argument('--rules', default='huanyu')
    args = parser.parse_args()

    from src.mahjong_engine import get as get_rules  # noqa: PLC0415

    rules = get_rules(args.rules)
    frame = cv2.imread(args.image)
    if frame is None:
        print(f'无法读取: {args.image}')
        sys.exit(1)

    if args.region:
        x1, y1, x2, y2 = (int(v) for v in args.region.split(','))
        region = (x1, y1, x2, y2)
    else:
        print('在窗口里框选亮牌区域(胡牌家手牌, 不含胡的那张), 回车确认')
        region_raw = cv2.selectROI('settle_check', frame, showCrosshair=False)
        cv2.destroyAllWindows()
        if region_raw[2] <= 0 or region_raw[3] <= 0:
            print('未选择区域')
            sys.exit(1)
        x, y, w, h = region_raw
        region = (x, y, x + w, y + h)

    from src.mahjong_cv.screen_vision import ScreenVision  # noqa: PLC0415

    # ScreenVision: 与实时同管道(ROI 精修 + 双框去重 + 背景类过滤) —
    # 裸 YOLO 会把同位置的重复框算两张, 亮牌 13 张变成 15 张
    sv = ScreenVision(args.model)
    dets = sv.process(frame, conf_threshold=0.2)
    x1, y1, x2, y2 = region
    tiles = [(d.tile, (d.x1 + d.x2) / 2, (d.y1 + d.y2) / 2)
             for d in dets
             if x1 <= (d.x1 + d.x2) / 2 <= x2
             and y1 <= (d.y1 + d.y2) / 2 <= y2]
    tiles.sort(key=lambda t: (t[2], t[1]))
    hand = [t for t, _cx, _cy in tiles]
    if not hand:
        print('区域内未检测到牌, 检查 --region 或截图清晰度')
        sys.exit(1)
    print('亮牌手牌: ' + ' '.join(tile_display(t) for t in hand))

    all_records = [json.loads(line)
                   for line in Path(args.log).read_text(encoding='utf-8')
                   .splitlines() if line.strip()]
    # 合并日志: 只取推断快照(type=infer; 旧格式无 type 字段默认 infer)
    records = [r for r in all_records if r.get('type', 'infer') == 'infer']
    if not records:
        print(f'日志无推断快照: {args.log}')
        sys.exit(1)
    last = records[-1]
    pred_prob = float(last.get('tenpai', {}).get(args.player, 0.0))
    pred_waiting = [int(t) for t in last.get('waiting', {}).get(args.player, [])]

    result = verify_settle(pred_waiting, pred_prob, hand, rules)
    for line in report(result, tenpai_curve(args.player, records),
                       args.player, rules):
        print(line)

    stats_path = Path(args.stats)
    stats = json.loads(stats_path.read_text(encoding='utf-8')) \
        if stats_path.exists() else {}
    stats = update_stats(stats, result)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=1),
                          encoding='utf-8')
    s = stats
    print(f"\n累计 {s['games']} 局: top3 命中 {s['top3_hit_n']}/{s['games']} "
          f"({s['top3_hit_n'] / s['games']:.0%})  "
          f"实际听牌 {s['tenpai_n']}/{s['games']}  平均推断概率 "
          f"{np.mean(s['probs']):.0%}")
    for b, e in sorted(s.get('buckets', {}).items()):
        if e['n']:
            print(f"  校准桶 {b}: n={e['n']} 实际听牌 {e['tenpai'] / e['n']:.0%}")


if __name__ == '__main__':
    main()
