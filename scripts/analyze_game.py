"""对局日志分析 — 归属链正确性诊断(run_assistant --trace 产物)。

用法:
    python scripts/analyze_game.py data/settle_logs/game_xxx.jsonl

输出:
    ① 归属链循环率(新牌 player 序列 vs 轮转期望, 含副露转移)
    ② 四家终局张数分布
    ③ 开局观察期(校准前事件应为 0)+ 回溯分配
    ④ 副露事件(碰/杠, 行动者转移)
    ⑤ 可疑冲突(连续同家打出, 非副露)
    ⑥ 动画忽略帧数
"""

import argparse
import json
import sys
from pathlib import Path

_ORDER = ['my_river', 'right_river', 'top_river', 'left_river']
_NAMES = {'my_river': '我', 'right_river': '右', 'top_river': '上',
          'left_river': '左'}


def _next_of(player: str) -> str:
    return _ORDER[(_ORDER.index(player) + 1) % 4]


def main() -> None:
    parser = argparse.ArgumentParser(description='对局日志归属链分析')
    parser.add_argument('log', help='game_*.jsonl 路径')
    args = parser.parse_args()

    lines = [json.loads(l) for l in
             Path(args.log).read_text(encoding='utf-8').splitlines()]
    traces = [d for d in lines if d['type'] == 'trace']
    infs = [d for d in lines if d['type'] == 'infer']
    print(f'日志 {len(lines)} 行: trace {len(traces)} 帧, '
          f'infer 快照 {len(infs)}')

    # ① 归属链: 每个确认的新牌按时间排列
    chain = []  # (player, tile, confirm/mode)
    pre_calib = 0
    ignored_total = 0
    melds = []
    for d in traces:
        for n in d.get('new', []):
            if n.get('mode') == 'spatial':
                chain.append((n['player'], n['tile'], '空间'))
            else:
                chain.append((n['player'], n['tile'], '时序'))
        ignored_total += d.get('ignored', 0)
        for m in d.get('melds', []):
            melds.append((m['player'], m['kind'], m['tile']))
        pre_calib += d.get('pre_assign', 0)

    # 循环率: 与轮转期望对比(副露转移会重锚, 见 ④ 输出辅助人工核对)
    expect = None
    ok = bad = 0
    mismatches = []
    for i, (p, t, mode) in enumerate(chain):
        if expect is not None and p != expect:
            bad += 1
            if len(mismatches) < 8:
                mismatches.append((i, expect, p, t, mode))
        else:
            ok += 1
        expect = _next_of(p)
    print(f'\n① 归属链: {len(chain)} 张新牌')
    print(f'   轮转符合 {ok} / 不符 {bad} → 循环率 '
          f'{ok / max(1, ok + bad):.1%}')
    for i, exp, got, t, mode in mismatches:
        print(f'   #{i}: 期望 {_NAMES[exp]} 实际 {_NAMES[got]}({mode}) '
              f'tile={t}')

    # ② 四家终局分布
    if infs:
        last = infs[-1]
        print(f'\n② 终局四家牌河(共 {last["events_total"]} 事件):')
        for p in _ORDER:
            r = last.get('rivers', {}).get(p, [])
            print(f'   {_NAMES[p]:2}: {len(r):2d} 张')

    # ③ 观察期 + 回溯
    print(f'\n③ 开局回溯分配 {pre_calib} 张; 动画忽略 {ignored_total} 框')

    # ④ 副露
    print(f'\n④ 副露 {len(melds)} 个:')
    for p, kind, t in melds:
        print(f'   {_NAMES[p]} {kind} tile={t}')

    # ⑤ 可疑冲突: 连续新牌同家(且非空间)
    conflicts = 0
    prev_p = None
    for p, t, mode in chain:
        if mode == '时序' and prev_p == p:
            conflicts += 1
        if mode == '时序':
            prev_p = p
    print(f'\n⑤ 可疑冲突(连续同家时序归属): {conflicts} 次')

    # 总体判断
    rate = ok / max(1, ok + bad)
    verdict = ('归属链健康(≥95%)' if rate >= 0.95 else
               '归属链有偏移(<95%) — 把日志发我定位')
    print(f'\n结论: {verdict}')


if __name__ == '__main__':
    main()
