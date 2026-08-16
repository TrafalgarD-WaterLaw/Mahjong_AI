"""事件流日志复盘 — 四家牌河时间线/副露/归属置信度。

用法: python scripts/dump_log.py [log.jsonl]  # 默认最新一局
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mahjong_core.tile import tile_display  # noqa: E402

PLAYERS = ('my_river', 'right_river', 'top_river', 'left_river')
_NAMES = {'my_river': '我', 'right_river': '右家',
          'top_river': '对家', 'left_river': '左家'}


def parse_log(lines: list[dict[str, Any]]) -> dict[str, Any]:
    """日志行 → 聚合摘要。

    freeze 行 = 牌河事件的最终归属; meld_assign 行 = 副露归属
    (由 MeldFormed 行补全 kind/tile)。
    """
    rivers: dict[str, list[int]] = {p: [] for p in PLAYERS}
    melds: list[tuple[str, str, int]] = []
    entropies: list[float] = []
    meld_info: dict[int, tuple[str, int]] = {}
    vanishes: list[tuple[str, int, int]] = []
    for line in lines:
        t = line.get('type')
        if t == 'MeldFormed':
            tiles = line['tiles']
            if len(tiles) >= 4:
                kind = 'kong'
            elif len(set(tiles)) == 1:
                kind = 'pong'
            else:
                kind = 'chi'
            meld_info[line['eid']] = (kind, tiles[0])
        elif t == 'meld_assign':
            if line['eid'] in meld_info:
                kind, tile = meld_info[line['eid']]
                melds.append((line['player'], kind, tile))
        elif t == 'freeze' and line.get('player') in rivers:
            rivers[line['player']].append(line['tile'])
            entropies.append(line.get('entropy', 0.0))
        elif t == 'TileVanished':
            vanishes.append((line['river'], line['tile'],
                             line['appeared_eid']))
    return {'rivers': rivers, 'melds': melds, 'n_events': len(lines),
            'entropies': entropies, 'vanishes': vanishes}


def main() -> None:
    parser = argparse.ArgumentParser(description='事件流日志复盘')
    parser.add_argument('log', nargs='?', default=None,
                        help='game_*.jsonl; 默认 data/settle_logs 最新')
    args = parser.parse_args()
    if args.log:
        path = Path(args.log)
    else:
        files = sorted(Path('data/settle_logs').glob('game_*.jsonl'),
                       key=lambda p: p.stat().st_mtime)
        if not files:
            print('没有找到 game_*.jsonl — 先跑一局 run_assistant.py')
            sys.exit(1)
        path = files[-1]
    lines = [json.loads(line) for line in path.read_text(encoding='utf-8')
             .splitlines() if line.strip()]
    summary = parse_log(lines)
    print(f'== {path.name} ==')
    print(f'事件 {summary["n_events"]} 行')
    for p in PLAYERS:
        tiles = ' '.join(tile_display(t) for t in summary['rivers'][p])
        print(f'{_NAMES[p]}({len(summary["rivers"][p])}): {tiles}')
    for _p, kind, tile in summary['melds']:
        print(f'副露: {kind} {tile_display(tile)}')
    for river, tile, appeared_eid in summary['vanishes']:
        print(f'消失: {_NAMES.get(river, river)} {tile_display(tile)} '
              f'被拿走(appeared {appeared_eid})')
    if summary['entropies']:
        avg = sum(summary['entropies']) / len(summary['entropies'])
        print(f'冻结归属平均熵: {avg:.3f}(低=可信)')


if __name__ == '__main__':
    main()
