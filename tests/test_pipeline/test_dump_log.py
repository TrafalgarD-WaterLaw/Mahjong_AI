"""复盘脚本单测 — 日志解析与聚合。"""

from scripts.dump_log import parse_log

_SAMPLE = [
    {'type': 'TileAppeared', 'eid': 1, 'tile': 5, 'track_id': 101,
     'cx': 700.0, 'cy': 740.0, 'conf': 0.8, 'frame': 0, 'ts': 0.0,
     'motion': None},
    {'type': 'freeze', 'eid': 1, 'tile': 5, 'player': 'my_river',
     'entropy': 0.1, 'logits': {}, 'ts': 0.1},
    {'type': 'MeldFormed', 'eid': 2, 'tiles': [8, 8, 8], 'cx': 1150.0,
     'cy': 550.0, 'bbox': [1100.0, 500.0, 1200.0, 600.0],
     'frame': 2, 'ts': 0.2},
    {'type': 'meld_assign', 'eid': 2, 'player': 'right_river', 'ts': 0.3},
]


def test_parse_log_aggregates_freezes():
    summary = parse_log(_SAMPLE)
    assert summary['rivers']['my_river'] == [5]
    assert summary['melds'] == [('right_river', 'pong', 8)]
    assert summary['n_events'] == 4
