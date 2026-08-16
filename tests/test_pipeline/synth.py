"""合成对局生成器 — 底牌剧本 + 噪声(漏检/位置抖动/碰截断)。

生成"谁在什么帧打了什么"的事件序列 + 真值标注, 供解码器端到端
回归(每次调窗口 K/束宽/权重都有硬指标)。
"""

from __future__ import annotations

import random

from src.mahjong_ai.pipeline.decoder import EvidenceModel
from src.mahjong_ai.pipeline.events import (
    HandChanged,
    MeldFormed,
    TileAppeared,
    TileClaimed,
)

_ROTATION = ('my_river', 'right_river', 'top_river', 'left_river')

_SEED = {
    'my_river': (600.0, 700.0, 800.0, 780.0),
    'right_river': (1100.0, 500.0, 1180.0, 600.0),
    'top_river': (500.0, 100.0, 700.0, 180.0),
    'left_river': (100.0, 500.0, 180.0, 600.0),
}
_MELD_SEED = {
    'my_river': (820.0, 700.0, 920.0, 780.0),
    'right_river': (1100.0, 620.0, 1200.0, 700.0),
    'top_river': (720.0, 100.0, 820.0, 180.0),
    'left_river': (80.0, 500.0, 160.0, 600.0),
}


def synth_game(seed: int = 0, miss_every: int = 9) -> list[tuple[str, object]]:
    """生成一局事件序列: [(ground_truth_player, event)]。

    剧本: 三轮正常轮转(12 张) + 第 2 轮右家碰我的第 2 张(截断, 右家
    接着打)。每 miss_every 张漏检一张(不生成事件); 落点加 ±15px 抖动。
    事件按帧序排列; 我的打出前后插 HandChanged(13→14 / 14→13)。
    """
    rng = random.Random(seed)
    seq: list[tuple[str, object]] = []
    eid = 0
    frame = 0

    def hand_draw() -> tuple[str, object]:
        """我摸牌(13→14): 相位硬锚 — 在打出事件之前。"""
        nonlocal eid
        eid += 1
        return ('my_river', HandChanged(eid=eid, n_old=13, n_new=14,
                                        frame=frame, ts=frame * 0.1))

    def hand_disc() -> tuple[str, object]:
        """我打出(14→13): 在打出事件之后。"""
        nonlocal eid
        eid += 1
        return ('my_river', HandChanged(eid=eid, n_old=14, n_new=13,
                                        frame=frame, ts=frame * 0.1))

    def appeared(player: str, tile: int) -> tuple[str, object]:
        nonlocal eid
        eid += 1
        x1, y1, x2, y2 = _SEED[player]
        cx = (x1 + x2) / 2 + rng.uniform(-15, 15)
        cy = (y1 + y2) / 2 + rng.uniform(-15, 15)
        return (player, TileAppeared(
            eid=eid, tile=tile, track_id=5000 + eid, cx=cx, cy=cy,
            conf=0.8, frame=frame, ts=frame * 0.1))

    played = 0
    order = list(_ROTATION) * 3      # 12 张(第 2 轮会被碰截断重排)
    skip_next = False
    for i, p in enumerate(order):
        if skip_next:
            skip_next = False        # 碰后右家已打, 跳过原计划该张
            continue
        if p == 'my_river':
            seq.append(hand_draw())  # 摸牌锚在打出之前
        frame += 1
        if (i + 1) % miss_every == 0:
            continue  # 漏检: 该打出没有事件
        ev = appeared(p, tile=(played % 30))
        seq.append(ev)
        if p == 'my_river':
            seq.append(hand_disc())  # 打出锚在打出之后
        played += 1
        # 我第 2 轮的打出(i==4)被右家碰 → 副露 + 被碰 + 右家接着打
        if i == 4:
            eid += 1
            meld_eid = eid
            mx1, my1, mx2, my2 = _MELD_SEED['right_river']
            seq.append(('right_river', MeldFormed(
                eid=meld_eid, tiles=(ev[1].tile,) * 3,
                cx=(mx1 + mx2) / 2, cy=(my1 + my2) / 2,
                bbox=(mx1, my1, mx2, my2), frame=frame, ts=frame * 0.1)))
            eid += 1
            seq.append(('right_river', TileClaimed(
                eid=eid, claimed=ev[1].eid, meld=meld_eid,
                frame=frame, ts=frame * 0.1)))
            frame += 1
            eid += 1
            seq.append(('right_river', TileAppeared(
                eid=eid, tile=(played % 30), track_id=5000 + eid,
                cx=(_SEED['right_river'][0] + _SEED['right_river'][2]) / 2,
                cy=(_SEED['right_river'][1] + _SEED['right_river'][3]) / 2,
                conf=0.8, frame=frame, ts=frame * 0.1)))
            played += 1
            skip_next = True
    return seq


def make_decoder(k: int = 10, beam: int = 16):
    """构造种好子的解码器(牌河种子 + 副露种子)。"""
    river_ev = EvidenceModel()
    river_ev.reseed(_SEED, 1280, 800)
    meld_ev = EvidenceModel()
    meld_ev.reseed(_MELD_SEED, 1280, 800)
    from src.mahjong_ai.pipeline.decoder import WindowDecoder

    return WindowDecoder(river_ev, meld_ev, k=k, beam=beam)
