"""牌河消失检测 — 各河最新落定牌连续缺失 → TileVanished(被碰/吃)。

只盯每河最近一次落定的 TileAppeared: 游戏里唯一会让它消失的
机制是被碰/吃拿走(该河最新一张, 不可能被别家覆盖)。整河检测
塌(抽风)不算消失 — "同河其他牌还在"才计数, 挡住假消失。
"""

from __future__ import annotations

from collections.abc import Callable

from src.mahjong_ai.pipeline.events import TileVanished
from src.mahjong_cv.detections import TileDet

#: 连续缺失帧数(≈2s @7fps — 被碰动画 1s + 检测抖动余量)
_VANISH_FRAMES = 15
#: 在场判定半径(像素 — 落定牌位置抖动 ±5px, 60 足够宽)
_PRESENT_RADIUS = 60.0
#: 区域判定外扩(与提取器区域门禁同口径)
_REGION_MARGIN = 20.0
#: 盯防牌最低置信: 低于此的"消失"不发事件 — 低置信误检的消失
#: 是误检自己淡出, 不是被碰(真实对局: 二万 conf 0.35 淡出 →
#: 对家假副露 → 听牌归零); 真牌落定后通常 ≥0.6。
_CLAIM_MIN_CONF = 0.5


class RiverWatcher:
    """逐河盯最新牌; 缺失 ≥_VANISH_FRAMES 且河还活着 → TileVanished。"""

    def __init__(self) -> None:
        self._regions: dict[str, tuple[float, float, float, float]] = {}
        #: river -> {'eid', 'tile', 'cx', 'cy', 'conf', 'missing'}
        self._newest: dict[str, dict[str, float | int]] = {}

    def set_regions(
        self, regions: dict[str, tuple[float, float, float, float]],
    ) -> None:
        """注入牌河区域(像素, 运行器重映射后随 set_regions 一起调用)。"""
        self._regions = dict(regions)
        self._newest = {k: v for k, v in self._newest.items()
                        if k in self._regions}

    def on_appeared(self, river: str, eid: int, tile: int,
                    cx: float, cy: float, conf: float = 1.0) -> None:
        """该河最新落定牌(提取器在 TileAppeared 时调用, 替换盯防对象)。

        conf = 落定时刻检测置信度(盯防期取最大值 — 单帧低不是问题,
        一直低 = 误检, 消失不发事件)。
        """
        self._newest[river] = {'eid': eid, 'tile': tile, 'cx': cx,
                               'cy': cy, 'conf': conf, 'missing': 0}

    def tick(self, dets: list[TileDet], frame: int, ts: float,
             next_eid: Callable[[], int]) -> list[TileVanished]:
        """逐帧: 在场清零/缺失累计; 达到阈值且同河其他牌在场 → 消失。"""
        out: list[TileVanished] = []
        for river, nw in list(self._newest.items()):
            tile = int(nw['tile'])
            cx, cy = float(nw['cx']), float(nw['cy'])
            region = self._regions[river]
            best: TileDet | None = None
            for d in dets:
                if d.tile == tile and (
                        (d.x1 + d.x2) / 2 - cx) ** 2 \
                        + ((d.y1 + d.y2) / 2 - cy) ** 2 \
                        < _PRESENT_RADIUS ** 2:
                    best = d
                    break
            if best is not None:
                nw['missing'] = 0
                nw['conf'] = max(float(nw['conf']), best.conf)
                continue
            alive = any(
                rx1 - _REGION_MARGIN <= (d.x1 + d.x2) / 2
                <= rx2 + _REGION_MARGIN
                and ry1 - _REGION_MARGIN <= (d.y1 + d.y2) / 2
                <= ry2 + _REGION_MARGIN
                for d in dets
                for rx1, ry1, rx2, ry2 in (region,))
            if not alive:
                continue  # 整河塌: 不计数(抽风假消失)
            nw['missing'] = int(nw['missing']) + 1
            if nw['missing'] >= _VANISH_FRAMES:
                if float(nw['conf']) < _CLAIM_MIN_CONF:
                    del self._newest[river]  # 低置信误检淡出: 不盯也不发
                    continue
                out.append(TileVanished(
                    eid=next_eid(), tile=tile, river=river,
                    appeared_eid=int(nw['eid']), frame=frame, ts=ts))
                del self._newest[river]
        return out
