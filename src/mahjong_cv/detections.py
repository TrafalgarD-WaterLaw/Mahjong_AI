"""检测结果数据类 — 屏幕视觉/模板匹配/状态机共用。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TileDet:
    """一张牌的检测结果。

    tile: 牌编码 0-33(与 mahjong_core.tile 一致)
    x1/y1/x2/y2: 像素坐标框
    conf: 置信度 0-1
    track_id: 官方跟踪器(ByteTrack)跨帧维持的目标 ID; 纯检测 None
    """

    tile: int
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    track_id: int | None = None
