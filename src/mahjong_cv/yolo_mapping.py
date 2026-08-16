"""YOLO 类序 → 项目牌编码映射(实物麻将数据集 Mahjong.v83i.yolov8)。

实物数据集 42 类(含 8 花): 类序 = 每数字一组 [条, 万, 饼, 花, 季]
(1-4 有花季, 5-9 只有条万饼), 最后 7 类 = 东南西北中发白。
项目牌编码: 0-8 万, 9-17 饼, 18-26 条, 27 东 28 南 29 西 30 北,
31 中 32 发 33 白, 34 花(8 种花合并为 34)。
"""

from __future__ import annotations

from src.mahjong_core.tile import (
    B1,
    B2,
    B3,
    B4,
    B5,
    B6,
    B7,
    B8,
    B9,
    BAI,
    BEI,
    CHUN,
    DONG,
    DONG_HUA,
    FA,
    JU,
    LAN,
    MEI,
    NAN,
    QIU,
    T1,
    T2,
    T3,
    T4,
    T5,
    T6,
    T7,
    T8,
    T9,
    W1,
    W2,
    W3,
    W4,
    W5,
    W6,
    W7,
    W8,
    W9,
    XI,
    XIA,
    ZHONG,
    ZHU,
)

#: YOLO 类序(42 类) → 项目牌编码。类序(实物数据集 Mahjong.v83i.yolov8):
#: [1条 1万 1饼 1花 1季, 2条 2万 2饼 2花 2季, ... 4 组,
#:  5-9 组只有条万饼, 最后 东南西北中发白]。
#: dict: 越界类 id 由 KeyError 表达非法(调用方捕获跳过)。
YOLO_TO_OUR: dict[int, int] = dict(enumerate([
    T1, W1, B1, MEI, CHUN,
    T2, W2, B2, LAN, XIA,
    T3, W3, B3, ZHU, QIU,
    T4, W4, B4, JU, DONG_HUA,
    T5, W5, B5,
    T6, W6, B6,
    T7, W7, B7,
    T8, W8, B8,
    T9, W9, B9,
    DONG, FA, BEI, ZHONG, NAN, BAI, XI,
]))

#: YOLO 类名(与实物数据集 data.yaml 一致; 1B=一条, 1C=一万,
#: 1D=一饼, 1F=花 1, 1S=季 1)
YOLO_CLASS_NAMES: list[str] = [
    *(f'{n}{s}' for n in range(1, 5) for s in ('B', 'C', 'D', 'F', 'S')),
    *(f'{n}{s}' for n in range(5, 10) for s in ('B', 'C', 'D')),
    'EW', 'GD', 'NW', 'RD', 'SW', 'WD', 'WW',
]


def yolo_to_tile(yolo_id: int) -> int:
    """YOLO 类 id(0-41) → 项目牌编码(0-34; 8 种花合并为 34)。"""
    return YOLO_TO_OUR[yolo_id]
