"""麻将牌常量定义与辅助函数。

牌使用整数编码，直接对应 ML 模型的分类输出:
    0-8:   万牌 (W1-W9)
    9-17:  饼牌 (B1-B9)
    18-26: 条牌 (T1-T9)
    27-30: 风牌 (东南西北)
    31-33: 箭牌 (中发白)
    34:    花牌 (梅兰竹菊春夏秋冬 8 种统一为 1 类 — 检测不需要区分,
           样本少的类别合并降低类别不平衡; 2026-08-09 决定)
"""


TOTAL_TILES = 35

# 花色常量
WAN = 0
BING = 1
TIAO = 2
FENG = 3
JIAN = 4
HUA = 5

# 具体牌常量
W1, W2, W3, W4, W5, W6, W7, W8, W9 = 0, 1, 2, 3, 4, 5, 6, 7, 8
B1, B2, B3, B4, B5, B6, B7, B8, B9 = 9, 10, 11, 12, 13, 14, 15, 16, 17
T1, T2, T3, T4, T5, T6, T7, T8, T9 = 18, 19, 20, 21, 22, 23, 24, 25, 26
DONG, NAN, XI, BEI = 27, 28, 29, 30
ZHONG, FA, BAI = 31, 32, 33
# 花牌统一为 1 类(35-41 的历史编号映射到 34, 实物模型 8 种花同理)
MEI = LAN = ZHU = JU = CHUN = XIA = QIU = DONG_HUA = 34

# 牌组范围
WAN_TILES = range(0, 9)
BING_TILES = range(9, 18)
TIAO_TILES = range(18, 27)
FENG_TILES = range(27, 31)
JIAN_TILES = range(31, 34)
HUA_TILES = range(34, 35)

_NUMBER_NAMES = ["一", "二", "三", "四", "五", "六", "七", "八", "九"]
_HONOR_NAMES = {27: "东", 28: "南", 29: "西", 30: "北", 31: "中", 32: "发", 33: "白"}
_HUA_NAMES = {34: "花"}
#: 历史花牌名(模板文件名等遗留)全部映射到统一花牌类 34
_HUA_LEGACY = {"梅": 34, "兰": 34, "竹": 34, "菊": 34,
               "春": 34, "夏": 34, "秋": 34, "冬": 34}


def tile_suit(tile: int) -> int:
    """返回牌的花色常量。"""
    if tile in WAN_TILES:
        return WAN
    if tile in BING_TILES:
        return BING
    if tile in TIAO_TILES:
        return TIAO
    if tile in FENG_TILES:
        return FENG
    if tile in JIAN_TILES:
        return JIAN
    if tile in HUA_TILES:
        return HUA
    raise ValueError(f"非法牌值: {tile}")


def tile_number(tile: int) -> int | None:
    """返回数牌的数字(1-9)，字牌和花牌返回 None。"""
    if tile in WAN_TILES:
        return (tile - 0) + 1
    if tile in BING_TILES:
        return (tile - 9) + 1
    if tile in TIAO_TILES:
        return (tile - 18) + 1
    if tile in range(27, TOTAL_TILES):
        return None
    raise ValueError(f"非法牌值: {tile}")


def tile_display(tile: int) -> str:
    """返回牌的中文显示名(花牌统一为"花")。"""
    if 0 <= tile <= 8:
        return _NUMBER_NAMES[tile - 0] + "万"
    if 9 <= tile <= 17:
        return _NUMBER_NAMES[tile - 9] + "饼"
    if 18 <= tile <= 26:
        return _NUMBER_NAMES[tile - 18] + "条"
    if 27 <= tile <= 33:
        return _HONOR_NAMES[tile]
    if tile == 34:
        return _HUA_NAMES[tile]
    raise ValueError(f"非法牌值: {tile}")


def tile_from_str(s: str) -> int:
    """从中文显示名解析牌整数(含花牌"花"与历史花名)。"""
    if s in _HUA_LEGACY:
        return _HUA_LEGACY[s]
    for t in range(0, TOTAL_TILES):
        if tile_display(t) == s:
            return t
    raise ValueError(f"无法识别的牌名: '{s}'")
