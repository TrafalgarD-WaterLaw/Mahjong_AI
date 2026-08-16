"""YOLO映射表的单元测试。"""
from src.mahjong_core.tile import (
    B1,
    B2,
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
    T9,
    W1,
    W2,
    W9,
    XI,
    XIA,
    ZHONG,
    ZHU,
)
from src.mahjong_cv.yolo_mapping import YOLO_CLASS_NAMES, YOLO_TO_OUR, yolo_to_tile


class TestYoloMapping:
    def test_all_42_mapped(self):
        assert len(YOLO_TO_OUR) == 42

    def test_class_names_count(self):
        assert len(YOLO_CLASS_NAMES) == 42

    def test_wan_mapping(self):
        assert yolo_to_tile(1) == W1
        assert yolo_to_tile(6) == W2
        assert yolo_to_tile(33) == W9

    def test_bing_mapping(self):
        assert yolo_to_tile(2) == B1
        assert yolo_to_tile(7) == B2
        assert yolo_to_tile(34) == B9

    def test_tiao_mapping(self):
        assert yolo_to_tile(0) == T1
        assert yolo_to_tile(5) == T2
        assert yolo_to_tile(32) == T9

    def test_feng_mapping(self):
        assert yolo_to_tile(35) == DONG
        assert yolo_to_tile(39) == NAN
        assert yolo_to_tile(41) == XI
        assert yolo_to_tile(37) == BEI

    def test_jian_mapping(self):
        assert yolo_to_tile(38) == ZHONG
        assert yolo_to_tile(36) == FA
        assert yolo_to_tile(40) == BAI

    def test_hua_and_season_mapping(self):
        assert yolo_to_tile(3) == MEI
        assert yolo_to_tile(8) == LAN
        assert yolo_to_tile(13) == ZHU
        assert yolo_to_tile(18) == JU
        assert yolo_to_tile(4) == CHUN
        assert yolo_to_tile(9) == XIA
        assert yolo_to_tile(14) == QIU
        assert yolo_to_tile(19) == DONG_HUA

    def test_no_duplicates(self):
        # 花牌 8 种 YOLO 类统一映射到 34(合并), 允许重复; 非花类不应重复
        values = list(YOLO_TO_OUR.values())
        dup = [v for v in set(values) if values.count(v) > 1]
        assert dup == [34], f'非花类不应重复: {dup}'
