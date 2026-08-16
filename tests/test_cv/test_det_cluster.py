"""det_cluster 单元测试 — DBSCAN 聚类归属(手牌/副露/牌河/噪声)。"""

from src.mahjong_cv.det_cluster import cluster_dets
from src.mahjong_cv.detections import TileDet


def hand_det(tile: int, x: int, conf: float = 0.9) -> TileDet:
    """手牌框: y=270-300(中心 285), 30x30。"""
    return TileDet(tile, float(x), 270.0, float(x + 30), 300.0, conf)


def river_det(tile: int, cx: float, cy: float) -> TileDet:
    """牌河小框: 15x15。"""
    return TileDet(tile, cx - 7.5, cy - 7.5, cx + 7.5, cy + 7.5, 0.9)


class TestHandClassification:
    def test_hand13_forms_single_cluster(self):
        # 手牌 13 张(间距 40 < eps 44) → 连通成一簇, 分类为手牌
        hand13 = [hand_det(t, 50 + i * 40) for i, t in enumerate(range(13))]
        hand, melds, rivers = cluster_dets(hand13)
        assert len(hand) == 13
        assert not melds and not rivers

    def test_broken_hand_merged_by_eps(self):
        # 手牌漏检断开(5+6, 间距仍 < eps) → DBSCAN 连通, 不断组
        hand_broken = [hand_det(t, 50 + i * 40)
                       for i, t in enumerate([*range(5), *range(5, 11)])]
        hand, _m, _r = cluster_dets(hand_broken)
        assert len(hand) == 11


class TestMeldClassification:
    def test_independent_meld_detected(self):
        # 独立副露(间隙 > eps): 3 张紧凑 + 尺寸≈手牌 → 副露
        hand11 = [hand_det(t, 50 + i * 40) for i, t in enumerate(range(11))]
        meld = [hand_det(31, 600 + i * 8) for i in range(3)]  # 中中中
        hand, melds, _r = cluster_dets(hand11 + meld)
        assert len(hand) == 11
        assert len(melds) == 1 and len(melds[0]) == 3

    def test_adjacent_meld_may_fuse(self):
        # 紧贴副露(间隙 10 < eps): 与手牌连成一簇(已知几何限制, 不崩溃)
        hand13 = [hand_det(t, 50 + i * 40) for i, t in enumerate(range(13))]
        meld = [hand_det(31, 540 + i * 10) for i in range(3)]
        hand, melds, _r = cluster_dets(hand13 + meld)
        assert len(hand) >= 13  # 手牌部分保留(可能含副露, 物理不可分)


class TestRiverAndNoise:
    def test_river_cluster_separated(self):
        # 牌河小牌紧凑堆(尺寸小, 不满足手牌/副露尺寸锚) → 牌河簇
        hand13 = [hand_det(t, 50 + i * 40) for i, t in enumerate(range(13))]
        river = [river_det(i, 100 + i * 16, 100) for i in range(5)]
        hand, melds, rivers = cluster_dets(hand13 + river)
        assert len(hand) == 13
        assert len(rivers) == 1 and len(rivers[0]) == 5

    def test_single_box_is_river(self):
        # 孤立单点(离手牌远, 超副露近距阈值) → 不是副露, 作为牌河单张
        hand13 = [hand_det(t, 50 + i * 40) for i, t in enumerate(range(13))]
        stray = hand_det(0, 900, conf=0.9)  # 距手牌 bbox 外缘 ~340 > 5×30
        hand, melds, rivers = cluster_dets(hand13 + [stray])
        assert len(hand) == 13
        assert not melds
        assert any(len(c) == 1 for c in rivers)  # 单张并入牌河(未成簇)


class TestConfFilter:
    def test_low_conf_excluded_by_default(self):
        hand11 = [hand_det(t, 50 + i * 40) for i, t in enumerate(range(11))]
        noise = hand_det(0, 200, conf=0.1)
        hand, _m, _r = cluster_dets(hand11 + [noise])
        assert len(hand) == 11

    def test_conf_min_zero_keeps_all(self):
        hand11 = [hand_det(t, 50 + i * 40) for i, t in enumerate(range(11))]
        noise = hand_det(0, 200, conf=0.1)
        hand, _m, _r = cluster_dets(hand11 + [noise], conf_min=0.0)
        assert len(hand) == 12  # 低置信度框并入(距手牌 150 < eps? 50+10*40=450, 200 距 250 > 44 → 噪声!)
