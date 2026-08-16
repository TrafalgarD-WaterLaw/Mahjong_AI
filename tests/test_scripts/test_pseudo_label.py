"""半监督伪标注工具的单元测试(纯函数部分)。"""
import tempfile
from pathlib import Path

from src.mahjong_cv.det_cluster import cluster_dets
from src.mahjong_cv.detections import TileDet


def _det(i: int, cx: float, cy: float, w: float = 50.0, h: float = 70.0,
         conf: float = 0.9) -> TileDet:
    """以中心点构造检测框(像素坐标)。"""
    return TileDet(i, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, conf)


def _hand_row(n: int, y: float = 900.0) -> list[TileDet]:
    """n 张手牌单行(中心距 60px, 底部)。"""
    return [_det(i, 100 + i * 60, y) for i in range(n)]


def _river_row(n: int, y: float = 550.0, x0: float = 200.0,
               gap: float = 40.0) -> list[TileDet]:
    """n 张牌河横排一行(中心距 gap, 紧凑)。"""
    return [_det(i + 30, x0 + i * gap, y, w=35.0, h=50.0) for i in range(n)]


class TestClusterDets:
    def test_hand_bottom_row_is_hand(self):
        dets = _hand_row(13)
        hand, _melds, rivers = cluster_dets(dets)
        assert len(hand) == 13
        assert rivers == []

    def test_hand_plus_own_river_two_rows(self):
        # 手牌 + 自己牌河 2 行(行距 90px)
        dets = _hand_row(13) + _river_row(6, y=700) + _river_row(6, y=790)
        hand, _melds, rivers = cluster_dets(dets)
        assert len(hand) == 13
        assert len(rivers) >= 1          # 2 行可能合并 1 簇或分 2 簇, 但都是牌河
        river_n = sum(len(r) for r in rivers)
        assert river_n == 12

    def test_side_river_vertical_column(self):
        # 左家竖列 5 张(x 固定, y 递增, 列距 60px)
        col = [_det(i + 30, 150, 300 + i * 60, w=35.0, h=50.0) for i in range(5)]
        dets = _hand_row(13) + col
        hand, _melds, rivers = cluster_dets(dets)
        assert len(hand) == 13
        assert sum(len(r) for r in rivers) == 5

    def test_isolated_single_box_is_other(self):
        dets = _hand_row(13) + [_det(35, 700, 400, w=35, h=50)]  # 桌面中央孤牌
        hand, _melds, rivers = cluster_dets(dets)
        assert len(hand) == 13
        assert any(len(r) == 1 for r in rivers)  # 单张并入牌河

    def test_empty(self):
        assert cluster_dets([]) == ([], [], [])

    def test_hand_missing_tiles_still_hand(self):
        # 手牌漏检剩 8 张, 仍应识别为手牌
        dets = _hand_row(8)
        hand, _m, _r = cluster_dets(dets)
        assert len(hand) == 8

    def test_meld_group_in_hand_band_excluded(self):
        # 副露 3 张与手牌同带(y 差 < 带容差)但分组独立(间隙 > 分组阈值)
        # → 单独返回为副露组, 不进手牌
        # 副露 = 亮出的牌, 与手牌同尺寸(50x70), 紧贴(间距 20)
        meld = [_det(i + 30, 50 + i * 20, 890, w=50.0, h=70.0) for i in range(3)]
        hand = [_det(t, 300 + i * 60, 920) for i, t in enumerate(range(13))]
        dets = hand + meld
        hand_out, melds, rivers = cluster_dets(dets)
        assert len(hand_out) == 13
        assert sum(len(m) for m in melds) == 3   # 副露组
        assert rivers == []

    def test_no_hand_only_rivers(self):
        # 无手牌帧: 竖列牌河(不满足水平线性) → 手牌为空
        col = [_det(i + 30, 150, 300 + i * 60, w=35.0, h=50.0) for i in range(5)]
        col2 = [_det(i + 40, 700, 300 + i * 60, w=35.0, h=50.0) for i in range(5)]
        dets = col + col2
        hand, _melds, rivers = cluster_dets(dets)
        assert hand == []
        assert sum(len(r) for r in rivers) == 10


from scripts.pseudo_label import pseudo_exists  # noqa: E402


class TestPseudoExists:
    def test_found_in_subdir(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            (out / 'images' / 'train').mkdir(parents=True)
            (out / 'images' / 'train' / 'pseudo_00003_shot_00042.jpg').touch()
            assert pseudo_exists(out, 'shot_00042') is True

    def test_not_found(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            (out / 'images').mkdir(parents=True)
            assert pseudo_exists(out, 'shot_00043') is False
