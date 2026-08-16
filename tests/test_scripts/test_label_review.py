"""人工审核打标工具的单元测试(纯函数部分)。"""
import tempfile
from pathlib import Path

import cv2
import numpy as np

from scripts.label_review import (
    _dedup_overlap,
    _load_review_labels,
    _match_inherited,
    _shot_ts,
    hit_test,
    resolve_tile,
    reviewed_exists,
)
from src.mahjong_core.tile import BAI, DONG, FA, MEI, W1, W5, ZHONG


class TestHitTest:
    def test_inside_box(self):
        assert hit_test(10, 10, (0, 0, 20, 20)) is True

    def test_outside_box(self):
        assert hit_test(50, 50, (0, 0, 20, 20)) is False

    def test_tolerance_around_box(self):
        # 容差 _CLICK_TOL=6: 框外 5px 也算命中
        assert hit_test(-4, 5, (0, 0, 20, 20)) is True
        assert hit_test(26, 5, (0, 0, 20, 20)) is True

    def test_far_outside(self):
        assert hit_test(-10, 5, (0, 0, 20, 20)) is False


class TestResolveTile:
    def test_wan(self):
        assert resolve_tile('1', '1') == W1
        assert resolve_tile('1', '5') == W5

    def test_feng_jian_hua(self):
        assert resolve_tile('4', '1') == DONG
        assert resolve_tile('5', '1') == ZHONG
        assert resolve_tile('5', '2') == FA
        assert resolve_tile('5', '3') == BAI
        assert resolve_tile('6', '1') == MEI

    def test_out_of_range(self):
        assert resolve_tile('4', '5') is None   # 风只有4个
        assert resolve_tile('5', '4') is None   # 箭只有3个
        assert resolve_tile('6', '9') is None   # 花只有8个
        assert resolve_tile('7', '1') is None   # 无7花色
        assert resolve_tile('1', '0') is None   # 无0


class TestReviewedExists:
    def test_reviewed_found(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            (out / 'images').mkdir(parents=True)
            (out / 'images' / 'review_00005_shot_00042.jpg').touch()
            assert reviewed_exists(out, 'shot_00042') is True

    def test_not_reviewed(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            (out / 'images').mkdir(parents=True)
            assert reviewed_exists(out, 'shot_00043') is False


def _img(value: int) -> np.ndarray:
    """单色图像: 0=黑, 255=白 — 同色 → 牌面一致, 异色 → 差异巨大。"""
    return np.full((400, 400, 3), value, dtype=np.uint8)


def _img_patch(base: int, patch: tuple[int, int, int, int, int]) -> np.ndarray:
    """底色 base + 一块异色区域(点数模拟): (x1,y1,x2,y2,色值)。"""
    img = np.full((400, 400, 3), base, dtype=np.uint8)
    x1, y1, x2, y2, val = patch
    img[y1:y2, x1:x2] = val
    return img


class TestMatchInherited:
    def test_matched_same_face_inherits(self):
        # 位置匹配 + 牌面一致(黑↔黑) → 继承用户修正标签
        inherited = [(35, 0, 0, 100, 100, 1.0)]
        preds = [(0, 0, 0, 100, 100, 0.9)]
        merged, n, kept = _match_inherited(inherited, preds,
                                           _img(0), _img(0))
        assert n == 1 and kept == 0
        assert len(merged) == 1
        assert merged[0][0] == 35  # 继承的牌标签优先
        assert merged[0][5] == 1.0
        assert merged[0][6] is True

    def test_matched_face_changed_uses_model(self):
        # 位置匹配但牌面变了(黑↔白, 摸入新牌占位) → 改用模型标签
        inherited = [(35, 0, 0, 100, 100, 1.0)]
        preds = [(0, 0, 0, 100, 100, 0.9)]
        merged, n, kept = _match_inherited(inherited, preds,
                                           _img(0), _img(255))
        assert n == 0 and kept == 0
        assert len(merged) == 1
        assert merged[0][0] == 0   # 模型标签
        assert merged[0][6] is False

    def test_unmatched_same_face_kept(self):
        # 无匹配但牌面一致(模型持续漏检, 牌还在) → 保留原坐标
        inherited = [(35, 300, 300, 400, 400, 1.0)]
        preds = [(0, 0, 0, 100, 100, 0.9)]
        merged, n, kept = _match_inherited(inherited, preds,
                                           _img(0), _img(0))
        assert n == 0 and kept == 1
        assert len(merged) == 2
        kept_box = [m for m in merged if m[6]][0]
        assert kept_box[0] == 35
        assert kept_box[1:5] == (300, 300, 400, 400)  # 原坐标保留

    def test_unmatched_face_changed_dropped(self):
        # 无匹配且牌面变了(牌已打出, 位置空了) → 丢弃
        inherited = [(35, 300, 300, 400, 400, 1.0)]
        preds = [(0, 0, 0, 100, 100, 0.9)]
        merged, n, kept = _match_inherited(inherited, preds,
                                           _img(0), _img(255))
        assert n == 0 and kept == 0
        assert len(merged) == 1
        assert merged[0][0] == 0
        assert merged[0][6] is False

    def test_unmatched_face_changed_dropped_everywhere(self):
        # 无布局区分: 任何区域牌面变了都不继承(布局文件失准 → 全区域统一验证)
        inherited = [(35, 0, 0, 100, 100, 1.0)]
        preds = [(0, 500, 500, 600, 600, 0.9)]
        merged, n, kept = _match_inherited(inherited, preds,
                                           _img(0), _img(255))
        assert n == 0 and kept == 0
        assert len(merged) == 1
        assert merged[0][0] == 0   # 模型标签
        assert merged[0][6] is False

    def test_no_double_use_of_prediction(self):
        # 两个继承框抢同一个预测 → 只能匹配一次; 败者无匹配但牌面一致 → 保留
        inherited = [(35, 0, 0, 100, 100, 1.0), (36, 10, 10, 110, 110, 1.0)]
        preds = [(0, 5, 5, 105, 105, 0.9)]
        merged, n, kept = _match_inherited(inherited, preds,
                                           _img(0), _img(0))
        assert n == 1 and kept == 1
        assert len(merged) == 2

    def test_similar_face_with_shifted_pips_rejected(self):
        # 长得像但点数位置不同的牌(七万 vs 九万模拟): 灰度均值差小,
        # 但分块差异显著 → 拒绝继承(分块验证抓住结构性差异)
        prev = _img_patch(200, (20, 20, 60, 60, 0))     # 点数在左上
        cur = _img_patch(200, (60, 60, 100, 100, 0))    # 点数在右下
        inherited = [(35, 0, 0, 100, 100, 1.0)]
        preds = [(0, 0, 0, 100, 100, 0.9)]
        merged, n, kept = _match_inherited(inherited, preds, prev, cur)
        assert n == 0
        assert merged[0][0] == 0  # 改用模型标签, 不继承


class TestDedupOverlap:
    def test_inherited_beats_model(self):
        # 双框: 继承框(标签人工确认过)与模型框重叠 → 保留继承
        dets = [(35, 0, 0, 100, 100, 1.0), (0, 5, 5, 105, 105, 0.9)]
        inherited = [True, False]
        out, out_ih = _dedup_overlap(dets, inherited)
        assert len(out) == 1
        assert out[0][0] == 35 and out_ih == [True]

    def test_manual_beats_inherited(self):
        # 人工添加框(conf=1.0 且非继承)优先
        dets = [(35, 0, 0, 100, 100, 1.0), (36, 5, 5, 105, 105, 1.0)]
        inherited = [True, False]
        out, out_ih = _dedup_overlap(dets, inherited)
        assert len(out) == 1
        assert out[0][0] == 36 and out_ih == [False]

    def test_far_boxes_untouched(self):
        dets = [(35, 0, 0, 100, 100, 1.0), (0, 300, 300, 400, 400, 0.9)]
        inherited = [True, False]
        out, out_ih = _dedup_overlap(dets, inherited)
        assert len(out) == 2


class TestShotTs:
    def test_timestamp_first(self):
        # 新格式时间戳排序在旧格式序号之前(时间序)
        assert _shot_ts('shot_20260808200541') < _shot_ts('shot_000042')

    def test_newer_timestamp_later(self):
        assert _shot_ts('shot_20260809010101') > _shot_ts('shot_20260808010101')


class TestLoadReviewLabels:
    def test_load_and_convert_to_pixels(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / 'out'
            (out / 'labels').mkdir(parents=True)
            (out / 'labels' / 'review_00001_shot_00099.txt').write_text(
                '0 0.25 0.5 0.2 0.2\n', encoding='utf-8')
            prev = Path(td) / 'shot_00099.png'
            cv2.imwrite(str(prev), np.zeros((100, 200, 3), dtype=np.uint8))
            loaded = _load_review_labels(out, 'shot_00099', prev)
            assert loaded is not None
            dets, img = loaded
            assert len(dets) == 1
            assert img.shape == (100, 200, 3)  # 牌面验证用的上一帧图像
            tile, x1, y1, x2, y2, conf = dets[0]
            assert tile == 0 and conf == 1.0
            assert (x1, y1, x2, y2) == (30.0, 40.0, 70.0, 60.0)

    def test_missing_label_returns_none(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / 'out'
            (out / 'labels').mkdir(parents=True)
            prev = Path(td) / 'shot_00420.png'
            cv2.imwrite(str(prev), np.zeros((100, 100, 3), dtype=np.uint8))
            assert _load_review_labels(out, 'shot_00420', prev) is None



