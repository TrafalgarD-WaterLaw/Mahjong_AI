"""ScreenVision 单元测试(注入假检测器, 不加载真实模型)。"""
from types import SimpleNamespace

import numpy as np
import pytest

from src.mahjong_cv.detections import TileDet
from src.mahjong_cv.screen_vision import ScreenVision, merge_duplicate_boxes


class TestMergeDuplicateBoxes:
    def test_same_tile_overlap_merged(self):
        # 同一张牌被拆成两框(1/3 空白 + 2/3 牌面), IoU > 0.35 → 只留高置信度
        dets = [
            TileDet(0, 0, 0, 100, 100, 0.9),   # 高置信度, 保留
            TileDet(0, 40, 0, 140, 100, 0.7),  # 与上一框重叠 60%(IoU 0.43)
        ]
        out = merge_duplicate_boxes(dets)
        assert len(out) == 1
        assert out[0].conf == 0.9

    def test_different_tiles_kept(self):
        # 不同类别(相邻真牌)不合并
        dets = [
            TileDet(0, 0, 0, 100, 100, 0.9),
            TileDet(1, 60, 0, 160, 100, 0.9),
        ]
        assert len(merge_duplicate_boxes(dets)) == 2

    def test_low_overlap_kept(self):
        # 同 tile 但 IoU 低(两张相邻同牌)不合并
        dets = [
            TileDet(0, 0, 0, 100, 100, 0.9),
            TileDet(0, 150, 0, 250, 100, 0.9),
        ]
        assert len(merge_duplicate_boxes(dets)) == 2

    def test_empty(self):
        assert merge_duplicate_boxes([]) == []


class FakeBoxes:
    def __init__(self, cls, conf, xyxy):
        self.cls = np.array(cls, dtype=np.float32).reshape(-1, 1)
        self.conf = np.array(conf, dtype=np.float32).reshape(-1, 1)
        self.xyxy = np.array(xyxy, dtype=np.float32)

    def __iter__(self):
        """模拟 ultralytics Boxes 迭代: 逐框产出 (cls, conf, xyxy) 行视图。"""
        for i in range(self.cls.shape[0]):
            yield SimpleNamespace(
                cls=self.cls[i : i + 1],
                conf=self.conf[i : i + 1],
                xyxy=self.xyxy[i : i + 1],
            )


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeDetector:
    def __call__(self, frame, conf=0.5, verbose=False, **kwargs):
        return [
            FakeResult(FakeBoxes(
                cls=[0, 9, 42],          # 一万 / 一饼 / 越界class(应跳过)
                conf=[0.9, 0.8, 0.7],
                xyxy=[[10, 20, 42, 52], [100, 200, 132, 232], [0, 0, 10, 10]],
            ))
        ]


class TestScreenVision:
    def test_maps_class_id_to_tile(self):
        dets = ScreenVision('unused.pt').process(
            np.zeros((300, 400, 3), dtype=np.uint8),
            detector=FakeDetector(),
        )
        assert [d.tile for d in dets] == [0, 9]

    def test_refine_roi_batches_crops(self, monkeypatch):
        """ROI 精修批量推理: 低置信框 N 框 = 一次分类调用。

        逐框调用曾是每帧 ~1s 的瓶颈(60 框 × 单次推理开销; 实测
        空桌 58ms → 开局有牌 1s — 帧率 5.4→1.1fps 的根因)。
        """
        vision = ScreenVision('unused.pt', roi_model_path=None)
        calls = []

        class FakeRes:
            def __init__(self, top1):
                self.probs = SimpleNamespace(top1=top1)
                self.names = {t: str(t) for t in range(36)}

        class FakeRoi:
            def predict(self, source, batch=1, verbose=False, **kwargs):
                calls.append((len(source), batch))
                return [FakeRes(0) for _ in source]

        monkeypatch.setattr(vision, '_load_roi', lambda: FakeRoi())
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        # 全部低置信(conf < 0.8): 必须精修, 一次批量调用
        dets = [TileDet(tile=5, x1=10 + i * 20, y1=10, x2=40 + i * 20,
                        y2=50, conf=0.5, track_id=100 + i)
                for i in range(8)]
        out = vision._refine_roi(frame, dets)  # noqa: SLF001
        assert len(calls) == 1 and calls[0] == (8, 8)  # 一次调用, batch=全部
        assert len(out) == 8
        assert [d.tile for d in out] == [0] * 8
        assert [d.track_id for d in out] == [100 + i for i in range(8)]

    def test_refine_roi_skips_high_conf(self, monkeypatch):
        """分级精修: conf ≥ 0.8 的框跳过 ROI 分类, 直接采用检测类别。

        实测高置信框仅 2.8% 被精修改类别, 跳过省 ~90% ROI 成本。
        """
        vision = ScreenVision('unused.pt', roi_model_path=None)
        calls = []

        class FakeRoi:
            def predict(self, source, batch=1, verbose=False, **kwargs):
                calls.append(len(source))
                return source  # 不应被调用

        monkeypatch.setattr(vision, '_load_roi', lambda: FakeRoi())
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        dets = [
            TileDet(tile=7, x1=10, y1=10, x2=40, y2=50, conf=0.95),
            TileDet(tile=8, x1=60, y1=10, x2=90, y2=50, conf=0.80),
        ]
        out = vision._refine_roi(frame, dets)  # noqa: SLF001
        assert calls == []  # 分类器完全没被调用
        assert [d.tile for d in out] == [7, 8]  # 检测类别原样保留

    def test_refine_roi_mixed_conf(self, monkeypatch):
        """分级精修混合: 只有低置信框进分类器, 高置信框原样保留。"""
        vision = ScreenVision('unused.pt', roi_model_path=None)
        calls = []

        class FakeRes:
            def __init__(self, top1):
                self.probs = SimpleNamespace(top1=top1)
                self.names = {t: str(t) for t in range(36)}

        class FakeRoi:
            def predict(self, source, batch=1, verbose=False, **kwargs):
                calls.append((len(source), batch))
                return [FakeRes(3) for _ in source]  # 精修为 3

        monkeypatch.setattr(vision, '_load_roi', lambda: FakeRoi())
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        dets = [
            TileDet(tile=7, x1=10, y1=10, x2=40, y2=50, conf=0.95),  # 跳过
            TileDet(tile=8, x1=60, y1=10, x2=90, y2=50, conf=0.40),  # 精修
            TileDet(tile=9, x1=110, y1=10, x2=140, y2=50, conf=0.30),  # 精修
        ]
        out = vision._refine_roi(frame, dets)  # noqa: SLF001
        assert calls == [(2, 2)]  # 只有 2 个低置信框进分类器
        assert [d.tile for d in out] == [7, 3, 3]  # 高置信保留, 低置信覆盖

    def test_box_coordinates_passed_through(self):
        dets = ScreenVision('unused.pt').process(
            np.zeros((300, 400, 3), dtype=np.uint8),
            detector=FakeDetector(),
        )
        assert dets[0].x1 == 10 and dets[0].y2 == 52
        assert dets[0].conf == pytest.approx(0.9)

    def test_invalid_imgsz_rejected(self):
        vision = ScreenVision('unused.pt')
        with pytest.raises(ValueError):
            vision.process(np.zeros((300, 400, 3), dtype=np.uint8),
                           detector=FakeDetector(), imgsz=100)  # 非32倍数
        with pytest.raises(ValueError):
            vision.process(np.zeros((300, 400, 3), dtype=np.uint8),
                           detector=FakeDetector(), imgsz=3000)  # 超范围

    def test_valid_imgsz_accepted(self):
        dets = ScreenVision('unused.pt').process(
            np.zeros((300, 400, 3), dtype=np.uint8),
            detector=FakeDetector(), imgsz=960)
        assert len(dets) == 2  # 正常推理路径

    def test_warmup_loads_both_models(self, monkeypatch):
        """冷启动预热: 触发检测器 + ROI 分类器加载并跑一次推理。"""
        vision = ScreenVision('unused.pt', roi_model_path='roi.pt')
        loaded = []

        def fake_load(self):
            loaded.append('detector')
            return object()

        def fake_load_roi(self):
            loaded.append('roi')
            return object()

        monkeypatch.setattr(ScreenVision, '_load', fake_load)
        monkeypatch.setattr(ScreenVision, '_load_roi', fake_load_roi)
        monkeypatch.setattr(ScreenVision, 'process',
                            lambda self, frame, conf_threshold=0.0: [])
        vision.warmup(np.zeros((300, 400, 3), dtype=np.uint8))
        assert loaded == ['detector', 'roi']  # 两个模型都加载了

    def test_warmup_no_roi_model_ok(self, monkeypatch):
        """ROI 分类器缺失时 warmup 不报错(降级路径)。"""
        vision = ScreenVision('unused.pt', roi_model_path=None)
        monkeypatch.setattr(ScreenVision, '_load',
                            lambda self: object())
        monkeypatch.setattr(ScreenVision, '_load_roi',
                            lambda self: None)
        monkeypatch.setattr(ScreenVision, 'process',
                            lambda self, frame, conf_threshold=0.0: [])
        vision.warmup(np.zeros((300, 400, 3), dtype=np.uint8))  # 不抛异常
