"""屏幕视觉 — 欢乐麻将截图牌检测。

屏幕模型约定: class id = 牌编码 0-34(万饼条风箭 + 花牌统一, 35 类)。
"""

from __future__ import annotations

import threading
from typing import Any

import cv2
import numpy as np

from src.mahjong_core.tile import TOTAL_TILES
from src.mahjong_cv.detections import TileDet

#: imgsz 合法性范围(ultralytics 要求 32 的倍数)
_IMGSZ_MIN = 128
_IMGSZ_MAX = 2560


def merge_duplicate_boxes(
    dets: list[TileDet], iou_thresh: float = 0.35,
) -> list[TileDet]:
    """同类重叠框合并: 同一张牌被模型拆成两框(如三分之一空白+牌面,
    或牌面高光分割)时只留置信度高的一个。

    不同类别或 IoU 低于阈值的框不动(相邻真牌 IoU 不会 >0.35, 不误合并)。
    合并后手牌计数正确(双框会让手牌虚多一张)。
    """
    merged: list[TileDet] = []
    for d in sorted(dets, key=lambda x: -x.conf):
        dup = False
        for m in merged:
            if m.tile != d.tile:
                continue
            ix1, iy1 = max(d.x1, m.x1), max(d.y1, m.y1)
            ix2, iy2 = min(d.x2, m.x2), min(d.y2, m.y2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            union = ((d.x2 - d.x1) * (d.y2 - d.y1)
                     + (m.x2 - m.x1) * (m.y2 - m.y1) - inter)
            if union > 0 and inter / union > iou_thresh:
                dup = True
                break
        if not dup:
            merged.append(d)
    return merged


class ScreenVision:
    """YOLO 屏幕牌检测封装(含官方跟踪 + ROI 牌面分类精修)。"""

    #: ROI 分类器默认路径(train_roi_cls.py 输出; 不存在则跳过精修)
    DEFAULT_ROI = 'data/models/roi_cls/weights/best.pt'

    def __init__(
        self, model_path: str,
        roi_model_path: str | None = DEFAULT_ROI,
    ) -> None:
        self._model_path = model_path
        self._model: Any = None
        self._roi_model_path = roi_model_path
        self._roi_model: Any = None
        # 推理锁: 当前单线程(后台推理线程)调用, 防御未来多线程
        self._lock = threading.Lock()

    def _load(self) -> Any:
        if self._model is None:
            # ultralytics 动态导出 YOLO, mypy 无法解析(uv 环境已安装)
            from ultralytics import YOLO  # type: ignore[attr-defined]

            self._model = YOLO(self._model_path)
        return self._model

    def process(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.5,
        detector: Any | None = None,
        imgsz: int | None = None,
    ) -> list[TileDet]:
        """检测一帧全部牌(纯检测, 无跟踪)。detector 用于测试注入。

        imgsz: 推理分辨率覆盖(默认用模型训练尺寸, 如 1280)。
        实时场景用小尺寸(640/960)提速, 标定/诊断用原尺寸。
        坐标自动映射回原图, 但小尺寸会降低小牌(牌河)召回。
        实时连续帧用 track() 获得跨帧 ID。
        """
        dets = self._infer(frame, conf_threshold, imgsz, detector)
        # 精修前按检测器类别合并(去重省算力);
        # 精修后按精修类别再合并(检测器双框粗类别不同 → 精修后同类别
        # → 必须二次合并, 否则同一张牌两个框, 一个匹配一个落空成 FP)
        return merge_duplicate_boxes(
            self._refine_roi(frame, merge_duplicate_boxes(dets)))

    def track(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.15,
        detector: Any | None = None,
        imgsz: int | None = None,
    ) -> list[TileDet]:
        """检测 + 官方跟踪(ByteTrack): 返回带 track_id 的框。

        必须在同一 ScreenVision 实例上逐帧连续调用(跟踪器状态在
        model 内部, persist=True 跨帧保持) — 帧间隔/漏检由跟踪器
        卡尔曼预测兜底。牌河框的 track_id 即身份(跨帧归属的依据)。

        conf_threshold 0.15(低阈值召回): 漏检的真牌(conf 0.15-0.30)
        被找回, 误检框由 ROI 分类器背景类(35)过滤。
        """
        if imgsz is not None and (imgsz % 32 != 0
                                  or not _IMGSZ_MIN <= imgsz <= _IMGSZ_MAX):
            raise ValueError(f'imgsz 必须为 32 的倍数且 '
                             f'[{_IMGSZ_MIN}, {_IMGSZ_MAX}], 当前 {imgsz}')
        model = detector if detector is not None else self._load()
        from pathlib import Path  # noqa: PLC0415

        # 自定义跟踪配置: new_track_thresh 0.25(默认 0.6 会把
        # conf 0.3-0.6 的新牌拒之门外, 表现为"检测不到")
        tracker_cfg = str(Path(__file__).parent / 'bytetrack_mahjong.yaml')
        kwargs: dict[str, Any] = {'conf': conf_threshold, 'verbose': False,
                                  'persist': True, 'tracker': tracker_cfg}
        if imgsz is not None:
            kwargs['imgsz'] = imgsz
        with self._lock:
            results = model.track(frame, **kwargs)
        # 精修后二次合并(见 process 注释)
        return merge_duplicate_boxes(
            self._refine_roi(frame, merge_duplicate_boxes(
                self._parse(results))))

    def _refine_roi(
        self, frame: np.ndarray, dets: list[TileDet],
    ) -> list[TileDet]:
        """ROI 牌面精修: 全部框裁剪放大 2 倍 → 单次批量分类 → 覆盖类别。

        解决小框(15px 牌河)牌面难分辨的类别混淆(检测器全图 68.7% →
        分类器放大特写 98.3%, 实验验证)。分类器缺失时原样返回。
        背景类(35)输出 → 丢弃该框: 低阈值(0.15)检测带来的误检框
        由分类器过滤 — 找回漏检的代价由背景类吸收。
        逐框调用曾是每帧 ~1s 的瓶颈(60 框 × 单次推理调用开销; 实测
        空桌 58ms → 开局有牌 1s — 帧率 5.4→1.1fps 的根因) → 批量:
        一次推理全部裁剪, 帧率恢复检测主导。
        """
        model = self._load_roi()
        if model is None or not dets:
            return dets
        fh, fw = frame.shape[:2]
        crops: list[np.ndarray] = []
        idx: list[int] = []          # crops[i] 对应 dets[idx[i]]
        passthrough: set[int] = set()  # 空裁剪原样保留的下标
        for i, d in enumerate(dets):
            x1, y1 = max(0, int(d.x1) - 4), max(0, int(d.y1) - 4)
            x2, y2 = min(fw, int(d.x2) + 4), min(fh, int(d.y2) + 4)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                passthrough.add(i)
                continue
            crops.append(cv2.resize(crop, (96, 96)))
            idx.append(i)
        refined: dict[int, TileDet] = {}
        if crops:
            with self._lock:
                # 列表源 + 显式 batch: ultralytics 各版本对 4D 数组源
                # 处理不一致(实测 ValueError: Expected a single (H,W,C)
                # image), 列表 + batch=N 是稳定路径 — 单次前向全部裁剪
                rs = model.predict(crops, batch=len(crops), verbose=False)
            # ImageFolder 类别 = 目录名排序(字符串序 '0','1','10'...),
            # 索引 ≠ 牌编码 — 用 names 映射回目录名再转 int
            for i, r in zip(idx, rs, strict=True):
                tile = int(r.names[r.probs.top1])
                if tile >= TOTAL_TILES:
                    continue  # 背景类(35): 误检框过滤
                d = dets[i]
                refined[i] = TileDet(tile, d.x1, d.y1, d.x2, d.y2, d.conf,
                                     d.track_id)
        return [refined.get(i, dets[i]) for i in range(len(dets))
                if i in refined or i in passthrough]

    def _load_roi(self) -> Any:
        """懒加载 ROI 分类器; 缺失/不可用返回 None(跳过精修, 降级)。"""
        if self._roi_model is None and self._roi_model_path:
            from pathlib import Path  # noqa: PLC0415

            if Path(self._roi_model_path).exists():
                try:
                    from ultralytics import YOLO  # type: ignore[attr-defined]  # noqa: PLC0415, E501

                    self._roi_model = YOLO(self._roi_model_path)
                except Exception:  # noqa: BLE001 — 分类器不可用 → 降级
                    self._roi_model = None
        return self._roi_model

    @staticmethod
    def _parse(results: Any) -> list[TileDet]:
        """解析检测结果(process/track 共用)。"""
        dets: list[TileDet] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                if not 0 <= cls_id < TOTAL_TILES:
                    continue  # 越界 class 防御
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].tolist()
                track_id = None
                if getattr(box, 'id', None) is not None:
                    track_id = int(box.id[0].item())
                dets.append(
                    TileDet(
                        tile=cls_id,
                        x1=xyxy[0],
                        y1=xyxy[1],
                        x2=xyxy[2],
                        y2=xyxy[3],
                        conf=conf,
                        track_id=track_id,
                    )
                )
        return dets

    def _infer(
        self,
        frame: np.ndarray,
        conf_threshold: float,
        imgsz: int | None,
        detector: Any | None,
    ) -> list[TileDet]:
        """纯检测推理(无跟踪)。"""
        if imgsz is not None and (imgsz % 32 != 0
                                  or not _IMGSZ_MIN <= imgsz <= _IMGSZ_MAX):
            raise ValueError(f'imgsz 必须为 32 的倍数且 '
                             f'[{_IMGSZ_MIN}, {_IMGSZ_MAX}], 当前 {imgsz}')
        model = detector if detector is not None else self._load()
        kwargs: dict[str, Any] = {'conf': conf_threshold, 'verbose': False}
        if imgsz is not None:
            kwargs['imgsz'] = imgsz
        with self._lock:  # YOLO predict 非线程安全, 防御性加锁
            results = model(frame, **kwargs)
        return self._parse(results)
