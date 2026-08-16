"""人工审核打标: YOLO 自动检测 → 点击删除错误框 → 其余保存为训练数据。

用法:
    python scripts/label_review.py <图片或目录> [--model best.pt] [--conf 0.25] [--out data/screen_dataset]
    # 审屏幕模型(best.pt 类 id 即牌编码)必须加 --identity:
    #   python scripts/label_review.py <目录> --model data/models/screen/.../best.pt --identity
    # 帧间继承(默认开): 连续帧自动继承上一张保存的修正(橙框),
    #   继承框与当前帧模型预测按 IoU 匹配(位置随当前帧), 修正的牌标签优先;
    #   全部区域统一做「牌面图像验证」后才继承(不再按布局分区 —
    #   布局文件失准会让手牌区跳过验证, 错位标签入库)。
    #   验证通过: 继承(手牌换牌/牌打出 → 区域图像大变 → 拒绝, 安全)。
    #   未匹配且验证通过 → 保留原坐标(模型持续漏检的牌标一次即持续继承);
    #   未匹配且验证失败 → 丢弃; 画面尺寸变化 → 整体跳过继承(像素坐标失效)。
    #   --no-inherit 关闭; --same-diff 调牌面一致阈值。
    # 交互(全窗口内, 无需终端):
    #   点击框内      → 删除该框(误检)
    #   a             → 添加模式: 拖框补漏检, 松开后两键输入牌名:
    #                     第1键: 1万 2饼 3条 4风 5箭 6花
    #                     第2键: 点数/序号(万饼条1-9, 风1-4, 箭1-3, 花1-8)
    #                     Esc 取消本次输入
    #   Esc           → 退出添加模式
    #   c             → 清空本图全部框重来
    #   l             → 标签开关(牌河牌小, 标签挡牌面时关掉只看框)
    #   h             → 左上角文字开关(提示/进度/图例)
    #   Enter / n     → 保存本图标签(保留 ≥1 个框), 下一张
    #   q             → 退出(当前图不保存)

标签: 默认按 best.pt 的 YOLO 42 类顺序映射到牌编码 0-41(实物模型);
      屏幕模型类 id 与牌编码恒等, 需 --identity 跳过映射。
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.capture_dataset import next_index, save_sample
from src.mahjong_ui.pil_text import render_text_overlay
from src.mahjong_core.tile import tile_display
from src.mahjong_cv.yolo_mapping import yolo_to_tile

MODEL = 'data/models/mahjong_tile_detector/weights/best.pt'
_CLICK_TOL = 6  # 点击容差像素

#: 两键输入映射: 花色键 → (起始牌编码, 最大序号, 花色名)
#: 花牌 8 种已统一为 1 类(34), 第 2 键输 1 即可
_SUITS = {
    '1': (0, 9, '万'), '2': (9, 9, '饼'), '3': (18, 9, '条'),
    '4': (27, 4, '风'), '5': (31, 3, '箭'), '6': (34, 1, '花'),
}
_LEGEND = [
    '添加模式: 拖框后两键输入牌名',
    ' 1=万  2=饼  3=条  4=风  5=箭  6=花   (第1键)',
    ' 万/饼/条 输1-9; 风输1-4(东南西北); 箭输1-3(中发白); 花输1',
    ' Esc 取消输入/退出添加模式',
]
_HINTS = ['[Enter/n] 保存   [q] 退出   [a] 添加   [c] 清空   [l] 标签   [h] 文字   [点击] 删除',
          '橙框=继承(牌面一致)   绿框=模型新检测']

_SAME_TILE_DIFF = 25.0  # 牌面灰度平均绝对差阈值: 小于此视为同一张牌


def hit_test(px: float, py: float, box: tuple[float, float, float, float]) -> bool:
    """点是否落在框内(带容差)。"""
    x1, y1, x2, y2 = box
    return x1 - _CLICK_TOL <= px <= x2 + _CLICK_TOL and y1 - _CLICK_TOL <= py <= y2 + _CLICK_TOL


def _iou(a: tuple[float, float, float, float],
         b: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = ((a[2] - a[0]) * (a[3] - a[1])
             + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


def _same_tile(img_a: np.ndarray, box_a: tuple[float, float, float, float],
               img_b: np.ndarray, box_b: tuple[float, float, float, float],
               thresh: float) -> bool:
    """两帧两个框区域是否为同一张牌: 灰度均值差 < thresh 且分块无显著差异。

    牌河位置固定、牌面不变 → 恒通过; 手牌打出/换牌 → 区域图像大变 → 拒绝。
    裁剪尺寸不一致时先 resize 到 64x64 再比较。

    分块检查(4x4): 点数/图案位置不同的牌(七万 vs 九万、四饼 vs 五饼 —
    灰度均值可能大体相似)在某个格子上差异显著; 同一张牌的均匀明暗变化
    (动画/高光)各格同涨, 块差≈均值差。整体均值骗不过结构性差异。
    """
    x1a, y1a, x2a, y2a = (int(v) for v in box_a)
    x1b, y1b, x2b, y2b = (int(v) for v in box_b)
    crop_a = img_a[y1a:y2a, x1a:x2a]
    crop_b = img_b[y1b:y2b, x1b:x2b]
    if crop_a.size == 0 or crop_b.size == 0:
        return False
    ga = cv2.resize(cv2.cvtColor(crop_a, cv2.COLOR_BGR2GRAY), (64, 64))
    gb = cv2.resize(cv2.cvtColor(crop_b, cv2.COLOR_BGR2GRAY), (64, 64))
    diff = np.abs(ga.astype(np.float32) - gb.astype(np.float32))
    if float(diff.mean()) >= thresh:
        return False
    # 4x4 分块均值差: 点数错位 → 某格显著; 静止牌面 → 各格≈0
    block_max = max(float(diff[i * 16:(i + 1) * 16, j * 16:(j + 1) * 16].mean())
                    for i in range(4) for j in range(4))
    return block_max < thresh * 2.5


def _match_inherited(
    inherited: list[tuple[int, float, float, float, float, float]],
    preds: list[tuple[int, float, float, float, float, float]],
    prev_img: np.ndarray,
    cur_img: np.ndarray,
    iou_thresh: float = 0.35,
    same_diff: float = _SAME_TILE_DIFF,
) -> tuple[list[tuple[int, float, float, float, float, float, bool]], int, int]:
    """继承框(上一帧修正)与当前帧预测按 IoU 贪心匹配, 全部区域统一牌面验证。

    继承与否取决于「两帧是否同一张牌」——不再按布局分区(布局文件失准,
    区域感知会让手牌区跳过验证, 错位标签直接入库): 宁可少继承, 不可错继承。
    匹配上且验证通过 → 用继承标签, 位置取当前帧预测框, conf 记 1.0;
    匹配上但验证失败 → 位置是新的牌, 改用模型标签;
    未匹配且验证通过 → 保留原坐标继承(模型持续漏检的牌);
    未匹配且验证失败 → 丢弃(牌已打出/移走);
    未匹配的预测 → 保留模型标签。
    画面尺寸变化由调用方处理(跳过继承), 像素坐标在尺寸不同时无意义。
    返回 (合并列表, 匹配继承数, 无匹配保留数); 列表元素末位 bool = 是否继承框。
    """
    used = [False] * len(preds)
    merged: list[tuple[int, float, float, float, float, float, bool]] = []
    n_inherited = 0
    n_kept = 0
    for it, ix1, iy1, ix2, iy2, _ic in inherited:
        ibox = (ix1, iy1, ix2, iy2)
        best_j, best_s = -1, iou_thresh
        for j, (_t, px1, py1, px2, py2, _pc) in enumerate(preds):
            if used[j]:
                continue
            s = _iou(ibox, (px1, py1, px2, py2))
            if s > best_s:
                best_j, best_s = j, s
        if best_j >= 0:
            used[best_j] = True
            t_pred, px1, py1, px2, py2, pc = preds[best_j]
            pbox = (px1, py1, px2, py2)
            if _same_tile(prev_img, ibox, cur_img, pbox, same_diff):
                merged.append((it, px1, py1, px2, py2, 1.0, True))
                n_inherited += 1
            else:
                merged.append((t_pred, px1, py1, px2, py2, pc, False))
        elif _same_tile(prev_img, ibox, cur_img, ibox, same_diff):
            merged.append((it, ix1, iy1, ix2, iy2, 1.0, True))
            n_kept += 1
    for j, (t, px1, py1, px2, py2, pc) in enumerate(preds):
        if not used[j]:
            merged.append((t, px1, py1, px2, py2, pc, False))
    return merged, n_inherited, n_kept


def _dedup_overlap(
    dets: list[tuple[int, float, float, float, float, float]],
    inherited: list[bool],
    iou_thresh: float = 0.5,
) -> tuple[list[tuple[int, float, float, float, float, float]], list[bool]]:
    """同位置重叠框去重(双框防护), 保留高优先级: 人工添加 > 继承 > 模型。

    双框来源: 继承框 IoU 匹配失败(>0.35 才匹配)但实际是同一张牌时,
    旧坐标继承框与模型新预测同存一位置 → 保存进训练集 = 同位置双 GT。
    保留人工添加的框(用户拖框确认过位置和标签); 其次继承框(标签是
    人工确认过的); 最后模型框。
    """
    keep = [True] * len(dets)

    def _pri(k: int) -> int:
        d, ih = dets[k], inherited[k]
        return 2 if (d[5] >= 1.0 and not ih) else (1 if ih else 0)

    for i in range(len(dets)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(dets)):
            if not keep[j]:
                continue
            if _iou(dets[i][1:5], dets[j][1:5]) < iou_thresh:
                continue
            if _pri(i) >= _pri(j):
                keep[j] = False
            else:
                keep[i] = False
    kept = [(d, ih) for d, ih, k in zip(dets, inherited, keep) if k]
    return [d for d, _ in kept], [ih for _, ih in kept]


def _load_review_labels(
    out: Path, source_stem: str, prev_path: Path,
    dataset: Path | None = None,
) -> tuple[list[tuple[int, float, float, float, float, float]], np.ndarray] | None:
    """加载某源文件的审核标签(归一化→像素), 断点续审继承用。

    搜索顺序: 审核暂存区 out → 数据集 dataset(已 --ingest 入库的审核版,
    入库是移动文件, 暂存区里不再有 → 链不断)。返回 (像素坐标的标签,
    该帧图像[牌面验证用]); None 表示没有审核标签。
    """
    for base in (out, dataset):
        if base is None:
            continue
        hits = list((base / 'labels').glob(f'**/review_*_{source_stem}.txt'))
        if not hits:
            continue
        img = cv2.imread(str(prev_path))
        if img is None:
            return None
        h, w = img.shape[:2]
        dets: list[tuple[int, float, float, float, float, float]] = []
        for line in hits[0].read_text(encoding='utf-8').splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            tile = int(parts[0])
            cx, cy, bw, bh = (float(v) for v in parts[1:])
            dets.append((tile,
                         (cx - bw / 2) * w, (cy - bh / 2) * h,
                         (cx + bw / 2) * w, (cy + bh / 2) * h, 1.0))
        return dets, img
    return None


def reviewed_exists(out: Path, source_stem: str) -> bool:
    """该源文件是否已有审核输出(review_*_<源文件名>.jpg)。

    拆分后输出可能在 images/train|val 子目录, 需递归搜索。
    """
    return bool(list((out / 'images').glob(f'**/review_*_{source_stem}.jpg')))


def resolve_tile(suit_key: str, rank_key: str) -> int | None:
    """两键 → 牌编码; 非法返回 None。"""
    suit = _SUITS.get(suit_key)
    if suit is None:
        return None
    base, max_rank, _name = suit
    if not rank_key.isdigit():
        return None
    rank = int(rank_key)
    if 1 <= rank <= max_rank:
        return base + rank - 1
    return None


class ReviewSession:
    """单张图的审核交互。"""

    def __init__(self) -> None:
        # dets: (tile, x1, y1, x2, y2, conf)
        self.dets: list[tuple[int, float, float, float, float, float]] = []
        self.inherited: list[bool] = []  # 与 dets 对齐: 是否继承上一帧
        self.show_labels = True  # 标签开关: 牌河牌小, 标签会挡牌面
        self.show_msg = True  # 左上角文字开关(提示/进度/图例)
        self.img: np.ndarray | None = None
        self.path = ''
        self.add_mode = False
        self.drag: list[tuple[int, int]] = []
        self.msg = ''
        self.quit = False
        self._mouse_down = False
        self._input_state: str | None = None  # None | 'suit' | 'rank'
        self._input_suit = ''
        self.dirty = True  # 画面是否需重绘(鼠标/按键事件置位)

    def _boxes(self) -> list[tuple[float, float, float, float]]:
        return [(d[1], d[2], d[3], d[4]) for d in self.dets]

    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if self.add_mode:
            if self._input_state is not None:
                return  # 输入牌名期间忽略鼠标
            if event == cv2.EVENT_LBUTTONDOWN:
                self._mouse_down = True
                self.drag = [(x, y)]
                self.dirty = True
            elif event == cv2.EVENT_MOUSEMOVE and self._mouse_down:
                # 只在按住左键时跟踪拖框(松开后光标移动不得改动待添加框)
                self.drag = [self.drag[0], (x, y)]
                self.dirty = True
            elif event == cv2.EVENT_LBUTTONUP and self._mouse_down:
                self._mouse_down = False
                self.drag = [self.drag[0], (x, y)]
                self._begin_input()
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            # 删除点中的框(从后往前, 点中最上层)
            for i in range(len(self.dets) - 1, -1, -1):
                if hit_test(x, y, self._boxes()[i]):
                    tile = self.dets[i][0]
                    del self.dets[i]
                    del self.inherited[i]
                    self.msg = f'已删除 {tile_display(tile)} (剩 {len(self.dets)} 个)'
                    self.dirty = True
                    return
            self.msg = f'未点中任何框 ({x},{y}), 剩 {len(self.dets)} 个'
            self.dirty = True

    def _begin_input(self) -> None:
        assert self.img is not None and len(self.drag) == 2
        (x1, y1), (x2, y2) = self.drag
        lx, rx = min(x1, x2), max(x1, x2)
        ty, by = min(y1, y2), max(y1, y2)
        if rx - lx < 12 or by - ty < 12:
            self.msg = '框太小, 重新拖'
            self.drag = []
            return
        self._input_state = 'suit'
        self._input_suit = ''
        self.msg = '第1键: 1万 2饼 3条 4风 5箭 6花 (Esc 取消)'

    def _handle_input_key(self, key: int) -> bool:
        """处理输入态按键, 返回是否已消费。"""
        if self._input_state == 'suit':
            ch = chr(key)
            if ch in _SUITS:
                self._input_suit = ch
                self._input_state = 'rank'
                self.msg = f'{_SUITS[ch][2]}: 输序号 1-{_SUITS[ch][1]} (Esc 取消)'
                return True
        elif self._input_state == 'rank':
            ch = chr(key)
            tile = resolve_tile(self._input_suit, ch)
            if tile is not None:
                assert self.img is not None
                (x1, y1), (x2, y2) = self.drag
                lx, rx = min(x1, x2), max(x1, x2)
                ty, by = min(y1, y2), max(y1, y2)
                self.dets.append((tile, lx, ty, rx, by, 1.0))
                self.inherited.append(False)
                self.msg = f'已添加 {tile_display(tile)} (共 {len(self.dets)} 个), 继续拖框'
            else:
                self.msg = f'序号 {ch} 超出范围, 重新输入 (Esc 取消)'
                self._input_state = 'suit'
                self._input_suit = ''
                return True
            self._input_state = None
            self.drag = []
            return True
        return False

    def _render(self, index: int, total: int) -> np.ndarray:
        """渲染一帧: 框用 cv2 画, 全部文字一次性 PIL 绘制(单次转换)。"""
        assert self.img is not None
        display = self.img.copy()
        _DARK = (25, 25, 25)
        _Overlay = tuple[int, int, str, tuple[int, int, int]] \
            | tuple[int, int, str, tuple[int, int, int], tuple[int, int, int]]
        overlays: list[_Overlay] = []

        for i, (tile, x1, y1, x2, y2, conf) in enumerate(self.dets):
            inherited = self.inherited[i]
            color = (0, 165, 255) if inherited else (0, 255, 0)
            bg = (0, 110, 170) if inherited else (0, 200, 0)
            cv2.rectangle(display, (int(x1), int(y1)), (int(x2), int(y2)),
                          color, 2)
            if self.show_labels:
                text = f'{tile_display(tile)} {conf:.2f}'
                label_y = int(y1) - 26
                if label_y < 0:
                    label_y = int(y1) + 2
                overlays.append((int(x1), label_y, text, (0, 0, 0), bg))
        # 拖框预览
        if self.add_mode and len(self.drag) == 2:
            (x1, y1), (x2, y2) = self.drag
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # 状态/进度/提示(深色底衬); 左上角文字可用 h 键关闭(挡牌面时)
        h_img, _w_img = display.shape[:2]
        if self.show_msg:
            overlays += [
                (10, 10, self.msg, (255, 255, 0), _DARK),
                (10, 36, f'第 {index}/{total} 张: {Path(self.path).name}',
                 (255, 255, 255), _DARK),
            ]
            if self.add_mode:
                for i, line in enumerate(_LEGEND):
                    overlays.append((10, 60 + i * 24, line, (0, 255, 0), _DARK))
        for i, line in enumerate(_HINTS):
            overlays.append((10, h_img - 30 - i * 24, line, (255, 255, 255), _DARK))
        return render_text_overlay(display, overlays)

    def process(self, image_path: Path, model: Any, conf: float,
                index: int = 0, total: int = 0,
                identity: bool = False,
                inherit_from: list[tuple[int, float, float, float, float, float]]
                | None = None,
                prev_img: np.ndarray | None = None,
                same_diff: float = _SAME_TILE_DIFF) -> bool:
        """审核一张图, 返回是否保存。

        实时检测(conf 过滤)出初始框 — 伪标签过滤后的框漏检太多,
        实时检测全量框以「删错为主」体验更好; 保存后删除同源伪标签,
        以审核版为准。

        identity: 类 id 即牌编码(屏幕模型)。
        inherit_from: 上一帧保存的修正框(会话内来自上一张; 断点续审
            由调用方从磁盘加载), 与当前帧预测 IoU 匹配 + 牌面图像验证后继承。
        prev_img: 继承框所在的上一帧图像(仅断点续审传入; 会话内自动取
            self.img)。继承且无上一帧图像时退化为全部用模型预测。
        same_diff: 牌面一致灰度阈值(_same_tile)。
        """
        prev = self.img if self.img is not None else prev_img
        img = cv2.imread(str(image_path))
        if img is None:
            print(f'无法读取: {image_path}')
            return False
        if inherit_from is not None and prev is not None \
                and prev.shape != img.shape:
            # 尺寸变化: 像素坐标继承全部失效(旧坐标错位), 整体退回模型预测
            print(f'尺寸变化 {prev.shape} → {img.shape}, 跳过继承(像素坐标失效)')
            inherit_from = None
        self.img = img
        self.path = str(image_path)
        self.dets = []
        self.inherited = []
        self.add_mode = False
        self.drag = []
        self.msg = ''
        self._input_state = None
        self.dirty = True

        results = model(str(image_path), conf=conf, verbose=False)
        preds: list[tuple[int, float, float, float, float, float]] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                conf_v = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                tile = cls_id if identity else yolo_to_tile(cls_id)
                preds.append((tile, x1, y1, x2, y2, conf_v))
        if inherit_from and prev is not None:
            merged, n_inh, n_kept = _match_inherited(
                inherit_from, preds, prev, img, same_diff=same_diff)
            self.msg = (f'继承 {n_inh + n_kept} 个(橙), '
                        f'模型新检测 {len(preds) - n_inh} 个(绿)')
            if n_kept:
                self.msg += f'; {n_kept} 个无匹配保留(牌面一致)'
        else:
            merged = [(t, x1, y1, x2, y2, c, False)
                      for t, x1, y1, x2, y2, c in preds]
        # 双框防护: 同一位置继承框与模型框重叠时只留一个(人工>继承>模型)
        dets = [(t, x1, y1, x2, y2, c)
                for t, x1, y1, x2, y2, c, _i in merged]
        self.dets, self.inherited = _dedup_overlap(
            dets, [m[-1] for m in merged])

        print(f'\n=== {image_path.name} === 共 {len(self.dets)} 个框')
        cv2.namedWindow('label_review')
        cv2.setMouseCallback('label_review', self._on_mouse)

        saved = False
        while True:
            # 画面有变化才重绘(空闲只轮询按键, 保证键盘响应)
            if self.dirty:
                display = self._render(index, total)
                cv2.imshow('label_review', display)
                self.dirty = False

            key = cv2.waitKey(30) & 0xFF
            if key == 255:  # 无按键
                continue
            self.dirty = True
            if self._input_state is not None:
                if key == 27:
                    self._input_state = None
                    self.drag = []
                    self.msg = '已取消输入, 继续拖框或 Esc 退出添加'
                else:
                    self._handle_input_key(key)
                continue
            if key == ord('a'):
                self.add_mode = not self.add_mode
                self.msg = '添加模式: 拖框后两键输牌名' if self.add_mode else '退出添加模式'
            elif key == 27 and self.add_mode:
                self.add_mode = False
                self.msg = '已退出添加模式'
            elif key == ord('l'):
                self.show_labels = not self.show_labels
                self.msg = ('标签已隐藏(只显示框)' if not self.show_labels
                            else '标签已显示')
            elif key == ord('h'):
                self.show_msg = not self.show_msg
                self.msg = ('左上角文字已隐藏(h 恢复)' if not self.show_msg
                            else '左上角文字已显示')
            elif key == ord('c'):
                self.dets = []
                self.inherited = []
                self.msg = '已清空全部框'
            elif key in (13, ord('n')):  # Enter / n: 保存并下一张
                if self.dets:
                    saved = True
                self.msg = f'保存 {len(self.dets)} 个框'
                break
            elif key == ord('q'):
                self.quit = True
                break

        cv2.destroyAllWindows()
        return saved


def _shot_ts(stem: str) -> tuple[int, str]:
    """文件排序键: 新格式时间戳(shot_YYYYMMDDHHMMSSffffff)按时间排序。

    混合新旧命名时, 字母序 ≠ 时间序(shot_000042 < shot_20260808...),
    相邻两帧实际不连续 → 继承错链。时间戳可解析 → 按时间; 否则字母序兜底。
    """
    m = re.search(r'shot_(\d{14,})', stem)
    return (0, m.group(1)) if m else (1, stem)


def main() -> None:
    parser = argparse.ArgumentParser(description='YOLO 自动检测 + 人工删错框 = 训练数据')
    parser.add_argument('images', help='图片路径或目录(递归找 png/jpg)')
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--out', default='data/screen_dataset')
    parser.add_argument('--identity', action='store_true',
                        help='屏幕模型: 类 id 即牌编码, 不做 YOLO 顺序映射')
    parser.add_argument('--no-inherit', action='store_true',
                        help='不继承上一帧修正的标签(默认继承)')
    parser.add_argument('--same-diff', type=float, default=_SAME_TILE_DIFF,
                        help='牌面一致灰度阈值(默认 25; 继承误杀调大, 错继承调小)')
    parser.add_argument('--dataset', default='data/screen_dataset',
                        help='已入库数据集(断点续审搜索已入库的审核版, 链不断)')
    parser.add_argument('--redo', action='store_true',
                        help='重新审核已处理过的文件(默认自动跳过)')
    args = parser.parse_args()

    from ultralytics import YOLO  # type: ignore[attr-defined]  # noqa: PLC0415

    model = YOLO(args.model)
    out = Path(args.out)
    dataset = Path(args.dataset)

    src = Path(args.images)
    files = sorted(src.glob('*.png'), key=lambda f: _shot_ts(f.stem)) \
        + sorted(src.glob('*.jpg'), key=lambda f: _shot_ts(f.stem)) \
        if src.is_dir() else [src]
    # 输入目录里的 review_* 文件若已是输出产物(同名存在于输出目录)则跳过,
    # 防止输入与输出同目录时把上次审核输出再当输入; 独立目录的 review_*
    # 源图(如 test_real 本身就是审核命名)必须正常审核。
    files = [f for f in files
             if not (f.stem.startswith('review_')
                     and (out / 'images' / f.name).exists())]
    if not files:
        print(f'没有找到图片: {args.images}')
        sys.exit(1)
    print(f'共 {len(files)} 张待审核')

    session = ReviewSession()
    saved_count = 0
    skipped = 0
    index = next_index(out / 'images', 'review')
    pending = files if args.redo else [
        f for f in files if not reviewed_exists(out, f.stem)
    ]
    skipped = len(files) - len(pending)
    if skipped:
        print(f'跳过 {skipped} 张已审核的图片(断点续审); --redo 可重新审核')
    pos = {f: k for k, f in enumerate(files)}
    prev_dets: list[tuple[int, float, float, float, float, float]] | None = None
    for i, f in enumerate(pending, start=1):
        if session.quit:
            break
        inherit = prev_dets if not args.no_inherit else None
        prev_img = None
        if inherit is None and not args.no_inherit:
            # 断点续审: 本张之前最近一张已审核过的图在磁盘上 → 加载它的修正
            # (先查审核暂存区, 再查已入库数据集 — 入库是移动, 暂存区会空)
            for k in range(pos[f] - 1, -1, -1):
                loaded = _load_review_labels(out, files[k].stem, files[k],
                                             dataset)
                if loaded is not None:
                    inherit, prev_img = loaded
                    break
        if session.process(f, model, args.conf, i, len(pending),
                           args.identity, inherit, prev_img,
                           same_diff=args.same_diff):
            from src.mahjong_cv.detections import TileDet

            assert session.img is not None  # process 已成功读图
            dets = [TileDet(t, x1, y1, x2, y2, c)
                    for t, x1, y1, x2, y2, c in session.dets]
            save_sample(out, session.img, dets, index,
                        name=f'review_{index:06d}_{f.stem}')
            print(f'已保存 review_{index:06d}_{f.stem}: {len(dets)} 个框')
            index += 1
            saved_count += 1
            prev_dets = list(session.dets)
            # 以审核为准: 若审核的是伪标注输出, 删除同源伪标签(图+标签),
            # 不再产生重复样本(拆分脚本的同源去重兜底)
            same_img = out / 'images' / f'{f.stem}.jpg'
            if f.stem.startswith('pseudo_') and same_img.exists():
                (out / 'labels' / f'{f.stem}.txt').unlink(missing_ok=True)
                same_img.unlink()
                print(f'  已删除同源伪标签 {f.stem}(以审核版为准)')
    print(f'\n完成: 保存 {saved_count}/{len(pending)} 张 -> {out}/images + labels')


if __name__ == '__main__':
    main()
