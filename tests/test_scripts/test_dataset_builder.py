"""数据管线纯函数的单元测试(合成数据)。"""
import tempfile
from pathlib import Path

import cv2
import numpy as np

from scripts.build_dataset_yaml import (
    _dedup_keep_review,
    _ingest,
    _source_key,
    split_and_write_yaml,
)
from scripts.capture_dataset import next_index, save_sample
from src.mahjong_core.tile import tile_display
from src.mahjong_cv.detections import TileDet


def make_dets() -> list[TileDet]:
    return [
        TileDet(0, 10, 20, 42, 52, 0.95),   # 一万
        TileDet(9, 100, 200, 132, 232, 0.9),  # 一饼
    ]


class TestSaveSample:
    def test_writes_image_and_label(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            frame = np.zeros((300, 400, 3), dtype=np.uint8)
            save_sample(out, frame, make_dets(), 7)

            img = out / 'images' / 'frame_000007.jpg'
            lbl = out / 'labels' / 'frame_000007.txt'
            assert img.exists() and lbl.exists()

            text = lbl.read_text(encoding='utf-8').strip().split('\n')
            assert len(text) == 2
            cls, cx, cy, bw, bh = text[0].split()
            assert cls == '0'
            assert 0 <= float(cx) <= 1 and 0 <= float(bw) <= 1


class TestNextIndex:
    """全局计数器: 只增不减, 删除数据不回退(测试隔离 seq 文件)。"""

    def _seq(self, td: str) -> Path:
        return Path(td) / '.seq_test.json'

    def test_empty_dir_starts_zero(self):
        with tempfile.TemporaryDirectory() as td:
            assert next_index(Path(td), 'shot', self._seq(td)) == 0

    def test_continues_after_max(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            for i in (3, 7, 12):
                cv2.imwrite(str(d / f'shot_{i:05d}.jpg'), np.zeros((10, 10, 3), np.uint8))
            # 其他前缀不影响
            cv2.imwrite(str(d / 'frame_000001.jpg'), np.zeros((10, 10, 3), np.uint8))
            assert next_index(d, 'shot', self._seq(td)) == 13

    def test_never_reuses_after_deletion(self):
        # 删除文件后计数器不回退 — 命名永不重名的核心
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            for i in (0, 1):
                cv2.imwrite(str(d / f'shot_{i:05d}.jpg'), np.zeros((10, 10, 3), np.uint8))
            assert next_index(d, 'shot', self._seq(td)) == 2
            for p in d.glob('shot_*.jpg'):
                p.unlink()  # 模拟用户删除旧数据
            assert next_index(d, 'shot', self._seq(td)) == 3  # 不回退!
            assert next_index(d, 'shot', self._seq(td)) == 4

    def test_multi_segment_names(self):
        # review_00005_shot_00042 这类带源文件名的输出, 取前缀后第一个数字段
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            for i in (0, 5, 23):
                cv2.imwrite(str(d / f'review_{i:05d}_shot_{i:05d}.jpg'),
                            np.zeros((10, 10, 3), np.uint8))
            assert next_index(d, 'review', self._seq(td)) == 24

    def test_ignores_unparsable_names(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / 'shot_abc.jpg').touch()
            (d / 'other.jpg').touch()
            assert next_index(d, 'shot', self._seq(td)) == 0


class TestSplitAndWriteYaml:
    def test_split_and_yaml(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # 简报原文未创建目录, cv2.imwrite 静默失败且 write_text 抛 FileNotFoundError,
            # 属测试脚手架缺陷; 补 mkdir 后不影响被测函数行为
            (root / 'images').mkdir(parents=True, exist_ok=True)
            (root / 'labels').mkdir(parents=True, exist_ok=True)
            for i in range(6):
                cv2.imwrite(str(root / 'images' / f'f{i}.jpg'), np.zeros((10, 10, 3), np.uint8))
                (root / 'labels' / f'f{i}.txt').write_text('0 0.5 0.5 0.1 0.1\n', encoding='utf-8')

            yaml_path = split_and_write_yaml(root, val_ratio=0.2)

            train_imgs = list((root / 'images' / 'train').glob('*.jpg'))
            val_imgs = list((root / 'images' / 'val').glob('*.jpg'))
            assert len(train_imgs) == 5 and len(val_imgs) == 1

            content = yaml_path.read_text(encoding='utf-8')
            assert 'nc: 35' in content
            assert tile_display(0) in content  # names 含 '一万'


class TestSourceKey:
    def test_shot_embedded(self):
        assert _source_key('pseudo_000042_shot_000055') == 'shot_000055'
        assert _source_key('review_000042_pseudo_000042_shot_000055') == 'shot_000055'
        assert _source_key('review_000042_shot_000055') == 'shot_000055'

    def test_no_shot_falls_back(self):
        assert _source_key('review_000042_demo') == 'demo'


class TestDedupKeepReview:
    def _dataset(self, td: str) -> Path:
        root = Path(td)
        (root / 'images').mkdir(parents=True)
        (root / 'labels').mkdir(parents=True)
        return root

    def test_pseudo_deleted_when_review_exists(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._dataset(td)
            for name in ('pseudo_000042_shot_000055',
                         'review_000042_pseudo_000042_shot_000055'):
                (root / 'images' / f'{name}.jpg').touch()
                (root / 'labels' / f'{name}.txt').touch()
            removed = _dedup_keep_review(root)
            assert removed == 1
            assert not (root / 'images' / 'pseudo_000042_shot_000055.jpg').exists()
            assert not (root / 'labels' / 'pseudo_000042_shot_000055.txt').exists()
            assert (root / 'images' / 'review_000042_pseudo_000042_shot_000055.jpg').exists()

    def test_pseudo_kept_without_review(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._dataset(td)
            (root / 'images' / 'pseudo_000042_shot_000055.jpg').touch()
            (root / 'labels' / 'pseudo_000042_shot_000055.txt').touch()
            assert _dedup_keep_review(root) == 0
            assert (root / 'images' / 'pseudo_000042_shot_000055.jpg').exists()

    def test_review_vs_review_same_source_both_kept(self):
        # 罕见情况: 同一源审了两遍(审伪标输出 + 审源图) → 都保留, 不去重
        with tempfile.TemporaryDirectory() as td:
            root = self._dataset(td)
            for name in ('review_000042_pseudo_000042_shot_000055',
                         'review_000043_shot_000055'):
                (root / 'images' / f'{name}.jpg').touch()
                (root / 'labels' / f'{name}.txt').touch()
            assert _dedup_keep_review(root) == 0


class TestIngest:
    def test_moves_images_and_labels(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / 'data'
            (root / 'images').mkdir(parents=True)
            (root / 'labels').mkdir(parents=True)
            staging = Path(td) / 'staging'
            (staging / 'images').mkdir(parents=True)
            (staging / 'labels').mkdir(parents=True)
            (staging / 'images' / 'pseudo_000001_shot_100002.jpg').touch()
            (staging / 'labels' / 'pseudo_000001_shot_100002.txt').touch()
            n = _ingest(root, staging)
            assert n == 1
            assert (root / 'images' / 'pseudo_000001_shot_100002.jpg').exists()
            assert (root / 'labels' / 'pseudo_000001_shot_100002.txt').exists()
            assert not (staging / 'images' / 'pseudo_000001_shot_100002.jpg').exists()

    def test_skips_missing_label(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / 'data'
            (root / 'images').mkdir(parents=True)
            (root / 'labels').mkdir(parents=True)
            staging = Path(td) / 'staging'
            (staging / 'images').mkdir(parents=True)
            (staging / 'labels').mkdir(parents=True)
            (staging / 'images' / 'pseudo_000001_shot_100002.jpg').touch()
            n = _ingest(root, staging)
            assert n == 0
            assert not (root / 'images' / 'pseudo_000001_shot_100002.jpg').exists()

    def test_none_staging_returns_zero(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / 'data'
            (root / 'images').mkdir(parents=True)
            (root / 'labels').mkdir(parents=True)
            assert _ingest(root, None) == 0


class TestSourceKeyNewFormat:
    def test_dated_batch_distinct_from_old(self):
        # 新格式(带日期批次)与旧格式(无批次)即使序号相同也不是同一源
        assert _source_key('shot_20260808_000042') != _source_key('shot_000042')
        assert _source_key('shot_20260808_000042') == 'shot_20260808_000042'

    def test_embedded_full_source(self):
        assert _source_key('pseudo_000042_shot_20260808_000042') == 'shot_20260808_000042'
        assert _source_key('review_000042_pseudo_000042_shot_20260808_000042') \
            == 'shot_20260808_000042'
        assert _source_key('review_000042_shot_000042') == 'shot_000042'
