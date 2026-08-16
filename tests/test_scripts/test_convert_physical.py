"""实物数据集转换器的单元测试(纯函数部分)。"""
from scripts.convert_physical_dataset import remap_label_line
from src.mahjong_core.tile import BAI, DONG, MEI, T1, W1, W9


class TestRemapLabelLine:
    def test_yolo_order_to_tile_encoding(self):
        # YOLO 类序: 1=1C(一万), 0=1B(一条), 35=EW(东), 40=WD(白), 3=1F(梅)
        assert remap_label_line('1 0.5 0.5 0.1 0.2') == f'{W1} 0.5 0.5 0.1 0.2'
        assert remap_label_line('0 0.5 0.5 0.1 0.2') == f'{T1} 0.5 0.5 0.1 0.2'
        assert remap_label_line('33 0.5 0.5 0.1 0.2') == f'{W9} 0.5 0.5 0.1 0.2'
        assert remap_label_line('35 0.5 0.5 0.1 0.2') == f'{DONG} 0.5 0.5 0.1 0.2'
        assert remap_label_line('40 0.5 0.5 0.1 0.2') == f'{BAI} 0.5 0.5 0.1 0.2'
        assert remap_label_line('3 0.5 0.5 0.1 0.2') == f'{MEI} 0.5 0.5 0.1 0.2'

    def test_malformed_lines_rejected(self):
        # 10 列粘连行 / 越界类 id / 字段数不对 → None(跳过)
        assert remap_label_line('1 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9') is None
        assert remap_label_line('42 0.5 0.5 0.1 0.2') is None
        assert remap_label_line('abc 0.5 0.5 0.1 0.2') is None
        assert remap_label_line('1 0.5 0.5 0.1') is None
