"""mahjong_cv — 麻将牌计算机视觉识别模块。

提供:
    - ScreenVision: YOLO 屏幕牌检测(35 类, class id = 牌编码)
    - det_cluster: 检测框空间聚类(手牌/牌河/副露归属)
    - TileDet: 检测结果数据类
    - yolo_to_tile: YOLO 类别ID → 牌整数映射(实物模型)
"""
