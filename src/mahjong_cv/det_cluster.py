"""检测框空间聚类 — DBSCAN 密度聚类 + 特征分类(手牌/副露/牌河)。

2026-08-11 由 y 分带方案整体替换为 DBSCAN:
    - eps 自适应(平均框宽 × 2.0, 保证手牌连通又不跨区域)
    - 簇特征: 中心/平均尺寸/数量/紧密度(packing)/线性度
    - 分类规则: 手牌(cy 最大+水平线性+count≥3+尺寸锚) → 副露
      (密集 2-4 张簇, 尺寸≈手牌, 不限位置 — 自己的/别家副露都识别)
      → 牌河(剩余簇, 含单张=未成簇的牌河牌)
    - 接口: cluster_dets(dets, conf_min) → (hand, melds, rivers)

已知限制: 副露紧贴手牌(间隙≈内部间距)时与手牌并簇(物理不可分);
单张(刚摸的第 14 张手牌)不判副露。
"""

from __future__ import annotations

from statistics import mean, pstdev

from src.mahjong_cv.detections import TileDet

#: 局部 eps 倍率: 两框中心距 < 此值 × min(两框宽) 则连通。
#: 透视补偿(近大远小): 大框用大邻域, 小框用小邻域 — 比全局均值
#: 更精准(全局均值被大框拉高, 小框区域阈值失真, 易跨区域误连)
_EPS_RATIO = 2.0
#: DBSCAN 最小簇框数(1: 单张也成簇 — 副露漏检成 1-2 张不能丢;
#: 单张分类靠尺寸锚: ≈手牌尺寸 → 副露, 小框 → 牌河)
_MIN_SAMPLES = 1
#: 手牌最少框数(漏检容差)
_HAND_MIN = 3
#: 手牌尺寸锚(弱防御): 均高 ≥ 此倍 × 全局平均框高(含自身)。
#: 手牌 30 vs 牌河 15 差 2 倍, 0.75 阈值防「无手牌时横排牌河被误判」;
#: 副露与手牌同尺寸, 不参与锚基准(按 cy 最大优先选手牌)
_HAND_SIZE_RATIO = 0.75
#: 水平线性度: y 中心标准差 < 此值 × 簇均高
_LINEAR_TOL = 0.5
#: 副露最大框数(碰/吃 3, 杠 4)
_MELD_MAX = 4
#: 副露紧密度阈值(packing = 框面积和 / bbox 面积)。
#: 框间无空隙(间距 ≤ 1x 框宽)packing ≥ 1.0 — 与框尺寸无关,
#: 屏幕/实物比例通用; 稀疏排列(牌河行/手牌)packing < 1, 不进副露
_MELD_PACKING = 1.0
#: 副露尺寸: 均高 ≥ 此倍 × 手牌均高。
#: 0.8 假设"副露与手牌同尺寸" — 但别家副露在桌面中央有透视缩小
#: (手牌在底部近处最大, 中央远), 0.8 卡边界致别家碰/吃漏判成牌河;
#: 0.7 覆盖透视, 牌河小框(15px vs 手牌 30px = 0.5)仍不会误判
_MELD_SIZE_RATIO = 0.7
#: 聚类入口置信度下限(低置信度噪声框不参与聚类; 标定脚本传 0.0 保口径)
_CONF_MIN = 0.25


def _cx(d: TileDet) -> float:
    return (d.x1 + d.x2) / 2


def _cy(d: TileDet) -> float:
    return (d.y1 + d.y2) / 2


def _w(d: TileDet) -> float:
    return d.x2 - d.x1


def _h(d: TileDet) -> float:
    return d.y2 - d.y1


def _dbscan(
    dets: list[TileDet], eps_ratio: float, min_samples: int,
) -> tuple[list[list[int]], list[int]]:
    """DBSCAN(局部 eps): 中心距 < eps_ratio × min(两框宽) 则连通。

    局部邻域半径(透视补偿): 大框(近)用大半径, 小框(远)用小半径 —
    不需要全局平均(全局均值被大框拉高, 小框区域阈值失真)。
    簇大小 < min_samples 为噪声。O(n²) 距离 — 每帧几十框, 可接受。
    """
    n = len(dets)
    neighbors: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        xi, yi = _cx(dets[i]), _cy(dets[i])
        wi = _w(dets[i])
        for j in range(i + 1, n):
            if ((xi - _cx(dets[j])) ** 2 + (yi - _cy(dets[j])) ** 2) ** 0.5 \
                    < eps_ratio * min(wi, _w(dets[j])):
                neighbors[i].append(j)
                neighbors[j].append(i)
    visited = [False] * n
    clusters: list[list[int]] = []
    noise: list[int] = []
    for i in range(n):
        if visited[i]:
            continue
        comp: list[int] = []
        stack = [i]
        visited[i] = True
        while stack:
            k = stack.pop()
            comp.append(k)
            for nb in neighbors[k]:
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)
        if len(comp) >= min_samples:
            clusters.append(comp)
        else:
            noise.extend(comp)
    return clusters, noise


def _features(dets: list[TileDet], idx: list[int]) -> dict[str, float | bool]:
    """簇特征: 中心 y/平均高/数量/紧密度/水平线性度。"""
    ys = [_cy(dets[i]) for i in idx]
    hs = [_h(dets[i]) for i in idx]
    avg_h = mean(hs)
    # bbox 按框外缘(中心跨度对同排簇为 0 → 紧密度除零)
    bbox_w = max(dets[i].x2 for i in idx) - min(dets[i].x1 for i in idx)
    bbox_h = max(dets[i].y2 for i in idx) - min(dets[i].y1 for i in idx)
    bbox_area = bbox_w * bbox_h
    return {
        'cy': mean(ys), 'avg_h': avg_h,
        'count': len(idx),
        # 紧密度 = 框面积和 / bbox 面积(紧贴无空隙 → ≥1, 与框尺寸无关)
        'packing': (sum(_w(dets[i]) * _h(dets[i]) for i in idx) / bbox_area)
        if bbox_area > 0 else 0.0,
        'h_linear': pstdev(ys) < _LINEAR_TOL * avg_h,   # 水平排列(y 集中)
    }


def _classify(
    dets: list[TileDet],
    clusters: list[list[int]],
    global_avg_h: float,
) -> tuple[list[TileDet], list[list[TileDet]], list[list[TileDet]]]:
    """簇 → (手牌, 副露组, 牌河簇)。牌河簇含单张(未成簇的牌河单张)。"""
    feats = [_features(dets, c) for c in clusters]

    # 1. 手牌: 水平线性 + 数量达标, 按 cy 降序取第一个满足尺寸锚的
    #    (牌河都在手牌上方/两侧, cy 最大天然锁定手牌;
    #     锚只做弱防御: 无手牌时横排牌河(小框)不被误判)
    hand_idx: list[int] | None = None
    cands = [c for c, f in zip(clusters, feats, strict=True)
             if f['count'] >= _HAND_MIN and f['h_linear']]
    for c in sorted(cands, key=lambda c: -_features(dets, c)['cy']):
        if _features(dets, c)['avg_h'] >= _HAND_SIZE_RATIO * global_avg_h:
            hand_idx = c
            break
    hand = [dets[i] for i in hand_idx] if hand_idx is not None else []
    hand_avg_h = _features(dets, hand_idx)['avg_h'] if hand_idx is not None \
        else global_avg_h

    # 2. 副露/牌河: 非手牌簇, 密集小簇(2-4张)+ 尺寸 ≈ 手牌 → 副露;
    #    其余(含单张/噪声) → 牌河(单张 = 未成簇的牌河牌)。
    #    不限位置: 自己的副露紧贴手牌, 别家副露在桌面(碰/杠亮出) —
    #    都从牌河剔除, 不再误计为打出。
    #    单张不判副露: 近手牌单张 = 刚摸的第 14 张手牌(悬浮), 判副露
    #    会让 tracker 误确认"碰", 手牌数错乱。
    melds: list[list[TileDet]] = []
    rivers: list[list[TileDet]] = []
    for c, f in zip(clusters, feats, strict=True):
        if c is hand_idx:
            continue
        if (_MELD_MAX >= f['count'] >= 2
                and f['packing'] >= _MELD_PACKING
                and f['avg_h'] >= _MELD_SIZE_RATIO * hand_avg_h):
            melds.append([dets[i] for i in c])
        else:
            rivers.append([dets[i] for i in c])
    return hand, melds, rivers


def cluster_dets(
    dets: list[TileDet],
    conf_min: float = _CONF_MIN,
) -> tuple[list[TileDet], list[list[TileDet]], list[list[TileDet]]]:
    """检测框 DBSCAN 聚类归属, 返回 (手牌, 副露组, 牌河簇)。

    牌河簇含单张(未成簇的牌河牌/噪声 — tracker 按簇差分出牌事件)。
    eps 自适应: 平均框宽 × _EPS_RATIO(保证手牌连通, 又不过度跨区域)。
    手牌未识别时(漏检严重/无手牌)返回空手牌, 其余按簇归类。
    conf_min: 聚类入口置信度过滤; 标定/分析脚本传 0.0 保持全量口径。
    """
    if not dets:
        return [], [], []
    if conf_min > 0:
        dets = [d for d in dets if d.conf >= conf_min]
    if not dets:  # 过滤后为空(全低 conf) — 与空输入同处理
        return [], [], []
    global_avg_h = mean(_h(d) for d in dets)
    clusters, _noise = _dbscan(dets, _EPS_RATIO, _MIN_SAMPLES)
    return _classify(dets, clusters, global_avg_h)
