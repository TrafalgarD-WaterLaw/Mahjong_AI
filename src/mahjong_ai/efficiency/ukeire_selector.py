"""出牌推荐(纯牌效) — 向听数优先 + 有效进张枚数。

现行 recommend_discard(向听 + lonely 启发式)同向听候选不看进张面,
推荐不符合牌理(设计: docs/superpowers/specs/2026-08-15-ukeire-discard-design.md)。
本模块打分: 1. 打出后向听数(小优先) 2. 有效进张总枚数(改善向听
的进张, 按剩余枚数计) 3. 保持向听的进张枚数(好形率近似)
4. lonely 兜底(排序稳定)。旧 recommend_discard 保留不动。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from src.mahjong_ai.efficiency.discard_selector import (
    DiscardRecommendation,
    _expand_melds,
)
from src.mahjong_ai.efficiency.shanten import calculate_shanten
from src.mahjong_ai.efficiency.tile_efficiency import _lonely
from src.mahjong_core.tile import tile_display
from src.mahjong_engine.judges.tenpai_judge import get_waiting_tiles


@lru_cache(maxsize=16384)
def _shanten_cached(tiles: tuple[int, ...]) -> int:
    """向听数(排序 tuple 缓存 — 进张枚举大量重复计算)。"""
    return calculate_shanten(list(tiles))


@dataclass
class _Score:
    tile: int
    shanten_after: int
    ukeire: int          # 有效进张总枚数(改善向听)
    keep: int            # 保持向听的进张枚数(好形率近似)
    lonely: int          # 搭子潜力(末位 tie-break, 沿用旧启发式)
    is_tenpai: bool = False
    acceptance: int = 0  # 听牌时的等待牌总枚数
    n_waits: int = 0     # 听牌时的等待牌种类数


def _ukeire(remaining: list[int], available: dict[int, int],
            shanten_after: int) -> tuple[int, int]:
    """进张枚数: (改善向听的枚数, 保持向听的枚数)。

    打出后 13 张 + 进张 = 14 张: calculate_shanten 对 14 张先做胡判
    (−1 = 胡), 所以听牌手牌(shanten_after=0)的"改善进张"就是等待牌。
    """
    base = tuple(sorted(remaining))
    improve = keep = 0
    for d, n in available.items():
        if n <= 0 or d == 34:
            continue
        s = _shanten_cached(tuple(sorted(base + (d,))))
        if s < shanten_after:
            improve += n
        elif s == shanten_after:
            keep += n
    return improve, keep


def recommend_by_ukeire(
    tiles: list[int],
    available: dict[int, int],
    enabled_tiles: frozenset[int] | None = None,
    melds: list[list[int]] | None = None,
    exclude: set[int] | None = None,
) -> DiscardRecommendation:
    """纯牌效出牌推荐(接口与 recommend_discard 同族, 多 available)。

    available: 每类牌剩余可摸枚数(4 − 我手牌 − 各河 − 已亮副露)。
    """
    meld_tiles = _expand_melds(melds)
    combined = tiles + meld_tiles
    if not 2 <= len(combined) <= 14:
        raise ValueError(f'推荐需要 2-14 张(含副露展开), 当前 {len(combined)} 张'
                         f'({len(tiles)} 手牌 + {len(meld_tiles)} 副露)')
    excluded = set(meld_tiles) | (exclude or set())
    avail = {t: max(0, available.get(t, 0)) for t in range(34)}
    scores: list[_Score] = []
    for tile in sorted(set(combined) - excluded):
        remaining = list(combined)
        remaining.remove(tile)
        sa = _shanten_cached(tuple(sorted(remaining)))
        improve, keep = _ukeire(remaining, avail, sa)
        score = _Score(tile=tile, shanten_after=sa, ukeire=improve,
                       keep=keep, lonely=_lonely(tile, combined))
        if sa == 0 and len(remaining) == 13 and enabled_tiles is not None:
            ws = get_waiting_tiles(remaining, enabled_tiles)
            score.acceptance = sum(avail.get(w.tile, 0) for w in ws)
            score.n_waits = len(ws)
            score.is_tenpai = True
        scores.append(score)
    if not scores:
        raise ValueError('检测到的牌均为副露牌, 无可打出的候选')
    scores.sort(key=lambda s: (s.shanten_after, -s.ukeire, -s.keep,
                               s.lonely))
    best = scores[0]
    reason_parts = []
    if best.is_tenpai:
        reason_parts.append(f'打出 {tile_display(best.tile)} 后听牌, '
                            f'听 {best.n_waits} 种 {best.acceptance} 枚')
    else:
        reason_parts.append(f'打出 {tile_display(best.tile)} '
                            f'后 {best.shanten_after} 向听, '
                            f'有效进张 {best.ukeire} 枚')
    if melds:
        reason_parts.append(f'(副露 {len(melds)} 组)')
    missing = 14 - len(combined)
    if missing > 0:
        reason_parts.append(f'(手牌 {len(tiles)} 张, 可能漏检)')
    alternatives = [(s.tile, float(s.ukeire)) for s in scores[1:4]]
    return DiscardRecommendation(
        tile=best.tile,
        reason=' '.join(reason_parts),
        shanten_before=best.shanten_after,
        shanten_after=best.shanten_after,
        acceptance=best.acceptance,
        alternatives=alternatives,
    )
