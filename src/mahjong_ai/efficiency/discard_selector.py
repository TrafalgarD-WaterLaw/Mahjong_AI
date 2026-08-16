"""出牌选择器 — 推荐最优出牌。"""

from dataclasses import dataclass, field

from src.mahjong_ai.efficiency.tile_efficiency import evaluate_discard
from src.mahjong_core.tile import tile_display


@dataclass
class DiscardRecommendation:
    """出牌推荐。"""
    tile: int                       # 推荐打出的牌
    reason: str = ""                # 推荐理由
    shanten_before: int = 0         # 打出前向听数
    shanten_after: int = 0          # 打出后向听数
    acceptance: int = 0             # 打出后若听牌的听牌面数
    alternatives: list[tuple[int, float]] = field(default_factory=list)


def _expand_melds(melds: list[list[int]] | None) -> list[int]:
    """副露展开: 每组亮出的牌(碰/杠 = 3 张同牌, 杠含 1 张别家)。

    展开后与手牌合并 = 标准 14 张(摸牌后)或 13 张(打出后),
    复用 13/14 张标准向听/推荐引擎 — 不需要副露专用公式。
    """
    if not melds:
        return []
    return [t for group in melds for t in group[:3]]


def recommend_discard(
    tiles: list[int],
    enabled_tiles: frozenset[int] | None = None,
    melds: list[list[int]] | None = None,
    exclude: set[int] | None = None,
) -> DiscardRecommendation:
    """从手牌中推荐最优出牌(支持副露)。

    Args:
        tiles: 手牌(14张摸牌后; 副露已剔除时 14-3k 张, 如碰1次=11张)
        enabled_tiles: 牌库启用的牌
        melds: 副露组(每组亮出的牌, 如碰=[t,t,t]); 手牌不含副露时
            展开合并成 14 张评估
        exclude: 候选排除的 tile(如副露牌 — 无论是否混入手牌都不能打)

    Returns:
        出牌推荐，含备选方案

    副露未识别时的处理: 副露牌混入手牌(如碰后检测 14 张含 3 张副露) —
    副露刻子天然构成 1 面, 不影响手牌构成评估, 直接按检测手牌计算,
    仅用 exclude 排除亮出的副露牌。
    """
    meld_tiles = _expand_melds(melds)
    combined = tiles + meld_tiles
    # 推荐是「现有牌里打哪张最优」的相对判断 — 与张数无关,
    # 两张牌也能比较; 少于 2 张没有选择可言
    if not 2 <= len(combined) <= 14:
        raise ValueError(f"推荐需要 2-14 张(含副露展开), 当前 {len(combined)} 张"
                         f"({len(tiles)} 手牌 + {len(meld_tiles)} 副露)")
    missing = 14 - len(combined)  # 漏检张数(推荐基于不完整手牌, 相对比较仍有效)

    # 评估所有候选(排除副露牌 — 亮出的牌不能再打)
    scores = evaluate_discard(combined, enabled_tiles)
    excluded = set(meld_tiles) | (exclude or set())
    scores = [s for s in scores if s.tile not in excluded]
    if not scores:
        # 检测到的牌全是副露牌(真手牌全漏检) → 没有可打出的候选
        raise ValueError('检测到的牌均为副露牌, 无可打出的候选')

    best = scores[0]
    shanten_before = best.shanten_after  # 当前向听数 = 打出任意牌后的最小向听数

    # 构建推荐
    reason_parts = []
    if best.is_tenpai:
        reason_parts.append(f"打出 {tile_display(best.tile)} 后听牌")
        if best.acceptance > 0:
            reason_parts.append(f"听 {best.acceptance} 种牌")
    else:
        reason_parts.append(f"打出 {tile_display(best.tile)} 向听数={best.shanten_after}")
    if melds:
        reason_parts.append(f"(副露 {len(melds)} 组)")
    if missing > 0:
        reason_parts.append(f"(手牌 {len(tiles)} 张, 可能漏检)")

    # 备选方案
    alternatives: list[tuple[int, float]] = []
    for i, s in enumerate(scores[:4]):
        if i == 0:
            continue
        score_val = float(10 - s.shanten_after * 3 + min(s.acceptance, 5))
        alternatives.append((s.tile, score_val))

    return DiscardRecommendation(
        tile=best.tile,
        reason=" ".join(reason_parts),
        shanten_before=shanten_before,
        shanten_after=best.shanten_after,
        acceptance=best.acceptance,
        alternatives=alternatives,
    )
