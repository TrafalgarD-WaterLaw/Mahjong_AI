"""AI 策略入口 — 依赖注入 IRuleSet。"""

from src.mahjong_ai.efficiency.discard_selector import DiscardRecommendation, recommend_discard
from src.mahjong_core import Hand
from src.mahjong_engine.rules.interface import IRuleSet


class StrategyEngine:
    """麻将AI策略引擎。

    通过 IRuleSet 接口注入规则，提供出牌推荐。
    """

    def __init__(self, rule_set: IRuleSet) -> None:
        self._rules = rule_set

    @property
    def rule_set(self) -> IRuleSet:
        return self._rules

    def recommend_discard(self, hand: Hand,
                          melds: list[list[int]] | None = None,
                          exclude: set[int] | None = None) -> DiscardRecommendation:
        """给定手牌，推荐最优出牌(支持副露)。

        Args:
            hand: 手牌(14张摸牌后; 副露已剔除时 14-3k 张)
            melds: 副露组(亮出的牌, 手牌不含副露时展开合并评估)
            exclude: 候选排除的 tile(副露牌混入手牌时用)

        Returns:
            出牌推荐
        """
        tiles = list(hand)
        enabled = self._rules.get_tile_set().enabled_tiles
        return recommend_discard(tiles, enabled, melds, exclude)
