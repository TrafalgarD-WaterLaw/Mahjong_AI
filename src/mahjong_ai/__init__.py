"""mahjong_ai — 麻将AI策略模块。

提供:
    - calculate_shanten: 向听数计算
    - evaluate_discard: 牌效评估
    - recommend_discard: 出牌推荐
    - StrategyEngine: 策略引擎(依赖注入IRuleSet)
"""
from src.mahjong_ai.efficiency.discard_selector import DiscardRecommendation, recommend_discard
from src.mahjong_ai.efficiency.shanten import calculate_shanten
from src.mahjong_ai.efficiency.tile_efficiency import TileScore, evaluate_discard
from src.mahjong_ai.strategy import StrategyEngine

__all__ = [
    'calculate_shanten',
    'evaluate_discard', 'TileScore',
    'recommend_discard', 'DiscardRecommendation',
    'StrategyEngine',
]
