"""欢乐推倒胡规则实现(腾讯欢乐麻将「推倒胡」模式)。"""

from src.mahjong_core import Hand
from src.mahjong_engine.judges.action_judge import can_chi
from src.mahjong_engine.rules.shaanxi_rules import ShaanxiRules


class HuanyuRules(ShaanxiRules):
    """欢乐推倒胡。

    与陕西推倒胡的唯一区别: 允许吃牌(上家打出时手牌可组成顺子)。
    其余规则(136张/碰/杠/七对/十三幺/无万能牌)完全继承。
    """

    def name(self) -> str:
        return 'huanyu'

    def can_chi(self, hand: Hand, tile: int) -> list[list[int]]:
        return can_chi(list(hand), tile)
