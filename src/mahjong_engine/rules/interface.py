"""IRuleSet 抽象接口。"""

from abc import ABC, abstractmethod

from src.mahjong_core import Hand, TileSetConfig
from src.mahjong_engine.judges.action_judge import KongResult
from src.mahjong_engine.judges.tenpai_judge import WaitingTile
from src.mahjong_engine.judges.win_judge import WinResult


class IRuleSet(ABC):
    """麻将规则集抽象接口。每种麻将玩法实现此接口。"""

    @abstractmethod
    def name(self) -> str:
        """规则名称，如 'shaanxi'。"""
        ...

    @abstractmethod
    def get_tile_set(self) -> TileSetConfig:
        """返回牌库配置。"""
        ...

    @abstractmethod
    def is_winning_hand(self, hand: Hand, new_tile: int | None = None) -> WinResult:
        """判断手牌是否胡牌。new_tile 为刚摸的牌(14张时4面子+1雀头用)。"""
        ...

    @abstractmethod
    def get_waiting_tiles(self, hand: Hand) -> list[WaitingTile]:
        """返回当前13张手牌的所有等待牌。"""
        ...

    @abstractmethod
    def can_pong(self, hand: Hand, tile: int) -> bool:
        """判断是否可以碰。"""
        ...

    @abstractmethod
    def can_kong(self, hand: Hand, tile: int) -> KongResult:
        """判断是否可以杠(明/暗/加杠)。"""
        ...

    @abstractmethod
    def can_chi(self, hand: Hand, tile: int) -> list[list[int]]:
        """判断是否可以吃，返回可行的顺子组合。"""
        ...

    @abstractmethod
    def is_wild_tile(self, tile: int) -> bool:
        """判断某牌是否为万能牌(赖子)。"""
        ...
