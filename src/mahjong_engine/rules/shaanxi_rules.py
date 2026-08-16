"""陕西推倒胡规则实现。"""

from src.mahjong_core import Hand, TileSetConfig
from src.mahjong_engine.judges.action_judge import KongResult, can_kong, can_pong
from src.mahjong_engine.judges.tenpai_judge import WaitingTile, get_waiting_tiles
from src.mahjong_engine.judges.win_judge import WinResult, is_winning_hand
from src.mahjong_engine.rules.interface import IRuleSet


class ShaanxiRules(IRuleSet):
    """陕西推倒胡。

    规则要点:
        - 136张牌 (万筒条各36 + 风16 + 箭12)
        - 不允许吃牌
        - 允许碰牌、杠牌(明/暗/加杠)
        - 允许特殊胡牌型: 七对、十三幺
        - 无万能牌
    """

    def name(self) -> str:
        return 'shaanxi'

    def get_tile_set(self) -> TileSetConfig:
        return TileSetConfig.standard_136()

    def is_winning_hand(self, hand: Hand, new_tile: int | None = None) -> WinResult:
        tiles = list(hand)
        if new_tile is not None:
            tiles = tiles + [new_tile]
        return is_winning_hand(tiles)

    def get_waiting_tiles(self, hand: Hand) -> list[WaitingTile]:
        enabled = self.get_tile_set().enabled_tiles
        return get_waiting_tiles(list(hand), enabled)

    def can_pong(self, hand: Hand, tile: int) -> bool:
        return can_pong(list(hand), tile)

    def can_kong(self, hand: Hand, tile: int) -> KongResult:
        return can_kong(list(hand), tile)

    def can_chi(self, hand: Hand, tile: int) -> list[list[int]]:
        # 陕西推倒胡不允许吃牌
        return []

    def is_wild_tile(self, tile: int) -> bool:
        return False
