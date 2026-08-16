"""听牌分析 — 基于胡牌判定的等待牌计算。"""

from dataclasses import dataclass

from src.mahjong_engine.judges.win_judge import WinResult, is_winning_hand


@dataclass
class WaitingTile:
    """等待牌信息。"""
    tile: int
    count_remaining: int = 0  # 理论剩余张数(后续对手建模时使用)


def get_waiting_tiles(
    hand_tiles: list[int],
    enabled_tiles: frozenset[int],
) -> list[WaitingTile]:
    """计算手牌的听牌列表。

    Args:
        hand_tiles: 13张手牌
        enabled_tiles: 牌库中启用的牌(只遍历这些)

    Returns:
        等待牌列表，按牌值排序
    """
    if len(hand_tiles) != 13:
        raise ValueError(f"听牌分析需要13张手牌，当前{len(hand_tiles)}张")

    waiting: list[WaitingTile] = []
    for t in sorted(enabled_tiles):
        candidate = hand_tiles + [t]
        result: WinResult = is_winning_hand(candidate)
        if result.can_win:
            waiting.append(WaitingTile(tile=t))

    return waiting
