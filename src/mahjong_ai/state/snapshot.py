"""牌局状态快照数据类 — 现行管道的共享类型。

GameStateTracker(旧状态机)已删除(2026-08-15 清理); 这些数据类
是现行管道与保留工具的共同契约, 内容未改只换了家。
"""

from __future__ import annotations

from dataclasses import dataclass, field

TURN_MY = 'MY_TURN'
TURN_WAITING = 'WAITING'
TURN_UNKNOWN = 'UNKNOWN'

RIVER_NAMES = ('my_river', 'right_river', 'top_river', 'left_river')


@dataclass
class DiscardEvent:
    """检测到某家牌河新增一张牌(该家打出)。"""

    tile: int
    player: str  # 'my_river' | 'right_river' | 'top_river' | 'left_river'


@dataclass
class Meld:
    """已确认的副露(碰/杠/吃)。

    x1/y1/x2/y2: 副露组 bbox(吸收迟到补全牌用 — 紧贴该区域的
    新 ID 是漏检的第 3/4 张, 不是打出的牌)。
    """

    kind: str  # 'pong' | 'kong' | 'chi'
    tile: int
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 0.0
    y2: float = 0.0
    tiles: tuple[int, ...] = ()  # 全部亮牌(碰=(7,7,7), 吃=(6,7,8)); 空=旧路径只有 tile


@dataclass
class PlayerState:
    """某家的持久状态(所有推断只读这里, 不读瞬时检测)。

    hand_count: 推断的手牌数(基线 13, 副露确认联动调整);
    melds: 已确认副露(提交后不回退 — 副露永久存在, 漏检只是没看见);
    river: 已提交牌河序列(不回退)。
    """

    hand_count: int = 13
    melds: list[Meld] = field(default_factory=list)
    river: list[int] = field(default_factory=list)
    river_events: list[DiscardEvent] = field(default_factory=list)


@dataclass
class GameSnapshot:
    """去抖后的牌局状态。

    players 是唯一事实源(river/melds/river_events 都从这里读),
    顶层字段只保留跨行计算的结果(my_hand/turn/meld_detected)。
    """

    my_hand: list[int] = field(default_factory=list)
    #: 手牌塌陷时的兜底显示(未知槽 = None; 检测健康 → None)
    my_hand_fallback: list[int | None] | None = None
    turn: str = TURN_UNKNOWN
    meld_detected: bool = False
    river_events: list[DiscardEvent] = field(default_factory=list)
    players: dict[str, PlayerState] = field(default_factory=dict)
