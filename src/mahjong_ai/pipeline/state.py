"""对局状态投影 — 解码器(事件日志)→ 玩家视图(纯函数)。

牌河 = 冻结事件按帧序投影(含被碰走的 — 打出是历史事实, 推断依据);
visible_river = 牌河中当前物理可见子集(被碰走的消失 — 显示用);
provisional = 未冻结事件按当前 MAP 投影(显示即时性: 打出 1-2 秒
即上屏, 冻结后转实; 推断/M3 也消费当前归属而非等冻结 —
15 秒的冻结等待会错过好几轮, 对手听牌推断失去时效)。
显示/推断拆分由事件驱动: 不依赖瞬时检测框, 无闪烁。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.mahjong_ai.pipeline.decoder import Attribution, WindowDecoder
from src.mahjong_ai.pipeline.events import PLAYERS


@dataclass
class MeldView:
    """已确认副露(投影视图)。kind: 'pong' | 'kong'。"""

    kind: str
    tiles: tuple[int, ...]


@dataclass
class PlayerView:
    """某家视角: 牌河历史(river)与物理可见牌河(visible_river)。"""

    player: str
    river: list[int] = field(default_factory=list)
    visible_river: list[int] = field(default_factory=list)
    melds: list[MeldView] = field(default_factory=list)


@dataclass
class GameView:
    """对局快照(投影结果 — 推断/显示只读这里)。"""

    my_hand: list[int]
    players: dict[str, PlayerView]
    unfrozen: list[Attribution]
    provisional: dict[str, list[int]] = field(default_factory=dict)
    #: 手牌塌陷时的兜底显示(未知槽 = None; 检测健康 → None)
    my_hand_fallback: list[int | None] | None = None


def project(decoder: WindowDecoder, my_hand: list[int],
            my_hand_fallback: list[int | None] | None = None) -> GameView:
    """解码器 → 游戏视图。

    provisional = 未冻结事件按当前 MAP 分家(帧序)— 回看纠错时
    待定牌会换家, 这正是软归属的语义; 冻结后进入 visible_river。
    """
    players = {p: PlayerView(player=p) for p in PLAYERS}
    claimed = decoder.claimed_ids()
    for at in sorted(decoder.frozen(), key=lambda a: a.frame):
        assert at.player is not None  # 冻结事件必有归属(解码器保证)
        pv = players[at.player]
        pv.river.append(at.tile)
        if at.eid not in claimed:
            pv.visible_river.append(at.tile)
    provisional: dict[str, list[int]] = {p: [] for p in PLAYERS}
    for at in sorted(decoder.unfrozen(), key=lambda a: a.frame):
        provisional[at.map()].append(at.tile)
    for player, ev in decoder.melds():
        tiles = ev.tiles
        if len(tiles) >= 4:
            kind = 'kong'
        elif len(set(tiles)) == 1:
            kind = 'pong'
        else:
            kind = 'chi'  # 三张不同牌面(真实对局: 吃被判成碰的修复)
        players[player].melds.append(MeldView(kind=kind, tiles=tiles))
    return GameView(my_hand=my_hand, players=players,
                    unfrozen=decoder.unfrozen(),
                    provisional=provisional,
                    my_hand_fallback=my_hand_fallback)
