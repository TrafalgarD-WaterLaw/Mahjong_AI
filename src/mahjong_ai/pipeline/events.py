"""事件流观测事件 — 感知层报"发生了什么", 不带归属结论。

事件单调追加、永不改写 — 事件日志即复盘文件(对局报告的直接输入)。
时间建模(轨迹连续性/确认窗口)由解码器承担, 提取器只做出现/成组/
手牌变化的确认。归属是解码器输出的概率, 不是事件的一部分。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

#: 四家(命名与现行一致, 便于复用客户端/推断)
PLAYERS = ('my_river', 'right_river', 'top_river', 'left_river')


@dataclass(frozen=True)
class MotionEvidence:
    """打出动画轨迹证据(可选, v1 提取器不产出): 起点坐标 + 置信度。

    未来线下适配器的主信号(动作轨迹起点 = 谁的手), PC 动画快检出率
    一般 — 解码器 v1 忽略此字段, 事件日志原样保留。
    """

    start_x: float
    start_y: float
    conf: float


@dataclass(frozen=True)
class TileAppeared:
    """桌面出现一张新牌(潜在打出)。cx/cy 为像素落点中心。"""

    eid: int
    tile: int
    track_id: int
    cx: float
    cy: float
    conf: float
    frame: int
    ts: float
    motion: MotionEvidence | None = None


@dataclass(frozen=True)
class MeldFormed:
    """一组紧贴亮牌出现(碰/杠候选)。bbox 为组外接框(像素)。

    player: 事件推断出的副露者(可选) — MeldInferrer 从牌河消失 +
    台阶/行动权推断, 比空间归属可靠(副露区域框可能压错); None =
    未知, 解码器回退空间软归属。
    """

    eid: int
    tiles: tuple[int, ...]
    cx: float
    cy: float
    bbox: tuple[float, float, float, float]
    frame: int
    ts: float
    player: str | None = None


@dataclass(frozen=True)
class TileClaimed:
    """刚出现的牌被副露吸收。claimed=None = 秒碰(该牌从未出现)。"""

    eid: int
    claimed: int | None
    meld: int
    frame: int
    ts: float


@dataclass(frozen=True)
class TileVanished:
    """牌河最新一张消失(被碰/吃拿走) — RiverWatcher 产出。

    river: 该牌所属牌河(PLAYERS 名); appeared_eid: 该牌对应的
    TileAppeared eid — TileClaimed 直接用它配对, 不再需要
    "最近同类打出"的 10 秒兜底。
    """

    eid: int
    tile: int
    river: str
    appeared_eid: int
    frame: int
    ts: float


@dataclass(frozen=True)
class HandChanged:
    """我的手牌去抖后张数变化。13→14 = 我摸牌, 14→13 = 我打出。"""

    eid: int
    n_old: int
    n_new: int
    frame: int
    ts: float


@dataclass(frozen=True)
class FlowerShown:
    """花牌亮出(不进归属, 仅日志 — 花牌是补牌, 不是打出)。"""

    eid: int
    tile: int
    cx: float
    cy: float
    frame: int
    ts: float


Event = (TileAppeared | MeldFormed | TileClaimed | TileVanished
         | HandChanged | FlowerShown)


def event_to_dict(ev: Event) -> dict[str, object]:
    """事件 → 日志行 dict(复盘/核对用)。"""
    d = asdict(ev)
    d['type'] = type(ev).__name__
    return d
