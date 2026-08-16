"""现行管道共享类型 — 一帧建议与对手推断快照。

Advice: build_advice 产物, 客户端渲染输入;
InferenceResult: M3 粒子滤波推断快照(听牌/等待/放铳率)。
旧 GameSession/AssistantSession 已删除(2026-08-15 清理)。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.mahjong_ai.efficiency.discard_selector import DiscardRecommendation
from src.mahjong_ai.state.snapshot import GameSnapshot
from src.mahjong_engine.judges.tenpai_judge import WaitingTile

#: 三家对手(自己不算对手)
OPPONENT_PLAYERS = ('right_river', 'top_river', 'left_river')


@dataclass
class InferenceResult:
    """三家对手推断快照(粒子滤波输出)。"""

    tenpai_probs: dict[str, float] = field(default_factory=dict)
    waiting: dict[str, list[int]] = field(default_factory=dict)
    discard_risk: dict[int, float] = field(default_factory=dict)
    risk_tiles: list[int] = field(default_factory=list)


@dataclass
class Advice:
    """一帧的完整建议(供客户端渲染)。"""

    snapshot: GameSnapshot
    discard: DiscardRecommendation | None = None
    waiting: list[WaitingTile] | None = None
    inference: InferenceResult | None = None
    risk_warning: str | None = None
