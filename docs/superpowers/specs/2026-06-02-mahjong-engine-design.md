# Phase 2 设计文档: mahjong_engine 规则引擎

- **日期**: 2026-06-02
- **状态**: 已批准
- **范围**: 麻将AI项目 Phase 2
- **依赖**: mahjong_core (Phase 1)

## 概述

实现规则引擎模块，通过 IRuleSet 宽接口封装胡牌判定、听牌分析、操作合法性判定。首发实现陕西推倒胡规则，架构支持后续扩展其他玩法。

## 设计决策

| 决策项 | 选择 | 理由 |
|--------|------|------|
| 胡牌算法 | 回溯法拆面子+雀头 | 直观可解释，14张牌实际性能足够 |
| 接口风格 | 宽接口 IRuleSet | 一个类包含全部判定，简单直观 |
| 听牌分析 | 遍历34种牌逐一尝试 | 简单可靠，仅对启用的牌遍历 |
| 碰/杠/吃 | 独立判定函数 | 逻辑简单，直接判断即可 |

## 1. IRuleSet 接口 (interface.py)

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from src.mahjong_core import Hand, TileSetConfig

@dataclass
class WinResult:
    can_win: bool
    pattern_type: str = ""          # "standard" | "seven_pairs" | "thirteen_orphans"
    breakdown: list[str] = field(default_factory=list)  # ["一万二万三万", ...]

@dataclass
class WaitingTile:
    tile: int
    count_remaining: int = 0        # 理论剩余张数(暂不计算)

@dataclass
class KongResult:
    can_kong: bool
    kong_type: str = ""             # "ming" | "an" | "jia"

class IRuleSet(ABC):
    @abstractmethod
    def name(self) -> str: ...
    @abstractmethod
    def get_tile_set(self) -> TileSetConfig: ...
    @abstractmethod
    def is_winning_hand(self, hand: Hand, new_tile: int | None = None) -> WinResult: ...
    @abstractmethod
    def get_waiting_tiles(self, hand: Hand) -> list[WaitingTile]: ...
    @abstractmethod
    def can_pong(self, hand: Hand, tile: int) -> bool: ...
    @abstractmethod
    def can_kong(self, hand: Hand, tile: int) -> KongResult: ...
    @abstractmethod
    def can_chi(self, hand: Hand, tile: int) -> list[list[int]]: ...
    @abstractmethod
    def is_wild_tile(self, tile: int) -> bool: ...
```

## 2. 回溯法胡牌算法 (win_judge.py)

核心逻辑:

```
is_winning_hand(hand, new_tile):
    1. 合并手牌 + new_tile (若提供)，得到14张
    2. 检查特殊牌型：
       - 七对: 7种牌各2张 → WinResult("seven_pairs")
       - 十三幺: 13种幺九牌+1张重复 → WinResult("thirteen_orphans")
    3. 统计每张牌数量，转为计数列表
    4. 回溯: 尝试每种数量≥2的牌作为雀头
       → 剩余牌中，从最小牌开始去除面子:
          - 刻子: 同牌≥3张 → 减3递归
          - 顺子: 同花色连续3张各≥1 → 各减1递归
       → 所有牌消完即为胡牌
    5. 记录拆解路径用于输出 breakdown
```

## 3. 听牌分析 (tenpai_judge.py)

```
get_waiting_tiles(hand):  # 13张手牌
    对牌库中每种启用的牌 t:
        if is_winning_hand(hand, t).can_win:
            加入等待列表
    返回等待牌列表
```

## 4. 陕西推倒胡 (shaanxi_rules.py)

- 牌库: 136张 (TileSetConfig.standard_136())
- 特殊牌型: 允许七对、十三幺
- 吃牌: 不允许 (can_chi 返回空)
- 碰牌: 手牌中有≥2张相同牌时可碰
- 杠牌: 明杠(手牌3张+别家打出) / 暗杠(手牌4张) / 加杠(已碰+手牌1张)
- 万能牌: 无

## 5. 模块文件结构

```
src/mahjong_engine/
├── __init__.py            # 导出
├── rules/
│   ├── interface.py       # IRuleSet + WinResult + WaitingTile + KongResult
│   ├── shaanxi_rules.py   # 陕西推倒胡
│   └── registry.py        # 规则注册表 {name: IRuleSet}
├── judges/
│   ├── win_judge.py       # 回溯法胡牌判定
│   ├── tenpai_judge.py    # 听牌分析
│   └── action_judge.py    # 碰/杠/吃
└── game_state.py          # 牌局状态(Phase 2 仅定义，不做完整实现)
```

## 6. 测试文件

```
tests/test_engine/
├── test_win_judge.py       # 标准胡/七对/十三幺/非胡/边界
├── test_tenpai_judge.py    # 单面听/多面听/不听
├── test_action_judge.py    # 碰/杠/吃
└── test_shaanxi_rules.py   # 陕西规则集成
```

## 7. Phase 2 不包含

- 牌河/对手追踪
- 牌局完整状态机
- JSON 规则配置加载
- 其他麻将玩法(四川、广东等)
