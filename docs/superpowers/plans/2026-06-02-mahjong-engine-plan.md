# mahjong_engine 实现计划

> **目标**: 实现规则引擎模块 — IRuleSet 接口 + 回溯法胡牌判定 + 听牌分析 + 操作判定 + 陕西推倒胡
> **TDD**: 每个任务先写测试(确认失败) → 实现(确认通过) → 提交

---

### 任务 1: 回溯法胡牌判定

**文件:** `tests/test_engine/test_win_judge.py` + `src/mahjong_engine/judges/win_judge.py`

**测试用例:**

| 用例 | 手牌 | 期望 |
|------|------|------|
| 标准胡-顺子 | 123万456筒789条东东+摸5万 | can_win=True, pattern=standard |
| 标准胡-刻子 | 111万222筒333条东东+摸9万 | can_win=True |
| 七对 | 11223344556677万 | can_win=True, pattern=seven_pairs |
| 十三幺 | 1万9万1筒9筒1条9条东南西北中发白+摸东 | can_win=True, pattern=thirteen_orphans |
| 不胡 | 123万456筒789条东西+摸白 | can_win=False |
| 不听 | 13张杂乱牌 | can_win=False |

**实现函数:**

```python
def is_winning_hand(tiles: list[int]) -> WinResult:
    """14张牌判断是否胡牌。返回 WinResult 含胡牌类型和拆解。"""
```

---

### 任务 2: 听牌分析

**文件:** `tests/test_engine/test_tenpai_judge.py` + `src/mahjong_engine/judges/tenpai_judge.py`

**实现函数:**

```python
def get_waiting_tiles(hand_tiles: list[int], enabled_tiles: frozenset[int], judge: Callable) -> list[WaitingTile]:
    """13张手牌，遍历每种启用牌逐一尝试 is_winning_hand"""
```

---

### 任务 3: 操作判定

**文件:** `tests/test_engine/test_action_judge.py` + `src/mahjong_engine/judges/action_judge.py`

**实现函数:**

```python
def can_pong(hand_tiles: list[int], tile: int) -> bool:
    """手牌中是否有≥2张相同牌"""

def can_kong(hand_tiles: list[int], tile: int, pongs: list[int]) -> KongResult:
    """暗杠(手牌4张) / 明杠(手牌3张+别家) / 加杠(已碰+手牌1张)"""

def can_chi(hand_tiles: list[int], tile: int, wild_tiles: frozenset) -> list[list[int]]:
    """返回可行的顺子组合，陕西规则返回空"""
```

---

### 任务 4: IRuleSet 接口 + 陕西规则实现

**文件:** `tests/test_engine/test_shaanxi_rules.py` + `src/mahjong_engine/rules/interface.py` + `src/mahjong_engine/rules/shaanxi_rules.py`

**陕西规则配置:**
- 牌库: 136张 (万筒条+风+箭)
- 特殊牌型: 七对、十三幺
- 吃牌: 不允许
- 碰牌: 允许
- 杠牌: 明/暗/加杠
- 万能牌: 无

---

### 任务 5: 模块导出 + 全面验证

**文件:** `src/mahjong_engine/__init__.py` + `src/mahjong_engine/rules/registry.py`

**registry.py:**

```python
from src.mahjong_engine.rules.shaanxi_rules import ShaanxiRules

_rules: dict[str, "IRuleSet"] = {}

def register(rule: "IRuleSet") -> None:
    _rules[rule.name()] = rule

def get(name: str) -> "IRuleSet":
    return _rules[name]

def list_names() -> list[str]:
    return list(_rules.keys())

# 内置注册
register(ShaanxiRules())
```

**最终验证:** pytest + ruff + mypy
