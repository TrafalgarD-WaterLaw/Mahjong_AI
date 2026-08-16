# 出牌推荐重设计(纯牌效)实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新出牌推荐算法: 向听数优先 + 有效进张枚数(纯牌效), 替换 build_advice 里的推荐调用; 旧 recommend_discard 保留不动。

**Architecture:** 新模块 `efficiency/ukeire_selector.py`(打分 = 打出后向听 ↑ → 进张枚数 ↓ → 保持枚数 ↓ → lonely 兜底); runner 切到新函数并复用已计算的 available; eval 模拟器加 --policy 做新旧 A/B(向听曲线)。

**Tech Stack:** Python 3.11+, lru_cache, pytest, ruff, mypy。

## Global Constraints

- 项目不使用 git — 无 commit 步骤,任务以门禁结束。
- 所有命令用 `uv run` 前缀。
- 旧代码不删: `efficiency/discard_selector.py`、`tile_efficiency.py` 原样保留(新模块复用其 `DiscardRecommendation`/`_expand_melds`/`_lonely`)。
- 门禁: pytest 全量除 test_cv 预存 1 失败外全绿; ruff 改动文件干净; mypy 新模块 strict 干净。
- 设计依据: `docs/superpowers/specs/2026-08-15-ukeire-discard-design.md`。

---

### Task 1: ukeire_selector.py + 经典牌理单测

**Files:**
- Create: `src/mahjong_ai/efficiency/ukeire_selector.py`
- Test: `tests/test_ai/test_ukeire_selector.py`(创建)

**Interfaces:**
- Consumes: `DiscardRecommendation`/`_expand_melds`(discard_selector)、`calculate_shanten`(shanten)、`_lonely`(tile_efficiency)、`get_waiting_tiles`(tenpai_judge)、`tile_display`
- Produces:
  - `_shanten_cached(tiles: tuple[int, ...]) -> int`(lru_cache)
  - `_ukeire(remaining: list[int], available: dict[int, int], shanten_after: int) -> tuple[int, int]` — (改善枚数, 保持枚数)
  - `recommend_by_ukeire(tiles, available, enabled_tiles=None, melds=None, exclude=None) -> DiscardRecommendation`

- [ ] **Step 1: 写失败测试**

`tests/test_ai/test_ukeire_selector.py`:

```python
"""纯牌效出牌推荐 — 经典牌理用例。"""

from src.mahjong_ai.efficiency.ukeire_selector import (
    _shanten_cached,
    _ukeire,
    recommend_by_ukeire,
)
from src.mahjong_engine import get as get_rules

_ENABLED = frozenset(get_rules('huanyu').get_tile_set().enabled_tiles)
_FULL = {t: 4 for t in range(34)}


def _rec(tiles, available=None, melds=None, exclude=None):
    return recommend_by_ukeire(tiles, available or _FULL, _ENABLED,
                               melds=melds, exclude=exclude)


def test_honor_lonely_first():
    """字牌孤张最优先(结构完整时)。"""
    tiles = [0, 0, 0, 1, 1, 1, 2, 2, 2, 27, 30, 33, 5, 6]
    assert _rec(tiles).tile == 33


def test_break_edge_over_middle_float():
    """拆边张搭(1,2 万)保中张孤张(5 万) — 进张面 8 vs 12。"""
    tiles = [1, 2, 5, 9, 9, 9, 18, 19, 20, 27, 27, 27, 30, 30]
    assert _rec(tiles).tile in (1, 2)


def test_keep_two_pairs():
    """两对子形态: 拆搭子保两对(碰 + 两面进张最大)。"""
    tiles = [0, 0, 5, 5, 9, 10, 18, 19, 20, 27, 27, 27, 30, 30]
    rec = _rec(tiles)
    assert rec.tile in (9, 10)  # 拆 9-10 搭, 不打对子牌


def test_acceptance_reflects_availability():
    """听牌时等待枚数 = 剩余可摸枚数(全见则少计)。"""
    # 3 面子 + 27 对 + 4 对 + 浮张 33(14 张) → 打 33 听牌, 等 4/27
    tiles = [0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 27, 4, 4, 33]
    rec = _rec(tiles)
    assert rec.is_tenpai if hasattr(rec, 'is_tenpai') else True  # 占位: 见 Step 1 修正
    assert rec.tile == 33
    assert rec.acceptance == 6  # 4×4 张 + 27×2 张
    avail = dict(_FULL)
    avail[4] = 0  # 4 全可见 → 只剩 27 两张
    assert _rec(tiles, avail).acceptance == 2


def test_ukeire_respects_availability():
    """进张枚数按剩余可摸枚数计(全见 → 不计)。"""
    # 3 面子 + 30 对 + 边张 1,2 → 一向听; 进张: 1(对子化) 2(对子化)
    # 3(边张成面子) 30(对碰) — 满牌池共 4+4+4+2 = 14 枚
    remaining = [1, 2, 9, 9, 9, 18, 19, 20, 27, 27, 27, 30, 30]
    sa = _shanten_cached(tuple(sorted(remaining)))
    assert sa == 1
    improve, _keep = _ukeire(remaining, _FULL, sa)
    assert improve == 14
    avail = dict(_FULL)
    avail[3] = 0
    improve0, _ = _ukeire(remaining, avail, sa)
    assert improve0 == 10


def test_meld_expansion_and_exclude():
    """副露展开评估 + 副露牌排除(与旧接口同语义)。"""
    # 手牌 11 张(碰 9 后) + 碰展开 3 张 = 14 评估
    tiles = [1, 2, 5, 18, 19, 20, 27, 27, 27, 30, 30]
    rec = _rec(tiles, melds=[[9, 9, 9]])
    assert rec.tile in (1, 2)
    # 混入副露牌时排除(9 不能打)
    rec2 = _rec(tiles + [9, 9, 9], melds=[[9, 9, 9]])
    assert rec2.tile != 9
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_ai/test_ukeire_selector.py -v`
Expected: FAIL(ModuleNotFoundError: ukeire_selector)

- [ ] **Step 3: 实现**

`src/mahjong_ai/efficiency/ukeire_selector.py`:

```python
"""出牌推荐(纯牌效) — 向听数优先 + 有效进张枚数。

现行 recommend_discard(向听 + lonely 启发式)同向听候选不看进张面,
推荐不符合牌理(设计: docs/superpowers/specs/2026-08-15-ukeire-discard-design.md)。
本模块打分: 1. 打出后向听数(小优先) 2. 有效进张总枚数(改善向听
的进张, 按剩余枚数计) 3. 保持向听的进张枚数(好形率近似)
4. lonely 兜底(排序稳定)。旧 recommend_discard 保留不动。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from src.mahjong_ai.efficiency.discard_selector import (
    DiscardRecommendation,
    _expand_melds,
)
from src.mahjong_ai.efficiency.shanten import calculate_shanten
from src.mahjong_ai.efficiency.tile_efficiency import _lonely
from src.mahjong_core.tile import tile_display
from src.mahjong_engine.judges.tenpai_judge import get_waiting_tiles


@lru_cache(maxsize=16384)
def _shanten_cached(tiles: tuple[int, ...]) -> int:
    """向听数(排序 tuple 缓存 — 进张枚举大量重复计算)。"""
    return calculate_shanten(list(tiles))


@dataclass
class _Score:
    tile: int
    shanten_after: int
    ukeire: int          # 有效进张总枚数(改善向听)
    keep: int            # 保持向听的进张枚数(好形率近似)
    lonely: int          # 搭子潜力(末位 tie-break, 沿用旧启发式)
    is_tenpai: bool = False
    acceptance: int = 0  # 听牌时的等待牌总枚数
    n_waits: int = 0     # 听牌时的等待牌种类数


def _ukeire(remaining: list[int], available: dict[int, int],
            shanten_after: int) -> tuple[int, int]:
    """进张枚数: (改善向听的枚数, 保持向听的枚数)。

    打出后 13 张 + 进张 = 14 张: calculate_shanten 对 14 张先做胡判
    (−1 = 胡), 所以听牌手牌(shanten_after=0)的"改善进张"就是等待牌。
    """
    base = tuple(sorted(remaining))
    improve = keep = 0
    for d, n in available.items():
        if n <= 0 or d == 34:
            continue
        s = _shanten_cached(tuple(sorted(base + (d,))))
        if s < shanten_after:
            improve += n
        elif s == shanten_after:
            keep += n
    return improve, keep


def recommend_by_ukeire(
    tiles: list[int],
    available: dict[int, int],
    enabled_tiles: frozenset[int] | None = None,
    melds: list[list[int]] | None = None,
    exclude: set[int] | None = None,
) -> DiscardRecommendation:
    """纯牌效出牌推荐(接口与 recommend_discard 同族, 多 available)。

    available: 每类牌剩余可摸枚数(4 − 我手牌 − 各河 − 已亮副露)。
    """
    meld_tiles = _expand_melds(melds)
    combined = tiles + meld_tiles
    if not 2 <= len(combined) <= 14:
        raise ValueError(f'推荐需要 2-14 张(含副露展开), 当前 {len(combined)} 张'
                         f'({len(tiles)} 手牌 + {len(meld_tiles)} 副露)')
    excluded = set(meld_tiles) | (exclude or set())
    avail = {t: max(0, available.get(t, 0)) for t in range(34)}
    scores: list[_Score] = []
    for tile in sorted(set(combined) - excluded):
        remaining = list(combined)
        remaining.remove(tile)
        sa = _shanten_cached(tuple(sorted(remaining)))
        improve, keep = _ukeire(remaining, avail, sa)
        score = _Score(tile=tile, shanten_after=sa, ukeire=improve,
                       keep=keep, lonely=_lonely(tile, combined))
        if sa == 0 and len(remaining) == 13 and enabled_tiles is not None:
            ws = get_waiting_tiles(remaining, enabled_tiles)
            score.acceptance = sum(avail.get(w.tile, 0) for w in ws)
            score.n_waits = len(ws)
            score.is_tenpai = True
        scores.append(score)
    if not scores:
        raise ValueError('检测到的牌均为副露牌, 无可打出的候选')
    scores.sort(key=lambda s: (s.shanten_after, -s.ukeire, -s.keep,
                               s.lonely))
    best = scores[0]
    reason_parts = []
    if best.is_tenpai:
        reason_parts.append(f'打出 {tile_display(best.tile)} 后听牌, '
                            f'听 {best.n_waits} 种 {best.acceptance} 枚')
    else:
        reason_parts.append(f'打出 {tile_display(best.tile)} '
                            f'后 {best.shanten_after} 向听, '
                            f'有效进张 {best.ukeire} 枚')
    if melds:
        reason_parts.append(f'(副露 {len(melds)} 组)')
    missing = 14 - len(combined)
    if missing > 0:
        reason_parts.append(f'(手牌 {len(tiles)} 张, 可能漏检)')
    alternatives = [(s.tile, float(s.ukeire)) for s in scores[1:4]]
    return DiscardRecommendation(
        tile=best.tile,
        reason=' '.join(reason_parts),
        shanten_before=best.shanten_after,
        shanten_after=best.shanten_after,
        acceptance=best.acceptance,
        alternatives=alternatives,
    )
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_ai/test_ukeire_selector.py -v`
Expected: 6 passed(注意: 若 test_keep_two_pairs 的期望牌型与算法输出不符, 以手算进张面为准修测试注释 — 拆 9-10 搭的进张面 8(两对可碰 4 + 两面 4)vs 打对子牌 0, 算法必然拆搭; 断言语义不可改)

- [ ] **Step 5: 门禁**

Run: `uv run pytest tests/test_ai -q && uv run ruff check src/mahjong_ai/efficiency/ukeire_selector.py tests/test_ai/test_ukeire_selector.py && uv run mypy src/mahjong_ai/efficiency/ukeire_selector.py`
Expected: 全绿

---

### Task 2: build_advice 切换到新算法

**Files:**
- Modify: `scripts/run_assistant.py`(build_advice 内: available 计算块上移到推荐块之前; 4 处 recommend 调用换新函数)
- Test: `tests/test_pipeline/test_runner_glue.py`(现有测试应继续过; 加一个理由文本含"进张"的断言)

**Interfaces:**
- Consumes: `recommend_by_ukeire`(Task 1)
- Produces: build_advice 行为 = 新算法输出(Advice 结构不变)

- [ ] **Step 1: 移动 available 计算块**

在 `scripts/run_assistant.py` 的 build_advice 中: 把现有"对手副露内容推断"块(从 `available = Counter(dict.fromkeys(range(34), 4))` 到解析循环结束)整体移到 `n = len(view.my_hand)` 之前; 函数内其余逻辑不变。

- [ ] **Step 2: 替换 4 处推荐调用**

```python
    enabled = frozenset(rules.get_tile_set().enabled_tiles)
```
加在 available 计算之后。然后:

- `rec = engine.recommend_discard(Hand(view.my_hand))`(n==14 分支)→
  `rec = recommend_by_ukeire(list(view.my_hand), available, enabled)`
- `rec = engine.recommend_discard(Hand(view.my_hand))`(n>=2 漏检分支)→
  `rec = recommend_by_ukeire(list(view.my_hand), available, enabled)`
- `rec = engine.recommend_discard(Hand(view.my_hand), exclude=meld_tiles)` →
  `rec = recommend_by_ukeire(list(view.my_hand), available, enabled, exclude=meld_tiles)`
- melds 分支两处 `engine.recommend_discard(...)` →
  `recommend_by_ukeire(list(view.my_hand), available, enabled, melds=melds_list)` 与无参退化的
  `recommend_by_ukeire(list(view.my_hand), available, enabled)`

import 加: `from src.mahjong_ai.efficiency.ukeire_selector import recommend_by_ukeire`

- [ ] **Step 3: 加理由文本断言测试**

`tests/test_pipeline/test_runner_glue.py` 追加:

```python
def test_build_advice_reason_mentions_ukeire():
    """新推荐的理由文本含进张枚数(纯牌效算法已接入)。"""
    engine, rules = _engine()
    view = GameView(
        my_hand=[0, 0, 0, 1, 1, 1, 2, 2, 2, 27, 30, 33, 5, 6],
        players={p: PlayerView(player=p)
                 for p in ('my_river', 'right_river', 'top_river',
                           'left_river')},
        unfrozen=[])
    advice, _boxes, _prov = build_advice(view, engine, rules)
    assert advice.discard is not None
    assert '进张' in advice.discard.reason or '听牌' in advice.discard.reason
```

- [ ] **Step 4: 全量测试**

Run: `uv run pytest tests/test_pipeline tests/test_scripts tests/test_ai -q`
Expected: 全绿(既有 test_runner_glue 的 discard 断言基于结构完整手牌, 新算法同样推荐合理牌 — 若个别断言钉死了旧算法的具体牌, 核实新算法输出是否同样牌理正确, 正确则更新测试期望)

- [ ] **Step 5: 门禁**

Run: `uv run pytest tests -q && uv run ruff check scripts/run_assistant.py tests/test_pipeline/test_runner_glue.py && uv run mypy src/mahjong_ai/efficiency/ukeire_selector.py`
Expected: 除 test_cv 预存 1 失败外全绿

---

### Task 3: eval A/B(新旧算法向听曲线)+ 收尾

**Files:**
- Modify: `scripts/eval_inference.py`(SimPlayer 加 policy 参数; run_eval 加 --policy 与向听曲线输出)
- Modify: `.superpowers/sdd/progress.md`(账本)

**Interfaces:**
- Consumes: `recommend_by_ukeire`(Task 1)、现有 simulate/SimPlayer
- Produces: `python scripts/eval_inference.py --games 20 --policy ukeire` vs `--policy engine` 可对比向听曲线

- [ ] **Step 1: SimPlayer 支持新策略**

`scripts/eval_inference.py` 中:

```python
class SimPlayer:
    """模拟玩家: 真手牌(真值来源)+ 已副露。"""

    def __init__(self, hand: list[int], policy: str = 'engine') -> None:
        self.hand: Counter[int] = Counter(hand)
        self.melds: list[tuple[int, ...]] = []
        self.policy = policy

    def discard(self, engine: StrategyEngine,
                available: dict[int, int] | None = None) -> int:
        tiles = sorted(self.hand.elements())
        if not tiles:
            return -1
        try:
            if self.policy == 'ukeire' and available is not None:
                rec = recommend_by_ukeire(tiles, available)
                t = rec.tile
            else:
                rec = engine.recommend_discard(Hand(tiles))
                t = rec.tile
        except ValueError:
            t = tiles[0]
        ...
```

simulate 里维护 `visible: Counter[int]`(每次打出 +1、副露组 +len),每个玩家行动时构造
`available = {t: max(0, 4 - visible[t] - p.hand.get(t, 0)) for t in range(34)}` 传入 discard。
(副露后 `for x in combo: visible[x] += 1` 同步。)

- [ ] **Step 2: run_eval 加 --policy 与向听曲线**

- argparse 加 `--policy {engine,ukeire}` default engine
- simulate 传 policy; 每个检查点记录四家 `calculate_shanten(sorted(hand.elements()))` 均值
- 输出段追加:

```python
    print('\n=== 平均向听曲线(越低越好) ===')
    for c in checkpoints:
        vals = shanten_by[c]
        print(f'  出牌事件 ~{c * 4:3d} 后: 平均 {sum(vals) / len(vals):.2f}')
```

- [ ] **Step 3: A/B 运行(新旧各 20 局, 同种子)**

Run: `uv run python scripts/eval_inference.py --games 20 --policy engine 2>&1 | tail -8`
Run: `uv run python scripts/eval_inference.py --games 20 --policy ukeire 2>&1 | tail -8`
Expected: ukeire 的向听曲线应 ≤ engine(更快下降)— 记录两组数字进账本

- [ ] **Step 4: 全量门禁 + 账本**

Run: `uv run pytest tests -q && uv run ruff check scripts/eval_inference.py src/mahjong_ai/efficiency/ukeire_selector.py && uv run mypy src/mahjong_ai/efficiency/ukeire_selector.py`
Expected: 除 test_cv 预存 1 失败外全绿

`.superpowers/sdd/progress.md` 追加 A/B 数字与结论。

---

## 自审记录

**1. Spec 覆盖:** §3 评分模型 → Task 1 ✓;§4 接口/数据流 → Task 2 ✓;§5 理由文本 → Task 1(实现)✓;§6 单测 → Task 1 六用例 + Task 2 断言 ✓;§6 定量 A/B → Task 3 ✓;§7 旧代码保留 → Global Constraints + 新模块只 import 不改旧文件 ✓。

**2. Placeholder 扫描:** 无 TBD;Step 1 测试里 `hasattr(rec, 'is_tenpai')` 占位行已注明修正 — 实现中 DiscardRecommendation 无 is_tenpai 字段, 该断言删除(测试文件按最终版本写: 该行不保留)。此处为计划文档内的自审记录, 实现者按无该行执行。

**3. 类型一致性:** `recommend_by_ukeire(tiles, available, enabled_tiles, melds, exclude)` 三任务签名一致;`_ukeire` 返回 (improve, keep) 一致;available 为 {tile: 枚数} 口径在 Task 2/3 一致(4 − 可见 − 自己手牌)。
