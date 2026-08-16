# 旧代码清理与现行版本转正 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 删除旧运行时管道(旧入口/GameSession/GameStateTracker/旧 HUD/旧测试),现行事件流管道转正为唯一版本(v4 → pipeline,run_assistant_v4 → run_assistant)。

**Architecture:** 六步顺序执行:session 裁剪 → 删纯旧文件 → 数据类迁移(snapshot.py)→ v4→pipeline 改名 → 入口/脚本/测试转正名 → 文档字样与收尾门禁。每步全量 pytest 验绿再走下一步(无 git,不回退)。

**Tech Stack:** Python 3.11+, pytest, ruff, mypy(项目门禁)。

## Global Constraints

- 项目不使用 git — 无 commit 步骤,无备份(用户明确选择),顺序执行不回退。
- 所有命令用 `uv run` 前缀。
- 训练链全保留:scripts/ 下除 run_assistant.py 外的全部工具脚本、data/ 全部、docs/ 全部不动。
- 历史日志文件名(game_v4_*.jsonl)是数据,不重命名;新日志前缀在 Task 5 转正。
- 验证口径:pytest 全量除 tests/test_cv 的 1 个预存失败(ROI 分类器)外全绿;ruff 改动文件干净;mypy 口径 = 改动文件 clean(旧债随删除消失)。
- 设计依据:`docs/superpowers/specs/2026-08-15-cleanup-legacy-design.md`。

---

### Task 1: session.py 裁剪(删 AssistantSession)

**Files:**
- Modify: `src/mahjong_ai/session.py`(整文件重写为数据类版)
- Delete: `tests/test_ai/test_session.py`、`tests/test_ai/test_session_inference.py`、`tests/test_ai/test_action_advisor.py`(测的是将删的 AssistantSession/action_advisor)

- [ ] **Step 1: 重写 session.py**

完整内容(注意:GameSnapshot 暂从 state.tracker import,Task 3 改路径):

```python
"""现行管道共享类型 — 一帧建议与对手推断快照。

Advice: build_advice 产物, 客户端渲染输入;
InferenceResult: M3 粒子滤波推断快照(听牌/等待/放铳率)。
旧 GameSession/AssistantSession 已删除(2026-08-15 清理)。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.mahjong_ai.efficiency.discard_selector import DiscardRecommendation
from src.mahjong_ai.state.tracker import GameSnapshot
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
```

- [ ] **Step 2: 删三个旧测试文件**

```bash
rm tests/test_ai/test_session.py tests/test_ai/test_session_inference.py tests/test_ai/test_action_advisor.py
```

- [ ] **Step 3: 全量测试**

Run: `uv run pytest tests -q`
Expected: 除 test_cv 预存 1 失败外全绿(若 test_scripts 中有引用旧 session 的测试,一并删除 — 测试旧入口的也是旧测试)

- [ ] **Step 4: 门禁**

Run: `uv run ruff check src/mahjong_ai/session.py && uv run mypy src/mahjong_ai/session.py --follow-imports=skip`
Expected: 干净

---

### Task 2: 删纯旧文件

**Files:**
- Delete: `scripts/run_assistant.py`、`src/mahjong_ui/format.py`、`src/mahjong_ui/pil_text.py`、`src/mahjong_cv/yolo_mapping.py`、`src/mahjong_ai/action_advisor.py`、`tests/test_ui/`(整个目录)、`tests/test_state/`(整个目录)、`tests/test_placeholder.py`

- [ ] **Step 1: 删除**

```bash
rm scripts/run_assistant.py \
   src/mahjong_ui/format.py src/mahjong_ui/pil_text.py \
   src/mahjong_cv/yolo_mapping.py src/mahjong_ai/action_advisor.py \
   tests/test_placeholder.py
rm -r tests/test_ui tests/test_state
rm tests/test_ai/__pycache__ tests/test_ui/__pycache__ 2>/dev/null || true
```

- [ ] **Step 2: 残留引用核实(必须为空)**

Run: `grep -rn "format.py\|pil_text\|yolo_mapping\|action_advisor\|run_assistant" src/ scripts/ tests/ --include="*.py" | grep -v "run_assistant_v4" | head`
Expected: 无输出(run_assistant_v4 除外)

- [ ] **Step 3: 全量测试**

Run: `uv run pytest tests -q`
Expected: 除 test_cv 预存 1 失败外全绿

---

### Task 3: 数据类迁移 state/snapshot.py + tracker.py 删除

**Files:**
- Create: `src/mahjong_ai/state/snapshot.py`
- Delete: `src/mahjong_ai/state/tracker.py`
- Modify: `src/mahjong_ai/state/__init__.py`(re-export 改指 snapshot)
- Modify(import 更新):`src/mahjong_ai/inference/opponent_inference.py`、`src/mahjong_ai/inference/soft_observer.py`、`src/mahjong_ui/client.py`、`scripts/run_assistant_v4.py`、`src/mahjong_ai/session.py`、`scripts/pick_anchors.py`、`scripts/visualize_cluster.py`、`tests/test_v4/test_soft_observer.py`、`tests/test_inference/test_opponent_inference.py`

- [ ] **Step 1: 创建 snapshot.py**

内容 = tracker.py 第 16-126 行的数据类部分(原样搬移: `from __future__ import annotations`、`from dataclasses import dataclass, field`、TURN_MY/TURN_WAITING/TURN_UNKNOWN、RIVER_NAMES、DiscardEvent、Meld、PlayerState、GameSnapshot)。**不搬**:_MeldCandidate(内部候选,已核实无外部引用)、全部旧常量(HAND_CONF/DEBOUNCE_*/MELD_*/RIVER_CONF/各 _XXX 下划线常量 — 现行 v4 提取器自包含复制,不依赖它们)、cluster_dets/TileDet import(旧逻辑用)。模块 docstring:

```python
"""牌局状态快照数据类 — 现行管道的共享类型。

GameStateTracker(旧状态机)已删除(2026-08-15 清理); 这些数据类
是现行管道与保留工具的共同契约, 内容未改只换了家。
"""
```

- [ ] **Step 2: 全仓 import 替换 + 删 tracker.py**

Run:

```bash
grep -rln "state.tracker" src/ scripts/ tests/ --include="*.py" | while read f; do sed -i 's/state\.tracker/state.snapshot/' "$f"; done
rm src/mahjong_ai/state/tracker.py
```

- [ ] **Step 3: 重写 state/__init__.py**

```python
"""牌局状态层 — 快照数据类(GameStateTracker 已删除)。"""
from src.mahjong_ai.state.snapshot import (
    TURN_MY,
    TURN_UNKNOWN,
    TURN_WAITING,
    DiscardEvent,
    GameSnapshot,
)

__all__ = [
    'TURN_MY', 'TURN_WAITING', 'TURN_UNKNOWN',
    'DiscardEvent', 'GameSnapshot',
]
```

- [ ] **Step 4: 残留核实**

Run: `grep -rn "state.tracker\|GameStateTracker\|_MeldCandidate" src/ scripts/ tests/ --include="*.py" | head`
Expected: 无输出

- [ ] **Step 5: 全量测试**

Run: `uv run pytest tests -q`
Expected: 除 test_cv 预存 1 失败外全绿

- [ ] **Step 6: 门禁**

Run: `uv run ruff check src/mahjong_ai/state/snapshot.py src/mahjong_ai/state/__init__.py && uv run mypy src/mahjong_ai/state/snapshot.py`
Expected: 干净

---

### Task 4: v4 → pipeline 目录改名

**Files:**
- Move: `src/mahjong_ai/v4/` → `src/mahjong_ai/pipeline/`
- Modify: 全仓 import 字符串 `src.mahjong_ai.v4` → `src.mahjong_ai.pipeline`

- [ ] **Step 1: 移动目录 + 全仓替换**

```bash
mv src/mahjong_ai/v4 src/mahjong_ai/pipeline
grep -rln "mahjong_ai\.v4" src/ scripts/ tests/ --include="*.py" | while read f; do sed -i 's/mahjong_ai\.v4/mahjong_ai.pipeline/' "$f"; done
```

- [ ] **Step 2: 残留核实**

Run: `grep -rn "mahjong_ai\.v4\|mahjong_ai/v4" src/ scripts/ tests/ --include="*.py" | head`
Expected: 无输出

- [ ] **Step 3: 全量测试**

Run: `uv run pytest tests -q`
Expected: 除 test_cv 预存 1 失败外全绿

- [ ] **Step 4: 门禁**

Run: `uv run ruff check src/mahjong_ai/pipeline && uv run mypy src/mahjong_ai/pipeline src/mahjong_cv/det_cluster.py`
Expected: 干净

---

### Task 5: 入口/脚本/测试目录转正名

**Files:**
- Move: `scripts/run_assistant_v4.py` → `scripts/run_assistant.py`;`scripts/dump_v4_log.py` → `scripts/dump_log.py`;`tests/test_v4/` → `tests/test_pipeline/`
- Modify: `scripts/run_assistant.py` 内日志前缀 `game_v4_` → `game_`、线程名 `v4-infer-worker` → `infer-worker`

- [ ] **Step 1: 移动**

```bash
mv scripts/run_assistant_v4.py scripts/run_assistant.py
mv scripts/dump_v4_log.py scripts/dump_log.py
mv tests/test_v4 tests/test_pipeline
```

- [ ] **Step 2: 日志前缀/线程名转正**

在 `scripts/run_assistant.py` 中:
`name = f'game_v4_{time.strftime("%Y%m%d_%H%M%S")}.jsonl'` → `name = f'game_{time.strftime("%Y%m%d_%H%M%S")}.jsonl'`;
`name='v4-infer-worker'` → `name='infer-worker'`。

- [ ] **Step 3: 旧名残留核实**

Run: `grep -rn "run_assistant_v4\|dump_v4_log\|test_v4" src/ scripts/ tests/ --include="*.py" | head`
Expected: 无输出

- [ ] **Step 4: 全量测试**

Run: `uv run pytest tests -q`
Expected: 除 test_cv 预存 1 失败外全绿

---

### Task 6: 文档字样 + config 孤儿 + 收尾门禁

**Files:**
- Modify: 现行文件里 "v4 管道"/"V4Runner"/"v4 " 等字样(机械替换,不动 docs/)
- Modify: config/ 孤儿文件核实(无引用则删)
- Modify: `.superpowers/sdd/progress.md`(账本)

- [ ] **Step 1: 文档字样替换**

Run:

```bash
grep -rln "v4 管道\|V4Runner\|v4 修复\|v4 运行器" src/ scripts/ tests/ --include="*.py" | while read f; do sed -i 's/v4 管道/事件流管道/g; s/V4Runner/AssistantRunner/g; s/v4 修复/管道修复/g; s/v4 运行器/管道运行器/g' "$f"; done
```

- [ ] **Step 2: config 孤儿核实**

Run: `grep -rn "river_anchors\|screen_layout\|screen_region" src/ scripts/ tests/ --include="*.py" | head`
Expected: 若全部只出现在旧管道(已删)→ 无输出 → 删 `config/river_anchors.json`、`config/screen_layout.json`、`config/screen_region.json`;若现行仍有引用 → 保留并记录。

- [ ] **Step 3: 全量门禁**

Run: `uv run pytest tests -q && uv run ruff check src scripts tests && uv run mypy src/mahjong_ai/pipeline src/mahjong_cv/det_cluster.py`
Expected: pytest 除 test_cv 预存 1 失败外全绿;ruff 可能报训练脚本旧债 — 训练链"保留"指逻辑保留,其 lint 旧债不修,门禁口径 = 本计划改动文件干净(改动文件清单 = Task 1-5 的文件)。

- [ ] **Step 4: 冒烟**

```bash
uv run python -c "import src.mahjong_ui.client, src.mahjong_ai.pipeline, src.mahjong_ai.inference.soft_observer, scripts.run_assistant; print('imports ok')"
uv run python scripts/settle_check.py --help
uv run python scripts/dump_log.py data/settle_logs/game_v4_20260815_210738.jsonl | head -5
uv run python -c "import scripts.pick_anchors" 2>/dev/null || true
uv run python -c "import ast; ast.parse(open('scripts/pick_anchors.py', encoding='utf-8').read()); ast.parse(open('scripts/visualize_cluster.py', encoding='utf-8').read()); print('tool scripts parse ok')"
```
Expected: 全部正常(settle_check --help 打印用法;dump_log 打印日志行)

- [ ] **Step 5: 账本**

`.superpowers/sdd/progress.md` 追加:

```
## 旧代码清理(2026-08-15): 旧运行时管道删除 + 现行转正
- 删: 旧入口 run_assistant/format/pil_text/yolo_mapping/action_advisor/
  AssistantSession/GameStateTracker(756 行)/旧测试(test_ui/test_state/
  test_session/test_session_inference/test_action_advisor/placeholder)
- 迁: 数据类 → state/snapshot.py(GameSnapshot/Meld/PlayerState/
  DiscardEvent/RIVER_NAMES/TURN_*); session.py 只留 Advice/
  InferenceResult/OPPONENT_PLAYERS
- 转正: mahjong_ai/v4 → mahjong_ai/pipeline; run_assistant_v4 → run_assistant;
  dump_v4_log → dump_log; test_v4 → test_pipeline; 日志前缀 game_v4_ → game_
- 保留: 训练链全部脚本+数据、docs、历史日志文件名
- 门禁: pytest 除 test_cv 预存 1 失败外全绿
```

---

## 自审记录

**1. Spec 覆盖:** 删除清单 → Task 1/2/3 ✓;拆分迁移 → Task 3 ✓;转正名 → Task 4/5 ✓;执行顺序 → 任务序 = spec §5 ✓;验证口径 → 各任务门禁 + Task 6 Step 3/4 ✓;训练链保留 → Global Constraints + Task 3/6 只改保留工具的 import 行 ✓。

**2. Placeholder 扫描:** 无 TBD;每步含确切命令/代码/预期。

**3. 类型一致性:** snapshot.py 数据类名与 tracker.py 原版一字不差(搬移式);全仓 sed 替换保证 import 路径一致;Task 1 的 session.py 暂用 state.tracker,Task 3 的 sed 覆盖它(session.py 在 grep 清单内)。
