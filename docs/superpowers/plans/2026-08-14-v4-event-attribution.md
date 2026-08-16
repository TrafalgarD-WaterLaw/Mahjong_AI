# v4 事件归属核心(M1+M2)实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 v4 事件流 + 滑动窗口全局归属解码管道(M1 解码器核心 + M2 PC 适配器),与旧管道并行运行,旧代码零改动。

**Architecture:** 检测/跟踪(复用)→ 事件提取器(纯确认,不归属)→ 窗口解码器(轮转自动机 + beam search + 空间似然,软归属回看纠错)→ 状态投影(事件日志→玩家视图)→ 复用牌效引擎给手牌建议。设计依据:`docs/superpowers/specs/2026-08-14-mahjong-assistant-v4-design.md`。M3(软观测推断接入)在本计划验收后另立计划。

**Tech Stack:** Python 3.12 / uv / PySide6 / ultralytics(YOLO+ByteTrack,复用)/ pytest+ruff+mypy

## Global Constraints

- **项目不用 git(用户要求)** — 本计划无 commit 步骤,每任务末尾以质量门禁收尾
- 所有命令加 `uv run` 前缀(项目 .venv 环境)
- 质量门禁(每任务末尾必跑,全部通过才算完成): `uv run pytest tests/test_v4 -q`、`uv run ruff check src/mahjong_ai/v4 tests/test_v4 scripts/run_assistant_v4.py scripts/dump_v4_log.py`、`uv run mypy --follow-imports=silent src/mahjong_ai/v4 scripts/run_assistant_v4.py scripts/dump_v4_log.py`(注: `--follow-imports=silent` 必须在文件列表前 — v4 导入 det_cluster 后 mypy 会跟随检查旧代码, 旧代码 det_cluster.py:109 有既存裸 `-> dict` 标注(旧代码零改动, 不修), 静默跟随只查 v4 自身)
- **旧代码零改动** — 唯一白名单修改: `src/mahjong_ui/client.py` 的 `_visible_tiles` 加回退分支(Task 8,规格 §12 明确授权);其余旧文件只读不写
- 代码风格与现有代码一致: 中文模块 docstring 与行内注释、`from __future__ import annotations`、模块常量 UPPER_CASE、dataclass frozen 优先
- 新代码全部落在: `src/mahjong_ai/v4/`、`tests/test_v4/`、`scripts/run_assistant_v4.py`、`scripts/dump_v4_log.py`
- 玩家键名与现行一致: `'my_river' | 'right_river' | 'top_river' | 'left_river'`;轮转表: 我→右→上→左(用户确认)

---

## File Structure

```
src/mahjong_ai/v4/
  events.py       # 事件数据类 + event_to_dict(日志序列化)— Task 1
  decoder.py      # TurnPhase 自动机 + EvidenceModel + WindowDecoder(beam) — Tasks 2-4
  state.py        # PlayerView/GameView + project() 投影纯函数 — Task 5
  extractor.py    # EventExtractor: 检测帧 → 事件(复用 det_cluster 聚类)— Task 7
tests/test_v4/
  test_events.py          # Task 1
  test_decoder_automaton.py  # Task 2
  test_decoder_evidence.py   # Task 3
  test_decoder_window.py     # Task 4
  test_state.py              # Task 5
  synth.py + test_end_to_end.py  # Task 6
  test_extractor.py          # Task 7
  test_runner_glue.py        # Task 9
src/mahjong_ui/client.py  # 白名单修改一处 — Task 8
scripts/run_assistant_v4.py  # Task 9
scripts/dump_v4_log.py       # Task 10
```

---

### Task 1: 事件数据类 events.py

**Files:**
- Create: `src/mahjong_ai/v4/__init__.py`(空文件)
- Create: `src/mahjong_ai/v4/events.py`
- Test: `tests/test_v4/__init__.py`(空文件)
- Test: `tests/test_v4/test_events.py`

**Interfaces:**
- Produces(后续任务全部依赖): `PLAYERS`、`TileAppeared(eid, tile, track_id, cx, cy, conf, frame, ts, motion=None)`、`MeldFormed(eid, tiles, cx, cy, bbox, frame, ts)`、`TileClaimed(eid, claimed, meld, frame, ts)`、`HandChanged(eid, n_old, n_new, frame, ts)`、`FlowerShown(eid, tile, cx, cy, frame, ts)`、`MotionEvidence(start_x, start_y, conf)`、`event_to_dict(ev) -> dict`

- [ ] **Step 1: 写失败测试**

`tests/test_v4/test_events.py`:

```python
"""v4 事件数据类单测。"""

from src.mahjong_ai.v4.events import (
    HandChanged, MeldFormed, TileAppeared, TileClaimed, event_to_dict,
)


def test_event_to_dict_includes_type():
    ev = TileAppeared(eid=1, tile=5, track_id=101, cx=700.0, cy=400.0,
                      conf=0.7, frame=10, ts=1.0)
    d = event_to_dict(ev)
    assert d['type'] == 'TileAppeared'
    assert d['eid'] == 1 and d['tile'] == 5 and d['cx'] == 700.0


def test_claimed_second_pong_supported():
    """秒碰: claimed=None(该牌从未作为 TileAppeared 出现)。"""
    meld = MeldFormed(eid=2, tiles=(5, 5), cx=300.0, cy=500.0,
                      bbox=(280, 480, 360, 560), frame=20, ts=2.0)
    claim = TileClaimed(eid=3, claimed=None, meld=2, frame=20, ts=2.0)
    assert claim.claimed is None and claim.meld == meld.eid


def test_hand_changed_draw_semantics():
    draw = HandChanged(eid=4, n_old=13, n_new=14, frame=30, ts=3.0)
    discard = HandChanged(eid=5, n_old=14, n_new=13, frame=31, ts=3.1)
    assert (draw.n_old, draw.n_new) == (13, 14)
    assert (discard.n_old, discard.n_new) == (14, 13)
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_v4/test_events.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.mahjong_ai.v4.events'`

- [ ] **Step 3: 写实现**

`src/mahjong_ai/v4/events.py`:

```python
"""v4 观测事件 — 感知层报"发生了什么", 不带归属结论。

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
    """一组紧贴亮牌出现(碰/杠候选)。bbox 为组外接框(像素)。"""

    eid: int
    tiles: tuple[int, ...]
    cx: float
    cy: float
    bbox: tuple[float, float, float, float]
    frame: int
    ts: float


@dataclass(frozen=True)
class TileClaimed:
    """刚出现的牌被副露吸收。claimed=None = 秒碰(该牌从未出现)。"""

    eid: int
    claimed: int | None
    meld: int
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


Event = TileAppeared | MeldFormed | TileClaimed | HandChanged | FlowerShown


def event_to_dict(ev: Event) -> dict:
    """事件 → 日志行 dict(复盘/核对用)。"""
    d = asdict(ev)
    d['type'] = type(ev).__name__
    return d
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_v4/test_events.py -q`
Expected: PASS(3 passed)

- [ ] **Step 5: 质量门禁**

Run: `uv run ruff check src/mahjong_ai/v4 tests/test_v4` 且 `uv run mypy src/mahjong_ai/v4`
Expected: 全部通过, 无输出错误

---

### Task 2: 轮转自动机(TurnPhase)

**Files:**
- Create: `src/mahjong_ai/v4/decoder.py`(本任务只写自动机部分)
- Test: `tests/test_v4/test_decoder_automaton.py`

**Interfaces:**
- Consumes: `PLAYERS`(Task 1)
- Produces: `next_in_rotation(player) -> str`、`TurnPhase(actor, played)` frozen dataclass、`TurnPhase.advance(player) -> (TurnPhase, float)`(新相位, 跳家惩罚)、`TurnPhase.claim(player) -> (TurnPhase, float)`(副露截断: 碰家接着打, 跳过的家本轮失去行动权, 无惩罚)、`phase_prior(phase, player) -> float`(对数先验, actor=0, 其他=-_TURN_PRIOR_GAP)

- [ ] **Step 1: 写失败测试**

`tests/test_v4/test_decoder_automaton.py`:

```python
"""轮转自动机单测 — 跳家惩罚/副露截断/整轮重置。"""

from src.mahjong_ai.v4.decoder import (
    TurnPhase, next_in_rotation, phase_prior,
)


def test_rotation_order():
    assert next_in_rotation('my_river') == 'right_river'
    assert next_in_rotation('right_river') == 'top_river'
    assert next_in_rotation('top_river') == 'left_river'
    assert next_in_rotation('left_river') == 'my_river'


def test_advance_in_turn_no_penalty():
    ph = TurnPhase(actor='my_river', played=frozenset())
    nph, penalty = ph.advance('my_river')
    assert penalty == 0
    assert nph.actor == 'right_river'
    assert nph.played == frozenset({'my_river'})


def test_advance_skip_penalty():
    """actor=我 但 top 打出 → 跳过 right(其打出漏检), 罚 1×跳家惩罚。"""
    ph = TurnPhase(actor='my_river', played=frozenset())
    nph, penalty = ph.advance('top_river')
    assert penalty > 0
    assert nph.actor == 'left_river'
    assert nph.played == frozenset({'right_river', 'top_river'})


def test_meld_interrupt_keeps_unplayed():
    """碰截断: top 碰(right 还没打)→ top 接着打; right 本轮失去行动权。"""
    ph = TurnPhase(actor='right_river',
                   played=frozenset({'my_river'}))
    nph, penalty = ph.claim('top_river')
    assert penalty == 0            # 碰是合法截断, 不罚
    assert nph.actor == 'top_river'  # 碰家接着打
    assert nph.played == frozenset({'my_river', 'right_river', 'top_river'})
    # top 接着打出: 无惩罚, 轮转从 top 继续
    nph2, penalty2 = nph.advance('top_river')
    assert penalty2 == 0
    assert nph2.actor == 'left_river'


def test_claim_by_next_player_no_skip():
    """我打出后下家(右)直接碰: actor 已是右, 无跳家。"""
    ph = TurnPhase(actor='right_river',
                   played=frozenset({'my_river'}))
    nph, penalty = ph.claim('right_river')
    assert penalty == 0
    assert nph.actor == 'right_river'
    assert nph.played == frozenset({'my_river', 'right_river'})


def test_round_resets_after_all_four():
    ph = TurnPhase(actor='my_river', played=frozenset())
    for p in ('my_river', 'right_river', 'top_river'):
        ph, _ = ph.advance(p)
    assert len(ph.played) == 3
    ph, _ = ph.advance('left_river')
    assert ph.played == frozenset()  # 整轮重置
    assert ph.actor == 'my_river'


def test_phase_prior_favors_actor():
    ph = TurnPhase(actor='top_river', played=frozenset())
    assert phase_prior(ph, 'top_river') > phase_prior(ph, 'my_river')
    assert phase_prior(ph, 'my_river') == phase_prior(ph, 'right_river')
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_v4/test_decoder_automaton.py -q`
Expected: FAIL — `ModuleNotFoundError` 或 `ImportError`(decoder.py 不存在)

- [ ] **Step 3: 写实现**

`src/mahjong_ai/v4/decoder.py`(本任务内容;后续任务在同一文件追加):

```python
"""归属解码器 — 滑动窗口联合解码(软归属, 回看纠错)。

轮转自动机(我→右→上→左, 副露截断)是全局约束而非硬链条:
每来一个 TileAppeared, 对窗口内全部未冻结事件重做 beam search,
输出每事件的四家归属分布; 事件出窗口时冻结(取 MAP)。

证据 = 空间似然(在线高斯, 框选区域做种子) + 轮转先验(自动机相位)。
手牌 13→14/14→13 是相位硬锚: 每次我摸牌都重置轮转相位 —
漂移不可能累积超过一轮。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from src.mahjong_ai.v4.events import PLAYERS

#: 轮转表(欢乐麻将: 我 → 右(下家) → 上(对家) → 左(上家), 用户确认)
_ROTATION = PLAYERS
#: 轮转跳一位的 log 惩罚(中间家的打出事件漏检时; 跳 N 位 = N×此值)
_SKIP_PENALTY = 1.5
#: 相位先验差距: log P(actor) − log P(其他家) ≈ P(actor)≈0.77
_TURN_PRIOR_GAP = 2.0


def next_in_rotation(player: str) -> str:
    """轮转下一位。"""
    return _ROTATION[(_ROTATION.index(player) + 1) % len(_ROTATION)]


@dataclass(frozen=True)
class TurnPhase:
    """轮转相位: actor = 下一位该行动者, played = 本轮已行动家。"""

    actor: str = 'my_river'
    played: frozenset[str] = frozenset()

    def advance(self, player: str) -> tuple['TurnPhase', int]:
        """player 行动(打出/碰) → (新相位, 跳家惩罚)。

        跳家 = actor 与 player 之间(轮转序)的家被跳过 — 他们应打出的
        牌漏检了: 隐式补记, 只罚分(轮转是约束不是链条)。
        """
        if player == self.actor:
            penalty = 0
            played = self.played | {player}
            actor = next_in_rotation(player)
        else:
            skipped: set[str] = set()
            cur = next_in_rotation(self.actor)  # 从 actor 的下一位数起
            while cur != player:
                skipped.add(cur)
                cur = next_in_rotation(cur)
            penalty = len(skipped) * _SKIP_PENALTY
            played = self.played | skipped | {player}
            actor = next_in_rotation(player)
        if len(played) >= len(_ROTATION):
            played = frozenset()
        return TurnPhase(actor=actor, played=played), penalty

    def claim(self, player: str) -> tuple['TurnPhase', float]:
        """副露截断: player 碰/杠 → 该家接着打(actor=碰家)。

        被截断的行动者起(含)到碰家止(不含)的家本轮失去行动权 —
        合法碰会消耗掉当前行动者的回合(actor 也被标记 played);
        碰是合法截断, 无惩罚。
        """
        if player == self.actor:
            played = self.played | {player}
            return TurnPhase(actor=player, played=played), 0.0
        skipped: set[str] = set()
        cur = self.actor  # 被截断的行动者本身也失去本轮行动权
        while cur != player:
            skipped.add(cur)
            cur = next_in_rotation(cur)
        played = self.played | skipped | {player}
        if len(played) >= len(_ROTATION):
            played = frozenset()
        return TurnPhase(actor=player, played=played), 0.0


def phase_prior(phase: TurnPhase, player: str) -> float:
    """log P(player | phase): actor 高, 其余均分(对数空间, 未归一)。"""
    return 0.0 if player == phase.actor else -_TURN_PRIOR_GAP
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_v4/test_decoder_automaton.py -q`
Expected: PASS(6 passed)

- [ ] **Step 5: 质量门禁**

Run: `uv run ruff check src/mahjong_ai/v4 tests/test_v4` 且 `uv run mypy src/mahjong_ai/v4`
Expected: 通过

---

### Task 3: 证据模型(EvidenceModel)

**Files:**
- Modify: `src/mahjong_ai/v4/decoder.py`(追加 EvidenceModel 与常量)
- Test: `tests/test_v4/test_decoder_evidence.py`

**Interfaces:**
- Consumes: `PLAYERS`(Task 1)
- Produces: `EvidenceModel()`、`reseed(regions | None, frame_w, frame_h) -> None`、`spatial_logp(player, cx, cy) -> float`、`update(player, cx, cy) -> None`(在线 EMA 学习)

- [ ] **Step 1: 写失败测试**

`tests/test_v4/test_decoder_evidence.py`:

```python
"""证据模型单测 — 种子高斯/在线学习。"""

from src.mahjong_ai.v4.decoder import EvidenceModel

_SEED = {
    'my_river': (600.0, 700.0, 800.0, 780.0),
    'right_river': (1100.0, 500.0, 1180.0, 600.0),
    'top_river': (500.0, 100.0, 700.0, 180.0),
    'left_river': (100.0, 500.0, 180.0, 600.0),
}


def test_seed_spatial_prefers_own_zone():
    ev = EvidenceModel()
    ev.reseed(_SEED, 1280, 800)
    my = ev.spatial_logp('my_river', 700.0, 740.0)
    right = ev.spatial_logp('right_river', 700.0, 740.0)
    assert my > right


def test_no_seed_flat_prior():
    """无框选配置 → 宽高斯, 各家空间分近似相等(轮转先验主导)。"""
    ev = EvidenceModel()
    ev.reseed(None, 1280, 800)
    a = ev.spatial_logp('my_river', 640.0, 400.0)
    b = ev.spatial_logp('right_river', 640.0, 400.0)
    assert abs(a - b) < 0.01


def test_update_pulls_mean_toward_landing():
    ev = EvidenceModel()
    ev.reseed(_SEED, 1280, 800)
    before = ev.spatial_logp('right_river', 1000.0, 450.0)
    for _ in range(10):
        ev.update('right_river', 1000.0, 450.0)
    after = ev.spatial_logp('right_river', 1000.0, 450.0)
    assert after > before  # 落点分布向实际落点漂移
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_v4/test_decoder_evidence.py -q`
Expected: FAIL — `ImportError: cannot import name 'EvidenceModel'`

- [ ] **Step 3: 写实现**

在 `decoder.py` 末尾追加(常量加在文件顶部常量区):

```python
#: 空间似然种子: 框 ≈ ±2σ → σ = 框半宽/2
_SEED_SIGMA = 4.0
#: 在线学习速率(冻结事件 → 该家落点高斯 EMA)
_LEARN_LR = 0.05


@dataclass
class _Gauss:
    mx: float = 0.0
    my: float = 0.0
    vx: float = 1.0
    vy: float = 1.0


class EvidenceModel:
    """空间证据: 每玩家落点高斯(框选区域种子 + 在线 EMA 学习)。

    种子来自 config/river_regions.json(旧 pick_regions 产物, 降级为
    先验而非权威); 冻结事件按高置信度在线更新 — 区域溢出/漂移由
    似然随数据漂移自然解决。无种子时宽高斯(空间退化为弱证据)。
    """

    def __init__(self) -> None:
        self._g: dict[str, _Gauss] = {p: _Gauss() for p in PLAYERS}
        self._seeded = False

    def reseed(
        self,
        regions: dict[str, tuple[float, float, float, float]] | None,
        frame_w: int,
        frame_h: int,
    ) -> None:
        """重置为种子高斯(像素)。窗口缩放/画面区域变化时由运行器调用。

        regions: {player: (x1, y1, x2, y2)} 像素框; None/缺家 → 该家
        用覆盖全画面的宽高斯(空间证据退化为近似均匀, 轮转先验主导)。
        """
        for p in PLAYERS:
            r = (regions or {}).get(p)
            if r is not None:
                x1, y1, x2, y2 = r
                self._g[p] = _Gauss(
                    mx=(x1 + x2) / 2, my=(y1 + y2) / 2,
                    vx=((x2 - x1) / _SEED_SIGMA) ** 2,
                    vy=((y2 - y1) / _SEED_SIGMA) ** 2)
            else:
                s = float(max(frame_w, frame_h))
                self._g[p] = _Gauss(mx=frame_w / 2, my=frame_h / 2,
                                    vx=s * s, vy=s * s)
        self._seeded = True

    def spatial_logp(self, player: str, cx: float, cy: float) -> float:
        """落点在该家牌河区的对数似然(高斯)。

        未播种 → 返回 0(空间证据中性): 构造默认是原点高斯, 若
        直接使用会对四家同值且绝对值荒谬 — 真实对局曾因此全部
        归我(守卫 bug); 中性化后未播种时由轮转先验主导。
        """
        if not self._seeded:
            return 0.0
        g = self._g[player]
        return (-0.5 * ((cx - g.mx) ** 2 / g.vx + (cy - g.my) ** 2 / g.vy)
                - math.log(2 * math.pi * math.sqrt(g.vx * g.vy)))

    def update(self, player: str, cx: float, cy: float) -> None:
        """在线学习(EMA): 该家落点分布向实际落点漂移。

        只在冻结归属高置信时由解码器调用(防错误样本反馈回路)。
        未播种时无样本可言, 不学习。
        """
        if not self._seeded:
            return
        g = self._g[player]
        ddx = cx - g.mx
        ddy = cy - g.my
        g.mx += _LEARN_LR * ddx
        g.my += _LEARN_LR * ddy
        g.vx = (1 - _LEARN_LR) * g.vx + _LEARN_LR * ddx * ddx
        g.vy = (1 - _LEARN_LR) * g.vy + _LEARN_LR * ddy * ddy
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_v4/test_decoder_evidence.py -q`
Expected: PASS(3 passed)

- [ ] **Step 5: 质量门禁**

Run: `uv run ruff check src/mahjong_ai/v4 tests/test_v4` 且 `uv run mypy src/mahjong_ai/v4`
Expected: 通过

---

### Task 4: 窗口解码器(WindowDecoder beam search + 冻结)

**Files:**
- Modify: `src/mahjong_ai/v4/decoder.py`(追加 Attribution/WindowDecoder/_logadd 与常量)
- Test: `tests/test_v4/test_decoder_window.py`

**Interfaces:**
- Consumes: Task 2/3 全部接口、`TileAppeared/MeldFormed/TileClaimed/HandChanged`(Task 1)
- Produces: `Attribution(eid, tile, cx, cy, frame, ts, logits, player, frozen)` 及方法 `probs()/map()/entropy()`;`WindowDecoder(river_ev, meld_ev, k=10, beam=16)`、`add(event) -> None`、`frozen() -> list[Attribution]`、`unfrozen() -> list[Attribution]`、`claimed_ids() -> set[int]`、`melds() -> list[tuple[str, MeldFormed]]`

- [ ] **Step 1: 写失败测试**

`tests/test_v4/test_decoder_window.py`:

```python
"""窗口解码器单测 — 软归属/回看纠错/相位硬锚/冻结学习。"""

from src.mahjong_ai.v4.decoder import EvidenceModel, WindowDecoder
from src.mahjong_ai.v4.events import HandChanged, TileAppeared

_SEED = {
    'my_river': (600.0, 700.0, 800.0, 780.0),
    'right_river': (1100.0, 500.0, 1180.0, 600.0),
    'top_river': (500.0, 100.0, 700.0, 180.0),
    'left_river': (100.0, 500.0, 180.0, 600.0),
}
_CENTERS = {p: ((r[0] + r[2]) / 2, (r[1] + r[3]) / 2)
            for p, r in _SEED.items()}


def _make(player: str, eid: int, frame: int = 0,
          jitter: tuple[float, float] = (0.0, 0.0)) -> TileAppeared:
    cx, cy = _CENTERS[player]
    return TileAppeared(eid=eid, tile=eid % 34, track_id=1000 + eid,
                        cx=cx + jitter[0], cy=cy + jitter[1],
                        conf=0.8, frame=frame, ts=frame * 0.1)


def _decoder(k: int = 4) -> WindowDecoder:
    ev = EvidenceModel()
    ev.reseed(_SEED, 1280, 800)
    mev = EvidenceModel()
    mev.reseed(None, 1280, 800)
    return WindowDecoder(ev, mev, k=k, beam=16)


def test_clean_rotation_all_correct():
    """位置清晰 + 轮转正常 → 全部归对。"""
    d = _decoder()
    for i, p in enumerate(('my_river', 'right_river', 'top_river', 'left_river',
                           'my_river', 'right_river', 'top_river', 'left_river',
                           'my_river', 'right_river', 'top_river', 'left_river')):
        d.add(_make(p, eid=i + 1, frame=i))
    got = {a.eid: a.map() for a in d.frozen()}
    truth = {1: 'my_river', 2: 'right_river', 3: 'top_river', 4: 'left_river',
             5: 'my_river', 6: 'right_river', 7: 'top_river', 8: 'left_river'}
    for eid, p in truth.items():
        assert got.get(eid) == p, f'eid {eid}: {got.get(eid)}'
    assert len(d.frozen()) == 8  # k=4: 12 张 → 冻结前 8 张


def test_missed_event_skip_penalty_path():
    """top 的打出漏检: my,right,left 序列 → left 仍归 left(不归 top)。"""
    d = _decoder(k=3)
    d.add(_make('my_river', 1, 0))
    d.add(_make('right_river', 2, 1))
    d.add(_make('left_river', 3, 2))
    d.add(_make('my_river', 4, 3))
    d.add(_make('right_river', 5, 4))
    a3 = [a for a in d.attributions() if a.eid == 3][0]
    assert a3.map() == 'left_river'


def test_retro_correction_on_new_evidence():
    """首事件位置歧义(桌面中央)先按轮转先验, 后续证据回看修正。"""
    d = _decoder(k=5)
    # 真值: e1=right(位置漂移大), e2=top, e3=left, e4=my
    d.add(TileAppeared(eid=1, tile=1, track_id=1001,
                       cx=640.0, cy=400.0,  # 中央歧义点
                       conf=0.5, frame=0, ts=0.0))
    before = {a.eid: dict(a.probs()) for a in d.attributions()}[1]
    d.add(_make('top_river', 2, 1))
    d.add(_make('left_river', 3, 2))
    d.add(_make('my_river', 4, 3))
    d.add(_make('right_river', 5, 4))
    d.add(_make('top_river', 6, 5))
    after = {a.eid: dict(a.probs()) for a in d.attributions()}[1]
    assert after != before  # 分布被重解(回看纠错发生)


def test_hand_anchor_resets_phase():
    """我摸牌(13→14)后相位硬锚: 后续新牌空间歧义时轮转先验归我。"""
    d = _decoder(k=3)
    d.add(_make('right_river', 1, 0))
    d.add(HandChanged(eid=90, n_old=13, n_new=14, frame=1, ts=0.1))
    # 空间歧义点(我的牌河上方 — 距我锚较近), 相位锚定后 → 归我
    d.add(TileAppeared(eid=2, tile=2, track_id=1002,
                       cx=700.0, cy=600.0, conf=0.5, frame=2, ts=0.2))
    a2 = [a for a in d.attributions() if a.eid == 2][0]
    assert a2.map() == 'my_river'


def test_entropy_low_for_clear_high_for_ambiguous():
    d = _decoder(k=2)
    d.add(_make('my_river', 1, 0))
    d.add(TileAppeared(eid=2, tile=2, track_id=1002,
                       cx=640.0, cy=400.0, conf=0.5, frame=1, ts=0.1))
    clear, amb = (a for a in d.unfrozen())
    assert clear.entropy() < amb.entropy()


def test_freeze_high_margin_learns_spatial():
    """高置信冻结 → 该家落点高斯在线学习(对实际落点似然上升)。"""
    ev = EvidenceModel()
    ev.reseed(_SEED, 1280, 800)
    d = WindowDecoder(ev, EvidenceModel(), k=1, beam=8)
    before = ev.spatial_logp('my_river', 700.0, 650.0)
    # 偏移落点(偏离种子中心): 连续 3 张我家的牌 → 冻结学习
    for i in range(3):
        d.add(TileAppeared(eid=i + 1, tile=i, track_id=2000 + i,
                           cx=700.0, cy=650.0,
                           conf=0.8, frame=i, ts=i * 0.1))
    after = ev.spatial_logp('my_river', 700.0, 650.0)
    assert after > before  # 均值向实际落点漂移(学习方向正确)
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_v4/test_decoder_window.py -q`
Expected: FAIL — `ImportError: cannot import name 'WindowDecoder'`

- [ ] **Step 3: 写实现**

在 `decoder.py` 末尾追加(顶部 import 区加 `TileClaimed/MeldFormed/HandChanged`,常量区加):

```python
#: 窗口大小(未冻结 TileAppeared 数)/ beam 宽度
_K_DEFAULT = 10
_BEAM_DEFAULT = 16
#: 冻结后在线学习的置信边距门槛(log 比): 低于此值不学习(防错误样本)
_LEARN_MIN_MARGIN = math.log(2.0)
```

追加实现:

```python
def _logadd(a: float, b: float) -> float:
    """log-sum-exp 两数版。"""
    if a == float('-inf'):
        return b
    if b == float('-inf'):
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


@dataclass
class Attribution:
    """单个 TileAppeared 的归属(软分布; 冻结后 player 为最终结论)。"""

    eid: int
    tile: int
    cx: float
    cy: float
    frame: int
    ts: float
    logits: dict[str, float] = field(default_factory=dict)
    player: str | None = None
    frozen: bool = False

    def probs(self) -> dict[str, float]:
        """归一化概率分布(softmax, 数值稳定: max 位移防 exp 下溢)。

        全 -inf(无证据) → 均匀分布 — 熵最大, 自诊断如实反映。
        """
        m = max(self.logits.values())
        if m == float('-inf'):
            return {p: 1.0 / len(self.logits) for p in self.logits}
        z = m + math.log(sum(math.exp(v - m) for v in self.logits.values()))
        return {p: math.exp(v - z) for p, v in self.logits.items()}

    def map(self) -> str:
        """MAP 归属。"""
        return max(self.logits, key=self.logits.get)

    def entropy(self) -> float:
        """分布熵(nats) — 自诊断: 熵低才可信, 熵高=证据不足。"""
        pr = self.probs()
        return -sum(q * math.log(q) for q in pr.values() if q > 0)


class WindowDecoder:
    """滑动窗口归属解码: beam search 全局重解 + 出窗冻结。

    窗口内事件每次重解(新证据可回看修正); 出窗口时冻结取 MAP —
    冻结时按置信边距决定是否喂给证据模型在线学习。
    副露/手牌变化是相位硬更新(不参与 beam 分支), 按时间序与
    TileAppeared 交错进入序列。
    """

    def __init__(self, river_ev: EvidenceModel, meld_ev: EvidenceModel,
                 k: int = _K_DEFAULT, beam: int = _BEAM_DEFAULT) -> None:
        self._river_ev = river_ev
        self._meld_ev = meld_ev
        self._k = k
        self._beam = beam
        self._phase_base = TurnPhase()   # 冻结前缀之后的相位
        self._unfrozen: list[tuple[str, object]] = []  # (kind, event)
        self._attribs: dict[int, Attribution] = {}
        self._meld_list: dict[int, MeldFormed] = {}
        self._meld_player: dict[int, str] = {}
        self._claimed: dict[int, int] = {}   # appeared eid -> meld eid
        self._claim_no_event: set[int] = set()  # 秒碰的 meld eid

    # ---- 事件入口 ----

    def add(self, event) -> None:
        """事件按时间序流入(提取器产出顺序)。FlowerShown 忽略。"""
        if isinstance(event, TileAppeared):
            self._unfrozen.append(('appeared', event))
            self._resolve()
            while self._n_appeared() > self._k:
                self._freeze_oldest()
        elif isinstance(event, MeldFormed):
            self._meld_list[event.eid] = event
            # 副露归属: 空间似然 argmax(位置强证据, 罕见事件, 不软)。
            # 退化保护: 四家得分完全相同(无种子/配置缺失)→ 不归属
            # (保持中性, 不产生 meld_assign — 绝不静默全归第一家)
            scores = {p: self._meld_ev.spatial_logp(p, event.cx, event.cy)
                      for p in PLAYERS}
            if len(set(scores.values())) == 1:
                self._meld_player[event.eid] = None
            else:
                self._meld_player[event.eid] = max(scores,
                                                   key=scores.get)
            self._unfrozen.append(('meld', event))
            self._resolve()
        elif isinstance(event, HandChanged):
            self._unfrozen.append(('hand', event))
            self._resolve()
        elif isinstance(event, TileClaimed):
            if event.claimed is None:
                self._claim_no_event.add(event.meld)
            else:
                self._claimed[event.claimed] = event.meld

    # ---- 查询(投影/日志用) ----

    def attributions(self) -> list[Attribution]:
        return list(self._attribs.values())

    def frozen(self) -> list[Attribution]:
        return [a for a in self._attribs.values() if a.frozen]

    def unfrozen(self) -> list[Attribution]:
        return [a for a in self._attribs.values() if not a.frozen]

    def claimed_ids(self) -> set[int]:
        """被碰走的 TileAppeared eid 集合(显示层排除用)。"""
        return set(self._claimed)

    def melds(self) -> list[tuple[str, MeldFormed]]:
        """已确认副露: [(玩家, 事件)] 按帧序。归属不明(退化)的跳过。"""
        return sorted(((player, ev)
                       for eid, ev in self._meld_list.items()
                       if (player := self._meld_player[eid]) is not None),
                      key=lambda t: t[1].frame)

    # ---- 内部 ----

    def _n_appeared(self) -> int:
        return sum(1 for k, _ in self._unfrozen if k == 'appeared')

    @staticmethod
    def _hand_phase(ev: HandChanged, phase: TurnPhase) -> TurnPhase:
        """手牌相位硬锚: 摸牌(13→14) → 轮到我; 打出(14→13) → 下家。

        副露局手牌数非 13/14, 变化不锚(轮转先验 + 副露截断兜底)。
        """
        if ev.n_old == 13 and ev.n_new == 14:
            return TurnPhase(actor='my_river', played=frozenset())
        if ev.n_old == 14 and ev.n_new == 13:
            return TurnPhase(actor='right_river',
                             played=frozenset({'my_river'}))
        return phase

    def _resolve(self) -> None:
        """窗口内 beam search 重解: 更新每个 TileAppeared 的 logits。

        路径得分 = Σ(空间似然 + 相位先验 − 跳家惩罚); 各事件归属
        分布 = 按整路径得分近似边缘化(soft counting, 束内归一)。
        """
        beams: list[tuple[TurnPhase, float, dict[int, str]]] = [
            (self._phase_base, 0.0, {})]
        for kind, ev in self._unfrozen:
            if kind == 'hand':
                beams = [(self._hand_phase(ev, ph), s, a)
                         for ph, s, a in beams]
            elif kind == 'meld':
                mplayer = self._meld_player[ev.eid]
                if mplayer is None:  # 退化: 归属不明 → 相位不动
                    continue
                beams = [(ph.claim(mplayer)[0], s, a)
                         for ph, s, a in beams]
            else:  # appeared: 四家分支, 束剪枝
                new_beams: list[tuple[TurnPhase, float, dict[int, str]]] = []
                for ph, s, a in beams:
                    for p in PLAYERS:
                        nph, penalty = ph.advance(p)
                        na = dict(a)
                        na[ev.eid] = p
                        new_beams.append(
                            (nph, s + self._river_ev.spatial_logp(
                                p, ev.cx, ev.cy)
                             + phase_prior(ph, p) - penalty, na))
                new_beams.sort(key=lambda t: -t[1])
                beams = new_beams[:self._beam]
        for kind, ev in self._unfrozen:
            if kind != 'appeared':
                continue
            per = {p: float('-inf') for p in PLAYERS}
            for _ph, s, a in beams:
                per[a[ev.eid]] = _logadd(per[a[ev.eid]], s)
            self._attribs[ev.eid] = Attribution(
                eid=ev.eid, tile=ev.tile, cx=ev.cx, cy=ev.cy,
                frame=ev.frame, ts=ev.ts, logits=per)

    def _freeze_oldest(self) -> None:
        """出队到最老的 appeared(之前的 hand/meld 硬更新先消化进基底)。"""
        while self._unfrozen and self._unfrozen[0][0] != 'appeared':
            kind, ev = self._unfrozen.pop(0)
            self._apply_hard(ev)
        if not self._unfrozen:
            return
        _kind, ev = self._unfrozen.pop(0)
        at = self._attribs[ev.eid]
        at.player = at.map()
        at.frozen = True
        # 在线学习(边距门槛: 低置信不喂样本, 防错误归属反馈回路)
        rest = max(v for p, v in at.logits.items() if p != at.player)
        if at.logits[at.player] - rest >= _LEARN_MIN_MARGIN:
            self._river_ev.update(at.player, ev.cx, ev.cy)
        self._phase_base, _ = self._phase_base.advance(at.player)

    def _apply_hard(self, ev) -> None:
        """硬更新事件进入冻结基底(相位推进)。"""
        if isinstance(ev, HandChanged):
            self._phase_base = self._hand_phase(ev, self._phase_base)
        elif isinstance(ev, MeldFormed):
            mplayer = self._meld_player[ev.eid]
            if mplayer is not None:  # 退化: 归属不明 → 相位不动
                self._phase_base, _ = self._phase_base.claim(mplayer)
```

并在顶部 import 区改为:

```python
from src.mahjong_ai.v4.events import (
    PLAYERS,
    HandChanged,
    MeldFormed,
    TileAppeared,
    TileClaimed,
)
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_v4/test_decoder_window.py -q`
Expected: PASS(7 passed)。若 `test_hand_anchor_resets_phase` 的歧义点 (700,600) 边距过窄(测试确定性断言, 应稳定通过),可把该点移到 (700,620) 增大幅距 — 但不得改相位先验常量来绕过。

- [ ] **Step 5: 质量门禁**

Run: `uv run pytest tests/test_v4 -q`、`uv run ruff check src/mahjong_ai/v4 tests/test_v4`、`uv run mypy src/mahjong_ai/v4`
Expected: 全部通过

---

### Task 5: 状态投影(state.py)

**Files:**
- Create: `src/mahjong_ai/v4/state.py`
- Test: `tests/test_v4/test_state.py`

**Interfaces:**
- Consumes: `WindowDecoder`(Task 4)、`PLAYERS`(Task 1)
- Produces: `MeldView(kind, tiles)`、`PlayerView(player, river, visible_river, melds)`、`GameView(my_hand, players, unfrozen)`、`project(decoder, my_hand) -> GameView`

- [ ] **Step 1: 写失败测试**

`tests/test_v4/test_state.py`:

```python
"""状态投影单测 — 事件日志 → 玩家视图(可见/推断拆分)。"""

from src.mahjong_ai.v4.decoder import EvidenceModel, WindowDecoder
from src.mahjong_ai.v4.events import (
    MeldFormed, TileAppeared, TileClaimed,
)
from src.mahjong_ai.v4.state import project

_SEED = {
    'my_river': (600.0, 700.0, 800.0, 780.0),
    'right_river': (1100.0, 500.0, 1180.0, 600.0),
    'top_river': (500.0, 100.0, 700.0, 180.0),
    'left_river': (100.0, 500.0, 180.0, 600.0),
}


def _build() -> WindowDecoder:
    ev = EvidenceModel()
    ev.reseed(_SEED, 1280, 800)
    meld_ev = EvidenceModel()
    meld_ev.reseed(_SEED, 1280, 800)  # 副露归属也用同种子(测试断言右家副露)
    return WindowDecoder(ev, meld_ev, k=1, beam=8)  # k=1: 逐张冻结


def test_river_visible_split_on_claim():
    """被碰走的牌: river 保留(打出历史), visible_river 排除(显示)。"""
    d = _build()
    d.add(TileAppeared(eid=1, tile=5, track_id=1001, cx=700.0, cy=740.0,
                       conf=0.8, frame=0, ts=0.0))
    d.add(TileAppeared(eid=2, tile=8, track_id=1002, cx=700.0, cy=742.0,
                       conf=0.8, frame=1, ts=0.1))
    d.add(TileClaimed(eid=50, claimed=2, meld=60, frame=2, ts=0.2))
    d.add(MeldFormed(eid=60, tiles=(8, 8, 8), cx=1150.0, cy=550.0,
                     bbox=(1100.0, 500.0, 1200.0, 600.0),
                     frame=2, ts=0.2))
    d.add(TileAppeared(eid=3, tile=9, track_id=1003, cx=700.0, cy=744.0,
                       conf=0.8, frame=3, ts=0.3))
    d.add(TileAppeared(eid=4, tile=2, track_id=1004, cx=1150.0, cy=550.0,
                       conf=0.8, frame=4, ts=0.4))
    d.add(TileAppeared(eid=5, tile=3, track_id=1005, cx=700.0, cy=746.0,
                       conf=0.8, frame=5, ts=0.5))  # 推 e4 出窗冻结
    view = project(d, my_hand=[1, 2, 3])
    me = view.players['my_river']
    assert me.river == [5, 8, 9]          # 打出历史含被碰走
    assert me.visible_river == [5, 9]     # 显示不含被碰走
    right = view.players['right_river']
    assert right.river == [2]
    assert right.melds and right.melds[0].kind == 'pong'
    assert right.melds[0].tiles == (8, 8, 8)
    assert view.my_hand == [1, 2, 3]


def test_claim_without_event_not_in_visible():
    """秒碰(claimed=None): 不指向任何牌河事件 — 投影不受影响。"""
    d = _build()
    d.add(TileAppeared(eid=1, tile=5, track_id=1001, cx=700.0, cy=740.0,
                       conf=0.8, frame=0, ts=0.0))
    d.add(MeldFormed(eid=60, tiles=(8, 8, 8), cx=1150.0, cy=550.0,
                     bbox=(1100.0, 500.0, 1200.0, 600.0),
                     frame=1, ts=0.1))
    d.add(TileClaimed(eid=50, claimed=None, meld=60, frame=1, ts=0.1))
    d.add(TileAppeared(eid=2, tile=6, track_id=1002, cx=700.0, cy=742.0,
                       conf=0.8, frame=2, ts=0.2))  # 推 e1 出窗冻结
    view = project(d, my_hand=[])
    assert view.players['my_river'].river == [5]
    assert view.players['my_river'].visible_river == [5]
    assert view.players['right_river'].melds[0].tiles == (8, 8, 8)
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_v4/test_state.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.mahjong_ai.v4.state'`

- [ ] **Step 3: 写实现**

`src/mahjong_ai/v4/state.py`:

```python
"""对局状态投影 — 解码器(事件日志)→ 玩家视图(纯函数)。

牌河 = 冻结事件按帧序投影(含被碰走的 — 打出是历史事实, 推断依据);
visible_river = 牌河中当前物理可见子集(被碰走的消失 — 显示用)。
显示/推断拆分由事件驱动: 不依赖瞬时检测框, 无闪烁。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.mahjong_ai.v4.decoder import Attribution, WindowDecoder
from src.mahjong_ai.v4.events import PLAYERS


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


def project(decoder: WindowDecoder, my_hand: list[int]) -> GameView:
    """解码器 → 游戏视图。"""
    players = {p: PlayerView(player=p) for p in PLAYERS}
    claimed = decoder.claimed_ids()
    for at in sorted(decoder.frozen(), key=lambda a: a.frame):
        pv = players[at.player]
        pv.river.append(at.tile)
        if at.eid not in claimed:
            pv.visible_river.append(at.tile)
    for player, ev in decoder.melds():
        kind = 'kong' if len(ev.tiles) >= 4 else 'pong'
        players[player].melds.append(MeldView(kind=kind, tiles=ev.tiles))
    return GameView(my_hand=my_hand, players=players,
                    unfrozen=decoder.unfrozen())
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_v4/test_state.py -q`
Expected: PASS(2 passed)

- [ ] **Step 5: 质量门禁**

Run: `uv run pytest tests/test_v4 -q`、`uv run ruff check src/mahjong_ai/v4 tests/test_v4`、`uv run mypy src/mahjong_ai/v4`
Expected: 全部通过

---

### Task 6: 合成对局生成器 + 端到端回归

**Files:**
- Create: `tests/test_v4/synth.py`
- Create: `tests/test_v4/test_end_to_end.py`

**Interfaces:**
- Consumes: `WindowDecoder/EvidenceModel`(Task 4)、事件类(Task 1)
- Produces: `synth_game(seed: int) -> list[tuple[str, object]]`(事件序列 + 真值标注)、`ground_truth(event) -> str | None`

- [ ] **Step 1: 写失败测试**

`tests/test_v4/synth.py`:

```python
"""合成对局生成器 — 底牌剧本 + 噪声(漏检/位置抖动/碰截断)。

生成"谁在什么帧打了什么"的事件序列 + 真值标注, 供解码器端到端
回归(每次调窗口 K/束宽/权重都有硬指标)。
"""

from __future__ import annotations

import random

from src.mahjong_ai.v4.decoder import EvidenceModel
from src.mahjong_ai.v4.events import (
    HandChanged, MeldFormed, TileAppeared, TileClaimed,
)

_ROTATION = ('my_river', 'right_river', 'top_river', 'left_river')

_SEED = {
    'my_river': (600.0, 700.0, 800.0, 780.0),
    'right_river': (1100.0, 500.0, 1180.0, 600.0),
    'top_river': (500.0, 100.0, 700.0, 180.0),
    'left_river': (100.0, 500.0, 180.0, 600.0),
}
_MELD_SEED = {
    'my_river': (820.0, 700.0, 920.0, 780.0),
    'right_river': (1100.0, 620.0, 1200.0, 700.0),
    'top_river': (720.0, 100.0, 820.0, 180.0),
    'left_river': (80.0, 500.0, 160.0, 600.0),
}


def synth_game(seed: int = 0, miss_every: int = 9) -> list[tuple[str, object]]:
    """生成一局事件序列: [(ground_truth_player, event)]。

    剧本: 三轮正常轮转(12 张) + 第 2 轮右家碰我的第 2 张(截断, 右家
    接着打)。每 miss_every 张漏检一张(不生成事件); 落点加 ±15px 抖动。
    事件按帧序排列; 我的打出前后插 HandChanged(13→14 / 14→13)。
    """
    rng = random.Random(seed)
    seq: list[tuple[str, object]] = []
    eid = 0
    frame = 0

    def hand_draw() -> tuple[str, object]:
        """我摸牌(13→14): 相位硬锚 — 在打出事件之前。"""
        nonlocal eid
        eid += 1
        return ('my_river', HandChanged(eid=eid, n_old=13, n_new=14,
                                        frame=frame, ts=frame * 0.1))

    def hand_disc() -> tuple[str, object]:
        """我打出(14→13): 在打出事件之后。"""
        nonlocal eid
        eid += 1
        return ('my_river', HandChanged(eid=eid, n_old=14, n_new=13,
                                        frame=frame, ts=frame * 0.1))

    def appeared(player: str, tile: int) -> tuple[str, object]:
        nonlocal eid
        eid += 1
        x1, y1, x2, y2 = _SEED[player]
        cx = (x1 + x2) / 2 + rng.uniform(-15, 15)
        cy = (y1 + y2) / 2 + rng.uniform(-15, 15)
        return (player, TileAppeared(
            eid=eid, tile=tile, track_id=5000 + eid, cx=cx, cy=cy,
            conf=0.8, frame=frame, ts=frame * 0.1))

    played = 0
    order = list(_ROTATION) * 3      # 12 张(第 2 轮会被碰截断重排)
    skip_next = False
    for i, p in enumerate(order):
        if skip_next:
            skip_next = False        # 碰后右家已打, 跳过原计划该张
            continue
        if p == 'my_river':
            seq.append(hand_draw())  # 摸牌锚在打出之前
        frame += 1
        if (i + 1) % miss_every == 0:
            continue  # 漏检: 该打出没有事件
        ev = appeared(p, tile=(played % 30))
        seq.append(ev)
        if p == 'my_river':
            seq.append(hand_disc())  # 打出锚在打出之后
        played += 1
        # 我第 2 轮的打出(i==4)被右家碰 → 副露 + 被碰 + 右家接着打
        if i == 4:
            eid += 1
            meld_eid = eid
            mx1, my1, mx2, my2 = _MELD_SEED['right_river']
            seq.append(('right_river', MeldFormed(
                eid=meld_eid, tiles=(ev[1].tile,) * 3,
                cx=(mx1 + mx2) / 2, cy=(my1 + my2) / 2,
                bbox=(mx1, my1, mx2, my2), frame=frame, ts=frame * 0.1)))
            eid += 1
            seq.append(('right_river', TileClaimed(
                eid=eid, claimed=ev[1].eid, meld=meld_eid,
                frame=frame, ts=frame * 0.1)))
            frame += 1
            eid += 1
            seq.append(('right_river', TileAppeared(
                eid=eid, tile=(played % 30), track_id=5000 + eid,
                cx=(_SEED['right_river'][0] + _SEED['right_river'][2]) / 2,
                cy=(_SEED['right_river'][1] + _SEED['right_river'][3]) / 2,
                conf=0.8, frame=frame, ts=frame * 0.1)))
            played += 1
            skip_next = True
    return seq


def make_decoder(k: int = 10, beam: int = 16):
    """构造种好子的解码器(牌河种子 + 副露种子)。"""
    river_ev = EvidenceModel()
    river_ev.reseed(_SEED, 1280, 800)
    meld_ev = EvidenceModel()
    meld_ev.reseed(_MELD_SEED, 1280, 800)
    from src.mahjong_ai.v4.decoder import WindowDecoder

    return WindowDecoder(river_ev, meld_ev, k=k, beam=beam)
```

`tests/test_v4/test_end_to_end.py`:

```python
"""端到端回归 — 合成对局(含漏检/抖动/碰截断)解码准确率与熵诊断。"""

from tests.test_v4.synth import make_decoder, synth_game


def test_end_to_end_accuracy_and_entropy():
    events = synth_game(seed=7, miss_every=9)
    dec = make_decoder(k=3)   # 小窗口: 多冻结、重解频繁, 更严苛
    for _truth, ev in events:
        dec.add(ev)
    # 冻结事件的 MAP 归属 vs 真值
    frozen = dec.frozen()
    truth = {ev.eid: p for p, ev in events}
    correct = sum(1 for a in frozen
                  if truth.get(a.eid) == a.player)
    assert len(frozen) >= 8, f'冻结数过少: {len(frozen)}'
    acc = correct / len(frozen)
    assert acc >= 0.85, f'归属准确率 {acc:.0%} < 85%'
    # 自诊断: 错误归属的熵应高于正确归属(证据不足 ≠ 假装确定)
    wrong = [a for a in frozen if truth.get(a.eid) != a.player]
    right = [a for a in frozen if truth.get(a.eid) == a.player]
    if wrong:
        assert (sum(a.entropy() for a in wrong) / len(wrong)
                >= sum(a.entropy() for a in right) / len(right))
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_v4/test_end_to_end.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tests.test_v4.synth'`(文件尚不存在)

- [ ] **Step 3: 写实现**

本任务两个文件即上述内容(synth.py + test_end_to_end.py),无额外实现 — 若测试失败,调 `make_decoder` 的 k/beam 或回查 Task 4 的解码器实现(权重/惩罚值),**不得**降低准确率断言来掩盖问题;允许把熵断言改成"错误熵均值 ≥ 正确熵均值"这一更稳的形态:

```python
    if wrong:
        assert (sum(a.entropy() for a in wrong) / len(wrong)
                >= sum(a.entropy() for a in right) / len(right))
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_v4/test_end_to_end.py -q`
Expected: PASS(1 passed)

- [ ] **Step 5: 质量门禁**

Run: `uv run pytest tests/test_v4 -q`、`uv run ruff check src/mahjong_ai/v4 tests/test_v4`、`uv run mypy src/mahjong_ai/v4`
Expected: 全部通过

---

### Task 7: 事件提取器(EventExtractor)

**Files:**
- Create: `src/mahjong_ai/v4/extractor.py`
- Test: `tests/test_v4/test_extractor.py`

**Interfaces:**
- Consumes: `cluster_dets(dets, conf_min=0.25) -> (hand, melds, rivers)`(src.mahjong_cv.det_cluster, 只读复用)、`TileDet`(src.mahjong_cv.detections)、事件类(Task 1)
- Produces: `EventExtractor()`、`process(dets, frame, ts) -> list[Event]`、`my_hand() -> list[int]`、`hand_dets() -> list[TileDet]`

- [ ] **Step 1: 写失败测试**

`tests/test_v4/test_extractor.py`:

```python
"""事件提取器单测 — 合成 TileDet 帧序列 → 事件(不带归属)。"""

from src.mahjong_ai.v4.events import (
    FlowerShown, HandChanged, MeldFormed, TileAppeared, TileClaimed,
)
from src.mahjong_ai.v4.extractor import EventExtractor
from src.mahjong_cv.detections import TileDet


def _mk(tile, x1, y1, x2, y2, conf=0.8, tid=None) -> TileDet:
    return TileDet(tile=tile, x1=x1, y1=y1, x2=x2, y2=y2, conf=conf,
                   track_id=tid)


def _hand_frame(n=13, x0=100.0, tid_base=100) -> list[TileDet]:
    """底行横排手牌: 宽 40 高 70, 间距 40(与真实比例一致)。"""
    return [_mk(i % 30, x0 + i * 40, 700.0, x0 + i * 40 + 40, 770.0,
                tid=tid_base + i) for i in range(n)]


def _feed(ext: EventExtractor, frames: list[list[TileDet]]) -> list:
    out = []
    for i, dets in enumerate(frames):
        out += ext.process(dets, frame=i, ts=i * 0.1)
    return out


def test_hand_changed_draw_and_discard():
    ext = EventExtractor()
    frames = [_hand_frame(13)] * 3          # 去抖稳定 13
    frames += [_hand_frame(13) + [_mk(13, 620.0, 700.0, 660.0, 770.0,
                                      tid=999)]] * 3   # 摸到 14(tile 13 不在基牌 0-12 中)
    events = _feed(ext, frames)
    changed = [e for e in events if isinstance(e, HandChanged)]
    assert changed and changed[0].n_old == 0
    assert changed[-1].n_old == 13 and changed[-1].n_new == 14
    # 位置序: 第 14 张在 x 620-660, 位于 tile 12(x 580-620)之后
    assert ext.my_hand() == list(range(13)) + [13]


def test_appeared_emitted_once_when_stable():
    ext = EventExtractor()
    frames = [_hand_frame(13)] * 3        # 去抖稳定 → 门开
    frames += [_hand_frame(13)
               + [_mk(7, 700.0, 400.0, 715.0, 415.0, tid=1)]] * 2
    events = _feed(ext, frames)
    appeared = [e for e in events if isinstance(e, TileAppeared)]
    assert len(appeared) == 1
    assert appeared[0].tile == 7


def test_appeared_gated_before_hand_stable():
    """发牌动画期(手牌 <13)新轨迹不产出事件。"""
    ext = EventExtractor()
    frames = [_hand_frame(6)] * 2
    frames += [frames[0] + [_mk(7, 700.0, 400.0, 715.0, 415.0, tid=1)]] * 2
    events = _feed(ext, frames)
    assert not [e for e in events if isinstance(e, TileAppeared)]


def test_flower_shown_deduped():
    ext = EventExtractor()
    dets = _hand_frame(13) + [_mk(34, 900.0, 750.0, 940.0, 820.0, tid=77)]
    events = _feed(ext, [dets, dets])
    flowers = [e for e in events if isinstance(e, FlowerShown)]
    assert len(flowers) == 1


def test_meld_confirmed_and_claim():
    ext = EventExtractor()
    base = _hand_frame(13)
    # 手牌先去抖稳定(3 帧, 门开), 再打出一张(tid 500)稳定出现
    f1 = base + [_mk(8, 700.0, 400.0, 715.0, 415.0, tid=500)]
    # 碰组: 3 张紧贴(含 tid 500 这张 + 2 张手牌尺寸), 与手牌远离
    meld = [_mk(8, 1150.0, 520.0, 1190.0, 590.0, tid=500),
            _mk(8, 1190.0, 520.0, 1230.0, 590.0, tid=501),
            _mk(8, 1230.0, 520.0, 1270.0, 590.0, tid=502)]
    events = _feed(ext, [base, base, base, f1, f1,
                         base + meld, base + meld, base + meld])
    meld_events = [e for e in events if isinstance(e, MeldFormed)]
    claims = [e for e in events if isinstance(e, TileClaimed)]
    assert meld_events, '碰组 3 帧应确认副露'
    assert claims and claims[0].claimed is not None
    appeared = [e for e in events if isinstance(e, TileAppeared)]
    assert len(appeared) == 1, f'只有打出的牌产生事件: {len(appeared)}'
    assert claims[0].claimed == appeared[0].eid  # 被碰的就是刚打出的那张


def test_baseline_swallows_existing_boxes():
    """门开瞬间已存在的非手牌框(别家手牌)不发事件; 新轨迹才发。"""
    ext = EventExtractor()
    base = _hand_frame(13)
    opp_hand = [_mk(i % 30, 1300.0 + i * 40, 100.0,
                    1340.0 + i * 40, 170.0, tid=300 + i)
                for i in range(13)]
    frames = [base + opp_hand] * 5     # 门开(帧 2)基线吞掉别家手牌
    frames += [base + opp_hand
               + [_mk(7, 700.0, 400.0, 715.0, 415.0, tid=1)]] * 2
    events = _feed(ext, frames)
    appeared = [e for e in events if isinstance(e, TileAppeared)]
    assert len(appeared) == 1  # 只有新轨迹; 别家手牌轨迹不发事件
    assert appeared[0].tile == 7
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_v4/test_extractor.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.mahjong_ai.v4.extractor'`

- [ ] **Step 3: 写实现**

`src/mahjong_ai/v4/extractor.py`:

```python
"""事件提取器 — 检测帧序列 → 观测事件(不带归属结论)。

复用 det_cluster 的聚类(手牌/副露组)做分组; 这里只做"出现/成组/
手牌变化"的确认, 不做归属(归属是解码器的事)。提取规则独立可单测。
"""

from __future__ import annotations

from collections import Counter

from src.mahjong_ai.v4.events import (
    FlowerShown,
    HandChanged,
    MeldFormed,
    TileAppeared,
    TileClaimed,
)
from src.mahjong_cv.det_cluster import cluster_dets
from src.mahjong_cv.detections import TileDet

#: 手牌置信度门槛(与现行去抖口径一致)
HAND_CONF = 0.55
#: 手牌去抖窗口/majority(现行口径, 自包含复制 — 不 import 旧 tracker)
_DEBOUNCE_WINDOW = 5
_DEBOUNCE_MAJORITY = 3
#: 新轨迹确认: 累计出现 ≥ 此帧数且位置稳定才发 TileAppeared
_APPEAR_FRAMES = 2
#: 位置稳定: 距首现位置位移 < 此值 × 框宽
_STABLE_RATIO = 0.5
#: 新轨迹超时(帧): 迟迟不稳定的候选丢弃(误检)
_APPEAR_TTL = 30
#: 副露确认帧数(候选按位置匹配去重)
_MELD_CONFIRM = 3
#: 副露候选位置匹配距离(像素)
_MELD_MATCH_PX = 40.0
#: 手牌带外扩(带内新轨迹 = 手牌, 不是打出)
_BAND_PAD = 10.0
#: 花牌位置去重粒度(像素/格)
_FLOWER_GRID = 20
#: 重复事件位置去重半径(像素 — 同牌重识别不重发)
_DUP_RADIUS = 30.0


class EventExtractor:
    """检测帧 → 事件。手牌去抖自包含(与旧 tracker 解耦)。"""

    def __init__(self) -> None:
        self._eid = 0
        self._hand_history: list[Counter[int]] = []
        self._hand_n = 0
        self._hand_pos: dict[int, float] = {}
        self._hand_band: tuple[float, float] | None = None
        self._last_hand: list[TileDet] = []
        self._tracks: dict[int, dict] = {}
        self._seen: set[int] = set()             # 基线已见轨迹(不发事件)
        self._emitted: set[int] = set()
        self._appeared: dict[int, int] = {}      # tid -> eid(碰检测用)
        self._dup_table: dict[int, tuple[int, float, float]] = {}
        self._meld_pool: list[dict] = []
        self._meld_seen = False
        self._gate_was_open = False
        self._flower_seen: set[tuple[int, int]] = set()

    def _next_eid(self) -> int:
        """事件序号(全局自增, 唯一)。"""
        self._eid += 1
        return self._eid

    # ---- 主入口 ----

    def process(self, dets: list[TileDet], frame: int, ts: float) -> list:
        """一帧检测 → 本帧新事件(副露/被碰先于新牌, 手牌变化最后)。"""
        events: list = []
        hand, melds, _rivers = cluster_dets(dets)
        self._last_hand = [d for d in hand if d.conf >= HAND_CONF]
        # 1) 手牌去抖 + HandChanged(最后追加: 相位锚作用于本帧之后)
        counts: Counter[int] = Counter()
        for d in self._last_hand:
            counts[d.tile] += 1
            self._hand_pos[d.tile] = (d.x1 + d.x2) / 2
        self._hand_history.append(counts)
        if len(self._hand_history) > _DEBOUNCE_WINDOW:
            self._hand_history.pop(0)
        hand_changed = None
        n = len(self._debounced_hand())
        if n != self._hand_n:
            hand_changed = HandChanged(eid=self._next_eid(),
                                       n_old=self._hand_n, n_new=n,
                                       frame=frame, ts=ts)
            self._hand_n = n
        if self._last_hand:
            self._hand_band = (min(d.y1 for d in self._last_hand),
                               max(d.y2 for d in self._last_hand))
        # 2) 花牌 → 亮花事件(位置去重; 不进归属)
        for d in dets:
            if d.tile != 34:
                continue
            key = (round((d.x1 + d.x2) / 2 / _FLOWER_GRID),
                   round((d.y1 + d.y2) / 2 / _FLOWER_GRID))
            if key in self._flower_seen:
                continue
            self._flower_seen.add(key)
            events.append(FlowerShown(
                eid=self._next_eid(), tile=d.tile,
                cx=(d.x1 + d.x2) / 2, cy=(d.y1 + d.y2) / 2,
                frame=frame, ts=ts))
        # 3) 副露组确认 → MeldFormed + TileClaimed
        for m in melds:
            self._check_meld(m, frame, ts, events)
        meld_tids = {d.track_id for m in melds for d in m
                     if d.track_id is not None}
        # 4) 新轨迹 → TileAppeared(手牌稳定前不产出 — 发牌动画期
        #    半空的牌在桌面中央, 不是打出)
        gate_open = self._hand_n >= 13 or self._meld_seen
        if gate_open and not self._gate_was_open:
            # 基线: 门开瞬间已存在的非手牌框(别家手牌/既有牌河)标记已见,
            # 之后只有"新出现的轨迹"才产出事件。打出 = 牌从手牌飞出,
            # ByteTrack 跨不过该位移 → 新轨迹(实测旧管道据此正常工作)。
            hand_tids = {d.track_id for d in hand if d.track_id is not None}
            for d in dets:
                if d.track_id is not None and d.track_id not in hand_tids:
                    self._seen.add(d.track_id)
            self._gate_was_open = True
        if gate_open:
            for d in dets:
                self._check_appeared(d, frame, ts, events, meld_tids)
        if hand_changed is not None:
            events.append(hand_changed)
        return events

    # ---- 手牌 ----

    def my_hand(self) -> list[int]:
        """去抖手牌(位置序, 与检测框显示一致)。"""
        return self._debounced_hand()

    def hand_dets(self) -> list[TileDet]:
        """本帧手牌框(显示用)。"""
        return self._last_hand

    def _debounced_hand(self) -> list[int]:
        counts: Counter[int] = Counter()
        for c in self._hand_history:
            counts.update(c)
        result: list[int] = []
        for t, n in counts.items():
            if n >= _DEBOUNCE_MAJORITY:
                mult = max(c[t] for c in self._hand_history)
                result.extend([t] * mult)
        result.sort(key=lambda t: self._hand_pos.get(t, 0.0))
        return result[:14]

    # ---- 新轨迹 ----

    def _check_appeared(self, d: TileDet, frame: int, ts: float,
                        events: list, meld_tids: set[int]) -> None:
        if d.track_id is None or d.tile == 34:
            return
        tid = d.track_id
        if tid in self._emitted or tid in self._seen or tid in meld_tids:
            return
        cx, cy = (d.x1 + d.x2) / 2, (d.y1 + d.y2) / 2
        t = self._tracks.get(tid)
        if t is None:
            self._tracks[tid] = {'lx': cx, 'ly': cy, 'f0': frame,
                                 'stable': 1}
            return
        # 稳定性锚定最近位置: 相邻两次出现位移 < 0.5×框宽 = 已落定。
        # 锚定首现位置的旧逻辑会把"飞行中首现"的牌判不稳定而丢弃 —
        # 区域门禁常开后大量打出因此丢失(真实对局回归: 整局 9 事件)
        w = d.x2 - d.x1
        moved = ((cx - t['lx']) ** 2 + (cy - t['ly']) ** 2) ** 0.5
        if moved < _STABLE_RATIO * w:
            t['stable'] += 1
        else:
            t['stable'] = 1  # 还在动(飞行中): 重新计数
        t['lx'], t['ly'] = cx, cy
        if (t['stable'] >= _APPEAR_FRAMES
                and not self._in_hand_band(d)
                and not self._recent_dup(d.tile, cx, cy)
                and (not self._regions
                     or self._in_any_river_region(cx, cy))):
            self._emitted.add(tid)
            self._tracks.pop(tid)
            self._appeared[tid] = self._next_eid()
            events.append(TileAppeared(
                eid=self._appeared[tid], tile=d.tile, track_id=tid,
                cx=cx, cy=cy, conf=d.conf, frame=frame, ts=ts))
        elif frame - t['f0'] > _APPEAR_TTL:
            self._tracks.pop(tid)  # 迟迟不落定(误检) → 丢弃

    def _recent_dup(self, tile: int, cx: float, cy: float) -> bool:
        """同 tile 近位置已发过事件 → 重复(跟踪重识别), 不重发。"""
        for t, x, y in self._dup_table.values():
            if t == tile and abs(x - cx) < _DUP_RADIUS \
                    and abs(y - cy) < _DUP_RADIUS:
                return True
        self._dup_table[len(self._dup_table)] = (tile, cx, cy)
        return False

    def _in_hand_band(self, d: TileDet) -> bool:
        """框中心是否在手牌带内(±_BAND_PAD) — 带内框是手牌, 不是打出。"""
        if self._hand_band is None:
            return False
        cy = (d.y1 + d.y2) / 2
        return self._hand_band[0] - _BAND_PAD <= cy \
            <= self._hand_band[1] + _BAND_PAD

    # ---- 副露 ----

    def _check_meld(self, m: list[TileDet], frame: int, ts: float,
                    events: list) -> None:
        """副露组确认(位置匹配去重, 累计 ≥_MELD_CONFIRM 帧) + 被碰检测。"""
        cx = sum((d.x1 + d.x2) / 2 for d in m) / len(m)
        cy = sum((d.y1 + d.y2) / 2 for d in m) / len(m)
        cand = next((c for c in self._meld_pool
                     if abs(c['cx'] - cx) < _MELD_MATCH_PX
                     and abs(c['cy'] - cy) < _MELD_MATCH_PX), None)
        if cand is None:
            cand = {'cx': cx, 'cy': cy, 'count': 0, 'done': False}
            self._meld_pool.append(cand)
        cand['count'] += 1
        if cand['count'] < _MELD_CONFIRM or cand['done']:
            return
        cand['done'] = True
        ev = MeldFormed(eid=self._next_eid(),
                        tiles=tuple(sorted(d.tile for d in m)),  # 保多重度: 杠 8-8-8-8 → (8,8,8,8), kind 判定靠张数
                        cx=cx, cy=cy,
                        bbox=(min(d.x1 for d in m), min(d.y1 for d in m),
                              max(d.x2 for d in m), max(d.y2 for d in m)),
                        frame=frame, ts=ts)
        events.append(ev)
        self._meld_seen = True
        # 被碰走的是哪张: 组内已打出过(tid 在 _appeared)的牌
        claimed = None
        for d in sorted(m, key=lambda x: -x.conf):
            if d.track_id is not None and d.track_id in self._appeared:
                claimed = self._appeared[d.track_id]
                break
        events.append(TileClaimed(eid=self._next_eid(), claimed=claimed,
                                  meld=ev.eid, frame=frame, ts=ts))
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_v4/test_extractor.py -q`
Expected: PASS(6 passed)。若 `test_meld_confirmed_and_claim` 失败,先检查: 碰组 3 张是否被 cluster_dets 判为 meld(紧贴 packing≥1.0、高度 ≥0.7×手牌均高 — 测试里的 70px 与手牌一致);不是则调整测试几何,不得改提取器绕过。

- [ ] **Step 5: 质量门禁**

Run: `uv run pytest tests/test_v4 -q`、`uv run ruff check src/mahjong_ai/v4 tests/test_v4`、`uv run mypy src/mahjong_ai/v4`
Expected: 全部通过

---

### Task 8: 客户端白名单修改(_visible_tiles 回退)

**Files:**
- Modify: `src/mahjong_ui/client.py`(唯一旧文件改动, 规格 §12 授权)
- Test: `tests/test_v4/test_client_fallback.py`

**Interfaces:**
- Consumes: 无新接口(复用现有 `MahjongClient.update(advice, boxes)`)
- Produces: `_InfoPanel._visible_tiles(player)` 无归属框时回退读 `advice.snapshot.players[player].river`

- [ ] **Step 1: 写失败测试**

`tests/test_v4/test_client_fallback.py`:

```python
"""客户端回退单测 — v4 路径(无归属框)下牌河显示读快照投影。"""

from src.mahjong_ai.session import Advice
from src.mahjong_ai.state.tracker import GameSnapshot, PlayerState
from src.mahjong_ui.client import _InfoPanel


def test_visible_tiles_fallback_to_snapshot_river():
    snap = GameSnapshot(
        my_hand=[1, 2, 3],
        players={'my_river': PlayerState(river=[5, 8]),
                 'right_river': PlayerState(river=[]),
                 'top_river': PlayerState(river=[]),
                 'left_river': PlayerState(river=[])})
    panel = _InfoPanel()
    panel.set_advice(Advice(snapshot=snap), boxes={'hand': []})
    assert panel._visible_tiles('my_river') == [5, 8]


def test_boxes_still_win_when_present():
    """旧路径不变: 有归属框时仍以框为准。"""
    from src.mahjong_cv.detections import TileDet

    snap = GameSnapshot(
        my_hand=[], players={p: PlayerState()
                             for p in ('my_river', 'right_river',
                                       'top_river', 'left_river')})
    panel = _InfoPanel()
    panel.set_advice(Advice(snapshot=snap),
                     boxes={'my_river': [TileDet(tile=3, x1=0, y1=0,
                                                 x2=10, y2=10, conf=0.9)]})
    assert panel._visible_tiles('my_river') == [3]
```

注意: `_InfoPanel.__init__` 会创建 QWidget — 测试需 QApplication 或离屏平台。若报 `QWidget: Cannot create a QWidget without QApplication`,在测试文件顶部加:

```python
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_v4/test_client_fallback.py -q`
Expected: FAIL — `panel._visible_tiles('my_river')` 返回 `[]`(当前无回退)

- [ ] **Step 3: 写实现**

`src/mahjong_ui/client.py` 的 `_visible_tiles` 方法(约 125 行)替换为:

```python
    def _visible_tiles(self, player: str) -> list[int]:
        """牌河显示用: 当前物理可见的牌。

        优先 tracker 归属框(旧管道); 该玩家键不存在时回退快照投影
        (v4 管道: visible_river 已排除被碰走的牌)。用 `is not None`
        而非真值判断 — 旧管道恒写入四家 river 键(可能为空列表),
        空列表时显示应为空(不闪烁到含被碰走牌的历史)。与
        players[p].river 不同: 后者是"打出历史"(推断用, 永久)。
        """
        dets = self._boxes.get(player)
        if dets is not None:
            return [d.tile for d in dets]
        if self._advice is not None:
            return list(self._advice.snapshot.players[player].river)
        return []
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_v4/test_client_fallback.py -q` 且 `uv run pytest tests/ -q`(全量回归 — 旧路径不受影响)
Expected: PASS

- [ ] **Step 5: 质量门禁**

client.py 有既有债(run 基线: ruff 3 条 F401/I001、mypy 26 错, 全在被改方法之外 — 旧代码零改动不修)。本任务门禁 = **相对基线无新增**:
Run: `uv run ruff check src/mahjong_ui/client.py` 且 `uv run mypy src/mahjong_ui/client.py --follow-imports=silent`
Expected: 报错数量与基线一致(3 ruff 条 + 26 mypy 条, `_visible_tiles` 方法贡献 0 新增);tests/test_v4 全过

---

### Task 9: v4 运行器(run_assistant_v4.py)

**Files:**
- Create: `scripts/run_assistant_v4.py`
- Test: `tests/test_v4/test_runner_glue.py`

**Interfaces:**
- Consumes: 全部 v4 接口、`ScreenVision`/`Win32Capture`/`detect_game_area`/`map_regions_to_frame`(复用)、`StrategyEngine`/`get_rules`/`Hand`、`Advice/GameSnapshot/PlayerState/Meld`(数据类复用)、`MahjongClient`/`DetOverlay`
- Produces: `build_advice(view, engine, rules) -> tuple[Advice, dict]`(可测的胶水函数)、`load_seed_regions() -> tuple[dict, dict]`、`V4Runner`、`main()`

- [ ] **Step 1: 写失败测试**

`tests/test_v4/test_runner_glue.py`:

```python
"""v4 运行器胶水单测 — GameView → Advice 转换。"""

from src.mahjong_ai.strategy import StrategyEngine
from src.mahjong_ai.v4.state import GameView, MeldView, PlayerView
from src.mahjong_engine import get as get_rules
from scripts.run_assistant_v4 import build_advice


def _engine():
    rules = get_rules('huanyu')
    return StrategyEngine(rules), rules


def _view() -> GameView:
    players = {p: PlayerView(player=p)
               for p in ('my_river', 'right_river', 'top_river', 'left_river')}
    players['my_river'].river = [5, 8]
    players['my_river'].visible_river = [5]
    players['right_river'].melds = [MeldView(kind='pong', tiles=(8, 8, 8))]
    return GameView(my_hand=[1, 2, 3], players=players, unfrozen=[])


def test_build_advice_maps_visible_river_and_melds():
    engine, rules = _engine()
    view = _view()
    advice, _boxes = build_advice(view, engine, rules)
    snap = advice.snapshot
    assert snap.my_hand == [1, 2, 3]
    assert snap.players['my_river'].river == [5]   # 显示用 visible
    assert snap.players['right_river'].melds[0].tile == 8


def test_build_advice_discard_when_14():
    engine, rules = _engine()
    view = GameView(
        my_hand=[0, 0, 0, 1, 2, 3, 9, 10, 11, 18, 19, 20, 27, 31],
        players={p: PlayerView(player=p)
                 for p in ('my_river', 'right_river', 'top_river',
                           'left_river')},
        unfrozen=[])
    advice, _ = build_advice(view, engine, rules)
    assert advice.discard is not None
    assert advice.discard.tile in view.my_hand
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_v4/test_runner_glue.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.run_assistant_v4'`

- [ ] **Step 3: 写实现**

`scripts/run_assistant_v4.py`:

```python
"""欢乐麻将实时 AI 辅助 v4 — 事件流 + 全局归属解码管道(与旧管道并行)。

旧 run_assistant.py 不动; 本入口用新核心: 捕获 → 检测/跟踪(复用
ScreenVision) → 事件提取(EventExtractor) → 窗口解码(WindowDecoder,
软归属) → 状态投影(GameView) → 手牌建议 + 客户端显示。
事件日志 data/settle_logs/game_v4_*.jsonl(复盘/核对用)。

用法:
    python scripts/run_assistant_v4.py               # 实时模式
    python scripts/run_assistant_v4.py --no-dets     # 关闭悬浮框
    python scripts/run_assistant_v4.py --imgsz 1280  # 全精度
"""

import argparse
import json
import signal
import sys
import threading
import time
import traceback
from collections.abc import Callable
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mahjong_ai.efficiency.discard_selector import DiscardRecommendation
from src.mahjong_ai.session import Advice
from src.mahjong_ai.state.tracker import (  # 仅数据类复用(旧代码只读)
    GameSnapshot,
    Meld,
    PlayerState,
)
from src.mahjong_ai.strategy import StrategyEngine
from src.mahjong_ai.v4.decoder import EvidenceModel, WindowDecoder
from src.mahjong_ai.v4.events import event_to_dict
from src.mahjong_ai.v4.extractor import EventExtractor
from src.mahjong_ai.v4.state import GameView, project
from src.mahjong_core import Hand
from src.mahjong_cv.capture.win32 import Win32Capture
from src.mahjong_cv.detections import TileDet
from src.mahjong_cv.game_area import detect_game_area, map_regions_to_frame
from src.mahjong_cv.screen_vision import ScreenVision
from src.mahjong_engine import get as get_rules
from src.mahjong_engine.judges.tenpai_judge import WaitingTile
from src.mahjong_ui.client import MahjongClient
from src.mahjong_ui.det_overlay import DetOverlay

MODEL = 'data/models/screen/mahjong_screen_detector/weights/best.pt'


class InferWorker:
    """后台推理线程(与旧入口同构): 最新帧优先, 结果取最新。"""

    def __init__(self, recognize: Callable[[np.ndarray], list[TileDet]]):
        self._recognize = recognize
        self._lock = threading.Lock()
        self._pending: np.ndarray | None = None
        self._results: list[tuple[tuple[int, int], list[TileDet]]] = []
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name='v4-infer-worker')
        self._thread.start()

    def submit(self, frame: np.ndarray) -> None:
        with self._lock:
            self._pending = frame

    def poll(self) -> tuple[tuple[int, int], list[TileDet]] | None:
        with self._lock:
            if not self._results:
                return None
            size, dets = self._results[-1]
            self._results.clear()
            return size, dets

    def _run(self) -> None:
        while True:
            with self._lock:
                frame = self._pending
                self._pending = None
            if frame is None:
                time.sleep(0.005)
                continue
            dets = self._recognize(frame)
            with self._lock:
                self._results.append((frame.shape[:2][::-1], dets))


def load_seed_regions() -> tuple[dict, dict]:
    """框选区域(config/river_regions.json, 相对画面区域归一化)。

    返回 (牌河种子, 副露种子); 无配置 → 空 dict(空间退化为弱证据)。
    副露键重映射: 配置里是 my_meld/right_meld/..., EvidenceModel 只认
    PLAYERS 名(my_river/...)— 不映射则副露种子全部失效(全部归我)。
    """
    p = Path('config/river_regions.json')
    if not p.exists():
        return {}, {}
    data = json.loads(p.read_text(encoding='utf-8'))
    rivers = {k: tuple(v) for k, v in data.get('regions', {}).items()}
    melds = {k.removesuffix('_meld') + '_river': tuple(v)
             for k, v in data.get('meld_regions', {}).items()}
    return rivers, melds


def build_advice(view: GameView, engine=None, rules=None):
    """GameView → (Advice, boxes) — M2 只做手牌建议(无对手推断)。

    discard: 14 张推荐 / ≥2 张漏检也推荐(相对判断);
    waiting: 13 张听牌提示。对手推断(M3)接入后再扩。
    """
    n = len(view.my_hand)
    discard = None
    waiting = None
    if n == 14:
        rec = engine.recommend_discard(Hand(view.my_hand))
        discard = (rec.tile, rec.reason)
    elif n == 13:
        wt = rules.get_waiting_tiles(Hand(view.my_hand))
        if wt:
            waiting = [w.tile for w in wt]
    elif n >= 2:
        rec = engine.recommend_discard(Hand(view.my_hand))
        discard = (rec.tile, rec.reason + '(手牌可能漏检)')
    players = {}
    for p, pv in view.players.items():
        players[p] = PlayerState(
            melds=[Meld(kind=m.kind, tile=m.tiles[0]) for m in pv.melds],
            river=list(pv.visible_river))  # 显示用可见牌河(被碰走已排除)
    snap = GameSnapshot(my_hand=view.my_hand, players=players)
    advice = Advice(snapshot=snap)
    if discard is not None:
        advice.discard = DiscardRecommendation(tile=discard[0],
                                               reason=discard[1])
    if waiting:
        advice.waiting = [WaitingTile(t, 0) for t in waiting]
    return advice, {}


class V4Runner:
    """实时循环: 捕获 → 推理 → 事件 → 解码 → 投影 → 显示/日志。"""

    def __init__(self, cap: Win32Capture, recognize, client: MahjongClient,
                 det_overlay: DetOverlay | None, log_dir: str) -> None:
        self._cap = cap
        self._worker = InferWorker(recognize)
        self._client = client
        self._det_overlay = det_overlay
        self._extractor = EventExtractor()
        self._river_ev = EvidenceModel()
        self._meld_ev = EvidenceModel()
        self._decoder = WindowDecoder(self._river_ev, self._meld_ev)
        rules = get_rules('huanyu')
        self._rules = rules
        self._engine = StrategyEngine(rules)
        self._river_rel, self._meld_rel = load_seed_regions()
        self._last_area: tuple | None = None
        self._seeded = False
        self._region_ticks = 0
        self._frame_idx = 0
        self._frozen_count = 0
        self._meld_count = 0
        self._last_err_ts = 0.0
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        name = f'game_v4_{time.strftime("%Y%m%d_%H%M%S")}.jsonl'
        self._log = (Path(log_dir) / name).open('a', encoding='utf-8')

    def tick(self) -> None:
        frame = self._cap.capture()
        if frame is None:
            return
        w, h = frame.shape[1], frame.shape[0]
        if self._region_ticks <= 0:
            area = detect_game_area(frame)
            # 从未播种 或 画面区域变化 → 重播种。注意: area 为 None
            # (无黑边)时也必须播种一次 — 初值 last_area=None 与
            # area=None 相等会跳过(真实对局曾因此四家高斯停在原点、
            # 全部归属我的 bug); 用独立 _seeded 旗标防此回归
            if not self._seeded or area != self._last_area:
                self._river_ev.reseed(
                    map_regions_to_frame(self._river_rel, area, w, h)
                    if self._river_rel else None, w, h)
                self._meld_ev.reseed(
                    map_regions_to_frame(self._meld_rel, area, w, h)
                    if self._meld_rel else None, w, h)
                self._last_area = area
                self._seeded = True
            self._region_ticks = 30
        self._region_ticks -= 1
        result = self._worker.poll()
        if result is not None:
            _size, dets = result
            try:
                self._handle(dets)
            except Exception:  # noqa: BLE001 — 单帧异常不杀主循环
                now = time.time()
                if now - self._last_err_ts > 2.0:
                    traceback.print_exc()
                    self._last_err_ts = now
        self._worker.submit(frame)

    def _handle(self, dets: list[TileDet]) -> None:
        frame, ts = self._frame_idx, time.time()
        self._frame_idx += 1
        for ev in self._extractor.process(dets, frame, ts):
            self._write_log(event_to_dict(ev))
            self._decoder.add(ev)
        frozen = self._decoder.frozen()
        for at in frozen[self._frozen_count:]:
            self._write_log({'type': 'freeze', 'eid': at.eid,
                             'tile': at.tile, 'player': at.player,
                             'entropy': round(at.entropy(), 3),
                             'logits': {p: round(v, 2)
                                        for p, v in at.logits.items()},
                             'ts': time.time()})
        self._frozen_count = len(frozen)
        melds = self._decoder.melds()
        for player, ev in melds[self._meld_count:]:
            self._write_log({'type': 'meld_assign', 'eid': ev.eid,
                             'player': player, 'ts': time.time()})
        self._meld_count = len(melds)
        view = project(self._decoder, self._extractor.my_hand())
        advice, _boxes = build_advice(view, self._engine, self._rules)
        boxes = {'hand': self._extractor.hand_dets()}
        self._client.update(advice, boxes)
        if self._det_overlay is not None:
            self._det_overlay.set_dets(boxes)

    def _write_log(self, record: dict) -> None:
        self._log.write(json.dumps(record, ensure_ascii=False) + '\n')
        self._log.flush()


def _build_capture(args: argparse.Namespace) -> Win32Capture:
    from src.mahjong_cv.capture.win32 import (  # noqa: PLC0415
        DEFAULT_TITLE_CANDIDATES,
        list_window_titles,
    )

    candidates = (DEFAULT_TITLE_CANDIDATES if args.title == '欢乐麻将'
                  else (args.title,))
    cap = Win32Capture(args.title, candidates)
    if cap.client_rect() is None:
        print('[窗口] 未找到游戏窗口, 当前可见窗口:')
        for t in list_window_titles():
            print(f'   - {t}')
        print('  请先启动欢乐麻将, 或使用 --title 指定标题')
        sys.exit(1)
    return cap


def main() -> None:
    parser = argparse.ArgumentParser(description='欢乐麻将实时 AI 辅助 v4')
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--log-dir', default='data/settle_logs')
    parser.add_argument('--no-dets', action='store_true',
                        help='关闭游戏上的悬浮检测框')
    parser.add_argument('--imgsz', type=int, default=960)
    parser.add_argument('--title', default='欢乐麻将')
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f'模型不存在: {args.model}')
        sys.exit(1)
    pipe = ScreenVision(args.model)

    def recognize(frame: np.ndarray) -> list[TileDet]:
        return pipe.track(frame, imgsz=args.imgsz)

    from PySide6.QtCore import QTimer  # noqa: PLC0415
    from PySide6.QtWidgets import QApplication  # noqa: PLC0415

    app = QApplication(sys.argv)
    cap = _build_capture(args)
    client = MahjongClient()
    det_overlay = None if args.no_dets else DetOverlay(cap.client_rect)
    client.set_status('模型加载中...')
    runner = V4Runner(cap=cap, recognize=recognize, client=client,
                      det_overlay=det_overlay, log_dir=args.log_dir)

    def _on_sigint(_signum: int, _frame: object) -> None:
        print('\n正在退出...')
        app.quit()

    signal.signal(signal.SIGINT, _on_sigint)
    timer = QTimer()
    timer.timeout.connect(runner.tick)
    timer.start(66)
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_v4/test_runner_glue.py -q`
Expected: PASS(2 passed)

- [ ] **Step 5: 质量门禁**

Run: `uv run pytest tests/test_v4 -q`、`uv run ruff check src/mahjong_ai/v4 tests/test_v4 scripts/run_assistant_v4.py`、`uv run mypy src/mahjong_ai/v4 scripts/run_assistant_v4.py`
Expected: 全部通过

---

### Task 10: 复盘脚本 + 真实对局验收(M2 完成)

**Files:**
- Create: `scripts/dump_v4_log.py`
- Test: `tests/test_v4/test_dump_log.py`

**Interfaces:**
- Consumes: 事件日志格式(Task 9 产出: event_to_dict 行 + freeze 行)
- Produces: `parse_log(lines) -> dict`(按类型聚合)、`render(summary) -> str`(四家牌河时间线文本)

- [ ] **Step 1: 写失败测试**

`tests/test_v4/test_dump_log.py`:

```python
"""复盘脚本单测 — 日志解析与聚合。"""

from scripts.dump_v4_log import parse_log


_SAMPLE = [
    {'type': 'TileAppeared', 'eid': 1, 'tile': 5, 'track_id': 101,
     'cx': 700.0, 'cy': 740.0, 'conf': 0.8, 'frame': 0, 'ts': 0.0,
     'motion': None},
    {'type': 'freeze', 'eid': 1, 'tile': 5, 'player': 'my_river',
     'entropy': 0.1, 'logits': {}, 'ts': 0.1},
    {'type': 'MeldFormed', 'eid': 2, 'tiles': [8, 8, 8], 'cx': 1150.0,
     'cy': 550.0, 'bbox': [1100.0, 500.0, 1200.0, 600.0],
     'frame': 2, 'ts': 0.2},
    {'type': 'meld_assign', 'eid': 2, 'player': 'right_river', 'ts': 0.3},
]


def test_parse_log_aggregates_freezes():
    summary = parse_log(_SAMPLE)
    assert summary['rivers']['my_river'] == [5]
    assert summary['melds'] == [('right_river', 'pong', 8)]
    assert summary['n_events'] == 4
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_v4/test_dump_log.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.dump_v4_log'`

- [ ] **Step 3: 写实现**

`scripts/dump_v4_log.py`:

```python
"""v4 事件日志复盘 — 四家牌河时间线/副露/归属置信度。

用法: python scripts/dump_v4_log.py [log.jsonl]  # 默认最新一局
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mahjong_core.tile import tile_display  # noqa: E402

PLAYERS = ('my_river', 'right_river', 'top_river', 'left_river')
_NAMES = {'my_river': '我', 'right_river': '右家',
          'top_river': '对家', 'left_river': '上家'}


def parse_log(lines: list[dict]) -> dict:
    """日志行 → 聚合摘要。

    freeze 行 = 牌河事件的最终归属; meld_assign 行 = 副露归属
    (由 MeldFormed 行补全 kind/tile)。
    """
    rivers: dict[str, list[int]] = {p: [] for p in PLAYERS}
    melds: list[tuple[str, str, int]] = []
    entropies: list[float] = []
    meld_info: dict[int, tuple[str, int]] = {}
    for line in lines:
        t = line.get('type')
        if t == 'MeldFormed':
            tiles = line['tiles']
            kind = 'kong' if len(tiles) >= 4 else 'pong'
            meld_info[line['eid']] = (kind, tiles[0])
        elif t == 'meld_assign':
            if line['eid'] in meld_info:
                kind, tile = meld_info[line['eid']]
                melds.append((line['player'], kind, tile))
        elif t == 'freeze' and line.get('player') in rivers:
            rivers[line['player']].append(line['tile'])
            entropies.append(line.get('entropy', 0.0))
    return {'rivers': rivers, 'melds': melds, 'n_events': len(lines),
            'entropies': entropies}


def main() -> None:
    parser = argparse.ArgumentParser(description='v4 事件日志复盘')
    parser.add_argument('log', nargs='?', default=None,
                        help='game_v4_*.jsonl; 默认 data/settle_logs 最新')
    args = parser.parse_args()
    if args.log:
        path = Path(args.log)
    else:
        files = sorted(Path('data/settle_logs').glob('game_v4_*.jsonl'),
                       key=lambda p: p.stat().st_mtime)
        if not files:
            print('没有找到 game_v4_*.jsonl — 先跑一局 run_assistant_v4.py')
            sys.exit(1)
        path = files[-1]
    lines = [json.loads(l) for l in path.read_text(encoding='utf-8')
             .splitlines() if l.strip()]
    summary = parse_log(lines)
    print(f'== {path.name} ==')
    print(f'事件 {summary["n_events"]} 行')
    for p in PLAYERS:
        tiles = ' '.join(tile_display(t) for t in summary['rivers'][p])
        print(f'{_NAMES[p]}({len(summary["rivers"][p])}): {tiles}')
    for _p, kind, tile in summary['melds']:
        print(f'副露: {kind} {tile_display(tile)}')
    if summary['entropies']:
        avg = sum(summary['entropies']) / len(summary['entropies'])
        print(f'冻结归属平均熵: {avg:.3f}(低=可信)')


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_v4/test_dump_log.py -q`
Expected: PASS(1 passed)

- [ ] **Step 5: 质量门禁**

Run: `uv run pytest tests/test_v4 -q`、`uv run ruff check src/mahjong_ai/v4 tests/test_v4 scripts/run_assistant_v4.py scripts/dump_v4_log.py`、`uv run mypy src/mahjong_ai/v4 scripts/run_assistant_v4.py scripts/dump_v4_log.py`
Expected: 全部通过

- [ ] **Step 6: 真实对局验收(M2 完成标准, 手动)**

1. 开欢乐麻将进对局 → `uv run python scripts/run_assistant_v4.py`
2. 打一整局。验收清单:
   - [ ] 客户端正常显示: 四家牌河随打出增长, 我的手牌/推荐正常
   - [ ] 碰牌时: 被碰的牌从打出家牌河显示消失, 碰家出现副露徽章(副露徽章必须出现在碰家 — 若全在我家, 说明副露种子键映射坏了)
   - [ ] `uv run python scripts/dump_v4_log.py` 输出的四家牌河与你实际看到的一致(逐张核对)
   - [ ] 日志里 freeze 行熵值分布: 位置清晰的牌熵低(<0.5), 歧义牌熵高
   - [ ] 观察对手摸牌: 若对手手牌位置出现误报 TileAppeared(别家摸的新牌被当成打出, 终审 I3 风险), 记录 jsonl 文件名 — 修法: 落点空间似然下限过滤或牌河区域预筛
3. 若有明显归属错误, 记录该局的 `game_v4_*.jsonl` 文件名作为后续调参依据(M1 参数: 窗口 K/束宽/跳家惩罚/先验差距 — 全部在 decoder.py 顶部常量区, 用合成回归测试保驾)。

---

## Self-Review Notes

- **规格覆盖**: M1(事件模型/自动机/证据/解码/投影/合成回归)= Tasks 1-6;M2(提取器/客户端入口/运行器/日志与验收)= Tasks 7-10;M3 不在本计划(已注明)。§6 花牌分流 = Task 7 FlowerShown;§7 相位硬锚 = Task 2/4;_LEARN_MIN_MARGIN 防反馈回路 = Task 4;§8 显示/推断拆分 = Task 5 + Task 8;§9 种子降级 = Task 3/9。
- **类型一致性**: `TileAppeared.eid` 全局唯一(提取器 `_next_eid` 自增);`TileClaimed.claimed` 引用 appeared eid;`Attribution.eid == TileAppeared.eid`;`project` 读 `decoder.claimed_ids()`;`build_advice` 的 `boxes` 仅 hand(客户端回退读快照)。
- **已知简化(有意)**: 副露归属 argmax 不软(罕见+位置强证据);`_dup_table` 位置去重 30px 为经验值;副露候选池 40px 匹配可能合并相邻两组副露(旧方案同病, 真实对局观察);动作证据 motion 字段保留但 v1 不产出;**基线机制** — 门开(手牌去抖≥13 或首副露)前别家的打出永久丢失(与旧方案开局观察期同代价),门开瞬间非手牌框标记已见;提取依赖"打出=新轨迹"(ByteTrack 跨不过手牌→牌河位移, 旧管道实测据此工作)。
