# 三区身份重设计(事件推断为主)实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 副露身份从"看对三张牌"(视觉检测)改为事件推断(牌河消失 + 手牌台阶 + 行动权),视觉检测降级为填充/兜底通道;顺手修"被碰走的牌还显示在丢弃者牌河"。

**Architecture:** 四个新组件:RiverWatcher(牌河消失检测)、HandTracker(我的手牌账本 + 台阶签名)、MeldInferrer(副露推断器,替代提取器旧副露机制)、视觉通道加固(≥3 张 + 组合合法 + 5 帧确认)。下游(解码器/状态投影/客户端)接口不变,只新增 TileVanished 事件与 MeldFormed.player 可选字段。

**Tech Stack:** Python 3.11+, dataclasses, pytest, ruff, mypy strict(项目门禁)。

## Global Constraints

- 项目不使用 git(用户要求)— 无 commit 步骤,每个任务以门禁结束。
- 所有命令用 `uv run` 前缀。
- 旧管道(v1-v3:tracker/session/旧 UI 逻辑)零改动;`state/tracker.py` 只允许加一个带默认值的字段;`client.py`、`run_assistant_v4.py` 为白名单文件。
- 质量门禁:pytest 相关目录全绿、`ruff check` 干净、`mypy` strict 干净。
- 代码注释/命名风格与现有 v4 一致(中文 docstring,79 列)。
- 设计依据:`docs/superpowers/specs/2026-08-15-zone-classification-design.md`(数值以本计划为准,spec 中"手牌差窗口 2 秒"在本计划放宽为 3 秒 — 稳定纪元需要 ~0.7s×5 帧 + 副露后出牌,2s 实测偏紧,见 Task 3)。

---

### Task 1: TileVanished 事件 + MeldFormed.player 字段

**Files:**
- Modify: `src/mahjong_ai/v4/events.py`
- Test: `tests/test_v4/test_events.py`(创建)

**Interfaces:**
- Consumes: 无。
- Produces:
  - `TileVanished(eid: int, tile: int, river: str, appeared_eid: int, frame: int, ts: float)` — river ∈ PLAYERS,appeared_eid = 被拿走那张牌的 TileAppeared eid。
  - `MeldFormed(..., player: str | None = None)` — 新可选字段(事件推断出的副露者;None = 未知,解码器走空间软归属)。
  - `Event` union 加入 `TileVanished`;`event_to_dict` 经 asdict 自动覆盖。

- [ ] **Step 1: 写失败测试**

`tests/test_v4/test_events.py`:

```python
"""事件数据类 — 序列化与字段契约。"""

from src.mahjong_ai.v4.events import (
    MeldFormed,
    TileVanished,
    event_to_dict,
)


def test_tile_vanished_serializes():
    ev = TileVanished(eid=7, tile=8, river='right_river',
                      appeared_eid=3, frame=101, ts=12.5)
    d = event_to_dict(ev)
    assert d['type'] == 'TileVanished'
    assert d['eid'] == 7 and d['tile'] == 8
    assert d['river'] == 'right_river'
    assert d['appeared_eid'] == 3


def test_meld_formed_player_defaults_none():
    ev = MeldFormed(eid=1, tiles=(8, 8, 8), cx=0.0, cy=0.0,
                    bbox=(0.0, 0.0, 0.0, 0.0), frame=1, ts=0.1)
    assert ev.player is None
    d = event_to_dict(ev)
    assert d['player'] is None


def test_meld_formed_player_carried():
    ev = MeldFormed(eid=1, tiles=(8, 8, 8), cx=0.0, cy=0.0,
                    bbox=(0.0, 0.0, 0.0, 0.0), frame=1, ts=0.1,
                    player='my_river')
    assert event_to_dict(ev)['player'] == 'my_river'
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_v4/test_events.py -v`
Expected: FAIL(ImportError: TileVanished 不存在)

- [ ] **Step 3: 实现**

在 `src/mahjong_ai/v4/events.py`:

在 `TileClaimed` 之后加:

```python
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
```

`MeldFormed` 加字段:

```python
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
```

`Event` union:

```python
Event = (TileAppeared | MeldFormed | TileClaimed | TileVanished
         | HandChanged | FlowerShown)
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_v4/test_events.py -v`
Expected: 3 passed

- [ ] **Step 5: 门禁**

Run: `uv run pytest tests/test_v4 -q && uv run ruff check src/mahjong_ai/v4/events.py tests/test_v4/test_events.py && uv run mypy src/mahjong_ai/v4/events.py`
Expected: 全绿(既有 test_v4 全过)

---

### Task 2: RiverWatcher — 牌河消失检测器

**Files:**
- Create: `src/mahjong_ai/v4/river_watcher.py`
- Test: `tests/test_v4/test_river_watcher.py`(创建)

**Interfaces:**
- Consumes: `TileVanished`(Task 1)、`TileDet`(src/mahjong_cv/detections)。
- Produces:
  - `RiverWatcher.set_regions(regions: dict[str, tuple[float, float, float, float]]) -> None`
  - `RiverWatcher.on_appeared(river: str, eid: int, tile: int, cx: float, cy: float) -> None` — 该河最新落定牌(提取器在 TileAppeared 时调用)。
  - `RiverWatcher.tick(dets: list[TileDet], frame: int, ts: float, next_eid: Callable[[], int]) -> list[TileVanished]` — 与提取器共享 eid 计数器(全局唯一)。

- [ ] **Step 1: 写失败测试**

`tests/test_v4/test_river_watcher.py`:

```python
"""牌河消失检测 — 最新落定牌连续缺失 → TileVanished(被碰/吃)。"""

from src.mahjong_ai.v4.river_watcher import RiverWatcher
from src.mahjong_cv.detections import TileDet

_REGION = {'right_river': (900.0, 300.0, 1200.0, 400.0)}


def _mk(tile, x1, y1, x2, y2, conf=0.8, tid=None) -> TileDet:
    return TileDet(tile=tile, x1=x1, y1=y1, x2=x2, y2=y2, conf=conf,
                   track_id=tid)


def _eids() -> list[int]:
    buf = [0]

    def nxt() -> int:
        buf[0] += 1
        return buf[0]
    return buf and []  # 占位, 下一行替换


def test_no_vanish_while_present():
    w = RiverWatcher()
    w.set_regions(_REGION)
    w.on_appeared('right_river', 3, 8, 950.0, 350.0)
    dets = [_mk(5, 1000.0, 310.0, 1030.0, 360.0, tid=2),
            _mk(8, 925.0, 320.0, 975.0, 370.0, tid=3)]
    events = []
    for i in range(30):
        events += w.tick(dets, frame=i, ts=i * 0.1, next_eid=lambda: 99)
    assert events == []


def test_vanish_after_sustained_absence():
    w = RiverWatcher()
    w.set_regions(_REGION)
    w.on_appeared('right_river', 3, 8, 950.0, 350.0)
    dets = [_mk(5, 1000.0, 310.0, 1030.0, 360.0, tid=2),
            _mk(8, 925.0, 320.0, 975.0, 370.0, tid=3)]
    w.tick(dets, frame=0, ts=0.0, next_eid=lambda: 99)
    # 之后 20 帧: 8 消失, 但同河 5 还在(河活着)
    dets_without = [_mk(5, 1000.0, 310.0, 1030.0, 360.0, tid=2)]
    events = [w.tick(dets_without, frame=i, ts=i * 0.1,
                     next_eid=lambda: 99) for i in range(1, 21)]
    fired = [e for sub in events for e in sub]
    assert len(fired) == 1
    assert fired[0].tile == 8
    assert fired[0].river == 'right_river'
    assert fired[0].appeared_eid == 3


def test_brief_occlusion_no_vanish():
    """飞牌遮挡 2 帧不算消失(阈值 15 帧吸收)。"""
    w = RiverWatcher()
    w.set_regions(_REGION)
    w.on_appeared('right_river', 3, 8, 950.0, 350.0)
    w.tick([_mk(5, 1000.0, 310.0, 1030.0, 360.0, tid=2),
            _mk(8, 925.0, 320.0, 975.0, 370.0, tid=3)],
           frame=0, ts=0.0, next_eid=lambda: 99)
    events = []
    for i in range(1, 40):
        dets = ([_mk(5, 1000.0, 310.0, 1030.0, 360.0, tid=2)]
                if i % 10 >= 2 else
                [_mk(5, 1000.0, 310.0, 1030.0, 360.0, tid=2),
                 _mk(8, 925.0, 320.0, 975.0, 370.0, tid=3)])
        events += w.tick(dets, frame=i, ts=i * 0.1, next_eid=lambda: 99)
    assert events == []   # 每 10 帧只缺 2 帧, 从未连续 15 帧


def test_whole_river_collapse_no_vanish():
    """整河检测塌(连 5 都检测不到)不算消失 — 挡住抽风假消失。"""
    w = RiverWatcher()
    w.set_regions(_REGION)
    w.on_appeared('right_river', 3, 8, 950.0, 350.0)
    w.tick([_mk(5, 1000.0, 310.0, 1030.0, 360.0, tid=2),
            _mk(8, 925.0, 320.0, 975.0, 370.0, tid=3)],
           frame=0, ts=0.0, next_eid=lambda: 99)
    events = []
    for i in range(1, 40):
        events += w.tick([], frame=i, ts=i * 0.1, next_eid=lambda: 99)
    assert events == []


def test_newest_replaced_by_later_appear():
    """新打出落定后, 盯防对象换成那张; 旧牌消失不再触发。"""
    w = RiverWatcher()
    w.set_regions(_REGION)
    w.on_appeared('right_river', 3, 8, 950.0, 350.0)
    w.tick([_mk(8, 925.0, 320.0, 975.0, 370.0, tid=3)],
           frame=0, ts=0.0, next_eid=lambda: 99)
    w.on_appeared('right_river', 4, 2, 1050.0, 350.0)
    events = []
    for i in range(1, 30):
        # 旧牌 8 消失, 新牌 2 在场
        events += w.tick([_mk(2, 1025.0, 320.0, 1075.0, 370.0, tid=4)],
                         frame=i, ts=i * 0.1, next_eid=lambda: 99)
    assert events == []
```

(Step 1 中 `_eids` 占位函数是笔误 — 直接删掉,不写进文件。)

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_v4/test_river_watcher.py -v`
Expected: FAIL(ModuleNotFoundError: river_watcher)

- [ ] **Step 3: 实现**

`src/mahjong_ai/v4/river_watcher.py`:

```python
"""牌河消失检测 — 各河最新落定牌连续缺失 → TileVanished(被碰/吃)。

只盯每河最近一次落定的 TileAppeared: 游戏里唯一会让它消失的
机制是被碰/吃拿走(该河最新一张, 不可能被别家覆盖)。整河检测
塌(抽风)不算消失 — "同河其他牌还在"才计数, 挡住假消失。
"""

from __future__ import annotations

from typing import Callable

from src.mahjong_ai.v4.events import TileVanished
from src.mahjong_cv.detections import TileDet

#: 连续缺失帧数(≈2s @7fps — 被碰动画 1s + 检测抖动余量)
_VANISH_FRAMES = 15
#: 在场判定半径(像素 — 落定牌位置抖动 ±5px, 60 足够宽)
_PRESENT_RADIUS = 60.0
#: 区域判定外扩(与提取器区域门禁同口径)
_REGION_MARGIN = 20.0


class RiverWatcher:
    """逐河盯最新牌; 缺失 ≥_VANISH_FRAMES 且河还活着 → TileVanished。"""

    def __init__(self) -> None:
        self._regions: dict[str, tuple[float, float, float, float]] = {}
        #: river -> {'eid', 'tile', 'cx', 'cy', 'missing'}
        self._newest: dict[str, dict[str, float | int]] = {}

    def set_regions(
        self, regions: dict[str, tuple[float, float, float, float]],
    ) -> None:
        """注入牌河区域(像素, 运行器重映射后随 set_regions 一起调用)。"""
        self._regions = dict(regions)
        self._newest = {k: v for k, v in self._newest.items()
                        if k in self._regions}

    def on_appeared(self, river: str, eid: int, tile: int,
                    cx: float, cy: float) -> None:
        """该河最新落定牌(提取器在 TileAppeared 时调用, 替换盯防对象)。"""
        self._newest[river] = {'eid': eid, 'tile': tile, 'cx': cx,
                               'cy': cy, 'missing': 0}

    def tick(self, dets: list[TileDet], frame: int, ts: float,
             next_eid: Callable[[], int]) -> list[TileVanished]:
        """逐帧: 在场清零/缺失累计; 达到阈值且同河其他牌在场 → 消失。"""
        out: list[TileVanished] = []
        for river, nw in list(self._newest.items()):
            tile = int(nw['tile'])
            cx, cy = float(nw['cx']), float(nw['cy'])
            region = self._regions[river]
            present = any(
                d.tile == tile
                and ((d.x1 + d.x2) / 2 - cx) ** 2
                + ((d.y1 + d.y2) / 2 - cy) ** 2 < _PRESENT_RADIUS ** 2
                for d in dets)
            if present:
                nw['missing'] = 0
                continue
            alive = any(
                rx1 - _REGION_MARGIN <= (d.x1 + d.x2) / 2
                <= rx2 + _REGION_MARGIN
                and ry1 - _REGION_MARGIN <= (d.y1 + d.y2) / 2
                <= ry2 + _REGION_MARGIN
                for d in dets
                for rx1, ry1, rx2, ry2 in (region,))
            if not alive:
                continue  # 整河塌: 不计数(抽风假消失)
            nw['missing'] = int(nw['missing']) + 1
            if nw['missing'] >= _VANISH_FRAMES:
                out.append(TileVanished(
                    eid=next_eid(), tile=tile, river=river,
                    appeared_eid=int(nw['eid']), frame=frame, ts=ts))
                del self._newest[river]
        return out
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_v4/test_river_watcher.py -v`
Expected: 5 passed

- [ ] **Step 5: 门禁**

Run: `uv run pytest tests/test_v4 -q && uv run ruff check src/mahjong_ai/v4/river_watcher.py tests/test_v4/test_river_watcher.py && uv run mypy src/mahjong_ai/v4/river_watcher.py`
Expected: 全绿

---

### Task 3: HandTracker — 手牌账本 + 台阶签名

**Files:**
- Create: `src/mahjong_ai/v4/hand_tracker.py`
- Test: `tests/test_v4/test_hand_tracker.py`(创建)

**Interfaces:**
- Consumes: 无(纯 Python)。
- Produces:
  - `StepSig(drop: int, ts: float)` — drop ∈ {2, 3}(2=碰/吃, 3=明杠)。
  - `HandTracker.tick(hand: list[int], ts: float) -> StepSig | None`
  - `HandTracker.on_my_discard(tile: int, ts: float) -> None`
  - `HandTracker.on_my_meld(delta: list[int], drop: int) -> None`
  - `HandTracker.hand_delta(ts: float, window: float = 3.0) -> list[int] | None` — 副露内容(前后稳定手牌差,剔除窗口内我的打出);任一侧缺失 → None。
  - `HandTracker.fallback_hand(ts: float) -> list[int | None] | None` — 塌陷兜底(未知槽 = None);检测健康(近 2s 有稳定快照)→ None。

- [ ] **Step 1: 写失败测试**

`tests/test_v4/test_hand_tracker.py`:

```python
"""手牌账本 + 台阶签名 — 塌陷兜底与副露内容推断。"""

from src.mahjong_ai.v4.hand_tracker import HandTracker, StepSig


def _stable(tracker: HandTracker, hand: list[int], t0: float,
            ticks: int = 5, dt: float = 0.1) -> float:
    """喂一个稳定纪元(计数连续不变 ticks 帧), 返回结束时刻。"""
    t = t0
    for _ in range(ticks):
        t += dt
        tracker.tick(hand, t)
    return t


def test_stable_epoch_builds_snapshot_and_fallback_none():
    tr = HandTracker()
    t = _stable(tr, list(range(13)), 0.0)
    assert tr.fallback_hand(t) is None  # 健康: 不需要兜底


def test_step_signature_on_drop_two():
    tr = HandTracker()
    t = _stable(tr, list(range(13)), 0.0)
    out = None
    for _ in range(50):  # 手牌 13→11(副露), 保持 11
        t += 0.1
        out = tr.tick(list(range(11)), t) or out
    assert out is not None
    assert out.drop == 2


def test_oscillation_no_step():
    """抽风振荡(13→7→4→10…)不构成台阶。"""
    tr = HandTracker()
    _stable(tr, list(range(13)), 0.0)
    seq = [7, 4, 10, 5, 0, 3, 2, 5, 4, 4, 4, 3, 0]
    t = 5.0
    steps = []
    for n in seq:
        t += 0.15
        s = tr.tick(list(range(max(n, 1))), t)
        if s:
            steps.append(s)
    assert steps == []


def test_fallback_hand_deductions():
    """塌陷兜底: 最后已知 − 我的打出 − 副露扣除, 未知槽 = None。"""
    tr = HandTracker()
    t = _stable(tr, list(range(13)), 0.0)
    # 塌陷后我打出 5, 副露扣除(内容未知, drop=2)
    tr.on_my_discard(5, t + 0.5)
    tr.on_my_meld([], 2)
    fb = tr.fallback_hand(t + 3.0)
    assert fb is not None
    assert 5 not in fb
    known = [x for x in fb if x is not None]
    assert len(known) == 10        # 13 − 1 − 2
    assert fb.count(None) == 0     # 未知槽 = 稳定计数 − 已知
    assert len(fb) == 10


def test_hand_delta_gives_meld_tiles():
    """副露内容 = 前后稳定手牌差, 剔除窗口内打出的牌。"""
    tr = HandTracker()
    hand13 = list(range(13))
    t = _stable(tr, hand13, 0.0)
    # 持续稳定期快照刷新(每 2s 重记)
    t = _stable(tr, hand13, t, ticks=20)
    vanish_ts = t + 0.5
    # 副露: 手牌 13 → 11(打出 8,8 进副露) → 稳定
    t = _stable(tr, [x for x in hand13 if x != 8], vanish_ts + 0.3)
    delta = tr.hand_delta(vanish_ts)
    assert delta == [8, 8]


def test_hand_delta_none_when_no_snapshot():
    tr = HandTracker()
    assert tr.hand_delta(1.0) is None
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_v4/test_hand_tracker.py -v`
Expected: FAIL(ModuleNotFoundError: hand_tracker)

- [ ] **Step 3: 实现**

`src/mahjong_ai/v4/hand_tracker.py`:

```python
"""我的手牌账本(塌陷兜底) + 副露台阶签名。

检测健康时手牌 = 检测(现行); 检测塌陷时用事件账本兜底:
最后已知手牌 − 我的打出 − 副露扣除, 摸进的新牌 = 未知槽。
台阶签名 = 稳定计数纪元掉 2(碰/吃)/3(明杠)且新纪元保持 ≥2 秒 —
抽风是振荡不落脚, 不构成台阶。手牌差 = 前后稳定手牌差(剔除
窗口内我打出的牌), 供 MeldInferrer 推断我的副露内容。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

#: 稳定纪元: 计数连续不变 ≥ 此帧数(≈0.7s @7fps)
_STABLE_TICKS = 5
#: 台阶保持: 新纪元持续 ≥ 此秒数才确认副露签名
_STEP_HOLD_S = 2.0
#: 快照/历史保留窗口(秒)
_SNAP_WINDOW = 10.0
#: 稳定期快照刷新间隔(秒 — 13 张手牌可能稳定几十秒, 快照必须新鲜)
_SNAP_REFRESH_S = 2.0
#: 台阶触发前纪元最小计数(排除发牌斜坡 10→13 等; 与消失配对后
#: 假台阶自然被过滤, 此阈值只省事)
_STEP_MIN_PREV = 8
#: 兜底判定: 近此秒数内有稳定快照 = 检测健康(兜底返回 None)
_HEALTHY_S = 2.0


@dataclass(frozen=True)
class StepSig:
    """副露台阶签名: 稳定计数掉 drop(2=碰/吃, 3=明杠), ts=台阶时刻。"""

    drop: int
    ts: float


class HandTracker:
    """手牌账本 + 台阶签名(纯 Python, 每帧 tick)。"""

    def __init__(self) -> None:
        self._last_good: Counter[int] = Counter()
        self._unknown = 0
        self._snapshots: list[tuple[float, Counter[int]]] = []
        self._discarded: list[tuple[int, float]] = []
        self._epoch_count: int | None = None
        self._epoch_ticks = 0
        self._pending: StepSig | None = None

    def tick(self, hand: list[int], ts: float) -> StepSig | None:
        """每帧: 纪元/台阶检测 + 稳定快照更新(可能返回台阶)。"""
        n = len(hand)
        if n != self._epoch_count:
            if (self._epoch_count is not None
                    and self._epoch_count >= _STEP_MIN_PREV
                    and self._epoch_count - n in (2, 3)):
                self._pending = StepSig(drop=self._epoch_count - n, ts=ts)
            self._epoch_count = n
            self._epoch_ticks = 0
        self._epoch_ticks += 1
        out = None
        if (self._pending is not None
                and ts - self._pending.ts >= _STEP_HOLD_S):
            out = self._pending
            self._pending = None
        if self._epoch_ticks == _STABLE_TICKS and hand:
            self._record(ts, hand)
        elif (self._epoch_ticks > _STABLE_TICKS and hand
              and self._snapshots
              and ts - self._snapshots[-1][0] >= _SNAP_REFRESH_S):
            self._record(ts, hand)  # 长稳定期快照刷新(手牌差要新鲜)
        return out

    def _record(self, ts: float, hand: list[int]) -> None:
        """稳定快照: 账本重建 + 快照历史(裁剪保留窗口)。"""
        self._last_good = Counter(hand)
        self._unknown = 0
        self._snapshots.append((ts, Counter(hand)))
        while self._snapshots and \
                ts - self._snapshots[0][0] > _SNAP_WINDOW:
            self._snapshots.pop(0)

    def on_my_discard(self, tile: int, ts: float) -> None:
        """我打出一张: 账本扣牌(已知扣已知, 未知扣未知槽)。"""
        self._discarded.append((tile, ts))
        while self._discarded and \
                ts - self._discarded[0][1] > _SNAP_WINDOW:
            self._discarded.pop(0)
        if self._last_good.get(tile, 0) > 0:
            self._last_good[tile] -= 1
            if self._last_good[tile] <= 0:
                del self._last_good[tile]
        else:
            self._unknown = max(0, self._unknown - 1)

    def on_my_meld(self, delta: list[int], drop: int) -> None:
        """我的副露扣除(塌陷兜底账本)。账本已被后续稳定检测重建时
        扣除已隐含 — 校验账本仍含 delta 才扣(防双扣)。"""
        if delta and all(self._last_good.get(t, 0) >= delta.count(t)
                         for t in set(delta)):
            for t in delta:
                self._last_good[t] -= 1
                if self._last_good[t] <= 0:
                    del self._last_good[t]
        elif not delta:
            self._unknown = max(0, self._unknown - drop)

    def hand_delta(self, ts: float, window: float = 3.0) -> list[int] | None:
        """ts 前后的稳定手牌差(净副露牌): 前 = ts 前最后快照,
        后 = [ts−1.5, ts+window] 内首张快照(副露后手牌在消失事件
        之前就稳定 — 消失检测滞后 ≈2s, 所以后窗口前伸 1.5s);
        差值剔除 [ts−window, ts+window] 内我打出的牌。任一缺失
        (检测塌陷)→ None(内容降级未知)。
        """
        before = after = None
        for t, c in reversed(self._snapshots):
            if t <= ts:
                before = c
                break
        for t, c in self._snapshots:
            if ts - 1.5 <= t <= ts + window:
                after = c
                break
        if before is None or after is None:
            return None
        delta = Counter(before)
        for t, c in after.items():
            delta[t] -= c
        for t, dts in self._discarded:
            if ts - window <= dts <= ts + window and delta.get(t, 0) > 0:
                delta[t] -= 1
        out: list[int] = []
        for t, k in delta.items():
            if k > 0:
                out.extend([t] * k)
        return sorted(out)

    def fallback_hand(self, ts: float) -> list[int | None] | None:
        """塌陷兜底: 最后已知 + 未知槽(None); 检测健康 → None。"""
        if self._snapshots and ts - self._snapshots[-1][0] < _HEALTHY_S:
            return None
        return sorted(self._last_good.elements()) + [None] * self._unknown
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_v4/test_hand_tracker.py -v`
Expected: 6 passed

- [ ] **Step 5: 门禁**

Run: `uv run pytest tests/test_v4 -q && uv run ruff check src/mahjong_ai/v4/hand_tracker.py tests/test_v4/test_hand_tracker.py && uv run mypy src/mahjong_ai/v4/hand_tracker.py`
Expected: 全绿

---

### Task 4: MeldInferrer — 副露推断器

**Files:**
- Create: `src/mahjong_ai/v4/meld_inferrer.py`
- Test: `tests/test_v4/test_meld_inferrer.py`(创建)

**Interfaces:**
- Consumes: `TileVanished`(Task 1)、`StepSig`/`HandTracker`(Task 3)。
- Produces:
  - `meld_valid(tiles: tuple[int, ...]) -> bool` — 碰=同牌×3、吃=同花连号(≤26)、杠=同牌×4;2 张/字牌连号/异花一律 False。
  - `meld_kind(tiles: tuple[int, ...]) -> str` — 'pong'|'chi'|'kong'。
  - `VisualMeld(player: str | None, tiles: tuple[int, ...], cx: float, cy: float, bbox: tuple[float, float, float, float])` — 视觉通道产出的合法组。
  - `MeldInferrer(next_eid: Callable[[], int])`;`set_meld_regions(regions) -> None`;
  - `MeldInferrer.tick(vanished: list[TileVanished], steps: list[StepSig], discards: list[tuple[str | None, int, float, int]], visual: list[VisualMeld], tracker: HandTracker, frame: int, ts: float) -> list[MeldFormed | TileClaimed]`
    — discards 元组 = (player, tile, ts, appeared_eid)(提取器近 10s 打出缓冲)。

- [ ] **Step 1: 写失败测试**

`tests/test_v4/test_meld_inferrer.py`:

```python
"""副露推断 — 消失 + 台阶/行动权 + 视觉填充, 不依赖看对三张牌。"""

from src.mahjong_ai.v4.events import MeldFormed, TileClaimed, TileVanished
from src.mahjong_ai.v4.hand_tracker import HandTracker
from src.mahjong_ai.v4.meld_inferrer import (
    MeldInferrer,
    VisualMeld,
    meld_kind,
    meld_valid,
)


def _eid() -> list[int]:
    buf = [100]

    def nxt() -> int:
        buf[0] += 1
        return buf[0]
    return nxt


def _vanish(tile=8, eid=3, ts=10.0) -> TileVanished:
    return TileVanished(eid=eid, tile=tile, river='right_river',
                        appeared_eid=5, frame=50, ts=ts)


def test_meld_valid_rules():
    assert meld_valid((8, 8, 8))            # 碰
    assert meld_valid((7, 8, 9))            # 吃
    assert meld_valid((0, 1, 2))            # 一万二万三万
    assert meld_valid((8, 8, 8, 8))         # 杠
    assert not meld_valid((8, 8))           # 2 张组(动画中)
    assert not meld_valid((27, 28, 29))     # 字牌不吃
    assert not meld_valid((8, 9, 10))       # 跨花色(8=九万, 9=一饼)
    assert not meld_valid((7, 8, 8))        # 非同牌非连号
    assert meld_kind((8, 8, 8, 8)) == 'kong'
    assert meld_kind((8, 8, 8)) == 'pong'
    assert meld_kind((7, 8, 9)) == 'chi'


def test_my_meld_from_step_and_delta():
    inf = MeldInferrer(_eid())
    tr = HandTracker()
    hand13 = list(range(13))
    t = 0.0
    for _ in range(5):
        t += 0.1
        tr.tick(hand13, t)
    v = _vanish(ts=t + 0.5)
    t += 1.0
    tr.tick(list(range(11)), t)  # 手牌 13→11(副露 8,8)
    for _ in range(4):
        t += 0.1
        tr.tick(list(range(11)), t)
    # 台阶(掉 2) + 消失 → 我的副露, 内容 = 8 + 手牌差(8,8)
    steps = []
    for _ in range(30):
        t += 0.1
        s = tr.tick(list(range(11)), t)
        if s:
            steps.append(s)
    events = inf.tick([v], steps, [], [], tr, frame=60, ts=t)
    melds = [e for e in events if isinstance(e, MeldFormed)]
    assert len(melds) == 1
    assert melds[0].tiles == (8, 8, 8)
    assert melds[0].player == 'my_river'
    claims = [e for e in events if isinstance(e, TileClaimed)]
    assert claims and claims[0].claimed == v.appeared_eid


def test_opponent_meld_from_first_discard():
    inf = MeldInferrer(_eid())
    tr = HandTracker()
    v = _vanish(ts=10.0)
    # 消失后第一家出牌 = right_river(行动权在副露者)
    discards = [('my_river', 3, 9.5, 30),      # 窗口外(碰前)
                ('right_river', 5, 11.0, 31),  # 第一家 → 副露者
                ('top_river', 6, 13.0, 32)]
    events = inf.tick([v], [], discards, [], tr, frame=60, ts=14.0)
    melds = [e for e in events if isinstance(e, MeldFormed)]
    assert len(melds) == 1
    assert melds[0].player == 'right_river'
    assert melds[0].tiles == (8,)          # 内容未知 → 只有被碰牌


def test_claimed_tile_itself_not_melder():
    """被碰那张的打出事件不能把丢弃者当成副露者。"""
    inf = MeldInferrer(_eid())
    tr = HandTracker()
    v = _vanish(ts=10.0)                    # appeared_eid=5
    discards = [('top_river', 8, 8.5, 5),   # 被碰牌自己的事件
                ('right_river', 5, 11.0, 31)]
    events = inf.tick([v], [], discards, [], tr, frame=60, ts=14.0)
    melds = [e for e in events if isinstance(e, MeldFormed)]
    assert len(melds) == 1
    assert melds[0].player == 'right_river'


def test_visual_fills_opponent_content():
    """视觉合法组(含被碰牌)在候选期内 → 填充对手副露内容。"""
    inf = MeldInferrer(_eid())
    tr = HandTracker()
    v = _vanish(ts=10.0)
    g = VisualMeld(player='right_river', tiles=(7, 8, 9),
                   cx=950.0, cy=400.0, bbox=(900.0, 360.0, 1000.0, 440.0))
    discards = [('right_river', 5, 11.0, 31)]
    events = inf.tick([v], [], discards, [g], tr, frame=60, ts=14.0)
    melds = [e for e in events if isinstance(e, MeldFormed)]
    assert melds[0].tiles == (7, 8, 9)      # 吃(视觉填充)
    assert melds[0].player == 'right_river'


def test_timeout_drops_candidate():
    """超时(5s)+宽限(3s)无证据 → 丢弃, 不发假事件。"""
    inf = MeldInferrer(_eid())
    tr = HandTracker()
    v = _vanish(ts=10.0)
    events = inf.tick([v], [], [], [], tr, frame=100, ts=10.5)
    assert events == []
    events = inf.tick([], [], [], [], tr, frame=200, ts=20.0)
    assert events == []


def test_visual_fallback_without_vanish():
    """消失信号漏掉: 合法视觉组兜底确认, 被碰牌用最近同类打出配对。"""
    inf = MeldInferrer(_eid())
    tr = HandTracker()
    g = VisualMeld(player='left_river', tiles=(8, 8, 8),
                   cx=500.0, cy=400.0, bbox=(450.0, 360.0, 550.0, 440.0))
    discards = [('top_river', 8, 9.0, 41)]  # 最近同类打出
    events = inf.tick([], [], discards, [g], tr, frame=60, ts=10.0)
    melds = [e for e in events if isinstance(e, MeldFormed)]
    claims = [e for e in events if isinstance(e, TileClaimed)]
    assert len(melds) == 1
    assert melds[0].player == 'left_river'
    assert claims[0].claimed == 41
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_v4/test_meld_inferrer.py -v`
Expected: FAIL(ModuleNotFoundError: meld_inferrer)

- [ ] **Step 3: 实现**

`src/mahjong_ai/v4/meld_inferrer.py`:

```python
"""副露推断 — 事件信号组合, 不依赖"看到三张牌"。

证据链:
  我的副露 = 牌河消失(X) + 手牌台阶(-2/-3); 内容 = X + 手牌前后差
  对手副露 = 牌河消失(X) + 消失后第一家出牌(行动权在副露者);
            内容 = X + 2 未知
  视觉合法组(≥3 张 + 组合合法 + 持续确认)只做内容填充与兜底。
确认超时(5s)→ 等视觉兜底 → 再 3s 无 → 丢弃。宁可缺事件, 不发假事件。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from src.mahjong_ai.v4.events import MeldFormed, TileClaimed, TileVanished
from src.mahjong_ai.v4.hand_tracker import HandTracker, StepSig

#: 台阶与消失的时间窗(秒)
_STEP_WINDOW = 2.0
#: 消失后第一家出牌窗口: [vanish−2(≈碰时刻, 消失检测滞后 2s), vanish+5]
_DISCARD_LO = 2.0
_DISCARD_HI = 5.0
#: 确认超时(秒): 无证据 → 等视觉兜底; 超时+宽限 → 丢弃
_CONFIRM_TIMEOUT = 5.0
_VISUAL_GRACE = 3.0
#: 我的副露等手牌差的最长宽限(秒) — 塌陷时差算不出, 按未知发射
_DELTA_WAIT = 2.0
#: 兜底配对(视觉副露的被碰牌): 最近同类打出的最大年龄(秒)
_CLAIM_HIST_MAX_AGE = 10.0


def meld_valid(tiles: tuple[int, ...]) -> bool:
    """副露组合合法: 碰=同牌×3, 吃=同花连号(数牌 0-26), 杠=同牌×4。"""
    if len(tiles) == 3:
        if len(set(tiles)) == 1:
            return True
        s = sorted(tiles)
        return (s[2] <= 26 and s[0] // 9 == s[2] // 9
                and s[1] == s[0] + 1 and s[2] == s[0] + 2)
    return len(tiles) == 4 and len(set(tiles)) == 1


def meld_kind(tiles: tuple[int, ...]) -> str:
    """副露 kind(pong/chi/kong) — 显示用。"""
    if len(tiles) >= 4:
        return 'kong'
    if len(set(tiles)) == 1:
        return 'pong'
    return 'chi'


@dataclass(frozen=True)
class VisualMeld:
    """视觉通道产出的合法副露组(已持续确认; 提取器生产)。"""

    player: str | None      # 组中心所在副露区域的归属; 无区域 → None
    tiles: tuple[int, ...]
    cx: float
    cy: float
    bbox: tuple[float, float, float, float]


@dataclass
class _Cand:
    """候选: 一个消失信号 + 逐步补齐的证据。"""

    v: TileVanished
    melder: str | None = None
    content: tuple[int, ...] | None = None


class MeldInferrer:
    """消失/台阶/出牌 → 副露事件。eid 与提取器共享(_next_eid)。"""

    def __init__(self, next_eid: Callable[[], int]) -> None:
        self._next_eid = next_eid
        self._cands: list[_Cand] = []
        self._used_steps: set[tuple[int, float]] = set()
        self._meld_regions: dict[str, tuple[float, float, float, float]] = {}

    def set_meld_regions(
        self, regions: dict[str, tuple[float, float, float, float]],
    ) -> None:
        """默认副露位置(无视觉组时事件 bbox 用该家区域中心)。"""
        self._meld_regions = dict(regions)

    def tick(
        self,
        vanished: list[TileVanished],
        steps: list[StepSig],
        discards: list[tuple[str | None, int, float, int]],
        visual: list[VisualMeld],
        tracker: HandTracker,
        frame: int,
        ts: float,
    ) -> list[MeldFormed | TileClaimed]:
        """每帧信号组合 → 本帧新副露事件。"""
        events: list[MeldFormed | TileClaimed] = []
        for v in vanished:
            self._cands.append(_Cand(v=v))
        # 视觉组分配: 含消失牌的组 → 填充该候选; 剩余 → 兜底确认
        unmatched: list[VisualMeld] = []
        for g in visual:
            target = next((c for c in self._cands
                           if c.v.tile in g.tiles), None)
            if target is not None:
                target.content = g.tiles
                target.melder = target.melder or g.player
            else:
                unmatched.append(g)
        for g in unmatched:
            events += self._fallback_meld(g, discards, frame, ts)
        # 台阶配对(我的副露)
        for c in self._cands:
            if c.melder is not None:
                continue
            for s in steps:
                if (s.drop, s.ts) in self._used_steps:
                    continue
                if abs(s.ts - c.v.ts) <= _STEP_WINDOW:
                    c.melder = 'my_river'
                    self._used_steps.add((s.drop, s.ts))
                    break
        # 出牌配对(对手副露 — 消失后第一家出牌)
        for c in self._cands:
            if c.melder is not None:
                continue
            for player, _tile, dts, eid in discards:
                if eid == c.v.appeared_eid:
                    continue  # 被碰那张自己的事件 → 丢弃者不是副露者
                if c.v.ts - _DISCARD_LO <= dts <= c.v.ts + _DISCARD_HI:
                    c.melder = player
                    break
        # 内容补全(我的: 手牌差)
        for c in self._cands:
            if c.melder == 'my_river' and c.content is None:
                delta = tracker.hand_delta(c.v.ts)
                if delta is not None:
                    c.content = (c.v.tile,) + tuple(delta)
        # 确认发射
        for c in list(self._cands):
            if c.melder is None:
                continue
            if c.content is None:
                if c.melder == 'my_river' and ts - c.v.ts < _DELTA_WAIT:
                    continue  # 等手牌差; 超宽限按未知发射
                c.content = (c.v.tile,)
            events += self._emit(c, frame, ts)
            self._cands.remove(c)
        # 超时清理(不发假事件)
        for c in list(self._cands):
            if ts - c.v.ts > _CONFIRM_TIMEOUT + _VISUAL_GRACE:
                self._cands.remove(c)
        return events

    def _emit(self, c: _Cand, frame: int, ts: float,
              ) -> list[MeldFormed | TileClaimed]:
        """候选确认 → MeldFormed + TileClaimed(位置默认该家区域中心)。"""
        content = c.content or (c.v.tile,)
        region = self._meld_regions.get(c.melder or '')
        if region is not None:
            rx1, ry1, rx2, ry2 = region
            cx, cy = (rx1 + rx2) / 2, (ry1 + ry2) / 2
            bbox = region
        else:
            cx, cy = c.v.cx, c.v.cy
            bbox = (c.v.cx, c.v.cy, c.v.cx, c.v.cy)
        ev = MeldFormed(eid=self._next_eid(), tiles=content,
                        cx=cx, cy=cy, bbox=bbox, frame=frame, ts=ts,
                        player=c.melder)
        return [ev, TileClaimed(eid=self._next_eid(),
                                claimed=c.v.appeared_eid, meld=ev.eid,
                                frame=frame, ts=ts)]

    def _fallback_meld(self, g: VisualMeld,
                       discards: list[tuple[str | None, int, float, int]],
                       frame: int, ts: float,
                       ) -> list[MeldFormed | TileClaimed]:
        """无消失证据的合法视觉组: 兜底确认(消失信号漏掉的副露)。

        被碰牌 = 最近同类打出(规则: 可被碰的只能是最近一次打出)。
        """
        claimed = None
        for _p, tile, dts, eid in reversed(discards):
            if tile in g.tiles and ts - dts <= _CLAIM_HIST_MAX_AGE:
                claimed = eid
                break
        ev = MeldFormed(eid=self._next_eid(), tiles=g.tiles,
                        cx=g.cx, cy=g.cy, bbox=g.bbox,
                        frame=frame, ts=ts, player=g.player)
        return [ev, TileClaimed(eid=self._next_eid(), claimed=claimed,
                                meld=ev.eid, frame=frame, ts=ts)]
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_v4/test_meld_inferrer.py -v`
Expected: 8 passed

- [ ] **Step 5: 门禁**

Run: `uv run pytest tests/test_v4 -q && uv run ruff check src/mahjong_ai/v4/meld_inferrer.py tests/test_v4/test_meld_inferrer.py && uv run mypy src/mahjong_ai/v4/meld_inferrer.py`
Expected: 全绿

---

### Task 5: 提取器重构 — 组件接线 + 视觉通道加固

**Files:**
- Modify: `src/mahjong_ai/v4/extractor.py`
- Test: `tests/test_v4/test_extractor.py`(更新 3 个副露测试到新语义)

**Interfaces:**
- Consumes: Task 1-4 全部接口;`TileDet`;`cluster_dets`。
- Produces(对外不变):
  - `EventExtractor.process(dets, frame, ts) -> list[Event]` — 事件流现在包含 TileVanished 与推断出的 MeldFormed/TileClaimed。
  - `EventExtractor.fallback_hand(ts: float) -> list[int | None] | None`(新)。
  - 其余公开接口不变(my_hand/hand_dets/hand_band/band_dets/reset_hand_band/set_regions/consume_new_game)。

**删除的旧机制**(被 Task 4 取代):`_check_meld`、`_meld_pool`、`_appeared`、`_appeared_history`、`_CLAIM_MAX_AGE`、`_MELD_CONFIRM`。保留:`_split_groups`、`_in_any_meld_region`、`_meld_seen`(门禁回退用)。

- [ ] **Step 1: 更新既有副露测试(新语义)**

`tests/test_v4/test_extractor.py` 中三个测试改为:

```python
def test_visual_meld_confirmed_and_claim_paired():
    """视觉通道: 合法 3 张组持续 ≥5 帧 → 兜底确认副露 + 最近同类配对。"""
    ext = EventExtractor()
    ext.set_regions(
        {'my_river': (600.0, 300.0, 800.0, 450.0),
         'right_river': (900.0, 300.0, 1200.0, 450.0)},
        {'my_river': (800.0, 600.0, 1000.0, 700.0)})
    base = _hand_frame(13)
    # 手牌先去抖稳定(门开), 再打出一张 8(tid 500)落进 right_river
    f1 = base + [_mk(8, 950.0, 400.0, 965.0, 415.0, tid=500)]
    # 碰组: 3 张紧贴同牌, 与手牌远离(独立簇)
    meld = [_mk(8, 700.0, 700.0, 740.0, 770.0, tid=501),
            _mk(8, 740.0, 700.0, 780.0, 770.0, tid=502),
            _mk(8, 780.0, 700.0, 820.0, 770.0, tid=503)]
    frames = [f1] * 2 + [base + meld] * 5   # 视觉组持续 5 帧
    events = _feed(ext, frames)
    melds = [e for e in events if isinstance(e, MeldFormed)]
    claims = [e for e in events if isinstance(e, TileClaimed)]
    assert melds and melds[0].tiles == (8, 8, 8)
    assert claims and claims[0].claimed is not None  # 最近同类打出配对


def test_invalid_visual_group_rejected():
    """南南北类假副露: 组合非法 → 永不确认。"""
    ext = EventExtractor()
    base = _hand_frame(13)
    bogus = [_mk(28, 700.0, 700.0, 740.0, 770.0, tid=601),
             _mk(30, 740.0, 700.0, 780.0, 770.0, tid=602),
             _mk(30, 780.0, 700.0, 820.0, 770.0, tid=603)]
    frames = [base] * 3 + [base + bogus] * 8
    events = _feed(ext, frames)
    assert not [e for e in events if isinstance(e, MeldFormed)]


def test_two_tile_group_not_confirmed():
    """2 张组(动画中/手牌碎片)不确认 — 旧门槛 ≥2 张的回归修复。"""
    ext = EventExtractor()
    base = _hand_frame(13)
    pair = [_mk(7, 700.0, 700.0, 740.0, 770.0, tid=701),
            _mk(8, 740.0, 700.0, 780.0, 770.0, tid=702)]
    frames = [base] * 3 + [base + pair] * 8
    events = _feed(ext, frames)
    assert not [e for e in events if isinstance(e, MeldFormed)]
```

删除旧测试 `test_meld_confirmed_and_claim`、`test_claim_fallback_by_recent_same_tile`(被上面的测试取代)、`test_region_meld_detected_when_cluster_merges`(区域路径在 Task 8 端到端覆盖)。

`test_flower_excluded_from_melds_and_hand` 保持,但帧数从 2-3 帧改为 `[base + meld] * 5`(确认门槛 3→5 帧),meld 变量里 4 张(含 34)按新语义清洗后为 3 张合法组。

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_v4/test_extractor.py -v`
Expected: FAIL(测试引用已删除/新行为未实现)

- [ ] **Step 3: 改造提取器**

`src/mahjong_ai/v4/extractor.py` 全部改动:

**(a) imports** — 加:

```python
from typing import Callable

from src.mahjong_ai.v4.hand_tracker import HandTracker, StepSig
from src.mahjong_ai.v4.meld_inferrer import (
    MeldInferrer,
    VisualMeld,
    meld_valid,
)
from src.mahjong_ai.v4.river_watcher import RiverWatcher
```

**(b) 常量** — 删除 `_MELD_CONFIRM`、`_CLAIM_MAX_AGE`,加:

```python
#: 视觉副露确认帧数(合法组持续 ≥ 此帧数才交给推断器)
_VISUAL_CONFIRM = 5
#: 打出缓冲/台阶缓冲保留窗口(秒)
_SIGNAL_WINDOW = 10.0
```

**(c) `__init__`** — 删除 `_appeared`、`_appeared_history`、`_meld_pool`,加:

```python
        self._watcher = RiverWatcher()
        self._tracker = HandTracker()
        self._inferrer = MeldInferrer(self._next_eid)
        self._visual_pool: list[dict[str, float | bool]] = []
        self._discard_buf: list[tuple[str | None, int, float, int]] = []
        self._step_buf: list[StepSig] = []
```

**(d) `set_regions`** — 末尾加:

```python
        self._watcher.set_regions(self._regions)
        self._inferrer.set_meld_regions(self._meld_regions)
```

**(e) `process`** — 第 3/4 步整体替换。删除旧副露段(第 3 步的 `_check_meld` 循环、第 3.5 步、`meld_tids` 归属段、第 4 步的 appeared 循环保留但加信号记录)。新代码:

```python
        # 3) 新轨迹 → TileAppeared(记录牌河归属 + 喂消失检测/打出缓冲)
        meld_tids = {d.track_id for m in melds for d in m
                     if d.track_id is not None}
        region_meld = [d for d in dets
                       if d.conf >= HAND_CONF
                       and d.tile != 34
                       and self._in_any_meld_region((d.x1 + d.x2) / 2,
                                                    (d.y1 + d.y2) / 2)]
        meld_tids |= {d.track_id for d in region_meld
                      if d.track_id is not None}
        gate_open = self._hand_n >= 13 or self._meld_seen
        if self._regions:
            gate_open = self._in_game
        if gate_open and not self._gate_was_open:
            hand_tids = {d.track_id for d in hand if d.track_id is not None}
            for d in dets:
                if d.track_id is not None and d.track_id not in hand_tids:
                    self._seen.add(d.track_id)
            self._gate_was_open = True
        if gate_open:
            for d in dets:
                self._check_appeared(d, frame, ts, events, meld_tids)
        # 4) 视觉副露候选(聚类组 + 区域组; ≥3 张且组合合法才进池)
        visual_confirmed: list[VisualMeld] = []
        for g in [m for m in melds if self._group_valid(m)]:
            visual_confirmed += self._visual_track(g)
        for g in self._split_groups(region_meld):
            if self._group_valid(g):
                visual_confirmed += self._visual_track(g)
        # 5) 牌河消失检测
        vanished: list[TileVanished] = []
        for v in self._watcher.tick(dets, frame, ts, self._next_eid):
            vanished.append(v)
            events.append(v)
        # 6) 手牌账本 + 台阶
        step = self._tracker.tick(self._debounced_hand(), ts)
        if step is not None:
            self._step_buf.append(step)
        while self._step_buf and ts - self._step_buf[0].ts > _SIGNAL_WINDOW:
            self._step_buf.pop(0)
        # 7) 副露推断(消费本帧全部信号)
        for ev in self._inferrer.tick(vanished, self._step_buf,
                                      self._discard_buf, visual_confirmed,
                                      self._tracker, frame, ts):
            events.append(ev)
            if isinstance(ev, MeldFormed):
                self._meld_seen = True
                if ev.player == 'my_river':
                    self._tracker.on_my_meld(list(ev.tiles[1:]),
                                             max(1, len(ev.tiles) - 1))
        if hand_changed is not None:
            events.append(hand_changed)
        return events
```

**(f) `_check_appeared`** — 发射分支替换为(记录区域 + 喂组件):

```python
        if (t['stable'] >= _APPEAR_FRAMES
                and not self._in_hand_band(d)
                and not self._recent_dup(d.tile, cx, cy)
                and (not self._regions
                     or self._in_any_river_region(cx, cy))):
            self._emitted.add(tid)
            self._tracks.pop(tid)
            eid = self._next_eid()
            events.append(TileAppeared(
                eid=eid, tile=d.tile, track_id=tid,
                cx=cx, cy=cy, conf=d.conf, frame=frame, ts=ts))
            reg = self._river_of(cx, cy)
            if reg is not None:
                self._watcher.on_appeared(reg, eid, d.tile, cx, cy)
                self._discard_buf.append((reg, d.tile, ts, eid))
                if reg == 'my_river':
                    self._tracker.on_my_discard(d.tile, ts)
            while self._discard_buf and \
                    ts - self._discard_buf[0][2] > _SIGNAL_WINDOW:
                self._discard_buf.pop(0)
```

**(g) 新 helpers**(`_split_groups` 之后,替换 `_check_meld`):

```python
    def _group_valid(self, g: list[TileDet]) -> bool:
        """视觉副露组合法性: 清洗花牌后 ≥3 张且组合合法。"""
        clean = tuple(sorted(d.tile for d in g if d.tile != 34))
        return len(clean) >= 3 and meld_valid(clean)

    def _visual_track(self, g: list[TileDet]) -> list[VisualMeld]:
        """视觉副露候选池: 位置匹配去重, 累计 ≥_VISUAL_CONFIRM 帧 → 确认。"""
        cx = sum((d.x1 + d.x2) / 2 for d in g) / len(g)
        cy = sum((d.y1 + d.y2) / 2 for d in g) / len(g)
        cand = next((c for c in self._visual_pool
                     if abs(float(c['cx']) - cx) < _MELD_MATCH_PX
                     and abs(float(c['cy']) - cy) < _MELD_MATCH_PX), None)
        if cand is None:
            cand = {'cx': cx, 'cy': cy, 'count': 0.0, 'done': False}
            self._visual_pool.append(cand)
        if bool(cand['done']):
            return []
        cand['count'] = float(cand['count']) + 1.0
        if cand['count'] < _VISUAL_CONFIRM:
            return []
        cand['done'] = True
        tiles = tuple(sorted(d.tile for d in g if d.tile != 34))
        bbox = (min(d.x1 for d in g), min(d.y1 for d in g),
                max(d.x2 for d in g), max(d.y2 for d in g))
        return [VisualMeld(player=self._meld_owner(cx, cy),
                           tiles=tiles, cx=cx, cy=cy, bbox=bbox)]

    def _meld_owner(self, cx: float, cy: float) -> str | None:
        """组中心所在副露区域的归属(PLAYERS 名); 不在任何区域 → None。"""
        for name, (rx1, ry1, rx2, ry2) in self._meld_regions.items():
            if (rx1 - _MELD_REGION_MARGIN <= cx <= rx2 + _MELD_REGION_MARGIN
                    and ry1 - _MELD_REGION_MARGIN <= cy
                    <= ry2 + _MELD_REGION_MARGIN):
                return name
        return None

    def _river_of(self, cx: float, cy: float) -> str | None:
        """落点所在牌河区域(PLAYERS 名); 不在任何区域 → None。"""
        for name, (rx1, ry1, rx2, ry2) in self._regions.items():
            if (rx1 - _REGION_MARGIN <= cx <= rx2 + _REGION_MARGIN
                    and ry1 - _REGION_MARGIN <= cy <= ry2 + _REGION_MARGIN):
                return name
        return None
```

**(h) 删除 `_check_meld` 整段,加 `fallback_hand` 公开接口**(`reset_hand_band` 后):

```python
    def fallback_hand(self, ts: float) -> list[int | None] | None:
        """塌陷兜底手牌(未知槽 = None); 检测健康 → None(显示用)。"""
        return self._tracker.fallback_hand(ts)
```

**(i) `import` 修正** — events import 加 `TileVanished`(process 返回类型用)。

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_v4/test_extractor.py -v`
Expected: 全过(含更新的 3 个副露测试)

- [ ] **Step 5: 门禁**

Run: `uv run pytest tests/test_v4 -q && uv run ruff check src/mahjong_ai/v4/extractor.py tests/test_v4/test_extractor.py && uv run mypy src/mahjong_ai/v4/extractor.py`
Expected: 全绿

---

### Task 6: 解码器 — TileVanished 消费 + MeldFormed.player 先验

**Files:**
- Modify: `src/mahjong_ai/v4/decoder.py`
- Test: `tests/test_v4/test_decoder_window.py`

**Interfaces:**
- Consumes: `TileVanished`、`MeldFormed.player`(Task 1)。
- Produces: `WindowDecoder.claimed_ids()` 现在返回 `_claimed` ∪ `_vanished`(显示层自动摘除被碰走的牌)。

- [ ] **Step 1: 写失败测试**

`tests/test_v4/test_decoder_window.py` 追加:

```python
def test_vanished_tile_removed_from_claimed_ids():
    from src.mahjong_ai.v4.events import TileVanished

    d = _decoder()
    d.add(_make('right_river', 1, 0))
    d.add(TileVanished(eid=99, tile=1, river='right_river',
                       appeared_eid=1, frame=1, ts=0.1))
    assert 1 in d.claimed_ids()


def test_meld_player_prior_overrides_spatial():
    """MeldFormed.player(事件推断)压过空间证据 — 副露区框错也不归错。"""
    d = _decoder()
    # 空间上远离 left_river 的副露, 但推断出副露者是 left_river
    d.add(MeldFormed(eid=60, tiles=(8, 8, 8), cx=1150.0, cy=550.0,
                     bbox=(1100.0, 500.0, 1200.0, 600.0),
                     frame=0, ts=0.0, player='left_river'))
    assert d.meld_player(60) == 'left_river'
    assert d.melds()[0][0] == 'left_river'


def test_meld_without_player_still_spatial():
    d = _decoder()
    d.add(MeldFormed(eid=61, tiles=(8, 8, 8), cx=1150.0, cy=550.0,
                     bbox=(1100.0, 500.0, 1200.0, 600.0),
                     frame=0, ts=0.0))
    assert d.meld_player(61) == 'right_river'  # 空间最近
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_v4/test_decoder_window.py -v`
Expected: FAIL(claimed_ids 不含 1;player 先验未生效)

- [ ] **Step 3: 实现**

`src/mahjong_ai/v4/decoder.py`:

**(a)** import 加 `TileVanished`;常量加:

```python
#: 事件推断副露者的先验强度(log) — 空间证据几乎不可能翻盘(软归属保留)
_MELD_PLAYER_BOOST = 10.0
```

**(b)** `__init__` 加:

```python
        self._vanished: set[int] = set()
```

**(c)** `add` 的 MeldFormed 分支改为:

```python
        elif isinstance(event, MeldFormed):
            self._meld_list[event.eid] = event
            scores = {p: self._meld_ev.spatial_logp(p, event.cx, event.cy)
                      for p in PLAYERS}
            if event.player is not None:
                # 事件推断出的副露者: 强先验(空间证据仍可覆盖极端矛盾)
                scores = {p: v + (_MELD_PLAYER_BOOST if p == event.player
                                  else 0.0)
                          for p, v in scores.items()}
            if len(set(scores.values())) == 1:
                self._meld_logits[event.eid] = {}
            else:
                self._meld_logits[event.eid] = scores
            self._unfrozen.append(('meld', event))
            self._resolve()
```

**(d)** `add` 末尾加分支(在 TileClaimed 分支后):

```python
        elif isinstance(event, TileVanished):
            self._vanished.add(event.appeared_eid)
```

**(e)** `claimed_ids`:

```python
    def claimed_ids(self) -> set[int]:
        """被碰走的 TileAppeared eid 集合(显示层排除用)。"""
        return set(self._claimed) | self._vanished
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_v4/test_decoder_window.py -v`
Expected: 全过

- [ ] **Step 5: 门禁**

Run: `uv run pytest tests/test_v4 -q && uv run ruff check src/mahjong_ai/v4/decoder.py tests/test_v4/test_decoder_window.py && uv run mypy src/mahjong_ai/v4/decoder.py`
Expected: 全绿

---

### Task 7: 手牌兜底显示链路(塌陷时暗牌位)

**Files:**
- Modify: `src/mahjong_ai/v4/state.py`、`src/mahjong_ai/state/tracker.py`(仅加字段)、`scripts/run_assistant_v4.py`、`src/mahjong_ui/client.py`
- Test: `tests/test_v4/test_state.py`

**Interfaces:**
- Consumes: `EventExtractor.fallback_hand(ts)`(Task 5)。
- Produces: `GameView.my_hand_fallback: list[int | None] | None`;`GameSnapshot.my_hand_fallback: list[int | None] | None = None`;客户端 `_display_hand() -> list[int | None]`(None = 暗牌位)。

- [ ] **Step 1: 写失败测试**

`tests/test_v4/test_state.py` 追加:

```python
def test_project_carries_hand_fallback():
    from src.mahjong_ai.v4.decoder import EvidenceModel, WindowDecoder

    ev = EvidenceModel()
    mev = EvidenceModel()
    ev.reseed({'my_river': (600.0, 700.0, 800.0, 780.0),
               'right_river': (1100.0, 500.0, 1180.0, 600.0),
               'top_river': (500.0, 100.0, 700.0, 180.0),
               'left_river': (100.0, 500.0, 180.0, 600.0)}, 1280, 800)
    mev.reseed(None, 1280, 800)
    d = WindowDecoder(ev, mev)
    view = project(d, [], [1, 2, None, 3])
    assert view.my_hand_fallback == [1, 2, None, 3]


def test_project_fallback_default_none():
    from src.mahjong_ai.v4.decoder import EvidenceModel, WindowDecoder

    ev = EvidenceModel()
    mev = EvidenceModel()
    ev.reseed(None, 1280, 800)
    mev.reseed(None, 1280, 800)
    d = WindowDecoder(ev, mev)
    view = project(d, [])
    assert view.my_hand_fallback is None
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_v4/test_state.py -v`
Expected: FAIL(project 参数不存在)

- [ ] **Step 3: 实现**

**(a)** `src/mahjong_ai/v4/state.py`:

```python
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
    ...
    return GameView(my_hand=my_hand, players=players,
                    unfrozen=decoder.unfrozen(),
                    provisional=provisional,
                    my_hand_fallback=my_hand_fallback)
```

**(b)** `src/mahjong_ai/state/tracker.py` — GameSnapshot 加字段(其余不动):

```python
    my_hand: list[int] = field(default_factory=list)
    #: 手牌塌陷时的兜底显示(未知槽 = None; 检测健康 → None)
    my_hand_fallback: list[int | None] | None = None
```

**(c)** `scripts/run_assistant_v4.py` — project 调用(第 350 行附近):

```python
        view = project(self._decoder, self._extractor.my_hand(),
                       self._extractor.fallback_hand(time.time()))
```

build_advice 的 GameSnapshot 构造(第 188 行附近):

```python
    snap = GameSnapshot(my_hand=view.my_hand,
                        my_hand_fallback=view.my_hand_fallback,
                        players=players)
```

**(d)** `src/mahjong_ui/client.py` — `_display_hand`:

```python
    def _display_hand(self) -> list[int | None]:
        """手牌显示(带保持 + 塌陷兜底): 检测抽风手牌塌 0 的数秒内
        冻结在最后一次非空手牌; 超过宽限后用账本兜底(未知槽 = None
        画暗牌位) — 不闪烁消失, 也不显示牌河误判的假手牌。"""
        import time  # noqa: PLC0415

        if self._advice is None:
            return []
        hand = self._advice.snapshot.my_hand
        if hand:
            self._held_hand = hand
            self._held_hand_ts = time.time()
            return hand
        if time.time() - self._held_hand_ts < _HAND_HOLD_SECONDS:
            return self._held_hand
        fb = self._advice.snapshot.my_hand_fallback
        return fb or []
```

`_draw_my_zone` 的手牌循环(第 495-505 行)改为(None → 暗牌位):

```python
        hand = self._display_hand()
        # 1) 手牌(最大牌块 — 打牌时最关心"现在"; 抽风期冻结/暗牌兜底)
        n = max(1, len(hand) or 1)
        gap = 3
        tw = min(40, max(22, (w - gap * (n - 1)) // n))
        th = int(tw * 1.4)
        for i, t in enumerate(hand):
            if (i + 1) * (tw + gap) > w:
                break
            if t is None:
                # 未知槽: 暗牌位(兜底手牌 — 摸进的新牌检测不到)
                painter.setPen(QColor(60, 64, 60))
                painter.setBrush(QColor(40, 44, 40))
                painter.drawRoundedRect(x + i * (tw + gap), y, tw, th, 4, 4)
                continue
            self._draw_tile(painter, x + i * (tw + gap), y, tw, th, t,
                            tw // 2)
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_v4 tests/test_scripts -q`
Expected: 全过

- [ ] **Step 5: 门禁**

Run: `uv run pytest tests/test_v4 tests/test_scripts -q && uv run ruff check src/mahjong_ai/v4/state.py src/mahjong_ai/state/tracker.py scripts/run_assistant_v4.py src/mahjong_ui/client.py && uv run mypy src/mahjong_ai/v4/state.py`
Expected: 全绿

---

### Task 8: 端到端合成回归 + 复盘脚本 + 全门禁

**Files:**
- Create: `tests/test_v4/test_e2e_zones.py`
- Modify: `scripts/dump_v4_log.py`(TileVanished 行)

**Interfaces:**
- Consumes: Task 1-7 全部。
- Produces: 端到端行为契约(我的碰 = 消失 + 台阶;被碰牌从牌河显示摘除)。

- [ ] **Step 1: 写失败测试**

`tests/test_v4/test_e2e_zones.py`:

```python
"""端到端: 事件推断副露 → 解码 → 投影(被碰牌摘除 + 副露内容)。"""

from src.mahjong_ai.v4.decoder import EvidenceModel, WindowDecoder
from src.mahjong_ai.v4.events import (
    MeldFormed,
    TileAppeared,
    TileClaimed,
    TileVanished,
)
from src.mahjong_ai.v4.extractor import EventExtractor
from src.mahjong_ai.v4.state import project
from src.mahjong_cv.detections import TileDet


def _mk(tile, x1, y1, x2, y2, conf=0.8, tid=None) -> TileDet:
    return TileDet(tile=tile, x1=x1, y1=y1, x2=x2, y2=y2, conf=conf,
                   track_id=tid)


def _hand13(t0: float = 100.0) -> list[TileDet]:
    """我的手牌: 13 张(含两张 8), 底行 y 700-770。"""
    return [_mk(i if i < 8 else i + 1, t0 + i * 40, 700.0,
                t0 + i * 40 + 40, 770.0, tid=1000 + i) for i in range(13)]


def _hand11() -> list[TileDet]:
    """副露后: 11 张(两张 8 进副露)。"""
    tiles = [i for i in range(14) if i != 8]
    return [_mk(t, 100.0 + i * 40, 700.0, 100.0 + i * 40 + 40, 770.0,
                tid=2000 + i) for i, t in enumerate(tiles)]


def test_my_pong_inferred_from_vanish_and_step():
    """我的碰: 右家打出 8 → 8 从牌河消失 + 我手牌 13→11 → 副露(8,8,8);
    被碰的 8 从右家 visible_river 摘除。"""
    ext = EventExtractor()
    ext.set_regions(
        {'my_river': (500.0, 300.0, 800.0, 450.0),
         'right_river': (900.0, 300.0, 1200.0, 450.0)},
        {'my_river': (800.0, 600.0, 1000.0, 700.0)})
    events = []
    # 阶段 1: 手牌 13 稳定(快照建立)
    for _ in range(8):
        events += ext.process(_hand13(), frame=0, ts=0.0)
    # 阶段 2: 右家打出 8(落进 right_river, 稳定 2 帧 → TileAppeared)
    disc = _mk(8, 950.0, 400.0, 965.0, 415.0, tid=500)
    events += ext.process(_hand13() + [disc], frame=1, ts=1.0)
    events += ext.process(_hand13() + [disc], frame=2, ts=1.1)
    appeared = [e for e in events if isinstance(e, TileAppeared)]
    assert appeared and appeared[-1].tile == 8
    # 阶段 3: 8 消失(被碰走), 右河其他牌还在; 我手牌 13→11
    other = _mk(5, 1000.0, 400.0, 1015.0, 415.0, tid=501)
    vanish_ts = 3.0
    for i in range(25):
        ts = vanish_ts + i * 0.1
        events += ext.process(_hand11() + [other], frame=10 + i, ts=ts)
    vanished = [e for e in events if isinstance(e, TileVanished)]
    assert len(vanished) == 1
    melds = [e for e in events if isinstance(e, MeldFormed)]
    assert len(melds) == 1
    assert melds[0].tiles == (8, 8, 8)
    assert melds[0].player == 'my_river'
    claims = [e for e in events if isinstance(e, TileClaimed)]
    assert claims and claims[0].claimed == appeared[-1].eid
    # 阶段 4: 解码 + 投影 — 被碰的 8 从右家 visible_river 摘除
    river_ev = EvidenceModel()
    meld_ev = EvidenceModel()
    river_ev.reseed({'my_river': (500.0, 300.0, 800.0, 450.0),
                     'right_river': (900.0, 300.0, 1200.0, 450.0),
                     'top_river': (500.0, 100.0, 700.0, 180.0),
                     'left_river': (100.0, 500.0, 180.0, 600.0)}, 1280, 800)
    meld_ev.reseed(None, 1280, 800)
    d = WindowDecoder(river_ev, meld_ev)
    for ev in events:
        d.add(ev)
    d.age_freeze(100.0)
    view = project(d, ext.my_hand())
    assert 8 not in view.players['right_river'].visible_river
    assert view.players['my_river'].melds[0].tiles == (8, 8, 8)
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_v4/test_e2e_zones.py -v`
Expected: FAIL 或行为断言失败(集成前链路未通)

- [ ] **Step 3: 复盘脚本 TileVanished 行**

`scripts/dump_v4_log.py` 的解析循环加(具体位置按文件内现有 type 分支风格,先读文件再插入):

```python
        if rec.get('type') == 'TileVanished':
            from src.mahjong_core.tile import tile_display
            print(f"  消失: {tile_display(rec['tile'])} 被拿走 "
                  f"@{rec['river']}(appeared {rec['appeared_eid']})")
            continue
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_v4 -q`
Expected: 全过(含端到端)

- [ ] **Step 5: 复盘脚本冒烟**

Run: `uv run python scripts/dump_v4_log.py data/settle_logs/game_v4_20260815_170938.jsonl | head -30`
Expected: 正常输出,不崩溃(旧日志无 TileVanished 记录,仅验证兼容)

- [ ] **Step 6: 全量门禁**

Run: `uv run pytest tests -q && uv run ruff check src scripts tests && uv run mypy src/mahjong_ai/v4 src/mahjong_ai/state/tracker.py src/mahjong_cv/det_cluster.py`
Expected: 全绿(pytest 全量、ruff、mypy v4 相关)

---

## 自审记录

**1. Spec 覆盖:**
- §4.1 RiverWatcher → Task 2 ✓;§4.2 HandTracker/台阶 → Task 3 ✓;§4.3 MeldInferrer → Task 4 ✓(台阶配对/行动权/超时丢弃/视觉填充/连续副露按时间序配对 — 多候选时按 tick 内顺序先后分配,时间序由候选列表顺序保证);§4.4 视觉通道加固(≥3+合法+5 帧)→ Task 5 ✓;§4.5 解码器 TileVanished → Task 6 ✓;§5 数据流 → Task 5 ✓;§7 测试 → Tasks 2-8 ✓;§6 退化矩阵各故障 → 对应测试覆盖(整河塌/超时/塌陷降级/非法组拒绝)。
- spec 的"手牌差窗口 2 秒"在 Task 3 放宽为 3 秒(实现分析:稳定纪元 ~0.7s + 副露后出牌,2s 偏紧),已在 Global Constraints 记录偏差。

**2. Placeholder 扫描:** 无 TBD/TODO;所有步骤含完整代码与命令。

**3. 类型一致性:** TileVanished/TileClaimed/MeldFormed 字段全任务一致;`discards` 元组 (player, tile, ts, appeared_eid) 在 Task 4/5 一致;`fallback_hand(ts)` 签名 Task 3/5/7 一致;`StepSig(drop, ts)` 一致;PLAYERS 名 'my_river'/'right_river'/'top_river'/'left_river' 与现有键一致(C1 重映射后 meld_regions 也以此为键)。
