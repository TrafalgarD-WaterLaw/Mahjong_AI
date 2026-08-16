"""副露推断 — 事件信号组合, 不依赖"看到三张牌"。

证据链:
  我的副露 = 牌河消失(X) + 手牌台阶(-2/-3); 内容 = X + 手牌前后差
  我的副露(快碰, 无消失) = 手牌台阶独立确认 — 摸打只 ±1, 掉 2/3
            只有副露一种可能; 内容 = 手牌差(合法性校验), 被碰牌由
            最近同类打出配对(吃=顺子补第三张; 暗杠无被碰牌)
  对手副露 = 牌河消失(X) + 消失后第一家出牌(行动权在副露者);
            内容 = X + 2 未知
  视觉合法组(≥3 张 + 组合合法 + 持续确认)只做内容填充与兜底。
确认超时(5s)→ 等视觉兜底 → 再 3s 无 → 丢弃。宁可缺事件, 不发假事件。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

from src.mahjong_ai.pipeline.events import MeldFormed, TileClaimed, TileVanished
from src.mahjong_ai.pipeline.hand_tracker import HandTracker, StepSig

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
    """候选: 消失信号(v 非空)或台阶独立(step 非空) + 补齐的证据。"""

    v: TileVanished | None
    melder: str | None = None
    content: tuple[int, ...] | None = None
    claimed: int | None = None    # 台阶路径的被碰牌 appeared eid(可 None)


def _my_meld_from_delta(delta: list[int], drop: int,
                        claimed_tile: int | None,
                        ) -> tuple[int, ...] | None:
    """台阶手牌差 → 我的副露内容; 不合法 → None(丢弃候选)。

    碰: delta=(X,X) → (X,X,X); 杠: delta=(X,X,X) → (X,X,X,X);
    吃: delta=(a,b) 同花 |a-b|≤2 → 补第三张(claimed_tile 或默认)。
    """
    if drop == 3:
        # 明杠差 3 张(13→10); 暗杠 13→9→补牌 10, 差 4 张 — 都合法
        return (delta[0],) * 4 if len(delta) in (3, 4) \
            and len(set(delta)) == 1 else None
    if drop == 2 and len(delta) == 2:
        if delta[0] == delta[1]:
            return (delta[0],) * 3
        a, b = sorted(delta)
        if a // 9 != b // 9 or b > 26 or b - a > 2:
            return None
        if b - a == 2:
            return (a, a + 1, b)
        third = (claimed_tile if claimed_tile is not None
                 and claimed_tile in (a - 1, b + 1) else None)
        if third is None:
            third = a - 1 if a % 9 > 0 else b + 1
        return tuple(sorted((a, b, third)))
    return None


def _claim_candidates(delta: list[int], drop: int) -> list[int]:
    """台阶副露的被碰牌候选(最近同类打出配对用)。"""
    if drop == 3 and len(delta) == 3 and len(set(delta)) == 1:
        return [delta[0]]
    if drop == 2 and len(delta) == 2:
        if delta[0] == delta[1]:
            return [delta[0]]
        a, b = sorted(delta)
        if a // 9 != b // 9 or b > 26:
            return []
        if b - a == 2:
            return [a + 1]
        if b - a == 1:
            out = []
            if a % 9 > 0:
                out.append(a - 1)
            if b % 9 < 8:
                out.append(b + 1)
            return out
    return []


def _sub_multi(a: tuple[int, ...], b: tuple[int, ...]) -> bool:
    """a 的多重集是否 ⊆ b — 副露去重匹配(同副露内容互为子集;
    相邻顺子(5,6,7) vs (6,7,8) 互不为子集, 不误伤)。"""
    cb = Counter(b)
    return all(cb.get(ti, 0) >= n for ti, n in Counter(a).items())


class MeldInferrer:
    """消失/台阶/出牌 → 副露事件。eid 与提取器共享(_next_eid)。"""

    def __init__(self, next_eid: Callable[[], int]) -> None:
        self._next_eid = next_eid
        self._cands: list[_Cand] = []
        self._used_steps: set[tuple[int, float]] = set()
        self._meld_regions: dict[str, tuple[float, float, float, float]] = {}
        #: 已确认副露 [(ts, tiles), 任意家] — 消失/台阶晚到不重复发
        #: (真实对局: 杠东风显示 5 张 = 视觉兜底 4 张 + 晚到消失 1 张)
        self._recent: list[tuple[float, tuple[int, ...]]] = []
        #: 本局已发副露(永久 — 副露是场景特征不会消失): 视觉兜底
        #: 可能隔几十秒才确认同一副露, 10 秒窗口挡不住(真实对局:
        #: 左家碰 9 消失路径先发, 49 秒后视觉组 (9,9,9) 又发 → 观察器
        #: 粒子缩两次 → 听牌恒 0)
        self._all_melds: list[tuple[int, ...]] = []

    def set_meld_regions(
        self, regions: dict[str, tuple[float, float, float, float]],
    ) -> None:
        """默认副露位置(无视觉组时事件 bbox 用该家区域中心)。"""
        self._meld_regions = dict(regions)

    def _note_recent(self, ts: float, tiles: tuple[int, ...]) -> None:
        """副露已发(任意家): 登记去重表(10s 窗口裁剪 + 永久记录)。"""
        self._recent.append((ts, tiles))
        self._all_melds.append(tiles)
        while self._recent and ts - self._recent[0][0] > _CLAIM_HIST_MAX_AGE:
            self._recent.pop(0)

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
            # 已确认过同一副露(视觉兜底/台阶路径, 消失晚到) → 不重复
            if any(_sub_multi((v.tile,), tiles)
                   for _ts, tiles in self._recent):
                continue
            self._cands.append(_Cand(v=v))
        # 视觉组分配: 含消失牌的组 → 填充该候选; 剩余 → 兜底确认
        unmatched: list[VisualMeld] = []
        for g in visual:
            if not meld_valid(g.tiles):
                continue  # 双保险: 提取器已筛, 此处不信任(真实对局假副露)
            target = next((c for c in self._cands
                           if c.v is not None and c.v.tile in g.tiles),
                          None)
            if target is not None:
                target.content = g.tiles
                target.melder = target.melder or g.player
            else:
                unmatched.append(g)
        for g in unmatched:
            # 同一副露的视觉晚到(副露是永久场景特征, 隔几十秒也可能)
            # → 对照本局已发副露永久去重(10 秒窗口挡不住)
            if any(_sub_multi(g.tiles, t) or _sub_multi(t, g.tiles)
                   for t in self._all_melds):
                continue
            events += self._fallback_meld(g, discards, frame, ts)
            self._note_recent(ts, g.tiles)
        # 台阶配对(我的副露 — 消失候选优先)
        for c in self._cands:
            if c.v is None or c.melder is not None:
                continue
            for s in steps:
                if (s.drop, s.ts) in self._used_steps:
                    continue
                if abs(s.ts - c.v.ts) <= _STEP_WINDOW:
                    c.melder = 'my_river'
                    self._used_steps.add((s.drop, s.ts))
                    break
        # 台阶独立确认(快碰: 被碰牌从未进牌河, 无消失) —
        # 摸打只 ±1, 手牌掉 2/3 只有副露一种可能; 手牌差合法性
        # 校验挡住检测抽风(差不成副露 → 不确认, 宁可缺事件)
        for s in steps:
            if (s.drop, s.ts) in self._used_steps:
                continue
            delta = tracker.hand_delta(s.ts)
            if delta is None:
                continue  # 塌陷期无稳定快照 → 不确认
            # 被碰牌先配对(吃的第三张由它定), 再组装最终内容 —
            # 默认第三张可能与别家副露撞牌, 先去重会误伤合法吃
            cands_tiles = _claim_candidates(delta, s.drop)
            claimed = claimed_tile = None
            for _p, tile, dts, eid in reversed(discards):
                if (tile in cands_tiles
                        and ts - dts <= _CLAIM_HIST_MAX_AGE):
                    claimed, claimed_tile = eid, tile
                    break
            content = _my_meld_from_delta(delta, s.drop, claimed_tile)
            if content is None:
                continue  # 差不成副露(检测污染) → 不确认
            if any(c.v is not None and c.v.tile in content
                   for c in self._cands):
                continue  # 消失候选在等 → 交给它(防双事件)
            if any(_sub_multi(content, tiles) or _sub_multi(tiles, content)
                   for _ts, tiles in self._recent):
                continue  # 视觉兜底/消失路径已发过同一副露 → 防双事件
            self._used_steps.add((s.drop, s.ts))
            self._cands.append(_Cand(v=None, melder='my_river',
                                     content=content, claimed=claimed))
        # 出牌配对(对手副露 — 消失后第一家出牌)
        for c in self._cands:
            if c.v is None or c.melder is not None:
                continue
            for player, _tile, dts, eid in discards:
                if eid == c.v.appeared_eid:
                    continue  # 被碰那张自己的事件 → 丢弃者不是副露者
                if c.v.ts - _DISCARD_LO <= dts <= c.v.ts + _DISCARD_HI:
                    c.melder = player
                    break
        # 内容补全(我的: 手牌差)
        for c in self._cands:
            if c.v is not None and c.melder == 'my_river' \
                    and c.content is None:
                delta = tracker.hand_delta(c.v.ts)
                if delta is not None and len(delta) in (2, 3):
                    cand = (c.v.tile,) + tuple(delta)
                    if meld_valid(cand):
                        c.content = cand
                    # 手牌差可能是另一次副露/检测抽风的(真实对局:
                    # 被碰 17 + 第二次副露的差 (20,21) → 假吃 (17,20,21))
                    # → 不采用, 超宽限按未知 (X,) 发射
        # 确认发射
        for c in list(self._cands):
            if c.melder is None:
                continue
            if c.content is None:
                if c.v is not None and ts - c.v.ts < _DELTA_WAIT:
                    continue  # 等手牌差; 超宽限按未知发射
                assert c.v is not None  # 台阶路径创建时必有内容
                c.content = (c.v.tile,)
            events += self._emit(c, frame, ts)
            self._cands.remove(c)
            self._note_recent(ts, c.content or ())
        # 超时清理(不发假事件)
        for c in list(self._cands):
            if c.v is not None and \
                    ts - c.v.ts > _CONFIRM_TIMEOUT + _VISUAL_GRACE:
                self._cands.remove(c)
        return events

    def _emit(self, c: _Cand, frame: int, ts: float,
              ) -> list[MeldFormed | TileClaimed]:
        """候选确认 → MeldFormed + TileClaimed(位置默认该家区域中心)。"""
        assert c.content is not None  # 发射前内容必已补齐(tick 保证)
        content = c.content
        region = self._meld_regions.get(c.melder or '')
        if region is not None:
            rx1, ry1, rx2, ry2 = region
            cx, cy = (rx1 + rx2) / 2, (ry1 + ry2) / 2
            bbox = region
        else:
            cx, cy = 0.0, 0.0   # 无区域 → 位置未知, 解码器走空间软归属
            bbox = (0.0, 0.0, 0.0, 0.0)
        ev = MeldFormed(eid=self._next_eid(), tiles=content,
                        cx=cx, cy=cy, bbox=bbox, frame=frame, ts=ts,
                        player=c.melder)
        claimed = (c.v.appeared_eid if c.v is not None else c.claimed)
        return [ev, TileClaimed(eid=self._next_eid(),
                                claimed=claimed, meld=ev.eid,
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
