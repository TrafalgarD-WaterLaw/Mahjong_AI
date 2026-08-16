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

from src.mahjong_ai.pipeline.events import (
    PLAYERS,
    HandChanged,
    MeldFormed,
    TileAppeared,
    TileClaimed,
    TileVanished,
)

#: 轮转表(欢乐麻将: 我 → 右(下家) → 上(对家) → 左(上家), 用户确认)
_ROTATION = PLAYERS
#: 轮转跳一位的 log 惩罚(中间家的打出事件漏检时; 跳 N 位 = N×此值)
_SKIP_PENALTY = 1.5
#: 相位先验差距: log P(actor) − log P(其他家) ≈ P(actor)≈0.77
_TURN_PRIOR_GAP = 2.0
#: 空间似然种子: 框 ≈ ±2σ → σ = 框半宽/2
_SEED_SIGMA = 4.0
#: 在线学习速率(冻结事件 → 该家落点高斯 EMA)
_LEARN_LR = 0.05
#: 窗口大小(未冻结 TileAppeared 数)/ beam 宽度
_K_DEFAULT = 10
_BEAM_DEFAULT = 16
#: 冻结后在线学习的置信边距门槛(log 比): 低于此值不学习(防错误样本)
_LEARN_MIN_MARGIN = math.log(2.0)
#: 事件推断副露者的先验强度(log) — 空间证据几乎不可能翻盘(软归属保留)
_MELD_PLAYER_BOOST = 10.0
#: 时间冻结阈值(秒): 事件超过此年龄即使窗口未满也冻结。
#: 显示已不依赖冻结(待定牌按当前 MAP 即时上屏), 冻结只为
#: 永久记录/在线学习服务 — 8s ≈ 1-2 个后手事件做回看纠错
_FREEZE_MAX_AGE = 8.0


def next_in_rotation(player: str) -> str:
    """轮转下一位。"""
    return _ROTATION[(_ROTATION.index(player) + 1) % len(_ROTATION)]


@dataclass(frozen=True)
class TurnPhase:
    """轮转相位: actor = 下一位该行动者, played = 本轮已行动家。"""

    actor: str = 'my_river'
    played: frozenset[str] = frozenset()

    def advance(self, player: str) -> tuple[TurnPhase, float]:
        """player 行动(打出/碰) → (新相位, 跳家惩罚)。

        跳家 = actor 与 player 之间(轮转序)的家被跳过 — 他们应打出的
        牌漏检了: 隐式补记, 只罚分(轮转是约束不是链条)。
        """
        if player == self.actor:
            penalty: float = 0
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

    def claim(self, player: str) -> tuple[TurnPhase, float]:
        """副露截断: player 碰/杠 → 该家接着打(actor=碰家)。

        跳过的家(轮转序上 actor 到 player 之间)本轮失去行动权
        (标记 played); 碰是合法截断, 无惩罚。
        """
        if player == self.actor:
            played = self.played | {player}
            return TurnPhase(actor=player, played=played), 0.0
        skipped: set[str] = set()
        cur = self.actor
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
        """落点在该家牌河区的对数似然(高斯); 未播种 → 返回 0(空间证据中性)。

        未播种时构造器原点高斯对四家完全相同且极端为负, 会污染整条
        beam 得分(真实对局曾因此全部归属 my_river); 中性化后未播种时
        由轮转先验主导。
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
        return max(self.logits, key=self.logits.__getitem__)

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
        self._meld_logits: dict[int, dict[str, float]] = {}
        self._claimed: dict[int, int] = {}   # appeared eid -> meld eid
        self._claim_no_event: set[int] = set()  # 秒碰的 meld eid
        self._vanished: set[int] = set()

    # ---- 事件入口 ----

    def add(self, event: object) -> None:
        """事件按时间序流入(提取器产出顺序)。FlowerShown 忽略。"""
        if isinstance(event, TileAppeared):
            self._unfrozen.append(('appeared', event))
            self._resolve()
            while self._n_appeared() > self._k:
                self._freeze_oldest()
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
        elif isinstance(event, HandChanged):
            self._unfrozen.append(('hand', event))
            self._resolve()
        elif isinstance(event, TileClaimed):
            if event.claimed is None:
                self._claim_no_event.add(event.meld)
            else:
                self._claimed[event.claimed] = event.meld
        elif isinstance(event, TileVanished):
            self._vanished.add(event.appeared_eid)

    # ---- 查询(投影/日志用) ----

    def attributions(self) -> list[Attribution]:
        return list(self._attribs.values())

    def frozen(self) -> list[Attribution]:
        return [a for a in self._attribs.values() if a.frozen]

    def unfrozen(self) -> list[Attribution]:
        return [a for a in self._attribs.values() if not a.frozen]

    def claimed_ids(self) -> set[int]:
        """被碰走的 TileAppeared eid 集合(显示层排除用)。"""
        return set(self._claimed) | self._vanished

    def melds(self) -> list[tuple[str, MeldFormed]]:
        """已确认副露: [(玩家, 事件)] 按帧序(MAP 归属)。归属不明的跳过。"""
        return sorted(((player, ev)
                       for eid, ev in self._meld_list.items()
                       if (player := self._meld_map(eid)) is not None),
                      key=lambda t: t[1].frame)

    def meld_probs(self, eid: int) -> dict[str, float]:
        """副露归属分布(softmax 归一; 空分布 → 空 dict)。

        M3 推断按分布加权扣牌池; 显示/相位用 melds() 的 MAP。
        """
        logits = self._meld_logits.get(eid, {})
        if not logits:
            return {}
        m = max(logits.values())
        z = math.log(sum(math.exp(v - m) for v in logits.values())) + m
        return {p: math.exp(v - z) for p, v in logits.items()}

    def _meld_map(self, eid: int) -> str | None:
        """副露 MAP 归属; 空分布(退化) → None。"""
        logits = self._meld_logits.get(eid, {})
        return max(logits, key=logits.__getitem__) if logits else None

    def meld_player(self, eid: int) -> str | None:
        """副露 MAP 归属(公开入口 — 显示/扣池用); 退化 → None。"""
        return self._meld_map(eid)

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
                assert isinstance(ev, HandChanged)
                beams = [(self._hand_phase(ev, ph), s, a)
                         for ph, s, a in beams]
            elif kind == 'meld':
                assert isinstance(ev, MeldFormed)
                logits = self._meld_logits.get(ev.eid, {})
                if not logits:  # 退化: 归属不明 → 相位不动
                    continue
                # 软归属: 副露处按四家分支(空间 logits 加权) —
                # 相位探索所有可能的碰家, 不做硬 argmax
                meld_beams = []
                for ph, s, a in beams:
                    for p, lp in logits.items():
                        meld_beams.append((ph.claim(p)[0], s + lp, a))
                meld_beams.sort(key=lambda t: -t[1])
                beams = meld_beams[:self._beam]
            else:  # appeared: 四家分支, 束剪枝
                assert isinstance(ev, TileAppeared)
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
            assert isinstance(ev, TileAppeared)
            per = {p: float('-inf') for p in PLAYERS}
            for _ph, s, a in beams:
                per[a[ev.eid]] = _logadd(per[a[ev.eid]], s)
            self._attribs[ev.eid] = Attribution(
                eid=ev.eid, tile=ev.tile, cx=ev.cx, cy=ev.cy,
                frame=ev.frame, ts=ev.ts, logits=per)

    def age_freeze(self, now: float) -> None:
        """时间冻结: 超过 _FREEZE_MAX_AGE 的未冻结事件即使窗口未满
        也冻结 — 运行器每 tick 用墙钟时间调用。

        窗口溢出才冻结的问题(真实对局): 事件间隔 3-13s, 显示滞后
        1-2 分钟; 对局结束不再有新事件 → 末 k 张永远不冻结。
        时间冻结把滞后压在 ~_FREEZE_MAX_AGE 秒内。
        """
        while True:
            # 最老的 appeared 定年龄(它前面的 hand/meld 硬更新不
            # 计年龄, 由 _freeze_oldest 顺带消化)
            oldest = next((ev for kind, ev in self._unfrozen
                           if kind == 'appeared'), None)
            if oldest is None:
                return
            assert isinstance(oldest, TileAppeared)
            if now - oldest.ts < _FREEZE_MAX_AGE:
                return
            self._freeze_oldest()

    def _freeze_oldest(self) -> None:
        """出队到最老的 appeared(之前的 hand/meld 硬更新先消化进基底)。"""
        while self._unfrozen and self._unfrozen[0][0] != 'appeared':
            kind, ev = self._unfrozen.pop(0)
            self._apply_hard(ev)
        if not self._unfrozen:
            return
        _kind, ev = self._unfrozen.pop(0)
        assert isinstance(ev, TileAppeared)
        at = self._attribs[ev.eid]
        at.player = at.map()
        at.frozen = True
        # 在线学习(边距门槛: 低置信不喂样本, 防错误归属反馈回路)
        rest = max(v for p, v in at.logits.items() if p != at.player)
        if at.logits[at.player] - rest >= _LEARN_MIN_MARGIN:
            self._river_ev.update(at.player, ev.cx, ev.cy)
        self._phase_base, _ = self._phase_base.advance(at.player)

    def _apply_hard(self, ev: object) -> None:
        """硬更新事件进入冻结基底(相位推进)。"""
        if isinstance(ev, HandChanged):
            self._phase_base = self._hand_phase(ev, self._phase_base)
        elif isinstance(ev, MeldFormed):
            mplayer = self._meld_map(ev.eid)
            if mplayer is not None:  # 退化: 归属不明 → 相位不动
                self._phase_base, _ = self._phase_base.claim(mplayer)
