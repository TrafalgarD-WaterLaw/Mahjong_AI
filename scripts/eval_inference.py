"""推断准确性离线评估 — 模拟对局 vs 粒子滤波推断。

方法: 生成合成对局(四家真手牌全程已知), 只把可观测事件
(谁打了什么 + 副露内容)喂给 SoftObserver, 在检查点对比推断与真值:

  - 手牌质量: 真手牌各张的推断概率之和(随机基线 ≈ 13/池·13,
    完美 = 13) — "根据出牌能推测出对手手牌吗"的定量答案
  - 听牌校准: 推断听牌概率分桶 vs 实际是否听牌
  - waiting 命中: 实际听牌时, 真等待牌是否在推断 top3

用法:
    python scripts/eval_inference.py [--games 10] [--steps 30]
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mahjong_ai.efficiency.shanten import calculate_shanten  # noqa: E402
from src.mahjong_ai.efficiency.ukeire_selector import (  # noqa: E402
    recommend_by_ukeire,
)
from src.mahjong_ai.inference.soft_observer import (  # noqa: E402
    OPPONENTS,
    MeldObs,
    SoftObserver,
)
from src.mahjong_ai.strategy import StrategyEngine  # noqa: E402
from src.mahjong_core import Hand  # noqa: E402
from src.mahjong_engine import get as get_rules  # noqa: E402

#: 花牌不进模拟(补花简化) — 34 类 × 4 = 136 张
_WALL_TILES = list(range(34))
_NAMES = ('my_river', 'right_river', 'top_river', 'left_river')


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
            return -1  # 无牌可打(极端副露局) → 调用方跳过
        try:
            if self.policy == 'ukeire' and available is not None:
                rec = recommend_by_ukeire(tiles, available)
                t = rec.tile
            else:
                rec = engine.recommend_discard(Hand(tiles))
                t = rec.tile
        except ValueError:
            t = tiles[0]  # 张数不足推荐下限 → 随便打
        self.hand[t] -= 1
        if self.hand[t] <= 0:
            del self.hand[t]
        return t

    def draw(self, wall: list[int]) -> None:
        if wall:
            self.hand[wall.pop()] += 1


def claim_candidates(hand: Counter[int], tile: int,
                     ) -> tuple[int, ...] | None:
    """该手牌能否碰/杠/吃 tile → 副露内容(优先杠 > 碰 > 吃)。"""
    if hand.get(tile, 0) >= 3:
        return (tile,) * 4
    if hand.get(tile, 0) >= 2:
        return (tile,) * 3
    if tile <= 26:
        suit = tile // 9 * 9
        for a in range(max(suit, tile - 2), min(tile, suit + 6) + 1):
            need = [t for t in (a, a + 1, a + 2) if t != tile]
            if all(hand.get(t, 0) > 0 for t in need):
                return (a, a + 1, a + 2)
    return None


def _meld_improves(hand: Counter[int], combo: tuple[int, ...]) -> bool:
    """副露决策(模拟真实玩家): 已近听(向听 ≤1)不碰 — 碰会破坏
    听牌结构; 向听 ≥2 的散牌阶段才考虑(真实玩家的碰吃都在中盘前)。
    """
    return calculate_shanten(sorted(hand.elements())) >= 2


def simulate(rng: random.Random, rules, engine: StrategyEngine,
             n_steps: int, policy: str = 'engine',
             checkpoints: list[int] | None = None,
             ) -> tuple[dict[str, SimPlayer], list[tuple[str, ...]],
                        dict[int, dict[str, Counter[int]]]]:
    """一局模拟 → (四家终局真手牌, 可观测事件流, 检查点手牌快照)。

    policy: 'engine'(旧推荐)或 'ukeire'(纯牌效新算法) — A/B 对比。
    snapshots[c]: 事件数达到 c 时四家真手牌的副本(检查点指标用 —
    终局手牌不能用于早期检查点对比)。
    """
    wall = [t for t in _WALL_TILES for _ in range(4)]
    rng.shuffle(wall)
    players = {n: SimPlayer([wall.pop() for _ in range(13)],
                            policy=policy)
               for n in _NAMES}
    visible: Counter[int] = Counter()  # 全桌可见牌(各河 + 副露)
    events: list[tuple[str, ...]] = []
    snapshots: dict[int, dict[str, Counter[int]]] = {}
    want = set(checkpoints or [])
    turn = 0
    for _ in range(n_steps):
        for _ in range(4):  # 一圈四家各行动一次
            if not wall:
                break  # 牌墙耗尽 → 对局结束
            name = _NAMES[turn % 4]
            turn += 1
            p = players[name]
            p.draw(wall)
            available = {t: max(0, 4 - visible[t] - p.hand.get(t, 0))
                         for t in range(34)}
            t = p.discard(engine, available)
            if t < 0:
                continue
            visible[t] += 1
            events.append(('discard', name, str(t)))
            if len(events) in want:
                snapshots[len(events)] = {n: Counter(pl.hand)
                                          for n, pl in players.items()}
            # 副露判定: 下家可吃, 任意家可碰/杠; 杠 > 碰 > 吃
            claims = []
            nxt = _NAMES[(turn) % 4]
            for cand_name in (nxt, *[n for n in _NAMES if n != nxt]):
                if cand_name == name:
                    continue
                combo = claim_candidates(players[cand_name].hand, t)
                if combo is not None:
                    claims.append((cand_name, combo))
                    break  # 只取第一个(杠/碰优先于吃)
            # 只在副露改善向听时才碰(真实玩家不碰坏牌效)
            claims = [cl for cl in claims
                      if _meld_improves(players[cl[0]].hand, cl[1])]
            if claims and rng.random() < 0.5:
                cname, combo = claims[0]
                cp = players[cname]
                for x in combo:  # 副露牌离开手牌(被碰张不在手牌中, 减成负即删)
                    cp.hand[x] -= 1
                    if cp.hand[x] <= 0:
                        del cp.hand[x]
                cp.melds.append(combo)
                for x in combo:
                    visible[x] += 1
                events.append(('meld', cname, str(t), ','.join(map(str, combo))))
                if len(events) in want:
                    snapshots[len(events)] = {n: Counter(pl.hand)
                                              for n, pl in players.items()}
                # 副露者接着打出一张
                available2 = {t: max(0, 4 - visible[t] - cp.hand.get(t, 0))
                              for t in range(34)}
                t2 = cp.discard(engine, available2)
                if t2 >= 0:
                    visible[t2] += 1
                    events.append(('discard', cname, str(t2)))
                    if len(events) in want:
                        snapshots[len(events)] = {n: Counter(pl.hand)
                                                  for n, pl in players.items()}
    return players, events, snapshots


def run_eval(n_games: int, n_steps: int, policy: str = 'engine',
             n_particles: int = 100) -> None:
    """N 局平均: 手牌质量 / 听牌校准 / waiting 命中 / 向听曲线。"""
    rules = get_rules('huanyu')
    engine = StrategyEngine(rules)
    checkpoints = list(range(5, n_steps + 1, 5))
    mass_by: dict[int, list[float]] = {c: [] for c in checkpoints}
    tenpai_buckets: list[tuple[float, int, int]] = []  # (prob, n, 实际听牌数)
    cal_by: dict[int, list[tuple[float, bool]]] = {c: [] for c in checkpoints}
    hit_by: dict[int, list[bool]] = {c: [] for c in checkpoints}
    shanten_by: dict[int, list[float]] = {c: [] for c in checkpoints}

    for g in range(n_games):
        rng = random.Random(100 + g)
        want = [c * 4 for c in checkpoints]
        players, events, snaps = simulate(rng, rules, engine, n_steps,
                                          policy=policy, checkpoints=want)
        obs = SoftObserver(rules, n_particles=n_particles, worker=False,
                           seed=200 + g)
        obs.reset(list(players['my_river'].hand.elements()), {}, {},
                  dict.fromkeys(OPPONENTS, 13))
        obs.apply_all()
        ev_i = 0
        for c in checkpoints:
            # 推进到检查点(批处理)
            while ev_i < len(events) and ev_i < c * 4:
                ev = events[ev_i]
                ev_i += 1
                if ev[0] == 'discard':
                    _kind, name, t = ev
                    obs.push_observation(ev_i, int(t), {name: 1.0})
                else:
                    _kind, name, t, combo_s = ev
                    combo = tuple(int(x) for x in combo_s.split(','))
                    obs.push_meld(MeldObs(meld_id=ev_i, tiles=combo,
                                          claimed_tile=int(t),
                                          melder=name))
            obs.apply_all()
            inf = obs.cached_inference()
            assert inf is not None
            hand_now = snaps.get(c * 4, {})  # 该检查点的真手牌快照
            for name in _NAMES:  # 向听曲线(四家真手牌; 越低越好)
                h = hand_now.get(name, Counter())
                if not h:  # 墙摸尽后的空手(终局极端副露)
                    continue
                shanten_by[c].append(float(calculate_shanten(
                    sorted(h.elements()))))
            for p in OPPONENTS:
                pi = players[p]
                true_hand = hand_now.get(p, Counter())
                dist = obs._infers[p].hand_distribution()  # noqa: SLF001
                if not true_hand:
                    continue  # 空手(终局极端副露)不参与手牌质量统计
                true_tiles = true_hand.elements()
                n_true = sum(true_hand.values())
                mass = sum(dist.get(t, 0.0) for t in true_tiles) / n_true
                mass_by[c].append(mass)
                # 实际听牌判定与系统同口径: 真手牌 + 副露展开 = 13
                expansion = [x for m in pi.melds for x in m[:3]]
                try:
                    actual_ws = rules.get_waiting_tiles(
                        Hand(sorted(true_hand.elements()) + expansion))
                except ValueError:
                    continue  # 张数不齐(摸打中间态) → 此检查点不评听牌
                prob = inf.tenpai_probs.get(p, 0.0)
                tenpai_buckets.append((prob, 1, int(bool(actual_ws))))
                cal_by[c].append((prob, bool(actual_ws)))
                if actual_ws:
                    actual = {w.tile for w in actual_ws}
                    top3 = set(inf.waiting.get(p, []))
                    hit_by[c].append(bool(top3 & actual))
        print(f'  game {g + 1}/{n_games} done')

    print('\n=== 手牌质量(真手牌每张的平均推断期望张数; 随机基线 ≈ '
          '手牌数/池 ≈ 0.11, 完美 = 1.0) ===')
    for c in checkpoints:
        m = mass_by[c]
        if not m:  # 墙摸尽(约 85-110 事件)后该检查点无样本
            print(f'  出牌事件 ~{c * 4:3d} 后: (无样本)')
            continue
        print(f'  出牌事件 ~{c * 4:3d} 后: 平均 {sum(m) / len(m):.2f}')

    print('\n=== 听牌校准(推断概率分桶 vs 实际听牌比例) ===')
    buckets: dict[str, list[tuple[int, int]]] = {}
    for prob, n, ten in tenpai_buckets:
        b = f'{prob // 0.2 * 0.2:.1f}-{prob // 0.2 * 0.2 + 0.2:.1f}'
        buckets.setdefault(b, []).append((n, ten))
    for b in sorted(buckets):
        n = sum(x[0] for x in buckets[b])
        ten = sum(x[1] for x in buckets[b])
        print(f'  [{b}]: n={n:4d}  实际听牌 {ten / n:.0%}')

    print('\n=== 逐检查点校准(预测听牌概率均值 vs 实际听牌比例) ===')
    for c in checkpoints:
        rows = cal_by[c]
        if not rows:
            print(f'  出牌事件 ~{c * 4:3d} 后: (无样本)')
            continue
        mean_p = sum(p for p, _ in rows) / len(rows)
        actual = sum(1 for _, ten in rows if ten) / len(rows)
        print(f'  出牌事件 ~{c * 4:3d} 后: 预测均值 {mean_p:.0%}  实际 {actual:.0%}'
              f'  (n={len(rows)})')

    print('\n=== waiting 命中(实际听牌时, 真等待牌在推断 top3 的比例) ===')
    for c in checkpoints:
        hits = hit_by[c]
        if hits:
            print(f'  出牌事件 ~{c * 4:3d} 后: {sum(hits) / len(hits):.0%} '
                  f'(n={len(hits)})')

    print('\n=== 平均向听曲线(四家真手牌; 越低越好 — 出牌算法 A/B) ===')
    for c in checkpoints:
        vals = shanten_by[c]
        if not vals:
            print(f'  出牌事件 ~{c * 4:3d} 后: (无样本)')
            continue
        print(f'  出牌事件 ~{c * 4:3d} 后: 平均 {sum(vals) / len(vals):.2f}')


def main() -> None:
    parser = argparse.ArgumentParser(description='推断准确性离线评估')
    parser.add_argument('--games', type=int, default=10)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--policy', choices=('engine', 'ukeire'),
                        default='engine', help='模拟玩家出牌算法')
    parser.add_argument('--particles', type=int, default=400,
                        help='每对手粒子数(生产默认 400, 评估对齐生产)')
    args = parser.parse_args()
    run_eval(args.games, args.steps, args.policy, args.particles)


if __name__ == '__main__':
    main()
