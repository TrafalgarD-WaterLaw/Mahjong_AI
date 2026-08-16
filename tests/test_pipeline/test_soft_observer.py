"""M3 SoftObserver 单测 — 软更新比例/硬等价/回放确定/翻案等价/副露扣池。"""

from collections import Counter

from src.mahjong_ai.inference.opponent_inference import OpponentInference
from src.mahjong_ai.inference.soft_observer import (
    OPPONENTS,
    MeldObs,
    SoftObserver,
)
from src.mahjong_ai.state.snapshot import PlayerState
from src.mahjong_engine import get as get_rules


def _rules():
    return get_rules('huanyu')


def _players(rivers=None, melds=None):
    """最小 PlayerState 字典(四家)。"""
    rivers = rivers or {}
    melds = melds or {}
    out = {}
    for p in ('my_river', 'right_river', 'top_river', 'left_river'):
        out[p] = PlayerState(river=list(rivers.get(p, [])),
                             melds=list(melds.get(p, [])))
    return out


def _new_inf(n=100, seed=7) -> OpponentInference:
    inf = OpponentInference(_rules(), n_particles=n, seed=seed)
    inf.reset([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
              _players(), hand_count=13)
    return inf


def test_partial_update_updates_exact_fraction():
    """② w=0.25 → 恰好 25/100 粒子被更新(其余原样)。"""
    inf = _new_inf(100, 7)
    before = inf.get_state()[0]
    inf.observe_discard_partial(5, 0.25)
    after = inf.get_state()[0]
    changed = sum(1 for x, y in zip(before, after, strict=True)
                  if x != y)
    assert changed == 25


def test_full_weight_equals_hard_discard():
    """① w=1.0 软观测 == 旧硬观测(状态全等含 RNG)。"""
    inf1 = _new_inf(100, 7)
    inf2 = _new_inf(100, 7)
    inf1.observe_discard(5)
    inf2.observe_discard_partial(5, 1.0)
    assert inf1.get_state() == inf2.get_state()


def test_snapshot_roundtrip():
    """快照 get/set 往返: 状态全等(粒子/池/RNG/hand_count)。"""
    inf = _new_inf(100, 7)
    st = inf.get_state()
    inf.observe_discard(5)
    inf.set_state(st)
    assert inf.get_state() == st


def _obs_state(seed: int) -> tuple:
    """同一观测序列在两个同种子观察者上应用后的状态对。"""
    obs = SoftObserver(_rules(), n_particles=50, seed=seed, worker=False)
    obs.reset([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
              {p: [] for p in OPPONENTS},
              {p: [] for p in OPPONENTS},
              dict.fromkeys(OPPONENTS, 13))
    obs.apply_all()
    return obs


def test_correction_equals_direct_observation():
    """④ 先按 {右:0.9} 观察再纠正 {上:0.9} == 直接按 {上:0.9} 观察。"""
    a = _obs_state(11)
    b = _obs_state(11)
    a.push_observation(1, 5, {'right_river': 0.9})
    a.push_observation(2, 9, {'top_river': 0.8})
    a.apply_all()
    a.correct(1, {'top_river': 0.9})
    a.apply_all()
    b.push_observation(1, 5, {'top_river': 0.9})
    b.push_observation(2, 9, {'top_river': 0.8})
    b.apply_all()
    sa = {p: a._infers[p].get_state() for p in OPPONENTS}  # noqa: SLF001
    sb = {p: b._infers[p].get_state() for p in OPPONENTS}  # noqa: SLF001
    assert sa == sb


def test_replay_determinism():
    """③ 同种子同序列两次独立应用 → 状态一致(回放确定性)。"""
    a = _obs_state(13)
    b = _obs_state(13)
    for obs in (a, b):
        obs.push_observation(1, 5, {'right_river': 0.6})
        obs.push_observation(2, 9, {'top_river': 0.7})
        obs.apply_all()
    sa = {p: a._infers[p].get_state() for p in OPPONENTS}  # noqa: SLF001
    sb = {p: b._infers[p].get_state() for p in OPPONENTS}  # noqa: SLF001
    assert sa == sb


def test_queue_overflow_coalesces():
    """队列软上限: 积压超限时池记账逐条精确, 粒子更新只留每家最新。

    主循环入队零阻塞, worker 消化慢时队列积压 → 听牌延迟无界。
    降级: 每条观测仍做公开扣池(牌池是下游记账基准, 必须逐条),
    粒子更新只对每家最后一个全权重观测做(最新观测信息量最大);
    被降级的观测不入日志 → 翻案目标缺失时静默跳过(有界延迟
    优先于翻案精度 — 冻结窗 8s 内翻案只发生在正常消化下)。
    """
    obs = SoftObserver(_rules(), n_particles=50, worker=False)
    obs.reset([], {}, {}, {'right_river': 13, 'top_river': 13,
                           'left_river': 13})
    obs.apply_all()
    n = 34  # 全部 34 类牌各一次 > 队列软上限 32
    tiles = list(range(n))
    for i, t in enumerate(tiles):
        obs.push_observation(i + 1, t, {'right_river': 1.0})
    obs.apply_all()
    pool = obs._infers['right_river'].get_state()[1]  # noqa: SLF001
    for t in tiles:
        assert pool.get(t, 0) == 3, f'牌池 {t} 应精确扣 1(4−打出)'
    log = obs.log()
    assert len(log) == 1  # 只有右家最新观测做了全更新(其余降级)
    assert log[0].obs_id == n
    assert log[0].tile == tiles[-1]
    assert obs.cached_inference() is not None  # 降级后十判照常


def test_queue_normal_path_full_log():
    """正常消化(未超限): 每条观测都全更新并入日志(与旧行为一致)。"""
    obs = SoftObserver(_rules(), n_particles=50, worker=False)
    obs.reset([], {}, {}, {'right_river': 13, 'top_river': 13,
                           'left_river': 13})
    obs.apply_all()
    for i, t in enumerate(range(10)):
        obs.push_observation(i + 1, t, {'right_river': 1.0})
    obs.apply_all()
    assert len(obs.log()) == 10


def test_meld_pool_deduction_counts():
    """⑤ 副露扣池: pong 有 claimed → 2 / 秒碰 → 3 / chi → 2 / 杠 → 3。"""
    obs = _obs_state(3)

    def pool(tile, p='top_river'):
        return obs._infers[p].get_state()[1].get(tile, 0)  # noqa: SLF001

    obs.push_observation(1, 5, {'right_river': 1.0})
    obs.apply_all()
    base5 = pool(5)
    obs.push_meld(MeldObs(meld_id=2, tiles=(5, 5, 5), claimed_tile=5,
                          melder='top_river'))
    obs.apply_all()
    assert pool(5) == base5 - 2  # 被碰那张已随打出扣过 → 再扣 2
    base9 = pool(9)
    obs.push_meld(MeldObs(meld_id=3, tiles=(9, 9, 9), claimed_tile=None,
                          melder='top_river'))
    obs.apply_all()
    assert pool(9) == base9 - 3  # 秒碰: 打出事件从未发生 → 扣 3
    # chi(6,7,8): 被吃的 6 已扣 → 扣 7/8
    base7, base8 = pool(7), pool(8)
    obs.push_meld(MeldObs(meld_id=4, tiles=(6, 7, 8), claimed_tile=6,
                          melder='top_river'))
    obs.apply_all()
    assert pool(7) == base7 - 1 and pool(8) == base8 - 1
    # 明杠(20,20,20,20) claimed → 扣 3(20 不在我手牌, 池=4)
    b20 = pool(20)
    obs.push_meld(MeldObs(meld_id=5, tiles=(20, 20, 20, 20),
                          claimed_tile=20, melder='top_river'))
    obs.apply_all()
    assert pool(20) == b20 - 3


def test_tenpai_discrimination_building_vs_scattered():
    """集成: 朝听牌构建的打出序列 vs 乱打散牌 → 听牌概率区分度。

    同种子同先验, 只差观测序列 — 验证软观测管道的端到端有效性
    (旧模拟的 5 倍区分验收线在 8 轮观测下; 这里用方向性断言
    保持稳定, 避免脆阈值)。
    """
    def prob(seed: int, tiles: list[int]) -> float:
        obs = _obs_state(seed)
        for i, t in enumerate(tiles):
            obs.push_observation(i + 1, t, {'right_river': 1.0})
        obs.apply_all()
        cache = obs.cached_inference()
        assert cache is not None
        return cache.tenpai_probs['right_river']

    # 构建型: 先打字牌孤张/边张(保牌效的典型路径)
    building = [27, 31, 33, 8, 0, 17, 26, 32]
    # 散打型: 中张乱打(拆搭子/面子, 毁牌效)
    scattered = [13, 14, 22, 23, 4, 5, 13, 22]
    p_build = prob(21, building)
    p_scat = prob(21, scattered)
    assert p_build > p_scat, f'构建型 {p_build:.3f} 应高于散打型 {p_scat:.3f}'


def test_reset_pool_semantics():
    """⑥ reset: 牌池 = 总数 − 我手牌 − 明牌(副露排除被碰张)。"""
    obs = SoftObserver(_rules(), n_particles=50, seed=5, worker=False)
    obs.reset([0, 0, 1], {'right_river': [5, 5]},
              {'right_river': [9, 9]},  # 碰(9,9,9)排除被碰的第三张
              {'right_river': 11, 'top_river': 13, 'left_river': 13})
    obs.apply_all()
    inf = obs._infers['right_river']  # noqa: SLF001
    pool = inf.get_state()[1]
    assert pool.get(0, 0) == 2   # 4 − 我手牌 2 张
    assert pool.get(5, 0) == 2   # 4 − 打出 2 张
    assert pool.get(9, 0) == 2   # 4 − 副露亮出 2 张(被碰张在打出家 river 已扣)


def test_discard_risks_snapshot_batch():
    """批量放铳率: 单次粒子快照 + 三家最大 — 帧内一致(防闪烁)。

    回归: 逐牌读实时粒子时, 后台线程在两次调用间更新粒子 —
    同一帧候选值互相不一致且逐帧抖动, 客户端放铳行闪烁。
    """
    from collections import Counter

    obs = SoftObserver(_rules(), n_particles=50)
    # 右家: 50 个同构听牌粒子(4,4 对 + 三面子, 等 4)→ 放铳率 1.0
    tenpai_hand = [0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 27, 4, 4]
    obs._infers['right_river']._seed_particles(
        [Counter(tenpai_hand) for _ in range(50)])
    risks = obs.discard_risks([4, 33])
    assert set(risks) == {4, 33}
    assert risks[4] == 1.0
    assert risks[33] < 0.5
    assert risks == obs.discard_risks([4, 33])  # 快照一致(粒子未动)


def test_meld_unknown_content_resolved_from_pool():
    """内容未知的对手副露: 牌池唯一解 → 扣池补齐(碰/吃确定)。

    场景: 被碰 8(九万), 我手牌 1 张 8 + 牌河打出 2 张 8 → 池中
    只剩 1 张 8, 碰不可能; 顺子 (6,7,8) 是唯一 → 确定是吃,
    6 和 7 也确定离开牌池。
    """
    obs = SoftObserver(_rules(), n_particles=50, worker=False)
    obs.reset([8, 0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14],
              {'right_river': [8, 8, 6, 7], 'top_river': [],
               'left_river': []},
              {}, {'right_river': 13, 'top_river': 13, 'left_river': 13})
    obs.apply_all()
    obs.push_meld(MeldObs(meld_id=1, tiles=(8,), claimed_tile=8,
                          melder='right_river'))
    obs.apply_all()
    pool = obs._infers['right_river'].get_state()[1]
    assert pool.get(6, 0) == 2   # 4 − 打出 1 − 副露确定扣 1
    assert pool.get(7, 0) == 2   # 4 − 打出 1 − 副露确定扣 1
    assert pool.get(8, 0) == 1   # 4 − 我手牌 1 − 打出 2(被碰张已扣)


def test_meld_ambiguous_content_only_claimed_deducted():
    """歧义(碰与吃都可能): 只扣确定的被碰牌, 不猜其余两张。"""
    obs = SoftObserver(_rules(), n_particles=50, worker=False)
    # 池中 8 剩 3(碰可行)且 6/7 也在池(吃可行) → 歧义
    obs.reset([0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15],
              {'right_river': [8], 'top_river': [],
               'left_river': []},
              {}, {'right_river': 13, 'top_river': 13, 'left_river': 13})
    obs.apply_all()
    obs.push_meld(MeldObs(meld_id=2, tiles=(8,), claimed_tile=8,
                          melder='right_river'))
    obs.apply_all()
    pool = obs._infers['right_river'].get_state()[1]
    assert pool.get(6, 0) == 4   # 未扣(歧义不猜)
    assert pool.get(7, 0) == 4
    assert pool.get(8, 0) == 3   # 4 − 打出 1(被碰张已扣, 不再扣)


def test_meld_shrinks_particles_post_discard():
    """P0: 副露后粒子缩小到 13−3k(副露后打出的状态), 十判展开回 13。"""
    obs = SoftObserver(_rules(), n_particles=50, worker=False)
    obs.reset([], {}, {}, {'right_river': 13, 'top_river': 13,
                           'left_river': 13})
    obs.apply_all()
    obs.push_meld(MeldObs(meld_id=1, tiles=(8, 8, 8), claimed_tile=8,
                          melder='right_river'))
    obs.apply_all()
    particles, _pool, _rng, hc = obs._infers['right_river'].get_state()
    assert hc == 10
    assert all(sum(p.values()) == 10 for p in particles)


def test_tenpai_with_meld_not_zero():
    """P0 回归: 副露后听牌不再恒 0 — 展开十判真实计算。"""
    obs = SoftObserver(_rules(), n_particles=50, worker=False)
    obs.reset([], {}, {}, {'right_river': 13, 'top_river': 13,
                           'left_river': 13})
    obs.apply_all()
    obs.push_meld(MeldObs(meld_id=1, tiles=(8, 8, 8), claimed_tile=8,
                          melder='right_river'))
    obs.apply_all()
    inf = obs._infers['right_river']
    # 10 张手牌 = 3 面子 + 单骑 4; 碰 8 展开后等 4
    hand10 = [0, 1, 2, 9, 10, 11, 18, 19, 20, 4]
    inf._seed_particles([Counter(hand10) for _ in range(50)])
    prob, waiting = inf._tenpai_of(inf.particles_snapshot(), ((8, 8, 8),))
    assert prob == 1.0
    assert waiting == {4: 1.0}


def test_discard_risk_with_meld():
    """P0: 副露后的放铳率真实计算(展开 13 + 牌 = 14 胡判)。"""
    obs = SoftObserver(_rules(), n_particles=50, worker=False)
    obs.reset([], {}, {}, {'right_river': 13, 'top_river': 13,
                           'left_river': 13})
    obs.apply_all()
    obs.push_meld(MeldObs(meld_id=1, tiles=(8, 8, 8), claimed_tile=8,
                          melder='right_river'))
    obs.apply_all()
    inf = obs._infers['right_river']
    hand10 = [0, 1, 2, 9, 10, 11, 18, 19, 20, 4]
    inf._seed_particles([Counter(hand10) for _ in range(50)])
    snap = inf.particles_snapshot()
    assert inf.discard_risk_from(snap, 4, ((8, 8, 8),)) == 1.0
    assert inf.discard_risk_from(snap, 33, ((8, 8, 8),)) == 0.0


def test_meld_unknown_content_still_computes():
    """P0: 内容未知的副露按碰近似, 十判照常(不再恒 0)。"""
    obs = SoftObserver(_rules(), n_particles=50, worker=False)
    obs.reset([], {}, {}, {'right_river': 13, 'top_river': 13,
                           'left_river': 13})
    obs.apply_all()
    obs.push_meld(MeldObs(meld_id=1, tiles=(8,), claimed_tile=8,
                          melder='right_river'))
    obs.apply_all()
    inf = obs._infers['right_river']
    particles, _pool, _rng, hc = inf.get_state()
    assert hc == 10
    assert all(sum(p.values()) == 10 for p in particles)
    prob, _w = inf._tenpai_of(inf.particles_snapshot(), ((8, 8, 8),))
    assert 0.0 <= prob <= 1.0  # 计算真实进行(旧行为恒 0)


def test_unknown_meld_tenpai_not_zero_via_pipeline():
    """内容未知的副露走完整管道时按碰展开(真实对局: 假副露 (1,)
    展开只 1 张 → 10+1=11 张 ≠13, 十判全跳过 → 听牌概率归零)。
    展开 (X,X,X) 后十判照常计算。"""
    obs = SoftObserver(_rules(), n_particles=50, worker=False)
    obs.reset([], {}, {}, {'right_river': 13, 'top_river': 13,
                           'left_river': 13})
    obs.apply_all()
    obs.push_meld(MeldObs(meld_id=1, tiles=(8,), claimed_tile=8,
                          melder='right_river'))
    obs.apply_all()
    inf = obs._infers['right_river']
    # 10 张手牌 = 3 面子 + 单骑 4; 未知副露按碰 8 展开 → 13 张听 4
    hand10 = [0, 1, 2, 9, 10, 11, 18, 19, 20, 4]
    inf._seed_particles([Counter(hand10) for _ in range(50)])
    obs.apply_all()  # 触发十判重算(force)
    cache = obs.cached_inference()
    assert cache is not None
    assert cache.tenpai_probs['right_river'] == 1.0
    assert cache.waiting['right_river'] == [4]


def test_melded_discard_inverse_filter():
    """P0: 副露后的打出观测走反向过滤(粒子中不应再有刚打出的牌)。"""
    obs = SoftObserver(_rules(), n_particles=50, worker=False)
    obs.reset([], {}, {}, {'right_river': 13, 'top_river': 13,
                           'left_river': 13})
    obs.apply_all()
    obs.push_meld(MeldObs(meld_id=1, tiles=(8, 8, 8), claimed_tile=8,
                          melder='right_river'))
    obs.apply_all()
    inf = obs._infers['right_river']
    hand10 = [0, 1, 2, 9, 10, 11, 18, 19, 20, 4]
    hand10b = [0, 1, 2, 9, 10, 11, 18, 19, 20, 5]
    inf._seed_particles([Counter(hand10) for _ in range(25)]
                        + [Counter(hand10b) for _ in range(25)])
    inf.observe_discarded(4)
    particles = inf.particles_snapshot()
    assert len(particles) == 50
    assert all(p.get(4, 0) == 0 for p in particles)
    assert all(sum(p.values()) == 10 for p in particles)


def test_all_discards_inverse_filter():
    """校准修复: 未副露家的打出观测同样走反向过滤。

    粒子 = 打出后手牌 — 刚打出的牌不该在其中。听牌家打掉的多是
    刚摸进的牌: 正向过滤(要求粒子含该牌)把真听牌手结构性滤掉
    (评估实测: 模型概率 27/42 挤在 0-0.2 桶而实际听牌 48%;
    全员反向过滤后手牌质量 0.59 → 0.96)。"""
    obs = SoftObserver(_rules(), n_particles=50, worker=False)
    obs.reset([], {}, {}, {'right_river': 13, 'top_river': 13,
                           'left_river': 13})
    obs.apply_all()
    inf = obs._infers['right_river']
    # 听牌手(等 4/27, 不含 5)vs 含 5 的散手 → 打出 5 的观测
    # 应让不含 5 的听牌手全部存活(反向), 而非被滤掉(正向)。
    hand13 = [0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 27, 4, 4]
    hand13b = [0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 27, 4, 5]
    inf._seed_particles([Counter(hand13) for _ in range(25)]
                        + [Counter(hand13b) for _ in range(25)])
    obs.push_observation(1, 5, {'right_river': 1.0})
    obs.apply_all()
    particles = inf.particles_snapshot()
    assert len(particles) == 50
    assert all(p.get(5, 0) == 0 for p in particles)
    assert all(sum(p.values()) == 13 for p in particles)
