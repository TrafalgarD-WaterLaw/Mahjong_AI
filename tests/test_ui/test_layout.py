"""布局规划纯函数单测 — 三档/降级链/任意尺寸不变量。"""

from src.mahjong_ui.layout import (
    MIN_H, MIN_W, band_fit, fit_count, plan_layout,
)


def _inside(rect, w, h) -> bool:
    x, y, rw, rh = rect
    return rw > 0 and rh > 0 and x >= 0 and y >= 0 \
        and x + rw <= w and y + rh <= h


def _overlap(a, b) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def test_fit_count_exact_fit():
    assert fit_count(86, 20, 2, 4) == (4, 0)   # 4×20+3×2=86


def test_fit_count_partial_and_rest():
    assert fit_count(50, 20, 2, 5) == (2, 3)


def test_fit_count_zero_guards():
    assert fit_count(0, 20, 2, 5) == (0, 5)
    assert fit_count(100, 0, 2, 5) == (0, 5)
    assert fit_count(100, 20, 2, 0) == (0, 0)


def test_band_fit_full_and_capped():
    # 放得下: 全部画, 无余数
    assert band_fit(120, 16, 2, 3) == (3, 0)
    # 放不下: 截断并给 +N 徽章让位(最后不足 26px 时再退一张)
    assert band_fit(50, 16, 2, 4) == (1, 3)  # 2 张=34px 余 14 < 26 → 退 1
    assert band_fit(46, 16, 2, 4) == (1, 3)  # 2 张=34px 余 12 < 26 → 退 1
    # 极窄: 一张都放不下(含 +N 预留) → 全记为余数
    assert band_fit(20, 16, 2, 3) == (0, 3)


def test_tier_thresholds():
    assert plan_layout(100, 100).tier == 'narrow'   # 入口钳制后仍窄
    assert plan_layout(419, 480).tier == 'narrow'
    assert plan_layout(420, 480).tier == 'medium'
    assert plan_layout(699, 480).tier == 'medium'
    assert plan_layout(700, 480).tier == 'wide'
    assert plan_layout(2000, 1200).tier == 'wide'


def test_narrow_structure():
    plan = plan_layout(380, 480)
    r = plan.rects
    assert set(r) >= {'opp_right', 'opp_top', 'opp_left', 'advice',
                      'my_hand', 'my_risk', 'my_meld'}
    # 对手条按序纵排, 其后推荐/手牌/放铳/副露依次
    assert r['opp_right'][1] < r['opp_top'][1] < r['opp_left'][1] \
        < r['advice'][1] < r['my_hand'][1] < r['my_risk'][1] < r['my_meld'][1]
    assert plan.show_waiting is False  # 窄档降级链 1: 听牌组合省略
    assert 'legend' not in r           # 窄档无图例(对手条有标签)
    assert 'my_river' in r             # 合法高度下窄档余量恒 ≥60
    assert plan.fonts == {'base': 9, 'small': 8, 'title': 10}


def test_medium_structure_and_legend_flag():
    plan = plan_layout(430, 600)
    r = plan.rects
    assert set(r) >= {'top', 'left', 'right', 'advice', 'my_hand',
                      'my_risk', 'my_meld', 'my_river'}
    assert 'legend' not in r  # W < 560 不画图例(降级链 2)
    assert plan.show_waiting is True
    assert plan.fonts == {'base': 10, 'small': 8, 'title': 11}
    assert 'legend' in plan_layout(560, 600).rects


def test_wide_all_shown():
    plan = plan_layout(900, 700)
    assert 'legend' in plan.rects
    assert plan.show_waiting is True
    assert plan.fonts == {'base': 12, 'small': 10, 'title': 14}


def test_must_keep_positive_at_min_size():
    plan = plan_layout(MIN_W, MIN_H)
    for k in ('my_hand', 'my_risk', 'advice'):
        x, y, w, h = plan.rects[k]
        assert w > 0 and h > 0, k
    assert plan.rects['opp_right'][3] > 0


def test_sweep_no_overflow_no_overlap():
    """任意尺寸: 所有区域不越界、两两不重叠(核心不变量)。"""
    for w in range(360, 2001, 80):
        for h in range(480, 1201, 80):
            plan = plan_layout(w, h)
            rects = list(plan.rects.values())
            for rect in rects:
                assert _inside(rect, w, h), f'{w}x{h}: {rect} 越界'
            for i, a in enumerate(rects):
                for b in rects[i + 1:]:
                    assert not _overlap(a, b), f'{w}x{h}: {a} 与 {b} 重叠'
