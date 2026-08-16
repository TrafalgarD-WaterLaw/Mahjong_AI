"""证据模型单测 — 种子高斯/在线学习。"""

from src.mahjong_ai.pipeline.decoder import EvidenceModel

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


def test_unseeded_spatial_neutral_and_no_learning():
    """未播种: 空间证据中性(0), 不学习 — 防原点高斯退化(真实对局 bug 防御)。"""
    ev = EvidenceModel()
    assert ev.spatial_logp('my_river', 700.0, 400.0) == 0.0
    assert ev.spatial_logp('right_river', 700.0, 400.0) == 0.0
    ev.update('my_river', 700.0, 400.0)   # 未播种不学习
    assert ev.spatial_logp('my_river', 700.0, 400.0) == 0.0
    ev.reseed(None, 1280, 800)            # 播种后恢复证据
    assert ev.spatial_logp('my_river', 700.0, 400.0) != 0.0
