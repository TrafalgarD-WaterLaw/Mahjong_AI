"""TileSetConfig 牌库配置的单元测试。"""
from dataclasses import FrozenInstanceError

import pytest

from src.mahjong_core.config import TileSetConfig
from src.mahjong_core.tile import DONG, MEI, W1, W9, ZHONG


class TestTileSetConfigBasic:
    def test_empty_config(self):
        cfg = TileSetConfig(enabled_tiles=frozenset())
        assert cfg.total_tiles == 0

    def test_contains(self):
        cfg = TileSetConfig(enabled_tiles=frozenset({W1, DONG}))
        assert cfg.contains(W1) is True
        assert cfg.contains(W9) is False

    def test_custom_tile_count(self):
        cfg = TileSetConfig(enabled_tiles=frozenset({W1, W9}), tile_count=3)
        assert cfg.total_tiles == 6

    def test_immutable(self):
        cfg = TileSetConfig(enabled_tiles=frozenset({W1}))
        with pytest.raises(FrozenInstanceError):
            cfg.enabled_tiles = frozenset()  # type: ignore


class TestWildTiles:
    def test_default_no_wild(self):
        cfg = TileSetConfig(enabled_tiles=frozenset({W1, DONG}))
        assert cfg.wild_tiles == frozenset()

    def test_with_wild_tiles(self):
        cfg = TileSetConfig(
            enabled_tiles=frozenset({W1, DONG, ZHONG}),
            wild_tiles=frozenset({ZHONG}),
        )
        assert ZHONG in cfg.wild_tiles
        assert W1 not in cfg.wild_tiles


class TestStandardConfigs:
    def test_standard_136(self):
        cfg = TileSetConfig.standard_136()
        assert cfg.total_tiles == 136
        assert len(cfg.enabled_tiles) == 34
        assert not cfg.contains(MEI)

    def test_standard_108(self):
        cfg = TileSetConfig.standard_108()
        assert cfg.total_tiles == 108
        assert len(cfg.enabled_tiles) == 27
        assert not cfg.contains(DONG)
        assert not cfg.contains(ZHONG)

    def test_standard_144(self):
        cfg = TileSetConfig.standard_144()
        assert cfg.total_tiles == 144
