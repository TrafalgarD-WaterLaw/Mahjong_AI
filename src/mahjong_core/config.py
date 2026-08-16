"""牌库配置 — 不可变白名单配置。"""

from dataclasses import dataclass, field

from src.mahjong_core.tile import (
    BING_TILES,
    FENG_TILES,
    HUA_TILES,
    JIAN_TILES,
    TIAO_TILES,
    WAN_TILES,
)


@dataclass(frozen=True)
class TileSetConfig:
    """麻将牌库配置。白名单方式，显式声明启用哪些牌。不可变。"""

    enabled_tiles: frozenset[int]
    tile_count: int = 4
    wild_tiles: frozenset[int] = field(default_factory=frozenset)
    _total_override: int | None = field(default=None, repr=False)

    @property
    def total_tiles(self) -> int:
        """牌库总张数。"""
        if self._total_override is not None:
            return self._total_override
        return len(self.enabled_tiles) * self.tile_count

    def contains(self, tile: int) -> bool:
        """判断某牌是否在牌库中。"""
        return tile in self.enabled_tiles

    @staticmethod
    def standard_136() -> 'TileSetConfig':
        """万+筒+条+风+箭，各4张 = 136张 (陕西推倒胡)"""
        tiles = (set(WAN_TILES) | set(BING_TILES) | set(TIAO_TILES)
                 | set(FENG_TILES) | set(JIAN_TILES))
        return TileSetConfig(enabled_tiles=frozenset(tiles), tile_count=4)

    @staticmethod
    def standard_108() -> 'TileSetConfig':
        """仅万+筒+条，各4张 = 108张 (四川血战)"""
        tiles = set(WAN_TILES) | set(BING_TILES) | set(TIAO_TILES)
        return TileSetConfig(enabled_tiles=frozenset(tiles), tile_count=4)

    @staticmethod
    def standard_144() -> 'TileSetConfig':
        """34种*4 + 花牌8种*1 = 144张 (广东麻将)"""
        tiles = (set(WAN_TILES) | set(BING_TILES) | set(TIAO_TILES)
                 | set(FENG_TILES) | set(JIAN_TILES) | set(HUA_TILES))
        return TileSetConfig(
            enabled_tiles=frozenset(tiles),
            tile_count=4,
            _total_override=144,
        )
