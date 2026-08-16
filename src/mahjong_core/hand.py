"""手牌类 — 排序列表表示，模拟玩家理牌行为。"""

import bisect
from collections.abc import Iterator


class Hand:
    """麻将手牌。内部使用排序列表存储牌整数。"""

    def __init__(self, tiles: list[int] | None = None) -> None:
        self._tiles: list[int] = sorted(tiles) if tiles else []

    def add(self, tile: int) -> None:
        """加入一张牌，保持排序顺序。"""
        bisect.insort(self._tiles, tile)

    def discard(self, tile: int) -> None:
        """打出一张牌。Raises ValueError 如果牌不在手牌中。"""
        try:
            self._tiles.remove(tile)
        except ValueError:
            raise ValueError(f'牌 {tile} not in hand') from None

    def count(self, tile: int) -> int:
        """返回手牌中某张牌的数量。"""
        return self._tiles.count(tile)

    def __len__(self) -> int:
        return len(self._tiles)

    def __getitem__(self, index: int) -> int:
        return self._tiles[index]

    def __iter__(self) -> Iterator[int]:
        return iter(self._tiles)

    def __contains__(self, tile: int) -> bool:
        return tile in self._tiles

    def __repr__(self) -> str:
        return f'Hand({self._tiles})'
