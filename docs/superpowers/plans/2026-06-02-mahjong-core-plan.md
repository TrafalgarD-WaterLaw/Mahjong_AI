# mahjong_core 实现计划

> **面向执行代理**: 使用 superpowers:subagent-driven-development 或 superpowers:executing-plans 按任务逐步实现。步骤使用 `- [ ]` 追踪。

**目标**: 实现 mahjong_core 模块——麻将AI项目的基础数据类型，零外部依赖。

**架构**: 三文件 + 一导出。整数编码表示牌型(0-41)，排序列表表示手牌，不可变白名单配置表示牌库。

**技术栈**: Python 3.11, 仅标准库(dataclass, bisect), pytest, mypy, ruff

**TDD 流程**: 每个任务: 写测试 → 确认失败(RED) → 写实现 → 确认通过(GREEN) → 提交

---

### 任务 1: 牌常量与辅助函数

**文件:**
- 创建: `src/mahjong_core/tile.py`

- [ ] **步骤 1: 创建 tile.py**

```python
"""麻将牌常量定义与辅助函数。

牌使用整数编码，直接对应 ML 模型的分类输出:
    0-8:   万牌 (W1-W9)
    9-17:  筒牌 (T1-T9)
    18-26: 条牌 (B1-B9)
    27-30: 风牌 (东南西北)
    31-33: 箭牌 (中发白)
    34-41: 花牌 (预留,未实现)
"""

from typing import Optional

TOTAL_TILES = 42

# 花色常量
WAN = 0
TONG = 1
TIAO = 2
FENG = 3
JIAN = 4
HUA = 5

# 具体牌常量
W1, W2, W3, W4, W5, W6, W7, W8, W9 = 0, 1, 2, 3, 4, 5, 6, 7, 8
T1, T2, T3, T4, T5, T6, T7, T8, T9 = 9, 10, 11, 12, 13, 14, 15, 16, 17
B1, B2, B3, B4, B5, B6, B7, B8, B9 = 18, 19, 20, 21, 22, 23, 24, 25, 26
DONG, NAN, XI, BEI = 27, 28, 29, 30
ZHONG, FA, BAI = 31, 32, 33
# 花牌预留 (Phase 1 不实现显示名)
MEI, LAN, ZHU, JU, CHUN, XIA, QIU, DONG_HUA = 34, 35, 36, 37, 38, 39, 40, 41

# 牌组范围
WAN_TILES = range(0, 9)
TONG_TILES = range(9, 18)
TIAO_TILES = range(18, 27)
FENG_TILES = range(27, 31)
JIAN_TILES = range(31, 34)
HUA_TILES = range(34, 42)

_NUMBER_NAMES = ["一", "二", "三", "四", "五", "六", "七", "八", "九"]
_HONOR_NAMES = {27: "东", 28: "南", 29: "西", 30: "北", 31: "中", 32: "发", 33: "白"}


def tile_suit(tile: int) -> int:
    """返回牌的花色常量。"""
    if tile in WAN_TILES:   return WAN
    if tile in TONG_TILES:  return TONG
    if tile in TIAO_TILES:  return TIAO
    if tile in FENG_TILES:  return FENG
    if tile in JIAN_TILES:  return JIAN
    if tile in HUA_TILES:   return HUA
    raise ValueError(f"非法牌值: {tile}")


def tile_number(tile: int) -> Optional[int]:
    """返回数牌的数字(1-9)，字牌和花牌返回 None。"""
    if tile in WAN_TILES:   return (tile - 0) + 1
    if tile in TONG_TILES:  return (tile - 9) + 1
    if tile in TIAO_TILES:  return (tile - 18) + 1
    if tile in range(27, 42):  return None
    raise ValueError(f"非法牌值: {tile}")


def tile_display(tile: int) -> str:
    """返回牌的中文显示名。花牌(34-41)抛出 NotImplementedError。"""
    if 0 <= tile <= 8:    return _NUMBER_NAMES[tile - 0] + "万"
    if 9 <= tile <= 17:   return _NUMBER_NAMES[tile - 9] + "筒"
    if 18 <= tile <= 26:  return _NUMBER_NAMES[tile - 18] + "条"
    if 27 <= tile <= 33:  return _HONOR_NAMES[tile]
    if 34 <= tile <= 41:  raise NotImplementedError(f"花牌显示名尚未实现: tile={tile}")
    raise ValueError(f"非法牌值: {tile}")


def tile_from_str(s: str) -> int:
    """从中文显示名解析牌整数。"""
    for t in range(0, 34):
        if tile_display(t) == s:
            return t
    raise ValueError(f"无法识别的牌名: '{s}'")
```

- [ ] **步骤 2: 验证导入**

```bash
cd /mnt/data/Mahjong_AI && conda run -n mahjong_ai python -c "
from src.mahjong_core.tile import W1, DONG, tile_suit, tile_display, WAN
print(tile_display(W1), tile_display(DONG), tile_suit(W1)==WAN)
"
```

预期输出: `一万 东 True`

- [ ] **步骤 3: 质量检查 + 提交**

```bash
cd /mnt/data/Mahjong_AI
conda run -n mahjong_ai ruff check src/mahjong_core/tile.py
conda run -n mahjong_ai mypy --strict src/mahjong_core/tile.py
git add src/mahjong_core/tile.py
git commit -m "[core] Add tile constants and helper functions"
```

---

### 任务 2: tile 辅助函数测试

**文件:**
- 创建: `tests/test_core/test_tile.py`

- [ ] **步骤 1: 写测试**

```python
"""tile.py 辅助函数的单元测试。"""
import pytest
from src.mahjong_core.tile import (
    W1, W9, T1, T9, B1, B9,
    DONG, NAN, XI, BEI, ZHONG, FA, BAI,
    WAN, TONG, TIAO, FENG, JIAN, HUA,
    WAN_TILES, TONG_TILES, TIAO_TILES,
    FENG_TILES, JIAN_TILES, HUA_TILES,
    TOTAL_TILES,
    tile_suit, tile_number, tile_display, tile_from_str,
)


class TestTileSuit:
    def test_all_suits(self):
        for t in WAN_TILES:   assert tile_suit(t) == WAN
        for t in TONG_TILES:  assert tile_suit(t) == TONG
        for t in TIAO_TILES:  assert tile_suit(t) == TIAO
        for t in FENG_TILES:  assert tile_suit(t) == FENG
        for t in JIAN_TILES:  assert tile_suit(t) == JIAN
        for t in HUA_TILES:   assert tile_suit(t) == HUA

    def test_invalid_raises(self):
        for v in [-1, 42, 100]:
            with pytest.raises(ValueError):
                tile_suit(v)


class TestTileNumber:
    def test_numeric_tiles(self):
        assert tile_number(W1) == 1
        assert tile_number(W9) == 9
        assert tile_number(T1) == 1
        assert tile_number(T9) == 9
        assert tile_number(B1) == 1
        assert tile_number(B9) == 9

    def test_honor_and_hua_return_none(self):
        for t in list(FENG_TILES) + list(JIAN_TILES) + list(HUA_TILES):
            assert tile_number(t) is None


class TestTileDisplay:
    def test_numbered_tiles(self):
        assert tile_display(W1) == "一万"
        assert tile_display(W9) == "九万"
        assert tile_display(T1) == "一筒"
        assert tile_display(B1) == "一条"

    def test_honor_tiles(self):
        assert tile_display(DONG) == "东"
        assert tile_display(NAN) == "南"
        assert tile_display(XI) == "西"
        assert tile_display(BEI) == "北"
        assert tile_display(ZHONG) == "中"
        assert tile_display(FA) == "发"
        assert tile_display(BAI) == "白"

    def test_hua_not_implemented(self):
        with pytest.raises(NotImplementedError):
            tile_display(34)


class TestTileFromStr:
    def test_roundtrip_all_standard(self):
        for t in range(0, 34):
            assert tile_from_str(tile_display(t)) == t

    def test_invalid_raises(self):
        for s in ["", "不是牌", "十萬"]:
            with pytest.raises(ValueError):
                tile_from_str(s)


class TestConstants:
    def test_total(self):
        assert TOTAL_TILES == 42

    def test_groups_disjoint(self):
        all_tiles = (list(WAN_TILES) + list(TONG_TILES) + list(TIAO_TILES) +
                     list(FENG_TILES) + list(JIAN_TILES) + list(HUA_TILES))
        assert len(all_tiles) == 42
        assert len(set(all_tiles)) == 42
```

- [ ] **步骤 2: 确认测试通过**

```bash
cd /mnt/data/Mahjong_AI && conda run -n mahjong_ai python -m pytest tests/test_core/test_tile.py -v
```

- [ ] **步骤 3: 提交**

```bash
git add tests/test_core/test_tile.py
git commit -m "[core] Add tile helper function tests"
```

---

### 任务 3: Hand 手牌类 (TDD)

**文件:**
- 创建: `tests/test_core/test_hand.py`
- 创建: `src/mahjong_core/hand.py`

- [ ] **步骤 1: 写测试 (tests/test_core/test_hand.py)**

```python
"""Hand 手牌类的单元测试。"""
import pytest
from src.mahjong_core.hand import Hand
from src.mahjong_core.tile import W1, W2, W3, T1, T9, DONG, ZHONG


class TestHandInit:
    def test_empty_hand(self):
        h = Hand()
        assert len(h) == 0
        assert list(h) == []

    def test_hand_sorts_on_init(self):
        h = Hand([W3, W1, W2])
        assert list(h) == [W1, W2, W3]

    def test_hand_with_duplicates(self):
        h = Hand([W1, W1, W2])
        assert list(h) == [W1, W1, W2]


class TestHandAdd:
    def test_add_maintains_sort(self):
        h = Hand([W1, W3])
        h.add(W2)
        assert list(h) == [W1, W2, W3]

    def test_add_to_empty(self):
        h = Hand()
        h.add(W1)
        assert len(h) == 1

    def test_add_duplicate(self):
        h = Hand([W1, W2])
        h.add(W1)
        assert list(h) == [W1, W1, W2]


class TestHandDiscard:
    def test_discard_removes_one(self):
        h = Hand([W1, W1, W2])
        h.discard(W1)
        assert list(h) == [W1, W2]

    def test_discard_not_present_raises(self):
        h = Hand([W1, W2])
        with pytest.raises(ValueError, match="not in hand"):
            h.discard(W3)

    def test_discard_last_occurrence(self):
        h = Hand([W1])
        h.discard(W1)
        assert len(h) == 0


class TestHandCount:
    def test_count(self):
        h = Hand([W1, W1, W2])
        assert h.count(W1) == 2
        assert h.count(W2) == 1
        assert h.count(W3) == 0


class TestHandDunder:
    def test_len(self):
        assert len(Hand([W1, W2, W3, T1, T9])) == 5

    def test_getitem(self):
        h = Hand([W2, W1])
        assert h[0] == W1
        assert h[1] == W2
        with pytest.raises(IndexError):
            _ = h[5]

    def test_iter(self):
        assert list(Hand([W3, W1, W2])) == [W1, W2, W3]

    def test_contains(self):
        h = Hand([W1, W2])
        assert W1 in h
        assert W3 not in h

    def test_repr(self):
        assert "Hand" in repr(Hand([W1, DONG]))


class TestHandEdgeCases:
    def test_14_tiles(self):
        assert len(Hand([W1] * 14)) == 14

    def test_large_hand_sort(self):
        from src.mahjong_core.tile import W1 as w1, W9 as w9, T1 as t1, T9 as t9
        h = Hand([t9, t1, w9, w1, DONG, ZHONG])
        assert list(h) == [w1, w9, t1, t9, DONG, ZHONG]
```

- [ ] **步骤 2: 运行测试，确认全部失败 (RED)**

```bash
cd /mnt/data/Mahjong_AI && conda run -n mahjong_ai python -m pytest tests/test_core/test_hand.py -v
```

- [ ] **步骤 3: 实现 Hand 类 (src/mahjong_core/hand.py)**

```python
"""手牌类 — 排序列表表示，模拟玩家理牌行为。"""

import bisect
from typing import Iterator, Optional


class Hand:
    """麻将手牌。内部使用排序列表存储牌整数。"""

    def __init__(self, tiles: Optional[list[int]] = None) -> None:
        self._tiles: list[int] = sorted(tiles) if tiles else []

    def add(self, tile: int) -> None:
        """加入一张牌，保持排序顺序。"""
        bisect.insort(self._tiles, tile)

    def discard(self, tile: int) -> None:
        """打出一张牌。Raises ValueError 如果牌不在手牌中。"""
        try:
            self._tiles.remove(tile)
        except ValueError:
            raise ValueError(f"牌 {tile} not in hand") from None

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
        return f"Hand({self._tiles})"
```

- [ ] **步骤 4: 运行测试，确认全部通过 (GREEN)**

```bash
cd /mnt/data/Mahjong_AI && conda run -n mahjong_ai python -m pytest tests/test_core/test_hand.py -v
```

- [ ] **步骤 5: 提交**

```bash
git add src/mahjong_core/hand.py tests/test_core/test_hand.py
git commit -m "[core] Add Hand class with tests"
```

---

### 任务 4: TileSetConfig 牌库配置 (TDD)

**文件:**
- 创建: `tests/test_core/test_config.py`
- 创建: `src/mahjong_core/config.py`

- [ ] **步骤 1: 写测试 (tests/test_core/test_config.py)**

```python
"""TileSetConfig 牌库配置的单元测试。"""
import pytest
from src.mahjong_core.config import TileSetConfig
from src.mahjong_core.tile import W1, W9, DONG, ZHONG, MEI


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
        with pytest.raises(Exception):
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
```

- [ ] **步骤 2: 确认失败 (RED)**

```bash
cd /mnt/data/Mahjong_AI && conda run -n mahjong_ai python -m pytest tests/test_core/test_config.py -v
```

- [ ] **步骤 3: 实现 TileSetConfig (src/mahjong_core/config.py)**

```python
"""牌库配置 — 不可变白名单配置。"""

from dataclasses import dataclass, field
from src.mahjong_core.tile import (
    WAN_TILES, TONG_TILES, TIAO_TILES,
    FENG_TILES, JIAN_TILES, HUA_TILES,
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
    def standard_136() -> "TileSetConfig":
        """万+筒+条+风+箭，各4张 = 136张 (陕西推倒胡)"""
        tiles = (set(WAN_TILES) | set(TONG_TILES) | set(TIAO_TILES)
                 | set(FENG_TILES) | set(JIAN_TILES))
        return TileSetConfig(enabled_tiles=frozenset(tiles), tile_count=4)

    @staticmethod
    def standard_108() -> "TileSetConfig":
        """仅万+筒+条，各4张 = 108张 (四川血战)"""
        tiles = set(WAN_TILES) | set(TONG_TILES) | set(TIAO_TILES)
        return TileSetConfig(enabled_tiles=frozenset(tiles), tile_count=4)

    @staticmethod
    def standard_144() -> "TileSetConfig":
        """34种*4 + 花牌8种*1 = 144张 (广东麻将)"""
        tiles = (set(WAN_TILES) | set(TONG_TILES) | set(TIAO_TILES)
                 | set(FENG_TILES) | set(JIAN_TILES) | set(HUA_TILES))
        return TileSetConfig(
            enabled_tiles=frozenset(tiles),
            tile_count=4,
            _total_override=144,
        )
```

- [ ] **步骤 4: 确认通过 (GREEN)**

```bash
cd /mnt/data/Mahjong_AI && conda run -n mahjong_ai python -m pytest tests/test_core/test_config.py -v
```

- [ ] **步骤 5: 提交**

```bash
git add src/mahjong_core/config.py tests/test_core/test_config.py
git commit -m "[core] Add TileSetConfig with factory methods"
```

---

### 任务 5: 模块导出 + 全面验证

**文件:**
- 修改: `src/mahjong_core/__init__.py`

- [ ] **步骤 1: 更新 __init__.py**

```python
"""mahjong_core — 麻将AI项目的基础数据类型。"""
from src.mahjong_core.tile import (
    TOTAL_TILES, WAN, TONG, TIAO, FENG, JIAN, HUA,
    W1, W2, W3, W4, W5, W6, W7, W8, W9,
    T1, T2, T3, T4, T5, T6, T7, T8, T9,
    B1, B2, B3, B4, B5, B6, B7, B8, B9,
    DONG, NAN, XI, BEI, ZHONG, FA, BAI,
    MEI, LAN, ZHU, JU, CHUN, XIA, QIU, DONG_HUA,
    WAN_TILES, TONG_TILES, TIAO_TILES,
    FENG_TILES, JIAN_TILES, HUA_TILES,
    tile_suit, tile_number, tile_display, tile_from_str,
)
from src.mahjong_core.hand import Hand
from src.mahjong_core.config import TileSetConfig

__all__ = [
    "TOTAL_TILES", "WAN", "TONG", "TIAO", "FENG", "JIAN", "HUA",
    "W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8", "W9",
    "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9",
    "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9",
    "DONG", "NAN", "XI", "BEI", "ZHONG", "FA", "BAI",
    "MEI", "LAN", "ZHU", "JU", "CHUN", "XIA", "QIU", "DONG_HUA",
    "WAN_TILES", "TONG_TILES", "TIAO_TILES",
    "FENG_TILES", "JIAN_TILES", "HUA_TILES",
    "tile_suit", "tile_number", "tile_display", "tile_from_str",
    "Hand", "TileSetConfig",
]
```

- [ ] **步骤 2: 全模块质量检查**

```bash
cd /mnt/data/Mahjong_AI
conda run -n mahjong_ai python -m pytest tests/test_core/ -v
conda run -n mahjong_ai ruff check src/mahjong_core/ tests/test_core/
conda run -n mahjong_ai mypy --strict src/mahjong_core/
```

预期: 全部测试通过, ruff 无报错, mypy 无报错

- [ ] **步骤 3: 提交**

```bash
git add src/mahjong_core/__init__.py
git commit -m "[core] Add __init__.py exports — Phase 1 complete"
```
