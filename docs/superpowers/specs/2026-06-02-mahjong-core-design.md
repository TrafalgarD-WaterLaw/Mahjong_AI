# Phase 1 Design: mahjong_core 基础数据结构

- **Date**: 2026-06-02
- **Status**: Approved
- **Scope**: Phase 1 of Mahjong AI project
- **Dependencies**: None (zero external dependencies)

## Overview

Define the three foundational types that all other modules (engine, ai, cv, ui) depend on.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Tile representation | Integer constants (0-41) | Direct mapping to ML model class indices, no Enum overhead |
| Hand storage | Sorted list[int] | Most intuitive — like a player holding sorted tiles |
| Tile config | Whitelist (enabled tiles set) | Explicit is better; each rule set declares what it uses |
| Wild tiles | frozenset in config | Supports 赖子/万能牌 in future rules |
| Flower tiles | Reserved 34-41, not in Phase 1 | Ready for future rules (广东麻将 etc.) |

## 1. Tile Constants ()

### Integer encoding



### Tile count: 42 total (34 standard + 8 flowers reserved)

### Public API



## 2. Hand ()

### Behavior

- Accepts a list of tile integers in constructor, sorts on init
- : inserts tile maintaining sorted order
- : removes one occurrence
- : returns count of a specific tile
- Supports len(), iteration, indexing

### Public API



### Invariants

- Tiles are always sorted (ascending integer order)
-  removes exactly one occurrence
-  should be 13 or 14 in normal gameplay

## 3. TileSetConfig ()

### Behavior

- Immutable (frozen dataclass)
- Whitelist approach: explicitly declare which tiles are enabled
- Factory methods for common configurations
- : check if a tile belongs to this tile set

### Public API



## 4. Module Layout



## 5. Testing

Tests mirror the module structure:



## 6. What's NOT in Phase 1

- Flower tile display strings (seats reserved, not implemented)
- Hand validation (must be called rules to validate 13/14 tiles)
- Game state (game_state.py in mahjong_engine, not core)
- Any UI or CV code
