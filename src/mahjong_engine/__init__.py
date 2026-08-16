"""mahjong_engine — 麻将规则引擎。

提供:
    - IRuleSet: 规则抽象接口
    - ShaanxiRules: 陕西推倒胡实现
    - WinResult / WaitingTile / KongResult: 判定结果数据类
    - 胡牌/听牌/碰杠吃 判定算法
"""
from src.mahjong_engine.judges.action_judge import KongResult, can_chi, can_kong, can_pong
from src.mahjong_engine.judges.tenpai_judge import WaitingTile, get_waiting_tiles
from src.mahjong_engine.judges.win_judge import WinResult, is_winning_hand
from src.mahjong_engine.rules.huanyu_rules import HuanyuRules
from src.mahjong_engine.rules.interface import IRuleSet
from src.mahjong_engine.rules.registry import get, list_names, register
from src.mahjong_engine.rules.shaanxi_rules import ShaanxiRules

__all__ = [
    'IRuleSet',
    'ShaanxiRules',
    'HuanyuRules',
    'register', 'get', 'list_names',
    'is_winning_hand', 'WinResult',
    'get_waiting_tiles', 'WaitingTile',
    'can_pong', 'can_kong', 'can_chi', 'KongResult',
]
