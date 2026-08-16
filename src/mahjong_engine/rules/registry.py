"""规则注册表 — 按名称查找规则集。"""

from src.mahjong_engine.rules.huanyu_rules import HuanyuRules
from src.mahjong_engine.rules.interface import IRuleSet
from src.mahjong_engine.rules.shaanxi_rules import ShaanxiRules

_rules: dict[str, IRuleSet] = {}


def register(rule: IRuleSet) -> None:
    """注册一个规则集。"""
    _rules[rule.name()] = rule


def get(name: str) -> IRuleSet:
    """按名称获取规则集。Raises KeyError 如果不存在。"""
    if name not in _rules:
        raise KeyError(f"规则 '{name}' 未注册。可用规则: {list(_rules.keys())}")
    return _rules[name]


def list_names() -> list[str]:
    """列出所有已注册规则名称。"""
    return list(_rules.keys())


# 内置规则注册
register(ShaanxiRules())
register(HuanyuRules())
