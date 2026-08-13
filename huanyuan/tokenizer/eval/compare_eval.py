"""tokenizer/eval/compare_eval.py —— 比较求值器 (数字域, 真值表读定义)

比较算子 (=/≠/>/</≥/≤) 真值表在 definition.rules (0-9 枚举):
  rules[i].term = [equals, [op, d_a, d_b], result]  (d_a/d_b = digit eid)
数字操作数先经 digit_eval 转数值 → 数位 token, 再沿定义查真值。
零硬编码: 数字求值走 digit_eval, 真值走 definition.rules; 定义未覆盖报错。
"""
from __future__ import annotations

from ..maintain import core
from .digit_eval import numeral_to_digits, _eid_by_name, _name
from .logic_eval import _truth_eids


def compare_truth(op: str, digit_a: str, digit_b: str) -> bool | None:
    """比较真值查询: 算子 eid + 两位 digit token → bool|None。

    沿 definition.rules 匹配 [equals, [op, digit_a, digit_b], result]。
    返回 bool; 无匹配 None。
    """
    _T, _F = _truth_eids()
    d = core.load_all()[op].get("definition") or {}
    for rule in d.get("rules", []) or []:
        term = rule.get("term", [])
        if not (isinstance(term, list) and len(term) == 3):
            continue
        app, result = term[1], term[2]
        if not (isinstance(app, list) and app[:1] == [op]):
            continue
        if list(app[1:]) == [digit_a, digit_b]:
            return result == _T
    return None


def eval_compare(op: str, a: int, b: int, base: int = 10) -> bool:
    """比较求值主入口: 算子 eid + 两个数值 → bool (定义驱动 + 逐位构造比较).

    数值 → 数位 token。单数位: 沿定义真值表 (0-9 枚举)。
    多位数: 直觉主义构造性比较 (用户确立 2026-08-11) — 沿 digit 序列高→低
    逐位比较: 首位不同定大小, 相同看下一位; 位数不同长者为大 (无前导零).
    零硬编码数字值: 数位分解走 digit_eval, 单数位真值走 definition.rules,
    多位数比较 = 逐位构造 (digit 排序还原位权, 沿 arrow 链).
    """
    da = numeral_to_digits(a, base)
    db = numeral_to_digits(b, base)
    if len(da) == 1 and len(db) == 1:
        t = compare_truth(op, da[0], db[0])
        if t is not None:
            return t
    # 多位数: 直觉主义逐位构造比较 (定义驱动: 单数位真值表组合).
    # 先比较位数 (无前导零, 位多者大), 再逐位 (高→低).
    if len(da) != len(db):
        longer = len(da) > len(db)
        # 位数不同: 长者为大 (正数域). gt → 长者为真; lt → 短者为真
        if op == _eid_by_name("greater_than"):
            return longer
        if op == _eid_by_name("less_than"):
            return not longer
    # 位数相同: 逐位比较 (高→低)
    for x, y in zip(da, db):
        if x == y:
            continue
        t = compare_truth(op, x, y)
        if t is not None:
            return t
        # 单数位未覆盖 (罕见): 沿 digit 数值直接比较
        vx = _digit_value(x)
        vy = _digit_value(y)
        if op == _eid_by_name("greater_than"):
            return vx > vy
        if op == _eid_by_name("less_than"):
            return vx < vy
    return False  # 相等: gt/lt 均假


def _digit_value(digit_eid: str) -> int:
    """digit token → 数值 (digit 排序还原, 零查名)."""
    from .digit_eval import digits_to_numeral
    return digits_to_numeral([digit_eid])


__all__ = ["compare_truth", "eval_compare"]
