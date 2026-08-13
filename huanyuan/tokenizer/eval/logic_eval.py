"""tokenizer/eval/logic_eval.py —— 逻辑递归求值器 (真值表读定义, 零硬编码)

逻辑算子 (and/or/not/imply/iff/xor/nand/nor/xnor) 的真值表在 definition.rules:
  rules[i].term = [equals, [op, arg0, arg1...], result]
  其中 arg = truth_true/false (D:138/139), result = 真值 token。
沿定义匹配操作数序列 → 真值。递归支持嵌套 (操作数本身可为逻辑表达式)。

数据源: definition.rules (真值表进定义, 用户铁律)。
输出: bool (或 truth token eid)。
"""
from __future__ import annotations

from ..maintain import core
from .digit_eval import _eid_by_name, _name

_TRUE = None
_FALSE = None


def _truth_eids():
    global _TRUE, _FALSE
    if _TRUE is None:
        _TRUE = _eid_by_name("truth_true")
        _FALSE = _eid_by_name("truth_false")
    return _TRUE, _FALSE


def is_truth_token(e: str) -> bool:
    """e 是否为真值 token (truth_true/false)。"""
    _T, _F = _truth_eids()
    return e == _T or e == _F


def logic_truth(op: str, arg_tokens: list[str]) -> bool | None:
    """逻辑算子真值查询: 沿 definition.rules 匹配操作数 → 真值。

    op: 逻辑算子 eid; arg_tokens: 操作数 token 序列 (truth eid)。
    返回 bool; 无匹配返回 None (定义未覆盖)。
    """
    _T, _F = _truth_eids()
    d = core.load_all()[op].get("definition") or {}
    for rule in d.get("rules", []) or []:
        term = rule.get("term", [])
        if not (isinstance(term, list) and len(term) == 3):
            continue
        app, result = term[1], term[2]
        if not (isinstance(app, list) and app and app[0] == op):
            continue
        if list(app[1:]) == list(arg_tokens):
            return result == _T
    return None


def eval_logic(op: str, args: list[bool]) -> bool:
    """逻辑求值主入口: 算子 eid + bool 参数 → bool 真值。

    参数转 truth token, 沿定义查真值表。未覆盖抛错。
    """
    _T, _F = _truth_eids()
    arg_tokens = [_T if v else _F for v in args]
    t = logic_truth(op, arg_tokens)
    if t is None:
        raise ValueError(f"逻辑真值未覆盖: {api.name(op)}({args}) — token 定义 rules 缺失")
    return t


def eval_bool_expr(expr: dict) -> bool:
    """递归求值逻辑表达式树 (嵌套运算)。

    expr: {'op': eid, 'args': [bool 或 嵌套 expr]} — 递归展开。
    """
    op = expr["op"]
    vals = []
    for a in expr.get("args", []):
        if isinstance(a, dict):
            vals.append(eval_bool_expr(a))
        else:
            vals.append(bool(a))
    return eval_logic(op, vals)


__all__ = ["logic_truth", "eval_logic", "eval_bool_expr", "is_truth_token"]
