"""选择器: 谓词层采样 (只采样运算子, fn 槽位 token)"""
from .._registry import register_selector
from ._roles import roles_of

LAYER = "predicate"


@register_selector("predicate")
def select_predicate(sequence, ctx=None):
    ctx = ctx or {}
    ast = ctx.get("ast")
    if ast is None:
        return list(sequence)
    roles = roles_of(ast)
    return [e for e in sequence if roles.get(e) == LAYER]
