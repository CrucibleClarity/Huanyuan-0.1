"""选择器: 名词层采样 (只采样操作数, arg/args 槽位 token)"""
from .._registry import register_selector
from ._roles import roles_of

LAYER = "noun"


@register_selector("noun")
def select_noun(sequence, ctx=None):
    ctx = ctx or {}
    ast = ctx.get("ast")
    if ast is None:
        return list(sequence)
    roles = roles_of(ast)
    return [e for e in sequence if roles.get(e) == LAYER]
