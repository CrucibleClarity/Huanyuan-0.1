"""选择器: 装饰层采样 (量化/修饰, binder/body 槽位 token)"""
from .._registry import register_selector
from ._roles import roles_of

LAYER = "decorator"


@register_selector("decorator")
def select_decorator(sequence, ctx=None):
    ctx = ctx or {}
    ast = ctx.get("ast")
    if ast is None:
        return list(sequence)
    roles = roles_of(ast)
    return [e for e in sequence if roles.get(e) == LAYER]
