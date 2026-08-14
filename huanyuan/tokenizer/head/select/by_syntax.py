"""选择器: 语法采样 (谓词采样 —— 按 AST 槽位角色判定参与)

对 assemble 产物 (AST) 按语法角色 (gtoken 槽位: fn/arg:N/args/binder/body) 采样参与 token。
ctx: {'ast': AST, 'roles': 可选角色过滤}; 无 ast 时退回全量。
"""
from ._registry import register_selector


def _walk(node, wanted, out):
    concept = node.get("concept")
    slots = list(node.get("slots", []))
    children = list(node.get("children", []))
    if concept and (wanted is None or "fn" in wanted):
        out.append(concept)
    if "fn" in slots:
        non_fn = [s for s in slots if s != "fn"]
        child_items = children[1:]
    else:
        non_fn = slots
        child_items = children
    sidx = 0
    for c in child_items:
        role = non_fn[sidx] if sidx < len(non_fn) else (non_fn[-1] if non_fn else "?")
        if isinstance(c, dict):
            _walk(c, wanted, out)
        elif isinstance(c, str) and c.startswith("D:"):
            if wanted is None or role in wanted:
                out.append(c)
        if role != "args":
            sidx = min(sidx + 1, len(non_fn) - 1) if non_fn else sidx


@register_selector("syntax")
def select_by_syntax(sequence, ctx=None):
    """语法采样: AST 槽位角色谓词 → 参与 token。ctx: {'ast':..., 'roles':...}。"""
    ctx = ctx or {}
    ast = ctx.get("ast")
    if ast is None:
        return list(sequence)
    out = []
    _walk(ast, ctx.get("roles"), out)
    return out
