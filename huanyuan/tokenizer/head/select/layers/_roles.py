"""select/layers/_roles.py —— 角色分层映射 (谓词/名词/装饰, 数据驱动槽位)

gtoken 槽位角色 → 层: fn→predicate (运算子), arg/args→noun (操作数),
binder/body→decorator (量化/修饰)。纯算术样本无 binder/body → 装饰层不激活。
"""
from __future__ import annotations


def roles_of(ast) -> dict:
    """AST → {概念 eid: 'predicate'|'noun'|'decorator'}。"""
    out = {}

    def walk(node):
        concept = node.get("concept")
        slots = list(node.get("slots", []))
        children = list(node.get("children", []))
        if concept:
            out[concept] = "predicate"
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
                walk(c)
            elif isinstance(c, str) and c.startswith("D:"):
                base = role.split(":")[0]
                out[c] = "decorator" if base in ("binder", "body") else "noun"
            if role != "args":
                sidx = min(sidx + 1, len(non_fn) - 1) if non_fn else sidx

    walk(ast)
    return out
