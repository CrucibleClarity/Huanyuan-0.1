"""shaper/orders.py —— 展平顺序策略 (AST → 概念序列, 可插拔)

preorder  先序 (节点先于子)
postorder 后序 (子先于节点)
level     层级序 (BFS 逐层)
infix     记法序 (呈现层 grammar linearize)
"""
from __future__ import annotations

from collections import deque

from tokenizer import api
from ._registry import register_order


def _walk(ast, visit):
    """分发: str=原子, dict=节点 (fn 前置冗余跳过)。visit(node, children), node 可为 str。"""
    if isinstance(ast, str):
        visit(ast, None)
        return
    children = ast.get("children", [])
    if "fn" in ast.get("slots", []):
        children = children[1:]
    visit(ast, children)


@register_order("preorder")
def preorder(ast):
    """先序: 节点概念先于子项。"""
    out = []

    def visit(node, children):
        if isinstance(node, str):
            if api.is_concept(node):
                out.append(node)
            return
        if node.get("concept"):
            out.append(node["concept"])
        for c in children:
            _walk(c, visit)

    _walk(ast, visit)
    return out


@register_order("postorder")
def postorder(ast):
    """后序: 子项先于节点。"""
    out = []

    def visit(node, children):
        if isinstance(node, str):
            if api.is_concept(node):
                out.append(node)
            return
        for c in children:
            _walk(c, visit)
        if node.get("concept"):
            out.append(node["concept"])

    _walk(ast, visit)
    return out


@register_order("level")
def level(ast):
    """层级序: BFS 逐层展开。"""
    out = []
    q = deque([ast])
    while q:
        node = q.popleft()
        if isinstance(node, str):
            if api.is_concept(node):
                out.append(node)
            continue
        if node.get("concept"):
            out.append(node["concept"])
        children = node.get("children", [])
        if "fn" in node.get("slots", []):
            children = children[1:]
        q.extend(children)
    return out


@register_order("infix")
def infix(ast):
    """记法序: 按呈现层 grammar 顺序 (arg:N 填子项, 符号字面量跳过)。"""
    out = []

    def visit(node, children):
        if isinstance(node, str):
            if api.is_concept(node):
                out.append(node)
            return
        concept = node.get("concept")
        pres = api.presentation_of(concept) if concept else None
        if pres:
            ci = 0
            emitted = False
            for g in pres["grammar"]:
                if g.startswith("arg:"):
                    if ci < len(children):
                        _walk(children[ci], visit)
                        ci += 1
                elif not emitted and concept:
                    out.append(concept)
                    emitted = True
        else:
            if concept:
                out.append(concept)
            for c in children:
                _walk(c, visit)

    _walk(ast, visit)
    return out
