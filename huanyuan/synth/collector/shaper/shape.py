"""shaper/shape.py —— 整形入口 (配置 order/encode/output)

output:
  sequence  多向量 [seq_len, dim(+通道)]  token 级序列 (标准 transformer 输入)
  layer     多向量 [n_layers, dim]        按嵌套深度逐层聚合 (结构感知)
  collapse  单向量 [dim]                  样本摘要 (复用 collector.collapse)
order/encode 可插拔 (见 _registry)。
"""
from __future__ import annotations

from tokenizer import api
from ._registry import get_order, get_encode, list_orders, list_encodes
from ..collapse import collapse


def _state(sample):
    space = api.eid_space()
    idx = {e: i for i, e in enumerate(space)}
    ast = sample["ast"]
    dm = api.depth_map(ast) if not isinstance(ast, str) else {e: 0 for e in sample["seq"]}
    vecs = {v["eid"]: v["counts"] for v in sample["vectors"]}
    roles = {}
    if not isinstance(ast, str):
        from tokenizer.head.select.layers import roles_of
        roles = roles_of(ast)
    return {"idx": idx, "dim": len(space), "vecs": vecs, "dm": dm, "roles": roles}


def _mean(rows, dim):
    out = [0.0] * dim
    n = max(len(rows), 1)
    for r in rows:
        for i, x in enumerate(r):
            out[i] += x / n
    return out


def shape(sample, order="preorder", encode="counts", output="sequence", collapse_method="mean"):
    """样本 → transformer 向量形态。order/encode/output 均可插拔。"""
    if output == "collapse":
        return collapse(sample, collapse_method)
    st = _state(sample)
    seq = get_order(order)(sample["ast"])
    enc = get_encode(encode)
    if output == "pairs":
        return _shape_pairs(seq, st)
    if output == "defexpand":
        return _defexpand(seq, st, exclude=sample.get("exclude"))
    rows = [enc(e, st) for e in seq]
    if output == "layer":
        layers = {}
        for e, row in zip(seq, rows):
            layers.setdefault(st["dm"].get(e, 0), []).append(row)
        n = (max(layers) + 1) if layers else 1
        return [_mean(layers.get(d, []), st["dim"]) for d in range(n)]
    return rows


def _shape_pairs(seq, st):
    """每 token → [content, role] 2 向量, 交错 [2*seq_len, dim]。

    content = counts 归一化 (内涵); role = 角色码 + 深度归一化 (外延/结构), pad 到 dim。
    """
    dim = st["dim"]
    out = []
    role_code = {"predicate": 1.0, "noun": 0.5, "decorator": 0.25}
    for e in seq:
        c = [0.0] * dim
        for k, v in st["vecs"].get(e, {}).items():
            if k in st["idx"]:
                c[st["idx"][k]] = v
        n = sum(x * x for x in c) ** 0.5
        if n > 0:
            c = [x / n for x in c]
        r = [0.0] * dim
        r[0] = role_code.get(st["roles"].get(e), 0.0)
        r[1] = min(st["dm"].get(e, 0) / 10.0, 1.0)
        out.append(c)
        out.append(r)
    return out


def _defexpand(seq, st, include_self=True, exclude=None):
    """展开一层定义的多向量: 每 token → [自身, 一层定义引用概念...] 各一向量。

    展开 = 中括号向量 [x] (一层定义引用, 不递归); 引用概念用其 counts 编码。
    exclude: 展开时跳过的 token (可解释泛化: 训练样本不含被排除目标)。
    输出 [sum(1+len(refs)), dim]。结构信息带进输入序列。
    """
    idx, dim = st["idx"], st["dim"]
    exclude = set(exclude or [])

    def vec(e):
        row = [0.0] * dim
        for k, c in api.counts(e).items():
            if k in idx:
                row[idx[k]] = c
        return row

    out = []
    for e in seq:
        if include_self and e not in exclude:
            out.append(vec(e))
        for r in api.bracket(e):
            if r in exclude:
                continue
            if api.is_concept(r):
                out.append(vec(r))
    return out


def shape_spec() -> dict:
    dim = len(api.eid_space())
    return {
        "sequence": ["seq_len", f"dim({dim})(+通道)"],
        "pairs": ["2*seq_len", f"dim({dim})", "(每 token 2 向量: content+role)"],
        "defexpand": ["sum(1+len(refs))", f"dim({dim})", "(每 token 展开一层定义引用, 各一向量)"],
        "layer": ["n_layers", dim],
        "collapse": [dim],
        "orders": list_orders(),
        "encodes": list_encodes(),
    }
