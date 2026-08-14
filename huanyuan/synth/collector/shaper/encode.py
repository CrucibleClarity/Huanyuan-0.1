"""shaper/encode.py —— 节点编码策略 (概念 → 向量行, 可插拔)

counts  counts 向量 (eid 空间, 结构编码)
depth  counts + 嵌套深度通道
role   counts + 角色通道 (predicate=1.0 / noun=0.5 / decorator=0.25)
"""
from __future__ import annotations

from ._registry import register_encode


def _row(vecs, e, idx, dim, extra=None):
    row = [0.0] * dim
    for k, c in vecs.get(e, {}).items():
        if k in idx:
            row[idx[k]] = c
    if extra is not None:
        row.append(extra)
    return row


@register_encode("counts")
def enc_counts(e, st):
    return _row(st["vecs"], e, st["idx"], st["dim"])


@register_encode("depth")
def enc_depth(e, st):
    return _row(st["vecs"], e, st["idx"], st["dim"], float(st["dm"].get(e, 0)))


@register_encode("role")
def enc_role(e, st):
    code = {"predicate": 1.0, "noun": 0.5, "decorator": 0.25}.get(st["roles"].get(e), 0.0)
    return _row(st["vecs"], e, st["idx"], st["dim"], code)
