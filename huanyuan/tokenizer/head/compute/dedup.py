"""算法: 消去 (连续相同 token 消去, 只保留不同 — 相邻比较注意力)

本质是另一种比较注意力: 相邻 token 比较, 相同消去, 保留变化点。
输入序列 → 压缩序列 (连续重复只留一个)。
"""
from ._registry import register_algorithm


@register_algorithm("dedup")
def dedup(eids, ctx=None):
    """消去连续相同, 只保留不同的 token (相邻比较, 压缩序列)。"""
    out = []
    prev = None
    for e in eids:
        if e != prev:
            out.append(e)
        prev = e
    return out
