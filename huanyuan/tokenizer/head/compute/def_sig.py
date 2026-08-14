"""算法: definition 引用签名聚类 (同类/异类)

同类/异类 由 C token 的定义引用 (definition) 决定 — 数据驱动, 不读散文。
def_sig(eid) = definition 引用的 eid 集合 (排除自身)。
同类 = 引用集 Jaccard 重叠 ≥ 阈值 (贪心聚类)。

实例 (5+3=8):
  [5][3][8] 定义都引用 successor → 同类 (数位类)
  [+]      定义引用 element/relation/自身 → 异类
  [=]      定义引用 digit/bool (签名) → 异类
"""
from collections import defaultdict

from ._registry import register_algorithm
from ..._register import token_of, definition_refs


def _jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@register_algorithm("def_sig")
def group_by_definition(eids, ctx=None, threshold=0.3):
    """按 definition 引用集 Jaccard 贪心聚类。返回 {类签名: [eids]}。"""
    feats = {e: set(definition_refs(token_of(e).definition)) - {e} for e in eids}
    groups = defaultdict(list)
    assigned = set()
    for e in eids:
        if e in assigned:
            continue
        sig = feats[e]
        groups[tuple(sorted(sig))].append(e)
        assigned.add(e)
        for other in eids:
            if other in assigned:
                continue
            if _jaccard(sig, feats[other]) >= threshold:
                groups[tuple(sorted(sig))].append(other)
                assigned.add(other)
    return dict(groups)
