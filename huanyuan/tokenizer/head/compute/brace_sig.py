"""算法: 大括号展开对比聚类 (同类/异类)

两两对比元素的【溯源展开 {x}】(沿 definition 递归到公设), 相似度 = 展开 eid 集 Jaccard。
{5}/{3}/{8} 共享整条 successor-counting 链 → 同类; {+} 是 Peano 展开 → 异类。

全 token 依据 (definition 引用网络), 无预定义类别。
"""
from collections import defaultdict

from ._registry import register_algorithm
from ..._register import token_of, definition_refs, is_axiomatic_eid


def _chain(eid, _seen=None):
    """溯源展开 {x}: 沿 definition 递归到公设 (不去重, 防环; 遇 axiomatic 终止; G 层无定义体终止)。"""
    if _seen is None:
        _seen = set()
    if eid in _seen:
        return [eid]
    _seen.add(eid)
    try:
        td = token_of(eid)
    except KeyError:
        return [eid]
    if is_axiomatic_eid(eid):
        return [eid]
    refs = definition_refs(td.definition)
    if not refs:
        return [eid]
    result = [eid]
    for r in refs:
        result.extend(_chain(r, _seen))
    return result


def _jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@register_algorithm("brace_sig")
def group_by_brace(eids, ctx=None, threshold=0.25):
    """按溯源展开集 Jaccard 贪心聚类。返回 {类签名: [eids]}。"""
    feats = {e: set(_chain(e)) for e in eids}
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
