"""算法: 外延相似度聚类 (概念在样本中的应用/共现上下文)

内涵 vs 外延:
  def_sig 内涵: definition 引用集 (定义内容)
  ext_sig 外延: 概念在样本中的共现上下文分布 (出现于哪些样本、与谁同现)

ctx 需含 samples (样本集, 每样本有 seq = 概念序列); 无样本时各自一类。
"""
from collections import Counter, defaultdict

from ._registry import register_algorithm


def _ext_feats(e, samples):
    """外延特征: 概念 e 在样本中的共现概念频率 (出现于同样本的其他概念)。"""
    co = Counter()
    for s in samples:
        seq = s.get("seq", [])
        if e in seq:
            for x in seq:
                if x != e:
                    co[x] += 1
    return co


def _jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@register_algorithm("ext_sig")
def group_by_extension(eids, ctx=None, threshold=0.3):
    """按外延 (共现上下文) 相似度贪心聚类。返回 {类签名: [eids]}。"""
    samples = (ctx or {}).get("samples")
    if not samples:
        return {e: [e] for e in eids}
    feats = {e: set(_ext_feats(e, samples)) for e in eids}
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
