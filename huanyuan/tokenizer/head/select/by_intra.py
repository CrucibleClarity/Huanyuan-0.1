"""选择器: 同类内差异采样 (比较同类差异 → 选择成员参与)

消费聚类结果 (ctx['cluster']), 计算类内成员与类质心的差异度 (brace 溯源特征),
按条件采样: mode (typical=典型成员/差异小, distinct=差异成员/差异大) + k (每类数量) 或 threshold。
"""
from ._registry import register_selector
from ...construct.expand import brace_logic


def _diffs(sequence, ctx):
    """类内差异度: {eid: 与类质心的 1-Jaccard}。"""
    cluster = ctx.get("cluster")
    if not cluster:
        return {e: 0.0 for e in sequence}
    feats = {e: set(brace_logic(e)) for e in sequence}
    diffs = {}
    for sig, members in cluster.items():
        centroid = set().union(*(feats.get(m, set()) for m in members)) if members else set()
        for m in members:
            f = feats.get(m, set())
            sim = len(f & centroid) / len(f | centroid) if (f and centroid) else 1.0
            diffs[m] = 1.0 - sim
    return diffs


@register_selector("intra")
def select_intra(sequence, ctx=None):
    """同类内差异采样。

    ctx: mode=typical/distinct (默认 distinct), k=每类采样数 或 threshold=差异阈值。
    缺省 k/threshold 时返回全量 (差异度注记进 ctx['intra_diffs'])。
    """
    ctx = ctx or {}
    cluster = ctx.get("cluster")
    if not cluster:
        return list(sequence)
    diffs = _diffs(sequence, ctx)
    ctx["intra_diffs"] = diffs
    mode = ctx.get("mode", "distinct")
    k = ctx.get("k")
    threshold = ctx.get("threshold")
    out = []
    for sig, members in cluster.items():
        sel = [m for m in members if m in diffs]
        if threshold is not None:
            if mode == "distinct":
                sel = [m for m in sel if diffs[m] >= threshold]
            else:
                sel = [m for m in sel if diffs[m] <= threshold]
        elif k:
            sel = sorted(sel, key=diffs.get, reverse=(mode == "distinct"))[:k]
        out.extend(sel)
    return out
