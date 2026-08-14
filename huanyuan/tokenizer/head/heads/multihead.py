"""heads/multihead.py —— 多头编排 (并行 + 归并)

多头 = Head 实例列表, 各自独立 run, 产物归并。
归并策略:
  mean     权重均值 (每 eid 各头权重平均)
  weighted 加权平均 (weights 每头系数)
  concat   多通道拼接 (每 eid → 各头权重列表)
  vote     聚类投票 (聚类产物: 每 eid 归属票数最多类)
"""
from __future__ import annotations

from collections import Counter


def _all_keys(results):
    return set().union(*[set(r) for r in results])


def combine_results(results, combine="mean", weights=None) -> dict:
    """归并多头产物。results: 各头输出列表 (权重 dict 或聚类 dict)。"""
    if not results:
        return {}
    if combine == "vote":
        votes = {}
        for r in results:
            for sig, eids in r.items():
                for e in eids:
                    votes.setdefault(e, Counter())[sig] += 1
        return {e: c.most_common(1)[0][0] for e, c in votes.items()}
    if combine == "concat":
        keys = _all_keys(results)
        return {e: [r.get(e, 0.0) for r in results] for e in keys}
    wts = weights or [1.0] * len(results)
    total = sum(wts)
    keys = _all_keys(results)
    return {e: sum(w * r.get(e, 0.0) for w, r in zip(wts, results)) / total for e in keys}


class MultiHead:
    """多头: heads 并行 run, combine 归并。"""

    def __init__(self, heads, combine="mean", weights=None):
        self.heads = list(heads)
        self.combine = combine
        self.weights = weights

    def run(self, sequence, ctx=None):
        results = [h.run(sequence, ctx) for h in self.heads]
        return combine_results(results, self.combine, self.weights)
