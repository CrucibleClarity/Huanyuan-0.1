"""verify/views.py —— 结果视图生成器

输出:
  overall_acc         整体正确率 (训练样本位置级)
  ood_acc             样本外正确率
  gen_acc             泛化样本整体正确率
  curve               逐 token 训练曲线 (每 epoch loss)
  per_token           逐 token 正确率/泛化成功率 {概念名: {correct, total, acc}}
"""
from __future__ import annotations

from tokenizer import api


def _acc(outs):
    total = sum(o.get("len", 0) for o in outs)
    match = sum(o.get("matched", 0) for o in outs)
    return match / total if total else 0.0


def _per_token(outs):
    """逐 token 正确率: {eid: {correct, total}} (原生化, 统计 OOD 所有 token)."""
    stat = {}
    for o in outs:
        for p, t in zip(o["pred"], o["true"]):
            s = stat.setdefault(t, {"correct": 0, "total": 0})
            s["total"] += 1
            if p == t:
                s["correct"] += 1
    return {api.name(e): {"correct": v["correct"], "total": v["total"],
                          "acc": v["correct"] / v["total"] if v["total"] else 0.0}
            for e, v in stat.items()}


def build_views(outs_train, outs_ood, outs_gen, losses, untrained=None, depth_outs=None) -> dict:
    """组装视图 dict.

    depth_outs: {depth: outs} 多层嵌套泛化.
    """
    depth_gen = {d: _acc(outs) for d, outs in (depth_outs or {}).items()}
    return {
        "overall_acc": _acc(outs_train),
        "ood_acc": _acc(outs_ood),
        "gen_acc": _acc(outs_gen),
        "depth_gen": depth_gen,
        "curve": {"losses": losses},
        "per_token_train": _per_token(outs_train),
        "per_token_gen": _per_token(outs_gen),
    }
