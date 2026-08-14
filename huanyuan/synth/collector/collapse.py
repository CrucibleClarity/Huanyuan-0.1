"""synth/collector/collapse.py —— 收拢器 (嵌套向量 → 最终向量)

将样本的嵌套向量 (每概念派生计数 counts) 合成为 transformer 接收的固定维向量
(eid 空间)。方法可插拔:
  sum            各 token counts 求和
  mean           各 token counts 平均
  depth_weighted 按嵌套深度加权 (浅层权重高)
  root           根概念 counts (不做聚合)
"""
from __future__ import annotations

from tokenizer import api

METHODS = ("sum", "mean", "depth_weighted", "root")


def _space():
    return api.eid_space()


def collapse(sample, method="sum") -> list[float]:
    """样本 (含 ast/seq/vectors) → 固定维向量。

    method 见 METHODS; 样本的 counts 分布在 eid 空间, 越界 eid 忽略。
    """
    if method not in METHODS:
        raise ValueError(f"未知向量合成方法: {method!r} ({METHODS})")
    space = _space()
    idx = {e: i for i, e in enumerate(space)}
    dim = len(space)
    vec = [0.0] * dim

    def add_counts(vectors, weight=1.0):
        for v in vectors:
            for e, c in v.get("counts", {}).items():
                if e in idx:
                    vec[idx[e]] += c * weight

    if method == "root":
        add_counts(sample["vectors"][:1])
    elif method == "sum":
        add_counts(sample["vectors"])
    elif method == "mean":
        n = max(len(sample["vectors"]), 1)
        add_counts(sample["vectors"], 1.0 / n)
    elif method == "depth_weighted":
        dm = api.depth_map(sample["ast"])
        for v in sample["vectors"]:
            w = 1.0 / (1.0 + dm.get(v["eid"], 0))
            for e, c in v.get("counts", {}).items():
                if e in idx:
                    vec[idx[e]] += c * w
    return vec


def vector_spec() -> dict:
    """向量维度规格 (供 transformer 配置)。"""
    space = _space()
    return {"dim": len(space), "space": space, "methods": list(METHODS)}
