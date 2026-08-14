"""shaper/unshape.py —— 输出解析 (模型向量 → 概念)

概念原型表 (codebook): 每概念 eid → 归一化 counts 向量。
模型输出向量 → 与全原型余弦 (点积, 原型已归一) → top-k 概念 eid。
双向可逆: counts(概念) ↔ 向量 ↔ 解析回概念。
"""
from __future__ import annotations

from tokenizer import api

_PROTO = None


def prototypes(force=False) -> dict:
    """概念原型表 {eid: 归一化 counts 向量} (缓存)。"""
    global _PROTO
    if _PROTO is None or force:
        space = api.eid_space()
        idx = {e: i for i, e in enumerate(space)}
        dim = len(space)
        tbl = {}
        for e in api.all_concepts():
            row = [0.0] * dim
            for k, c in api.counts(e).items():
                if k in idx:
                    row[idx[k]] = c
            n = sum(x * x for x in row) ** 0.5
            if n > 0:
                tbl[e] = [x / n for x in row]
        _PROTO = tbl
    return _PROTO


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def unshape(vec, k=1) -> list[str]:
    """输出向量 → top-k 最接近概念 eid (原型已归一, 点积=余弦)。"""
    tbl = prototypes()
    return sorted(tbl, key=lambda e: _dot(vec, tbl[e]), reverse=True)[:k]


def reset() -> None:
    global _PROTO
    _PROTO = None
