"""infer/decode.py —— 推理解码 (模型输出 → 概念序列)

logits [L, V] → argmax → 概念 eid (rev_vocab 反查)。
"""
from __future__ import annotations


def decode(logits, rev_v, k=1) -> list:
    """logits [L, V] → 概念 eid 列表 (每位置 argmax)。"""
    import torch
    idxs = torch.argmax(logits, dim=-1).tolist()
    return [rev_v.get(i, i) for i in idxs]
