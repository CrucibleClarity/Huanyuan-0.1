"""infer/ —— 推理模块

输入: 模型 (train 产物) + 样本。
输出: 每样本 {notation, pred, true, matched} — 预测概念序列 + 真值 + 逐位置匹配数。
数据管线供 verify 消费 (pred/true/matched)。

接口:
  infer(model_or_ckpt, samples, shape_method) → 推理结果列表
"""
from __future__ import annotations

import torch

from synth import shape
from .decode import decode
from train import collate, rev_vocab


def infer(model, samples, shape_method=None, order="preorder", encode="counts",
          ckpt=None, expand=None, batch_size=None) -> list[dict]:
    """推理接口: 样本 → 预测概念序列。

    model: train 产物 dict 或 torch Module; samples: synth 样本列表。
    expand: 显式语法展开深度 (训练形态一致); 缺省从 model dict 读, Module 默认 None。
    batch_size: 分批推理 (流式输出, 防长序列大样本 OOM; None=一次全批)。
    输出每样本 {notation, pred, true, matched} (verify 数据管线)。
    """
    if isinstance(model, dict):
        if model.get("shape_method"):
            shape_method = shape_method or model["shape_method"]
        order = model.get("order", order)
        encode = model.get("encode", encode)
        input_mode = model.get("input_mode", "vector")
        expand = expand if expand is not None else model.get("expand")
        if "model" in model and hasattr(model["model"], "state_dict"):
            model = model["model"]
        elif "state" in model:
            from train import TokenTransformer
            m = TokenTransformer(
                dim=next(iter(model["state"].values())).shape[-1],
                num_concepts=len(model["vocab"]),
                input_mode=model.get("input_mode", "vector"),
                causal=model.get("causal", False),
            )
            m.load_state_dict(model["state"])
            model = m
        else:
            raise TypeError("不认识的 model dict (需 train 产物或 ckpt {state, vocab})")
    else:
        input_mode = getattr(model, "input_mode", "vector")
        expand = None
    sm = shape_method or "sequence_counts"
    model.eval()
    rv = rev_vocab()
    rv_map = [rv.get(i, i) for i in range(len(rv))]   # id→token 列表 (批量反查, 替代逐 token dict)
    results = []
    B = len(samples)
    step = batch_size or B
    with torch.no_grad():
        for i0 in range(0, B, step):
            chunk = samples[i0:i0 + step]
            batch = collate(chunk, output=sm, order=order, encode=encode,
                            input_mode=input_mode, expand=expand)
            logits, _ = model(batch["inputs"], mask=batch["mask"])
            all_idx = torch.argmax(logits, dim=-1)          # 每批一次 argmax [b,L]
            for j, s in enumerate(chunk):
                ln = batch["lengths"][j]
                pred = [rv_map[k] for k in all_idx[j][:ln].tolist()]
                true = list(s["seq"])
                # 题型样本 (fill/choose): gap 位置评估答案 (与 collate 训练目标一致)
                if "gap_pos" in s and "answer" in s:
                    for jj, a in enumerate(s["answer"]):
                        idx = s["gap_pos"] + jj
                        if idx < len(true):
                            true[idx] = a
                results.append({
                    "notation": s.get("notation"),
                    "pred": pred,
                    "true": true,
                    "matched": sum(1 for p, t in zip(pred, true) if p == t),
                    "len": len(true),
                })
    return results


__all__ = ["infer"]
