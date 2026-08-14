"""lab/exp/eval_helpers.py —— 实验统一评估工具 (run_exp _judge_eval 权威口径)

所有实验脚本共用:
  load_model(run_dir)     加载训练模型 (causal=False 与 train_seq 一致)
  judge_batch(model, ss)  批量判定口径 (_judge_eval, 全序列重建)
  judge_full(model, seq)  单序列全位置预测 (诊断用)
"""
from __future__ import annotations

import torch

from train.data import vocab, collate, rev_vocab
from train.model import TokenTransformer


def load_model(run_dir: str, dim: int = 64, num_layers: int = 2) -> TokenTransformer:
    """加载 run_dir 的 model.pt (TokenTransformer, causal=False 与训练一致).

    注意: model.pt 是 state_dict; 架构须与训练 config 一致 (dim/layers).
    """
    v = vocab()
    m = TokenTransformer(dim=dim, num_concepts=len(v), num_layers=num_layers,
                         input_mode="ids", causal=False).eval()
    m.load_state_dict(torch.load(f"{run_dir}/model.pt", weights_only=False,
                                 map_location="cpu"))
    return m


def judge_batch(model, samples):
    """批量判定口径 (run_exp._judge_eval): 全序列重建, 全对才计正确.

    权威口径 (用户确立): 每样本非 padding 全部位置预测与真实 seq 一致.
    返回 (acc, 判真率, 判假率, 一致性).
    """
    from lab import run_exp
    return run_exp._judge_eval(model, samples)


def judge_full(model, seq):
    """单序列全位置预测 (诊断用): 返回 (是否全对, 真实 names, 预测 names)."""
    from tokenizer import api
    batch = collate([{"seq": seq, "valid": 1}], input_mode="ids")
    with torch.no_grad():
        logits, _ = model(batch["inputs"], mask=batch["mask"])
    rv = rev_vocab()
    rl = batch["lengths"][0]
    preds = [rv[p] for p in logits[0, :rl].argmax(dim=1).tolist()]
    names = [api.name(x) for x in seq]
    return preds == names, names, preds


def judge_many(model, seqs):
    """批量判定多序列 (避免单样本 collate 边界问题)."""
    samples = [{"seq": s, "valid": 1} for s in seqs]
    if len(samples) == 1:
        samples = samples + [{"seq": seqs[0], "valid": 1}]
    return judge_batch(model, samples)
