"""train/data.py —— 数据管线: 样本集 → 训练批次张量

vocab: 概念全集 → {eid: idx} (与 unshape 原型空间一致, 固定排序)。
collate: samples (synth 产出) → 张量批次 {inputs, targets, lengths, mask}。
  inputs  [B, L, dim] 整形向量; targets [B, L] 概念 idx (pad=-100, CE ignore);
  mask    [B, L] padding 标记 (True=忽略); lengths [B] 每样本真实长度。
"""
from __future__ import annotations

import torch

from tokenizer import api
from synth import shape


def vocab() -> dict:
    """概念全集 → {eid: idx} (排序固定, 供输入/输出/解析共用)。

    C 层概念 + A 层箭头 (coercion 提升等箭头 token 必须可预测/编码)。
    """
    eids = set(api.all_concepts())
    eids |= set(api.all_arrows())
    return {e: i for i, e in enumerate(sorted(eids))}


def rev_vocab() -> dict:
    return {i: e for e, i in vocab().items()}


def head_weights(seq):
    """每位置注意力先验权重 (用户读文字的注意力管线)。

    谓词 (有排列方法的运算子/关系词) 高权重; 数字/原子低权重。
    谓词权重 + 同类/异类比较: 运算子异类突出, 数字同类平凡。
    """
    w = []
    for e in seq:
        if api.is_concept(e) and api.arrange_of(e):
            w.append(2.0)
        else:
            w.append(1.0)
    return w


def _expanded_seq(s, depth=1):
    """语法展开序列: 每 token + 沿定义引用展开 depth 层的概念 (跳过 exclude)。"""
    exclude = set(s.get("exclude") or [])

    def refs(e, d):
        if d <= 0:
            return []
        out = []
        for r in api.bracket(e):
            if api.is_concept(r) and r not in exclude:
                out.append(r)
                out.extend(refs(r, d - 1))
        return out

    out = []
    for e in s["seq"]:
        if e not in exclude:
            out.append(e)
        out.extend(refs(e, depth))
    return out


def collate(samples, output="sequence_counts", order="preorder", encode="counts",
            input_mode="vector", max_len=None, expand=None, shift=False) -> dict:
    """样本集 → 批次张量。

    input_mode='vector': inputs [B,L,dim] 整形向量 (shaper);
    input_mode='ids':    inputs [B,L] 概念 id 序列 (Embedding + 位置)。
    expand: ids 模式下语法展开深度 (None=平铺序列, 1+=沿定义引用展开)。
    shift:  自回归目标 (targets[j]=seq[j+1], 预测下一个 token; 需配合 causal mask)。
    返回 {inputs, targets, lengths, mask}; targets pad=-100 (CE ignore)。
    """
    v = vocab()
    seqs = [_expanded_seq(s, expand) if (expand and input_mode == "ids") else list(s["seq"])
            for s in samples]
    L = max((len(x) for x in seqs), default=0)
    if max_len:
        L = min(L, max_len)
    B = len(samples)
    targets = torch.full((B, L), -100, dtype=torch.long)
    valid = torch.ones(B, dtype=torch.long)
    v_get = v.get
    lengths = []
    for i, seq in enumerate(seqs):
        ln = min(len(seq), L)
        lengths.append(ln)
        if samples[i].get("valid") == 0:
            valid[i] = 0
            continue
        for j, e in enumerate(seq[:ln]):
            tgt = j + 1 if shift else j
            if tgt < ln and (vid := v_get(seq[tgt])) is not None:
                targets[i, j] = vid
        # 题型样本 (fill/choose): gap 位置监督答案 (question token 体系)
        if "gap_pos" in samples[i] and "answer" in samples[i]:
            gp = samples[i]["gap_pos"]
            for jj, a_tok in enumerate(samples[i]["answer"]):
                if gp + jj < L and (aid := v_get(a_tok)) is not None:
                    targets[i, gp + jj] = aid

    if input_mode == "ids":
        inputs = torch.zeros(B, L, dtype=torch.long)
        v_get = v.get
        for i, seq in enumerate(seqs):
            ids = [v_get(e, 0) for e in seq[:L]]
            inputs[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    else:
        vecs = [shape(s, output=output, order=order, encode=encode) for s in samples]
        dim = len(vecs[0][0]) if vecs and vecs[0] else 0
        inputs = torch.zeros(B, L, dim)
        for i, vv in enumerate(vecs):
            ln = min(len(vv), L)
            for j in range(ln):
                inputs[i, j] = torch.tensor(vv[j], dtype=torch.float)

    mask = torch.zeros(B, L, dtype=torch.bool)
    hw = torch.zeros(B, L)
    for i, seq in enumerate(seqs):
        ln = min(len(seq), L)
        mask[i, ln:] = True
        hw[i, :ln] = torch.tensor(head_weights(seq[:ln]), dtype=torch.float)
    return {"inputs": inputs, "targets": targets, "lengths": lengths, "mask": mask,
            "valid": valid, "head_weight": hw}
