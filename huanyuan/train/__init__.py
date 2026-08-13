"""train/ —— 训练模块

输入: 待训练 token (外部给定) → synth 样本集。
输出: {model, losses, ckpt_path} — 训练好的模型 + 统计 + 检查点 (供 infer/verify)。

接口:
  train(token_eid, n_samples, epochs, shape_method, ckpt_path, ...) → 训练产物
  ckpt 含 {state, vocab, shape_method, token, losses}。
"""
from __future__ import annotations

import torch

from synth import build_sample_set
from .data import vocab, collate, rev_vocab
from .model import TokenTransformer
from .loop import train_epoch, evaluate
from archive import run_dir as _archive_run_dir, save_training


def train(token_eid, n_samples=50, epochs=3, seed=None, shape_method="sequence_counts",
          order="preorder", encode="counts", depth=1, ckpt_path=None, lr=1e-3,
          num_layers=2, archive_dir=None, exclude=None, tokens=None,
          input_mode="vector", dim=None, expand=None) -> dict:
    """训练接口: 合成样本集 (单或多 token) → 训练 → 归档。

    输出 {model, ckpt_path, losses, acc, vocab, shape_method, run_dir, config}。
    input_mode: 'vector' (整形向量) / 'ids' (token id 序列 + 位置, 排序/位置表达结构)。
    tokens: 待训练 ctoken 列表; exclude: 排除 token (拼装泛化目标)。
    """
    toks = list(tokens) if tokens else [token_eid]
    samples = []
    for t in toks:
        ss = build_sample_set(t, n_synth=n_samples, depth=depth, seed=seed, exclude=exclude)
        samples.extend(ss["synth_samples"])
    batch = collate(samples, output=shape_method, order=order, encode=encode,
                    input_mode=input_mode, expand=expand)
    num_concepts = len(vocab())
    dim = dim or (batch["inputs"].shape[2] if input_mode != "ids" else 64)
    model = TokenTransformer(dim=dim, num_concepts=num_concepts, num_layers=num_layers,
                             input_mode=input_mode)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    for _ in range(epochs):
        losses.append(train_epoch(model, batch, opt))
    loss, pos_acc, val_acc = evaluate(model, batch)

    config = {
        "token": token_eid,
        "tokens": toks,
        "synth": {"n_synth": n_samples, "depth": depth, "seed": seed,
                  "exclude": list(exclude) if exclude else []},
        "shaper": {"output": shape_method, "order": order, "encode": encode, "expand": expand},
        "model": {"num_layers": num_layers, "lr": lr, "dim": dim, "num_concepts": num_concepts,
                  "input_mode": input_mode},
        "train": {"epochs": epochs},
    }
    if archive_dir is None:
        archive_dir = _archive_run_dir(token_eid)
    save_training(
        archive_dir, config, samples, model.state_dict(),
        {"losses": losses, "acc": pos_acc, "valid_acc": val_acc},
    )

    if ckpt_path:
        torch.save({
            "state": model.state_dict(),
            "vocab": vocab(),
            "shape_method": shape_method,
            "order": order,
            "encode": encode,
            "token": token_eid,
            "losses": losses,
        }, ckpt_path)
    return {
        "model": model,
        "ckpt_path": ckpt_path,
        "losses": losses,
        "acc": pos_acc,
        "valid_acc": val_acc,
        "vocab": vocab(),
        "shape_method": shape_method,
        "token": token_eid,
        "run_dir": archive_dir,
        "config": config,
    }


def _qat_round_weights(model, bits):
    """int8 量化感知 (QAT): 权重约束到量化格点 (straight-through).

    每 epoch 后把权重量化到 int 格点再反量化, 模拟 int8 训练误差.
    bits: 量化位宽 (8 = int8). 用权重逐张量的 scale 对称量化.
    原地修改 model 参数 (浮点权重被夹到量化格点, 训练继续).
    """
    import torch
    qmax = 2 ** (bits - 1) - 1
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "weight" not in name or p.dim() < 1:
                continue
            scale = p.abs().max().clamp(min=1e-12) / qmax
            q = torch.round(p / scale).clamp(-qmax, qmax)
            p.copy_(q * scale)


def train_seq(samples, epochs=10, lr=1e-3, num_layers=2, dim=64, seed=None,
              archive_dir=None, token="multi", max_n=None, causal=False, shift=False,
              valid_w=0.5, batch_size=512, profile=False, epoch_eval_fn=None,
              qat_bits=None) -> dict:
    """序列样本训练 (语义正负例, 如 multidigit 等式).

    训练: 位置 CE (正例, 奖励对的) + 合法性 CE (对错区分)。
    causal+shift: 自回归 (位置 i 只看 ≤i, 目标=下一个 token, 根除 copy 捷径)。
    batch_size: mini-batch 大小 (GPU 内存友好, 长序列大样本防 OOM)。
    epoch_eval_fn: 每 epoch 后调用 (model → 泛化指标 float), 逐 epoch 泛化曲线
      (精细化控制训练量 + 反馈监督训练基础设施)。
    profile: 对训练跑 cProfile, 热点持久化归档 (TRAIN_PROFILE=1 环境变量等价).
    输出 {model, run_dir, losses, acc, valid_acc, epoch_gen, config}。
    """
    import os
    if profile or os.environ.get("TRAIN_PROFILE"):
        import cProfile
        import io as _io
        import pstats
        pr = cProfile.Profile()
        pr.enable()
        res = _train_seq_impl(samples, epochs, lr, num_layers, dim, seed,
                              archive_dir, token, max_n, causal, shift, valid_w,
                              batch_size, epoch_eval_fn, qat_bits)
        pr.disable()
        s = _io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(30)
        _append_train_profile(res["run_dir"], s.getvalue())
        return res
    return _train_seq_impl(samples, epochs, lr, num_layers, dim, seed,
                           archive_dir, token, max_n, causal, shift, valid_w,
                           batch_size, epoch_eval_fn, qat_bits)


def _train_seq_impl(samples, epochs, lr, num_layers, dim, seed,
                    archive_dir, token, max_n, causal, shift, valid_w,
                    batch_size=512, epoch_eval_fn=None, qat_bits=None) -> dict:
    import random
    import torch
    if max_n and len(samples) > max_n:
        rng = random.Random(seed)
        samples = rng.sample(samples, max_n)
    if seed is not None:
        torch.manual_seed(seed)
    num_concepts = len(vocab())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TokenTransformer(dim=dim, num_concepts=num_concepts, num_layers=num_layers,
                             input_mode="ids", causal=causal).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    B = len(samples)
    # 流式训练: 分批 collate (不一次性全量 batch, 长序列大样本防 OOM)
    losses = []
    epoch_gen = []
    for ep in range(epochs):
        for i in range(0, B, batch_size):
            mb = collate(samples[i:i + batch_size], input_mode="ids", shift=shift)
            mb = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in mb.items()}
            losses.append(train_epoch(model, mb, opt, valid_w=valid_w))
        # int8 量化感知训练 (QAT): 每 epoch 后把权重量化回 int8 格点
        # (straight-through: 权重约束到量化格点, 模拟 int8 训练精度损失).
        if qat_bits:
            _qat_round_weights(model, qat_bits)
        if epoch_eval_fn:
            epoch_gen.append(epoch_eval_fn(model))
    # evaluate 流式 (分批平均, 防长序列大样本 OOM)
    tot_loss = tot_pos = tot_valid = 0.0
    tot_n = 0
    for i in range(0, B, batch_size):
        mb = collate(samples[i:i + batch_size], input_mode="ids", shift=shift)
        mb = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in mb.items()}
        l, pa, va = evaluate(model, mb, valid_w=valid_w)
        tot_loss += l
        tot_pos += pa
        tot_valid += va
        tot_n += 1
    loss, pos_acc, val_acc = tot_loss / tot_n, tot_pos / tot_n, tot_valid / tot_n

    config = {
        "token": token,
        "synth": {"kind": "seq", "n": len(samples)},
        "shaper": {"output": "ids", "input_mode": "ids", "shift": shift},
        "model": {"num_layers": num_layers, "lr": lr, "dim": dim,
                  "num_concepts": num_concepts, "input_mode": "ids", "causal": causal},
        "train": {"epochs": epochs},
    }
    if archive_dir is None:
        archive_dir = _archive_run_dir(token)
    save_training(archive_dir, config, samples, model.state_dict(),
                  {"losses": losses, "acc": pos_acc, "valid_acc": val_acc,
                   "epoch_gen": epoch_gen})
    return {"model": model, "run_dir": archive_dir, "losses": losses,
            "acc": pos_acc, "valid_acc": val_acc, "epoch_gen": epoch_gen,
            "config": config}


def _append_train_profile(run_dir, profile_text):
    """训练 profile 持久化: 追加到归档 config.json['train_profile'] (性能观测留存)."""
    import json
    import os
    p = os.path.join(run_dir, "config.json")
    cfg = {}
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as f:
            cfg = json.load(f)
    cfg["train_profile"] = profile_text
    with open(p, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


__all__ = ["train", "vocab", "collate", "rev_vocab", "TokenTransformer"]
