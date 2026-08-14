"""docs/paper_data/scripts/quant_q12.py —— 现状最优模型 q1/q2 量化评估

方案 1 (用户选择): PyTorch 手写块量化 (零外部依赖, 无 bitsandbytes):
  q1: 每块 1 bit 对称量化 (scale 存 fp16, 权重存 1bit)
  q2: 每块 2 bit 对称量化

对 exp02_supervised_s2 (最优模型, 判定 1.000) 量化, 测量化前后判定口径
(run_exp _judge_eval 权威口径) — 测 1bit/2bit 量化的精度损失.

用法: PYTHONPATH=. python -m docs.paper_data.scripts.quant_q12
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from tokenizer import api
from train.data import vocab
from train.model import TokenTransformer

# ---- 块量化原语 (对称, 零外部依赖) ----

def block_quantize(tensor, bits, block=64):
    """对称块量化: 权重按 block 分块, 每块 scale=max_abs/(2^(bits-1)-1),
    量化到 [-2^(bits-1)+1, 2^(bits-1)-1] 整数, 反量化回 fp32.

    返回 (q_int, scale) — 可精确反量化.
    """
    flat = tensor.reshape(-1)
    n = flat.numel()
    n_blocks = (n + block - 1) // block
    # pad 到 block 整数倍
    pad = n_blocks * block - n
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype)])
    blocks = flat.view(n_blocks, block)
    qmax = 2 ** (bits - 1) - 1
    scale = blocks.abs().max(dim=1).values / qmax
    scale = torch.clamp(scale, min=1e-12)
    q = torch.round(blocks / scale.unsqueeze(1)).clamp(-qmax, qmax)
    return q, scale, n, pad


def block_dequantize(q, scale, n, pad):
    """反量化 (块量化重建)."""
    blocks = q * scale.unsqueeze(1)
    flat = blocks.reshape(-1)
    if pad:
        flat = flat[:n]
    return flat


def quantize_state_dict(sd, bits, block=64, min_params=1000):
    """state_dict 块量化 → 新 state_dict (权重层量化, bias/scale 层保留).

    只量化参数数 ≥ min_params 的权重层 (线性/注意力大层); norm/bias 保留 fp32
    (量化无意义且破坏稳定性). 量化权重用 int16 存 (块量化值范围小).
    """
    qsd = {}
    meta = {}
    for k, v in sd.items():
        if v.dim() >= 1 and "weight" in k and v.numel() >= min_params:
            q, scale, n, pad = block_quantize(v, bits, block)
            qsd[k] = q.to(torch.int16)
            qsd[f"{k}_scale"] = scale
            qsd[f"{k}_qn"] = torch.tensor([n])
            qsd[f"{k}_qpad"] = torch.tensor([pad])
            qsd[f"{k}_shape"] = torch.tensor(list(v.shape))
            meta[k] = (q.dtype, v.dtype)
        else:
            qsd[k] = v
    return qsd, meta


def dequantize_state_dict(qsd):
    """反量化回 fp32 state_dict (加载后前向)."""
    sd = {}
    for k, v in qsd.items():
        if k.endswith(("_scale", "_qn", "_qpad", "_shape")):
            continue
        if f"{k}_scale" in qsd:
            scale = qsd[f"{k}_scale"]
            n = int(qsd[f"{k}_qn"].item())
            pad = int(qsd[f"{k}_qpad"].item())
            orig_shape = tuple(int(x) for x in qsd[f"{k}_shape"].tolist())
            sd[k] = block_dequantize(v, scale, n, pad).reshape(orig_shape).to(torch.float32)
        else:
            sd[k] = v
    return sd


# ---- 评估 ----

def build_ood(samples_list):
    return [{"seq": s["seq"], "valid": 1} for s in samples_list]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="archive/log/train/exp02_supervised_s2_20260811_081412")
    ap.add_argument("--block", type=int, default=64)
    args = ap.parse_args()
    run = args.run

    v = vocab()
    # 原模型
    m = TokenTransformer(dim=64, num_concepts=len(v), num_layers=2,
                         input_mode="ids", causal=False).eval()
    sd = torch.load(f"{run}/model.pt", weights_only=False, map_location="cpu")
    m.load_state_dict(sd)

    # 评估样本 (同 exp10_impl 的 math 命题 + balanced)
    from lab import synth_core
    from lab import run_exp
    from eval_helpers import judge_many

    ss_arith = synth_core.logic_arith_samples(ops=["logical_and", "logical_or", "logical_imply"], hi=9, seed=1, notation="prefix")[0]
    ss_bal = synth_core.balanced_samples(max_depth=2, hi=9, op=api.eid_by_name("addition"), neg_mode=1)[0][:200]
    test = build_ood(list(ss_arith) + list(ss_bal))
    print(f"测试样本: {len(test)} (logic 算术 + addition)")

    # 基线判定
    a0, *_ = judge_many(m, [s["seq"] for s in test])
    print(f"基线 (fp32): acc={a0:.4f}")

    # q1 / q2
    for bits, label in ((1, "q1"), (2, "q2")):
        qsd, meta = quantize_state_dict(sd, bits, args.block)
        dq = dequantize_state_dict(qsd)
        mq = TokenTransformer(dim=64, num_concepts=len(v), num_layers=2,
                              input_mode="ids", causal=False).eval()
        mq.load_state_dict(dq)
        a, *_ = judge_many(mq, [s["seq"] for s in test])
        # 存储大小
        import io
        buf = io.BytesIO()
        torch.save(qsd, buf)
        size = buf.tell()
        print(f"{label} (block={args.block}): acc={a:.4f} (Δ={a - a0:+.4f}) 量化后大小={size:,}B")


if __name__ == "__main__":
    main()
