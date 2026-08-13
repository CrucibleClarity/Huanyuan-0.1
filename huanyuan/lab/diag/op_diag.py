#!/usr/bin/env python3
"""op_diag.py —— 单算子训练/OOD 样本对比诊断 (持久化工具)

诊断算子为何未达标 (0-acc): 对比该算子 token 在 训练样本 vs OOD 判定样本
中的 上下文搭配 (左右邻 token 类别/具体 token) + 出现频率 + 求值有效性.

用法: PYTHONPATH=src python -m lab.diag.op_diag --op root
       PYTHONPATH=src python -m lab.diag.op_diag --op tetration
"""
import argparse
from collections import Counter

from tokenizer import api
from lab import synth_core
from lab.synth_core import token_category


def analyze(samples, op_eid):
    """算子 token 在样本集合中的上下文统计."""
    n_samples = n_pos = n_neg = 0
    left_ctx = Counter()
    right_ctx = Counter()
    left_cat = Counter()
    right_cat = Counter()
    for s in samples:
        seq = s["seq"]
        if op_eid not in seq:
            continue
        n_samples += 1
        if s.get("truth"):
            n_pos += 1
        else:
            n_neg += 1
        for i, e in enumerate(seq):
            if e != op_eid:
                continue
            if i > 0:
                l = seq[i - 1]
                left_ctx[api.name(l)] += 1
                left_cat[token_category(l)] += 1
            if i < len(seq) - 1:
                r = seq[i + 1]
                right_ctx[api.name(r)] += 1
                right_cat[token_category(r)] += 1
    return {"n": n_samples, "pos": n_pos, "neg": n_neg,
            "left_ctx": left_ctx.most_common(8), "right_ctx": right_ctx.most_common(8),
            "left_cat": left_cat.most_common(5), "right_cat": right_cat.most_common(5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", required=True)
    ap.add_argument("--hi", type=int, default=9)
    ap.add_argument("--config", default="lab/configs/numeral_v4.json")
    args = ap.parse_args()

    import json
    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    train, npos, nneg = synth_core.compose_samples(samples=cfg["synth"]["samples"],
                                                   seed=cfg.get("seed", 0))
    op_eid = api.eid_by_name(args.op)

    # 训练样本 (从完整配置抽取, 含该算子)
    print(f"===== {args.op} ({op_eid}) 诊断 =====")
    print(f"\n[训练样本] 总数 {len(train)} (真{npos} 假{nneg})")
    tr = analyze(train, op_eid)
    print(f"  含 {args.op} 的样本: {tr['n']} (真{tr['pos']} 假{tr['neg']})")
    print(f"  左邻: {tr['left_ctx']}")
    print(f"  右邻: {tr['right_ctx']}")
    print(f"  左类: {tr['left_cat']}  右类: {tr['right_cat']}")

    # OOD 判定样本
    print(f"\n[OOD 判定样本] (mixed 5位)")
    for mode in ("mixed",):
        ood, op_n, on_neg = synth_core.ood_samples(op=args.op, digits=5, n=300,
                                                   mode=mode, neg_mode=1, seed=12345)
        od = analyze(ood, op_eid)
        print(f"  含 {args.op} 的 OOD: {od['n']} (真{od['pos']} 假{od['neg']})")
        print(f"  左邻: {od['left_ctx']}")
        print(f"  右邻: {od['right_ctx']}")
        print(f"  左类: {od['left_cat']}  右类: {od['right_cat']}")


if __name__ == "__main__":
    main()
