#!/usr/bin/env python3
"""op_cooccur.py —— 0-acc 算子样本搭配诊断 (对比 1-acc 算子)

视角:
  1. 出现频次       算子 token 的样本数/出现次数/左右邻
  2. n-gram 搭配量  含算子的 2/3/4-gram 计数分布 (多样性/低频)
  3. token 类搭配   含算子的 n-gram → 类别序列 (token_category) 分布 (语义多样性)
  4. 横向替换       固定算子位置上下文, 同槽位还出现过哪些其他 token
                    (左右全同但算子被替换 — 揭示该位置的 token 区分难度)

用法: PYTHONPATH=src/llm_research_v5 python -m lab.diag.op_cooccur --config lab/configs/numeral_v3.json
"""
import argparse
import json
from collections import Counter, defaultdict

from tokenizer import api
from lab import synth_core
from lab.synth_core import token_category


def build_samples(cfg):
    s = cfg["synth"]
    train, npos, nneg = synth_core.compose_samples(samples=s["samples"], seed=cfg.get("seed", 0))
    return train


def op_stats(samples, op_name, ctx_w=2):
    """目标算子 token 的样本搭配统计 (含类搭配 + 横向替换)."""
    op_eid = api.eid_by_name(op_name)
    total = 0          # 算子 token 总出现次数
    n_samples = 0      # 含算子 token 的样本数
    ngrams = defaultdict(int)          # 含算子的 n-gram → 计数
    class_grams = defaultdict(int)     # 含算子的 n-gram → 类别序列计数
    left_ctx = Counter()   # 紧邻左 token
    right_ctx = Counter()  # 紧邻右 token
    slots = defaultdict(Counter)   # (左ctx, 右ctx) 窗口 → 该槽位出现过的 token 计数

    for s in samples:
        seq = s["seq"]
        if op_eid not in seq:
            continue
        n_samples += 1
        for i, e in enumerate(seq):
            if e != op_eid:
                continue
            total += 1
            if i > 0:
                left_ctx[api.name(seq[i - 1])] += 1
            if i < len(seq) - 1:
                right_ctx[api.name(seq[i + 1])] += 1
            # 横向替换: (左 ctx 窗口, 右 ctx 窗口) → 槽位 token
            lw = tuple(api.name(x) for x in seq[max(0, i - ctx_w):i])
            rw = tuple(api.name(x) for x in seq[i + 1:i + 1 + ctx_w])
            slots[(lw, rw)][op_name] += 1
            # n-gram + 类序列
            for n in (2, 3, 4):
                for j in range(max(0, i - n + 1), min(i, len(seq) - n + 1)):
                    tup = tuple(api.name(x) for x in seq[j:j + n])
                    if op_name in tup:
                        ngrams[(n, tup)] += 1
                        cseq = tuple(token_category(x) for x in tup)
                        class_grams[(n, cseq)] += 1
    # 横向替换聚合: 每个窗口槽位, 收集该槽位所有候选 token (横向对比)
    # 单一 token 槽位也显示 (标识: 该位置无替换候选 = 槽位专属, 模型无横向竞争)
    slot_view = []
    for (lw, rw), cnt in slots.items():
        others = {t: c for t, c in cnt.items() if t != op_name}
        slot_view.append((lw, rw, dict(cnt), sum(c for c in others.values())))
    slot_view.sort(key=lambda x: (-x[3], -sum(x[2].values())))
    # 分 n 汇总
    by_n = {}
    for n in (2, 3, 4):
        cnts = [c for (nn, _), c in ngrams.items() if nn == n]
        by_n[n] = {"uniq": len(cnts), "min": min(cnts) if cnts else 0,
                   "total": sum(cnts), "low": sum(1 for c in cnts if c < 3)}
    by_nc = {}
    for n in (2, 3, 4):
        cnts = [c for (nn, _), c in class_grams.items() if nn == n]
        by_nc[n] = {"uniq": len(cnts), "min": min(cnts) if cnts else 0,
                    "total": sum(cnts)}
    return {"eid": op_eid, "n_samples": n_samples, "total": total,
            "left_ctx": left_ctx.most_common(6), "right_ctx": right_ctx.most_common(6),
            "by_n": by_n, "by_n_class": by_nc, "slots": slot_view[:5]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    train = build_samples(cfg)
    print(f"训练样本: n={len(train)}")

    zero_ops = ["multiplication", "subtraction", "power", "root", "tetration", "neg"]
    one_ops = ["addition", "division"]
    print("\n===== 0-acc 算子 =====")
    for op in zero_ops:
        st = op_stats(train, op)
        print(f"\n--- {op} (eid={st['eid']}) 样本数={st['n_samples']} 出现次数={st['total']} ---")
        print(f"  左邻居: {st['left_ctx']}")
        print(f"  右邻居: {st['right_ctx']}")
        for n in (2, 3, 4):
            b = st["by_n"][n]
            print(f"  {n}-gram: {b['uniq']} 种 / 最少 {b['min']} / 总 {b['total']} / 低频(<3) {b['low']}")
        for n in (2, 3, 4):
            b = st["by_n_class"][n]
            print(f"  类{n}-gram: {b['uniq']} 种 / 最少 {b['min']} / 总 {b['total']}")
        for lw, rw, cnt, n_other in st["slots"]:
            print(f"  槽位 {'|'.join(lw)} [{'/'.join(cnt.keys())}] {'|'.join(rw)}"
                  f" (本算子{cnt[op]} + 其他{n_other})")

    print("\n===== 1-acc 算子 (对比基准) =====")
    for op in one_ops:
        st = op_stats(train, op)
        print(f"\n--- {op} (eid={st['eid']}) 样本数={st['n_samples']} 出现次数={st['total']} ---")
        print(f"  左邻居: {st['left_ctx']}")
        print(f"  右邻居: {st['right_ctx']}")
        for n in (2, 3, 4):
            b = st["by_n"][n]
            print(f"  {n}-gram: {b['uniq']} 种 / 最少 {b['min']} / 总 {b['total']} / 低频(<3) {b['low']}")
        for n in (2, 3, 4):
            b = st["by_n_class"][n]
            print(f"  类{n}-gram: {b['uniq']} 种 / 最少 {b['min']} / 总 {b['total']}")
        for lw, rw, cnt, n_other in st["slots"]:
            print(f"  槽位 {'|'.join(lw)} [{'/'.join(cnt.keys())}] {'|'.join(rw)}"
                  f" (本算子{cnt[op]} + 其他{n_other})")


if __name__ == "__main__":
    main()
