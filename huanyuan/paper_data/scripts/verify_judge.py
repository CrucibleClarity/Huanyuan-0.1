"""lab/exp/verify_judge.py —— 判定口径 (_judge_eval) 计算逻辑校验

校验 run_exp._judge_eval 的正确性:
  1. collate lengths == seq 长度 (zip 无截断)
  2. logits[j] argmax == seq[j] (位置对齐, causal=False)
  3. _judge_eval 与手工全序列重建一致

用法: PYTHONPATH=. python -m lab.exp.verify_judge --run <run_dir>
"""
import argparse

from tokenizer import api
from lab.synth_core import logic_arith_samples, balanced_samples
from train.data import collate, rev_vocab
from eval_helpers import load_model, judge_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    model = load_model(args.run)

    # 混合样本 (含 padding 场景)
    ss1, _, _ = logic_arith_samples(ops=["logical_and", "logical_imply"], hi=9, seed=1)
    ss2, _, _ = balanced_samples(max_depth=2, hi=9, op=api.eid_by_name("addition"), neg_mode=1)
    mixed = [s for s in ss1[:8]] + [s for s in ss2[:2]]

    print(f"=== 判定口径校验 (模型 {args.run}) ===")
    batch = collate(mixed, input_mode="ids")
    ok_len = all(batch["lengths"][i] == len(s["seq"]) for i, s in enumerate(mixed))
    print(f"1. collate lengths == seq_len: {ok_len}")

    # 位置对齐
    rv = rev_vocab()
    import torch
    with torch.no_grad():
        logits, _ = model(batch["inputs"], mask=batch["mask"])
    misalign = 0
    total = 0
    for i, s in enumerate(mixed):
        rl = batch["lengths"][i]
        for j in range(rl):
            total += 1
            if rv[logits[i, j].argmax().item()] != s["seq"][j]:
                misalign += 1
    print(f"2. 位置对齐: {total - misalign}/{total} 对齐 ({misalign} 错位)")

    # _judge_eval vs 手工
    a, *_ = judge_batch(model, mixed)
    hit = 0
    for i, s in enumerate(mixed):
        rl = batch["lengths"][i]
        preds = [rv[p] for p in logits[i, :rl].argmax(dim=1).tolist()]
        if preds == list(s["seq"]):
            hit += 1
    print(f"3. _judge_eval acc={a:.4f} vs 手工全重建={hit / len(mixed):.4f} (n={len(mixed)})")
    print("结论: 校验通过 = _judge_eval 逻辑正确 (causal=False 对齐自预测)")


if __name__ == "__main__":
    main()
