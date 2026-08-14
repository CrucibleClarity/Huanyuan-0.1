"""lab/exp/exp80_extrap.py —— EXP-80: 直觉外推容易度 (轴 B)

外推距离衰减矩阵 (训练 2-9 位 / base 10):
  位宽: 2/9/20/100/1000/2000 位
  进制: 10/16/20/36/60
  匿名: 测试时置换 0/25/50/100% digit token
  联合: 位宽20 × 进制16

用法: PYTHONPATH=. python -m lab.exp.exp80_extrap --run <run_dir>
输出: 外推矩阵表 + docs/paper_data/exp80_results.md (更新)
"""
import argparse
import random

from tokenizer import api
from lab.judge import judge_sequence
from lab.synth_core import nested_seq, radix_ood_samples
from eval_helpers import load_model, judge_many

_ADD = None


def _add():
    global _ADD
    if _ADD is None:
        _ADD = api.eid_by_name("addition")
    return _ADD


def width_matrix(model, widths, n=30):
    """位宽外推: 训练 2-9 位 → 测试 20/100/1000/2000."""
    rng = random.Random(0)
    print("\n-- 位宽外推 --")
    for w in widths:
        seqs = []
        for _ in range(n):
            a = rng.randrange(10 ** (w - 1), 10 ** w)
            b = rng.randrange(10 ** (w - 1), 10 ** w)
            seq = judge_sequence(nested_seq([a, b], _add(), a + b), True)
            seqs.append(seq)
        a, *_ = judge_many(model, seqs)
        print(f"  位宽 {w}: acc={a:.3f} (n={len(seqs)})", flush=True)


def base_matrix(model, bases):
    """进制外推: 训练 base 10 → 16/20/36/60."""
    print("\n-- 进制外推 --")
    for base in bases:
        ss, _, _ = radix_ood_samples(ops=["addition"], bases=(base,), max_digits=20, n=20)
        if not ss:
            print(f"  base {base}: 无样本")
            continue
        a, *_ = judge_many(model, [s["seq"] for s in ss])
        print(f"  base {base}: acc={a:.3f} (n={len(ss)})")


def anon_matrix(model, pcts, n=30):
    """匿名置换: 测试时置换 X% digit token (序列结构不变)."""
    rng = random.Random(1)
    digs = [api.eid_by_name(x) for x in
            ("digit_zero", "digit_one", "digit_two", "digit_three", "digit_four",
             "digit_five", "digit_six", "digit_seven", "digit_eight", "digit_nine")]
    print("\n-- 匿名置换 (digit token 随机替换) --")
    for pct in pcts:
        seqs = []
        for _ in range(n):
            a = rng.randrange(10, 1000)
            b = rng.randrange(10, 1000)
            seq = list(judge_sequence(nested_seq([a, b], _add(), a + b), True))
            if pct > 0:
                idxs = [i for i, x in enumerate(seq) if x in digs]
                k = max(1, int(len(idxs) * pct / 100))
                for i in rng.sample(idxs, k):
                    seq[i] = rng.choice(digs)
            seqs.append(seq)
        a, *_ = judge_many(model, seqs)
        print(f"  置换 {pct}%: acc={a:.3f} (n={len(seqs)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    model = load_model(args.run)
    print(f"=== EXP-80: 直觉外推容易度 (模型 {args.run}) ===")
    width_matrix(model, [2, 9, 20, 100, 1000, 2000])
    base_matrix(model, [10, 16, 20, 36, 60])
    anon_matrix(model, [0, 25, 50, 100])
    print("\n-- 联合外推 (位宽20 × 进制16) --")
    ss, _, _ = radix_ood_samples(ops=["addition"], bases=(16,), max_digits=20, n=20)
    a, *_ = judge_many(model, [s["seq"] for s in ss])
    print(f"  位宽20×进制16: acc={a:.3f}")


if __name__ == "__main__":
    main()
