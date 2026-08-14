"""lab/exp/exp01_matrix.py —— EXP-01 主 OOD 矩阵 + EXP-50 三通道一致性

EXP-01: OOD 总样本按维度分组判定口径 (logic_arith / radix / cartesian).
EXP-50: 三通道一致性 (构造=token逐位 / 形式=eval_op / 直觉=模型) + 快慢路径成本.

用法:
  PYTHONPATH=. python -m lab.exp.exp01_matrix --run <run_dir>          # EXP-01 矩阵
  PYTHONPATH=. python -m lab.exp.exp01_matrix --run <run_dir> --three-channel   # EXP-50
"""
import argparse
import json
import time
from collections import defaultdict

from tokenizer import api
from tokenizer.eval.token_iterator import token_add
from tokenizer.eval.digit_eval import numeral_to_digits, digits_to_numeral
from tokenizer.eval.engine import eval_op
from lab.judge import judge_sequence
from lab.synth_core import nested_seq, numeral_of, _numeral_value_of
from eval_helpers import load_model, judge_many
from archive import load_config, load_samples

_ADD = None


def _add():
    global _ADD
    if _ADD is None:
        _ADD = api.eid_by_name("addition")
    return _ADD


def exp01(model, run_dir):
    """EXP-01 主 OOD 矩阵 (按序列长度分组)."""
    arch = load_config(run_dir)
    exp = arch.get("exp", {})
    cfg = {"synth": {"samples": exp.get("synth", {}).get("samples", [])},
           "verify": exp.get("verify", {})}
    train_samples = load_samples(run_dir)
    from lab import run_exp
    ood = run_exp._make_verify_fn(cfg, train_samples, 0)(12345, [], None)
    a, *_ = judge_many(model, [s["seq"] for s in ood])
    print(f"=== EXP-01: 主 OOD 矩阵 (总 n={len(ood)}, 判定 {a:.3f}) ===")
    bins = defaultdict(list)
    for s in ood:
        L = len(s["seq"])
        if L < 20:
            bins["logic_arith/短判定"].append(s)
        elif L < 60:
            bins["radix/cartesian 中长"].append(s)
        else:
            bins["radix 20位结果/长序列"].append(s)
    for b, ss in sorted(bins.items()):
        acc, *_ = judge_many(model, [s["seq"] for s in ss])
        print(f"  {b}: acc={acc:.3f} (n={len(ss)})")


def three_channel(model):
    """EXP-50 三通道一致性 + 快慢路径成本."""
    print("\n=== EXP-50: 三通道一致性 (构造/形式/直觉) ===")
    tests = [(12, 34), (123456789, 987654321), (int("9" * 20), 1), (int("9" * 2000), 1)]
    samples = []
    for a, b in tests:
        # 构造通道 (oracle)
        oc = digits_to_numeral(token_add(numeral_to_digits(a), numeral_to_digits(b)))
        # 形式通道
        of = _numeral_value_of(eval_op(_add(), [numeral_of(a), numeral_of(b)]))
        # 直觉通道 (模型)
        seq = judge_sequence(nested_seq([a, b], _add(), oc), True)
        samples.append({"seq": seq, "valid": 1})
        print(f"  a_len={len(str(a))}: 构造={oc} 形式={of} 一致={oc == of}")
    a, *_ = judge_many(model, [s["seq"] for s in samples])
    print(f"  直觉通道 (模型批量判定): acc={a:.3f} (n={len(samples)})")

    # 快慢路径成本
    print("\n-- 快慢路径成本 (EXP-51) --")
    a2000, b2000 = int("9" * 2000), 1
    t0 = time.perf_counter()
    token_add(numeral_to_digits(a2000), numeral_to_digits(b2000))
    t_construct = time.perf_counter() - t0
    t0 = time.perf_counter()
    eval_op(_add(), [numeral_of(a2000), numeral_of(b2000)])
    t_formal = time.perf_counter() - t0
    seq2k = judge_sequence(nested_seq([a2000, b2000], _add(),
                                      int("1" + "0" * 2000)), True)
    t0 = time.perf_counter()
    judge_many(model, [seq2k])
    t_intuition = time.perf_counter() - t0
    print(f"  2000位: 构造={t_construct:.4f}s 形式={t_formal:.4f}s 直觉={t_intuition:.4f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--three-channel", action="store_true")
    args = ap.parse_args()
    model = load_model(args.run)
    exp01(model, args.run)
    if args.three_channel:
        three_channel(model)


if __name__ == "__main__":
    main()
