"""docs/paper_data/scripts/expc1_eval.py —— EXP-C1 阶数/进制外推评估

对 C1a-c 去因子化模型测外推 (run_exp _judge_eval):
- 阶数外推: balanced 小范围 (addition/multiplication/power/root/tetration)
- 进制外推: radix base 16/60

用法: PYTHONPATH=. python -m docs.paper_data.scripts.expc1_eval --run <dir>
"""
import argparse

from tokenizer import api
from lab import synth_core

# eval_helpers 在同目录
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_helpers import load_model, judge_many

_OPS = [("addition", 9), ("multiplication", 5), ("power", 5),
        ("root", 5), ("tetration", 3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    model = load_model(args.run)
    print(f"=== EXP-C1: 阶数外推 (模型 {args.run}) ===")
    for op, hi in _OPS:
        try:
            ss, _, _ = synth_core.balanced_samples(max_depth=2, hi=hi,
                                                   op=api.eid_by_name(op), neg_mode=1)
            a, *_ = judge_many(model, [s["seq"] for s in ss[:200]])
            print(f"  {op} (hi={hi}): acc={a:.3f} (n={min(200, len(ss))})")
        except Exception as e:
            print(f"  {op}: ERR {str(e)[:50]}")

    print("\n=== 进制外推 (base 16/60) ===")
    for base in (16, 60):
        try:
            ss, _, _ = synth_core.radix_ood_samples(ops=["addition"], bases=(base,),
                                                    max_digits=20, n=20)
            a, *_ = judge_many(model, [s["seq"] for s in ss])
            print(f"  base {base}: acc={a:.3f} (n={len(ss)})")
        except Exception as e:
            print(f"  base {base}: ERR {str(e)[:50]}")


if __name__ == "__main__":
    main()
