"""lab/exp/exp20_gates.py —— EXP-20: 平行概念移出 (各门判定)

对模型逐门测 math 命题 OOD 判定口径 (run_exp _judge_eval):
  删某门族监督 → 该门 0 + 其他门降 (门族互训断裂).

用法: PYTHONPATH=. python -m lab.exp.exp20_gates --run <run_dir>
"""
import argparse

from lab.synth_core import logic_arith_samples
from eval_helpers import load_model, judge_many

_GATES = ["logical_and", "logical_or", "logical_imply", "logical_iff", "logical_xor",
          "logical_nand", "logical_nor", "logical_xnor", "logical_not"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    model = load_model(args.run)
    print(f"=== EXP-20: 各门 math 命题判定 (模型 {args.run}) ===")
    for g in _GATES:
        ss, _, _ = logic_arith_samples(ops=[g], hi=9, seed=1, notation="prefix")
        a, *_ = judge_many(model, [s["seq"] for s in ss])
        print(f"  {g}: acc={a:.3f} (n={len(ss)})")


if __name__ == "__main__":
    main()
