"""docs/paper_data/scripts/exp10_impl.py —— EXP-10: imply 语义评估

覆盖 (全部 run_exp _judge_eval 权威口径):
  1. imply 裸真值表 (prefix/infix 逐行)
  2. (A⇒B)⇒(C⇒D) 深嵌套 + 深度3 级联
  3. imply math 命题 OOD
  4. 训练集内 vs OOD (区分记忆 vs 泛化)
  5. 标准逻辑门/算术对照

用法: PYTHONPATH=. python -m docs.paper_data.scripts.exp10_impl --run <run_dir> [--op logical_imply]
"""
import argparse
from itertools import product

from tokenizer import api
from tokenizer.eval.logic_eval import logic_truth
from lab.judge import judge_sequence
from lab.synth_core import logic_arith_samples, logic_interdef_samples, logic_samples
from eval_helpers import load_model, judge_many

_TRUTH = None


def _toks():
    global _TRUTH
    if _TRUTH is None:
        _TRUTH = api.role_token("truth")
    return _TRUTH


def truth_table_rows(op_name, notation="prefix"):
    """裸真值表判定序列 (logic_samples 沿定义, prefix/infix)."""
    ss, _, _ = logic_samples(op=op_name, seed=0, notation=notation)
    return ss


def deep_nest_impl():
    """(A⇒B)⇒(C⇒D) 深嵌套 + 深度3 级联判定序列."""
    impl = api.eid_by_name("logical_imply")
    t, f = _toks()
    deep2 = []
    deep3 = []
    for a, b, c, d in product([True, False], repeat=4):
        ab = logic_truth(impl, [t if a else f, t if b else f])
        cd = logic_truth(impl, [t if c else f, t if d else f])
        outer = logic_truth(impl, [t if ab else f, t if cd else f])
        deep2.append(judge_sequence(api.assemble_seq(impl, [t if ab else f, t if cd else f]), outer))
    for a, b, c, d, e in product([True, False], repeat=5):
        ab = logic_truth(impl, [t if a else f, t if b else f])
        cd = logic_truth(impl, [t if c else f, t if d else f])
        cde = logic_truth(impl, [t if cd else f, t if e else f])
        outer = logic_truth(impl, [t if ab else f, t if cde else f])
        deep3.append(judge_sequence(api.assemble_seq(impl, [t if ab else f, t if cde else f]), outer))
    return deep2, deep3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="模型归档目录")
    ap.add_argument("--op", default="logical_imply")
    args = ap.parse_args()
    model = load_model(args.run)
    op = args.op

    print(f"=== EXP-10: {op} 语义评估 (模型 {args.run}) ===")

    # 1. 裸真值表 (prefix + infix 对照)
    for notation in ("prefix", "infix"):
        ss = truth_table_rows(op, notation)
        a, t, f, c = judge_many(model, [s["seq"] for s in ss])
        print(f"[{notation}] 裸真值表: acc={a:.3f} (n={len(ss)})")
        for s in ss:
            names = " ".join(api.name(x) for x in s["seq"])
            print(f"    {'T' if s['truth'] else 'F'} | {names}")

    # 2. 深嵌套 (imply 专测)
    if op == "logical_imply":
        deep2, deep3 = deep_nest_impl()
        a2, *_ = judge_many(model, deep2)
        a3, *_ = judge_many(model, deep3)
        print(f"[deep2] (A⇒B)⇒(C⇒D): acc={a2:.3f} (n={len(deep2)})")
        print(f"[deep3] (A⇒B)⇒((C⇒D)⇒E): acc={a3:.3f} (n={len(deep3)})")

    # 3. math 命题 OOD (logic_arith)
    ss = logic_arith_samples(ops=[op], hi=9, seed=1, notation="prefix")[0]
    a, *_ = judge_many(model, [s["seq"] for s in ss])
    print(f"[arith] {op} math 命题 OOD: acc={a:.3f} (n={len(ss)})")

    # 4. 训练集内 vs OOD (interdef)
    for seed, tag in ((0, "训练内"), (5, "OOD")):
        ss = logic_interdef_samples(ops=[op], seed=seed, imply_judge=False)[0]
        if ss:
            a, *_ = judge_many(model, [s["seq"] for s in ss])
            print(f"[interdef-{tag}] {op}: acc={a:.3f} (n={len(ss)})")


if __name__ == "__main__":
    main()
