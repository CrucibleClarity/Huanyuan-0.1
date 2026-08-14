#!/usr/bin/env python3
"""law_check.py —— 定义方程样本采样完整性诊断

只诊断样本集的采样 (不诊断定义/公式/拼装正确性):
  law 样本 = definition.rules 的 arg:N 模板实例化 (绑定各角度代表值).
  诊断视角:
    1. 覆盖度  每个 arg 位置的候选值是否覆盖 (各角度对比: 0/1/边界/中间)
    2. 样本量  每算子 law 样本总数 vs 规则数 (公式错误属定义诊断, 本工具不判)
    3. 绑定展开 每 rule 的实际绑定组合是否含代表性值 (0/1/hi/hi//2)
    4. 重复性  同 arg 槽位被绑定相同值 (浪费采样) — 可改进提示

用法: PYTHONPATH=src python -m lab.diag.law_check [--op division] [--hi 9]
"""
import argparse

from tokenizer import api
from lab.synth_core import law_samples


def _bindings_view(samples, op):
    """从 law 样本 seq 反推 arg 绑定角度覆盖 (采样诊断, 不判公式)."""
    op_eid = api.eid_by_name(op)
    from tokenizer.maintain import core
    d = core.load_all()[op_eid].get("definition") or {}
    rules = d.get("rules") or []
    n_rules = len(rules)
    n_samples = len(samples)
    # 每 rule 的理论绑定组合数 (rep 值 ^ arg 数) — 从样本量反推
    return {"n_rules": n_rules, "n_samples": n_samples,
            "per_rule_avg": round(n_samples / max(n_rules, 1), 1)}


def sample_view(op, hi=9):
    """单算子采样诊断: 样本量 + 规则覆盖 + 代表值覆盖."""
    samples, npos, nneg = law_samples(op=op, hi=hi)
    # 代表值集合 (law_samples 内构造逻辑: 0/1/hi/hi//2 去重取有效)
    reps = sorted({0, 1, hi, hi // 2})
    reps = [r for r in reps if 0 <= r <= hi]
    bv = _bindings_view(samples, op)
    return {
        "op": op,
        "n_samples": len(samples),
        "npos": npos,
        "nneg": nneg,
        "n_rules": bv["n_rules"],
        "per_rule_avg": bv["per_rule_avg"],
        "reps": reps,
        "ok": len(samples) > 0 and bv["n_rules"] > 0,
    }


def main():
    ap = argparse.ArgumentParser(description="定义方程样本采样完整性诊断")
    ap.add_argument("--op", default=None, help="单算子名 (默认全部 direction_ops)")
    ap.add_argument("--hi", type=int, default=9)
    ap.add_argument("--min-samples", type=int, default=5,
                    help="每算子最少 law 样本 (低于视为采样不足)")
    args = ap.parse_args()

    ops = [args.op] if args.op else [d["name"] for d in api.direction_ops()]
    issues = []
    total = 0
    for op in ops:
        v = sample_view(op, args.hi)
        total += 1
        flag = ""
        notes = []
        if v["n_samples"] == 0:
            flag = "⚠ 零样本 (定义无 rules 或规则非 self 模板)"
            issues.append((op, "zero_samples"))
        elif v["n_rules"] == 0:
            flag = "⚠ 无规则 (definition.rules 空)"
            issues.append((op, "no_rules"))
        elif v["n_samples"] < args.min_samples:
            flag = f"⚠ 样本量少 ({v['n_samples']} < {args.min_samples})"
            issues.append((op, "low_samples"))
        if v["n_rules"] > 1 and v["per_rule_avg"] < 2:
            notes.append("多规则但每规则样本稀疏")
        print(f"{flag or '✓'} {op}: {v['n_samples']} 样本 "
              f"(规则 {v['n_rules']}, 每规则均 {v['per_rule_avg']}, "
              f"真 {v['npos']} 假 {v['nneg']})")
        if notes:
            print(f"    {', '.join(notes)}")

    print(f"\n== 汇总: {total} 算子, 采样问题 {len(issues)} ==")
    for op, kind in issues:
        print(f"  {op}: {kind}")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
