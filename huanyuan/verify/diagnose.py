#!/usr/bin/env python3
"""verify/diagnose.py — 训练诊断 CLI (verify 功能集: 模型 OOD 判定 + 样本对比/覆盖/配置诊断).

对比归档训练样本 + 模型 OOD 判定 + token 覆盖 + 按配置样本类型诊断
(复现/定位训练问题, 零临时拼装). 从项目根运行:
  PYTHONPATH=src/llm_research_v5 python -m verify.diagnose --old <归档> [--new <归档>] [--ood-op addition]
"""
import argparse
import json
import glob
import os
import sys

from tokenizer import api
from train.data import collate, rev_vocab


def load_samples(run_dir):
    """归档 → 样本列表."""
    p = os.path.join(run_dir, "samples.jsonl")
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def seq_name(seq):
    return [api.name(e) for e in seq]


def compare(old_dir, new_dir):
    """准确对比样本 token 序列一致性 (multiset, 含重复度): 非集合/分布对比.

    对 OLD/NEW 每条 token 序列精确计数 (Counter of tuple(seq)),
    差异 = 计数不同的序列 (含 token 序列全文 + 两边计数), 零近似.
    """
    from collections import Counter
    old = load_samples(old_dir)
    new = load_samples(new_dir)
    old_c = Counter(tuple(s["seq"]) for s in old)
    new_c = Counter(tuple(s["seq"]) for s in new)
    print(f"OLD {len(old)} 样本 {len(old_c)} 唯一 | NEW {len(new)} 样本 {len(new_c)} 唯一")
    same = {k for k in old_c if old_c[k] == new_c[k]}
    print(f"序列精确一致 (token 序列+计数全同): {len(same)}")
    diffs = []
    for k in old_c | new_c:
        if old_c[k] != new_c[k]:
            diffs.append((k, old_c.get(k, 0), new_c.get(k, 0)))
    print(f"token 序列差异 (计数不同): {len(diffs)}")
    for k, oc, nc in diffs[:10]:
        names = " ".join(seq_name(k))
        print(f"  OLDx{oc} NEWx{nc}: {names}")
    return old, new


def token_coverage(samples):
    """token 训练覆盖 (计数)."""
    from collections import Counter
    cnt = Counter()
    for s in samples:
        for e in s["seq"]:
            cnt[e] += 1
    return cnt


def op_report(cfg_path, samples, run_dir):
    """按实验配置样本类型诊断: 每类样本的样本量 + 对应算子泛化 acc.

    样本类型列表优先读归档 config.json["exp"] (run_exp 已归档完整实验配置);
    cfg_path = 实验配置 (lab/configs/*.json) 作回退 (旧归档无 exp 字段时用).
    样本量 = 该 op token 在训练样本中的频次; acc = views.json per_token_gen.
    """
    from collections import Counter
    cfg = None
    vp = os.path.join(run_dir, "config.json")
    if os.path.isfile(vp):
        arch = json.load(open(vp, encoding="utf-8"))
        cfg = arch.get("exp")
    if not cfg and cfg_path:
        cfg = json.load(open(cfg_path, encoding="utf-8"))
    if not cfg:
        print("归档无 exp 实验配置, 且未提供 --config")
        return
    cnt = Counter()
    for s in samples:
        for e in s["seq"]:
            cnt[e] += 1
    vp = os.path.join(run_dir, "views.json")
    pt = json.load(open(vp, encoding="utf-8")).get("per_token_gen", {}) if os.path.isfile(vp) else {}
    specs = cfg.get("synth", {}).get("samples", [])
    print(f"按配置 {len(specs)} 类样本诊断 (样本量=op token 频次, acc=per_token_gen):")
    print(f"{'#':>2} {'kind':12} {'op':14} {'hi':>4} {'样本量':>6} {'泛化acc':>9}")
    for i, spec in enumerate(specs, 1):
        kind = spec.get("kind")
        op = spec.get("op")
        hi = spec.get("hi", "-")
        n = acc = None
        if op:
            try:
                eid = api.eid_by_name(op)
                n = cnt.get(eid, 0)
                p = pt.get(op)
                acc = p["acc"] if p else None
            except ValueError:
                pass
        acc_s = f"{acc:.3f}" if acc is not None else "N/A"
        n_s = str(n) if n is not None else "-"
        print(f"{i:>2} {kind:12} {str(op):14} {str(hi):>4} {n_s:>6} {acc_s:>9}")


def coverage_gap(samples, run_dir=None):
    """训练不足诊断: vocab 全集 (含 0 覆盖) vs 训练频次 + per-token 泛化正确率.

    样本量来自 samples.jsonl (训练覆盖频次), 正确率来自归档 views.json
    per_token_gen (逐 token 泛化 acc). 两指标并列展示.
    """
    from train.data import vocab as _vocab
    v = _vocab()
    cnt = token_coverage(samples)
    pt = {}
    if run_dir:
        vp = os.path.join(run_dir, "views.json")
        if os.path.isfile(vp):
            pt = json.load(open(vp, encoding="utf-8")).get("per_token_gen", {})
    rows = []
    for e in v:
        n = cnt.get(e, 0)
        name = api.name(e)
        p = pt.get(name)
        acc = p["acc"] if p else None
        rows.append((name, n, acc))
    return rows


def ood_judge_acc(model_path, ood_ops=("addition", "multiplication"), digits=5, n=100, seed=12345):
    """OOD 判定 acc (全真值): 训练模型对多位数 OOD 的完整序列重建, 逐 op 报告."""
    import torch
    from train import TokenTransformer
    from lab.synth_core import ood_samples
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    from train.data import vocab as _vocab
    v = _vocab()
    m = TokenTransformer(dim=64, num_concepts=len(v), input_mode="ids", causal=False)
    m.load_state_dict(state)
    rv = rev_vocab()
    total_hit = total_n = 0
    for op in ood_ops:
        ood = ood_samples(op=op, digits=digits, n=n, mode="mixed", seed=seed)[0]
        batch = collate(ood, input_mode="ids")
        m.eval()
        with torch.no_grad():
            logits, _ = m(batch["inputs"], mask=batch["mask"])
        lens = batch["lengths"]
        hit = 0
        for i, s in enumerate(ood):
            rl = lens[i]
            preds = [rv[p] for p in logits[i, :rl].argmax(dim=1).tolist()]
            if preds == s["seq"]:
                hit += 1
        total_hit += hit
        total_n += len(ood)
        print(f"  OOD 判定 {op:15} acc={hit/len(ood):.3f} ({hit}/{len(ood)})")
    return total_hit / total_n, total_n


def epoch_target_stats(target=0.98):
    """按 token 扫描全部归档的逐 epoch per-token 泛化曲线, 统计各 token 达 target 的第一 epoch.

    只看达到的 (未达标 token 不计); 返回 {token: [(run, epoch, acc)]}, 平均达标 epoch per token.
    反馈监督基础设施: 每 token 达标 epoch 揭示哪些 token 慢/不达标 (定义问题).
    """
    runs = sorted(glob.glob("src/llm_research_v5/archive/log/train/*/"))
    per_token = {}
    for r in runs:
        p = os.path.join(r, "metrics.json")
        if not os.path.isfile(p):
            continue
        m = json.load(open(p, encoding="utf-8"))
        eg = m.get("epoch_gen") or []   # [ {token: acc}, ... ]
        name = os.path.basename(r.rstrip("/"))
        for ep, tok_acc in enumerate(eg, 1):
            if not isinstance(tok_acc, dict):
                continue   # 旧格式 (整体 float) 跳过
            for tok, acc in tok_acc.items():
                if acc >= target:
                    per_token.setdefault(tok, []).append((name, ep, acc))
                    break   # 每 run 每 token 只记首个达标 epoch
    avg = {t: sum(h for _, h, _ in v) / len(v) for t, v in per_token.items()}
    return per_token, avg


def _last_token_acc():
    """最新归档 (含 per-token epoch_gen) 的每 token 最终泛化 acc.

    用于未达标 token 分类: 0-acc (完全没学) vs 部分学习 (>0 但 <target).
    """
    runs = sorted(glob.glob("src/llm_research_v5/archive/log/train/*/"))
    for r in reversed(runs):
        p = os.path.join(r, "metrics.json")
        if not os.path.isfile(p):
            continue
        m = json.load(open(p, encoding="utf-8"))
        eg = m.get("epoch_gen") or []
        last = next((d for d in reversed(eg) if isinstance(d, dict)), None)
        if last:
            return last
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default=None, help="旧归档目录")
    ap.add_argument("--new", default=None, help="新归档目录 (默认最新)")
    ap.add_argument("--model", default=None, help="model.pt 路径 (测 OOD 判定)")
    ap.add_argument("--ood-op", default="addition", help="OOD 算子 (逗号分隔)")
    ap.add_argument("--config", default=None, help="实验配置 JSON (lab/configs/*.json), 按样本类型诊断")
    ap.add_argument("--epoch-target", type=float, default=None,
                    help="逐 epoch 泛化达标统计: 准确率达该值的第一 epoch (只看达到的, 输出平均)")
    args = ap.parse_args()

    if args.epoch_target is not None:
        per_token, avg = epoch_target_stats(args.epoch_target)
        print(f"按 token 达标(≥{args.epoch_target})统计: {len(avg)} 个 token 达标")
        for tok in sorted(avg, key=lambda t: avg[t]):
            v = per_token[tok]
            print(f"  {tok:24} 平均达标 epoch {avg[tok]:.1f} (达标 {len(v)} run)")
        # 未达标 token 分类: 0-acc (完全没学) vs 部分学习 (>0 但 <target)
        zero_acc = [t for t in _last_token_acc() if t not in avg and _last_token_acc()[t] == 0.0]
        partial = [t for t in _last_token_acc() if t not in avg and _last_token_acc()[t] > 0.0]
        print(f"剔除 0-acc 后达标: 总 token {len(_last_token_acc())} | "
              f"达标 {len(avg)} | 0-acc 未学 {len(zero_acc)} | 部分学习 {len(partial)}")
        print("平均达标 epoch (全 token):", f"{sum(avg.values())/len(avg):.1f}" if avg else "无")

    runs = sorted(glob.glob("src/llm_research_v5/archive/log/train/*/"), reverse=True)
    new_dir = args.new or (runs[0] if runs else None)
    if args.old and new_dir:
        compare(args.old, new_dir.rstrip("/"))
    elif new_dir:
        new_dir = new_dir.rstrip("/")
        samples = load_samples(new_dir)
        if args.config:
            op_report(args.config, samples, new_dir)
        else:
            rows = coverage_gap(samples, new_dir)
            n_zero = sum(1 for _, n, _ in rows if n == 0)
            print(f"最新归档 {new_dir}: {len(samples)} 样本, "
                  f"vocab {len(rows)} (0 覆盖 {n_zero})")
            print(f"{'token':28} {'样本量':>6} {'泛化acc':>9}  (正确/总数)")
            zero_rows = [r for r in rows if r[1] == 0]
            for name, n, acc in zero_rows:
                print(f"  {name:26} {n:>6} {'N/A':>9}")
            for name, n, acc in sorted([r for r in rows if r[1] > 0], key=lambda x: x[1]):
                acc_s = f"{acc:.3f}" if acc is not None else "N/A"
                print(f"  {name:26} {n:>6} {acc_s:>9}")
    if args.model:
        acc, n = ood_judge_acc(args.model, ood_ops=args.ood_op.split(","))
        print(f"OOD 判定 acc: {acc:.3f} ({n} 样本, op={args.ood_op})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
