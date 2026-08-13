"""lab/diag/analyze.py —— 闭环编排 (覆盖 → 学习 → 设计 + 验证覆盖审计)

串联: coverage → learncurve → design, 并附加第4项:
  **verify 覆盖审计**: 训练后验证集是否充分覆盖 (相对训练), 是否需补跑。
  原则: 验证样本若含训练中未充分覆盖的 token/视角, 结果不可信, 需补跑验证。

观测基础 (第4项):
  1. 验证集覆盖审计: 验证样本的 token 覆盖 vs 训练样本, 找"训练未覆盖但验证用了"的 token。
  2. 剥离影响: 单 token 学习报告默认混入其他 token 影响; design 的"方法/理论问题"判定
     需排除"该 token 与未充分训练 token 共现"的干扰 (见 design.report 的 co-dependence 检查)。
"""
from __future__ import annotations

import time

from tokenizer import api
from .coverage import coverage_report
from .learncurve import full_report
from .design import plan


def _save(path, data):
    """延迟导入 save_report (避免与 __init__ 循环导入)。"""
    from . import save_report
    return save_report(path, data)


def analyze(samples, judge, tokens=None, save_path=None, ood_samples=None):
    """完整闭环: 覆盖 + 学习 + 设计 + 验证覆盖审计。

    ood_samples: 验证集 (样本外), 用于覆盖审计 (验证集是否引入训练未覆盖 token)。
    """
    t0 = time.time()
    cov = coverage_report(samples)
    lrn = full_report(judge, samples, tokens)
    groups = plan(cov, lrn)
    result = {
        "coverage": _summary(cov),
        "learn": lrn,
        "design": groups,
        "n_samples": len(samples),
        "elapsed_s": round(time.time() - t0, 1),
    }
    # 第4项: 验证覆盖审计
    if ood_samples is not None:
        result["verify_coverage"] = verify_coverage_audit(samples, ood_samples)
    if save_path:
        _save(save_path, result)
    return result


def verify_coverage_audit(train_samples, verify_samples):
    """验证覆盖审计: 验证集相对训练集引入了哪些新 token (未训练覆盖)。

    返回 {new_tokens: [...], under_covered: [...], advice: str}。
    new_tokens: 验证集有但训练集无的 token (判定不可信)。
    under_covered: 训练集覆盖 < 阈值 但验证集使用的 token。
    """
    train_toks = {e for s in train_samples for e in s["seq"]}
    verify_toks = {e for s in verify_samples for e in s["seq"]}
    new = sorted(verify_toks - train_toks, key=lambda e: api.name(e))
    under = sorted(
        (e for e in verify_toks if e in train_toks and _cover_count(train_samples, e) < 10),
        key=lambda e: api.name(e))
    advice = []
    if new:
        advice.append(f"验证集引入训练未覆盖 token: {[api.name(e) for e in new]} — 结果不可信, 需补跑训练样本")
    if under:
        advice.append(f"验证集使用训练覆盖不足 token: {[api.name(e) for e in under]} (<10次) — 建议补样本后重验")
    if not advice:
        advice.append("验证集覆盖充分 (无新增/不足 token)")
    return {
        "new_tokens": [api.name(e) for e in new],
        "under_covered": [api.name(e) for e in under],
        "advice": advice,
    }


def _cover_count(samples, e):
    return sum(1 for s in samples if e in s["seq"])


def _summary(cov):
    out = {}
    for eid, r in cov.items():
        out[eid] = {
            "name": r["name"],
            "roles": dict(r["roles"]),
            "depth": dict(r["freq_by_depth"]),
            "pos_neg": dict(r["pos_neg"]),
            "n_total": sum(r["roles"].values()),
        }
    return out


def print_report(result):
    """可读输出: 设计分组 + 验证覆盖审计。"""
    groups = result["design"]
    order = ["覆盖不足", "方法/理论问题", "已学会", "无样本", "未评估"]
    print(f"样本数: {result['n_samples']}  耗时: {result['elapsed_s']}s\n")
    for g in order:
        items = groups.get(g)
        if not items:
            continue
        print(f"[{g}] ({len(items)})")
        for d in items:
            lv = d["learn_level"] or "-"
            print(f"  {d['name']:<16} 覆盖={d['n_cover']:<5} 学习={lv:<6} → {d['advice']}")
        print()
    if "verify_coverage" in result:
        vc = result["verify_coverage"]
        print("[验证覆盖审计]")
        for a in vc["advice"]:
            print(f"  {a}")
