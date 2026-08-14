"""lab/diag/learncurve.py —— Token 学习曲线观测器 (工具2: 分视角学习)

输入: 训练产物 (Judge) + 分视角样本集
输出: 每 ctoken 学习报告 (分视角 acc/判真判假/分布, 全结构化零硬编码)。

注意 (第4项观测基础): 观测需剥离其他 token 影响 — 分视角样本是"含该 token
的所有样本", 一次失败可能是多种原因共同产生。设计上:
  - single/nested 分离排除结构变量
  - 判真/判假分解排除分布偏置
  - verify 覆盖观测 (第4项): 训练后验证集本身是否充分覆盖, 由 coverage 检查
"""
from __future__ import annotations

from collections import Counter

from tokenizer import api
from .roles import sequence_structure


def _truth(seq):
    """真值: 判定序列末 token (合成器放真值的位置)。"""
    _, _, truth = sequence_structure(seq)
    return truth is not None and truth == seq[-1]


def token_report(judge, samples, token_eid):
    """单 token 学习报告: 分视角准确率 + 判真/判假 + 分布。"""
    name = api.name(token_eid)
    report = {"eid": token_eid, "name": name}
    related = [s for s in samples if token_eid in s["seq"]]
    total = len(related)
    # 分视角: 括号层数 1 = 单层, ≥2 = 嵌套 (结构化, 零硬编码)
    single = [s for s in related if _depth(s["seq"]) <= 1]
    nested = [s for s in related if _depth(s["seq"]) > 1]
    report["n_related"] = total
    report["gen_by_view"] = {
        "single": _acc(judge, single) if single else None,
        "nested": _acc(judge, nested) if nested else None,
    }
    pos = [s for s in related if _truth(s["seq"])]
    neg = [s for s in related if not _truth(s["seq"])]
    report["true_acc"] = _acc(judge, pos) if pos else None
    report["false_acc"] = _acc(judge, neg) if neg else None
    dist = Counter()
    for s in related:
        p = judge.answer(s["seq"][1:-1])
        dist[api.name(p)] += 1
    report["pred_dist"] = dict(dist)
    return report


def _depth(seq):
    """嵌套深度: 统计 paren 类 token (结构化识别)。"""
    from .coverage import _depth as _d
    return _d(seq)


def _acc(judge, samples):
    if not samples:
        return None
    hit = sum(1 for s in samples
              if judge.answer(s["seq"][1:-1]) == s["seq"][-1])
    return hit / len(samples)


def full_report(judge, samples, tokens=None):
    """全部 (或指定) ctoken 的学习报告。"""
    tokens = tokens or sorted({e for s in samples for e in s["seq"]})
    return {t: token_report(judge, samples, t) for t in tokens}


def summarize(judge, samples, tokens=None, top=10):
    """可读摘要: 每 ctoken 分视角 acc + 判真判假。"""
    rep = full_report(judge, samples, tokens)
    print(f"样本数: {len(samples)}")
    print(f"\n{'token':<18}{'single':<9}{'nested':<9}{'判真':<9}{'判假':<9}{'n':<6}")
    for eid, r in sorted(rep.items(), key=lambda x: -x[1]["n_related"])[:top]:
        def f(x):
            return f"{x:.2f}" if isinstance(x, float) else "-"
        print(f"{r['name']:<18}{f(r['gen_by_view']['single']):<9}"
              f"{f(r['gen_by_view']['nested']):<9}{f(r['true_acc']):<9}"
              f"{f(r['false_acc']):<9}{r['n_related']:<6}")
