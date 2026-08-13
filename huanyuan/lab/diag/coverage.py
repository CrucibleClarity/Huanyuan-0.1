"""lab/diag/coverage.py —— Token 覆盖分析器 (工具1: 样本视角覆盖)

输入: 判定样本集 (judge_sequence 格式)
输出: 每 ctoken 的覆盖报告 (角色/深度/正负/共现)。

角色识别沿合成逻辑反查 (lab.diag.roles):
  judge_head = 序列首 token, truth = 序列末 token, 中间按 arrange 分类。
  深度 = 括号 token 数 (嵌套层数), 正负 = 样本 truth。
"""
from __future__ import annotations

from collections import Counter

from tokenizer import api
from .roles import roles_of_sequence


def _depth(seq):
    """嵌套深度 = 括号 token 数 + 1 (括号由 tokenizer 数据识别: dtype=dual 且非判定头)。"""
    from .roles import sequence_structure
    head, _, _ = sequence_structure(seq)
    d = 0
    for e in seq:
        td = api.token_of(e)
        if getattr(td, "dtype", "") == "dual" and e != head:
            d += 1
    return d + 1


def _truth_of(seq):
    """真值: 序列末 token 即判定结果 (合成逻辑反查)。"""
    from .roles import sequence_structure
    _, _, truth = sequence_structure(seq)
    return truth is not None


def coverage_report(samples, token_eid=None, label_fn=None):
    """全样本 → 覆盖报告。返回 {eid: {roles, freq_by_depth, pos_neg, cooccur}}。"""
    report = {}
    for s in samples:
        seq = s["seq"]
        truth = s.get("truth", _truth_of(seq))
        d = _depth(seq)
        roles = roles_of_sequence(seq, label_fn)
        for e in set(seq):
            if token_eid is not None and e != token_eid:
                continue
            r = report.setdefault(e, {
                "eid": e, "name": api.name(e),
                "roles": Counter(), "freq_by_depth": Counter(),
                "pos_neg": Counter(), "cooccur": Counter(),
            })
            r["roles"][roles.get(e, "context")] += 1
            r["freq_by_depth"][d] += 1
            r["pos_neg"]["pos" if truth else "neg"] += 1
            for o in seq:
                if o != e:
                    r["cooccur"][api.name(o)] += 1
    return report


def summarize(samples, top=10, label_fn=None):
    """可读摘要: 每个 ctoken 的角色分布 + 正负 + 深度。"""
    rep = coverage_report(samples, label_fn=label_fn)
    print(f"样本数: {len(samples)} 涉及 ctoken: {len(rep)}")
    print(f"\n{'token':<18}{'角色分布':<30}{'正/负':<10}{'深度':<12}")
    for eid, r in sorted(rep.items(), key=lambda x: -sum(x[1]["roles"].values()))[:top]:
        roles = " ".join(f"{k}:{v}" for k, v in r["roles"].most_common(3))
        pn = f"{r['pos_neg']['pos']}/{r['pos_neg']['neg']}"
        dep = " ".join(f"d{k}:{v}" for k, v in sorted(r["freq_by_depth"].items()))
        print(f"{r['name']:<18}{roles:<30}{pn:<10}{dep:<12}")
