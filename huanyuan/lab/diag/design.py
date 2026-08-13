"""lab/diag/design.py —— 实验设计器 (工具3: 覆盖→实验建议)

输入: 覆盖报告 (lab.diag.coverage) + 学习报告 (lab.diag.learncurve)
输出: 实验设计建议 (补样本 / 平衡调整 / 方法-理论问题判定)

决策逻辑:
  coverage 充足 + 学不好 + 无共现干扰 → 方法/理论问题 (非样本问题)
  coverage 充足 + 学不好 + 共现干扰   → 可能是共现 token 拖累, 需剥离验证
  coverage 不足 + 学不好              → 建议补缺失视角样本
  coverage 充足 + 学得好              → 已充分训练 (可减样本)

第4项 (剥离影响): 学不好时检查该 token 的共现 token 是否覆盖不足,
  若是 → 判定"共现干扰" (非该 token 自身问题), 建议先补共现 token 样本。
"""
from __future__ import annotations


def decide(cover_rep, learn_rep, thresholds=None):
    """单个 ctoken 的决策 (含共现干扰检查)。"""
    t = thresholds or {"coverage": 20, "learn": 0.7, "co_cover": 10}
    out = []
    for eid, lr in learn_rep.items():
        cr = cover_rep.get(eid, {})
        roles = cr.get("roles", {})
        total = sum(roles.values())
        cov_level = "充足" if total >= t["coverage"] else "不足"
        accs = [v for v in lr.get("gen_by_view", {}).values() if isinstance(v, float)]
        learn = max(accs) if accs else None
        if learn is None:
            verdict = "无样本" if total == 0 else "未评估"
            advice = "该 token 未出现在样本中, 需加入样本包"
        elif learn >= t["learn"]:
            verdict = "已学会"
            advice = "覆盖充分且学得好; 可减样本验证鲁棒性" if cov_level == "充足" else \
                     "样本高效, 保持当前覆盖"
        else:
            # 学不好: 检查共现干扰 (第4项)
            coo = cr.get("cooccur", {})
            co_under = [n for n, c in coo.items() if c < t["co_cover"]]
            if co_under:
                verdict = "共现干扰"
                advice = (f"学不好但共现 token 覆盖不足: {co_under}; "
                          f"可能是共现拖累, 非自身问题 — 先补共现样本再评估")
            elif cov_level == "充足":
                verdict = "方法/理论问题"
                advice = "覆盖充足且无共现干扰但学不好 — 非样本问题, 需查监督/架构/理论"
            else:
                verdict = "覆盖不足"
                missing = {k: v for k, v in roles.items() if v == 0}
                advice = f"覆盖不足 (共{total}); 建议补缺失视角: {missing or '增加深度/正负例'}"
        out.append({
            "eid": eid, "name": lr.get("name", eid),
            "coverage_level": cov_level, "n_cover": total,
            "learn_level": f"{learn:.2f}" if learn is not None else None,
            "verdict": verdict, "advice": advice,
        })
    return out


def plan(coverage_report, learn_report, thresholds=None):
    """全 token 设计: 按 verdict 分组。"""
    dec = decide(coverage_report, learn_report, thresholds)
    groups = {}
    for d in dec:
        groups.setdefault(d["verdict"], []).append(d)
    return groups


def summarize(coverage_report, learn_report, thresholds=None):
    """可读摘要: 分组 + 建议。"""
    groups = plan(coverage_report, learn_report, thresholds)
    print("=== 实验设计建议 ===")
    order = ["覆盖不足", "共现干扰", "方法/理论问题", "已学会", "无样本", "未评估"]
    for g in order:
        items = groups.get(g)
        if not items:
            continue
        print(f"\n[{g}] ({len(items)} 个)")
        for d in items:
            print(f"  {d['name']:<16} 覆盖={d['n_cover']:<5} "
                  f"学习={d['learn_level'] or '-':<6} → {d['advice']}")
