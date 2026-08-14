#!/usr/bin/env python3
"""arrow_completeness.py —— 对称组 arrow 完整性检查 (标准命名法基准)

范畴论对称组标准结构: 每组对称性应有 4 个 arrow —
  基点→A, 基点→B (生成方向), A→B, B→A (互逆方向).
本工具列出每个对称组已配齐 / 缺失的 arrow (供补齐).

用法: PYTHONPATH=src python -m lab.diag.arrow_completeness
"""
from tokenizer import api


def groups():
    """对称组定义: 组名 → (正端 name, 负端 name). 每个成员需已注册."""
    cands = [
        ("sign", "sign_pos", "sign_neg"),
        ("succ_pred", "succ", "pred"),
        ("increase", "increase", "decrease"),
        ("imply_iff", "imply_direction", "iff_direction"),
        ("translation", "translation", "inversion"),
        ("logical", "logical_and", "logical_or"),
        ("compare", "greater_than", "less_than"),
    ]
    out = []
    for label, pn, nn in cands:
        try:
            p, n = api.eid_by_name(pn), api.eid_by_name(nn)
        except KeyError:
            continue
        out.append((label, pn, nn, p, n))
    return out


def main():
    base = api.eid_by_name("basepoint")
    arrows = list(api.all_arrows())

    def has(s, t):
        return any(api.source_of(e) == s and api.target_of(e) == t for e in arrows)

    print("=== 对称组 arrow 完整性 (4-arrow 结构) ===\n")
    total_missing = 0
    for label, pn, nn, p, n in groups():
        need = [("base→" + pn, base, p), ("base→" + nn, base, n),
                (pn + "→" + nn, p, n), (nn + "→" + pn, n, p)]
        missing = [lbl for lbl, s, t in need if not has(s, t)]
        present = 4 - len(missing)
        total_missing += len(missing)
        mark = "✓" if not missing else "✗"
        print(f"{mark} {label} ({pn}/{nn}): {present}/4")
        for lbl, s, t in need:
            ok = has(s, t)
            print(f"    {'有' if ok else '缺'}  {lbl:30s} concept=?" )
        if missing:
            print()
    print(f"\n== 总计缺失 arrow: {total_missing} ==")


if __name__ == "__main__":
    main()
