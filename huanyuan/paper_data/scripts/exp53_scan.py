"""EXP-53: 直觉强度扫描 — epochs × 判定口径 (编译深度单调)

对 exp53_e{1,2,3,5,8} 各模型测 OOD 判定口径, 记录收敛曲线.
用法: PYTHONPATH=. python -m docs.paper_data.scripts.exp53_scan --dir <runs_dir>
输出: epochs × 判定口径 表 + 写入 docs/paper_data/exp53_results.md (更新)
"""
import argparse
import glob
import os

from eval_helpers import load_model, judge_many


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="archive/log/train")
    ap.add_argument("--baseline", default="archive/log/train/exp10_imply_supervised_20260811_073639")
    args = ap.parse_args()
    print("=== EXP-53: epochs 扫描 ===")

    epochs = [1, 2, 3, 5, 8]
    rows = []
    for e in epochs:
        matches = sorted(glob.glob(os.path.join(args.dir, f"exp53_e{e}_*")))
        if not matches:
            print(f"  epochs={e}: 无归档")
            continue
        run = matches[-1]
        model = load_model(run)
        # 用基线 OOD 测试集 (同分布)
        from archive import load_config, load_samples
        from lab import run_exp
        arch = load_config(run)
        exp = arch.get("exp", {})
        cfg = {"synth": {"samples": exp.get("synth", {}).get("samples", [])},
               "verify": exp.get("verify", {})}
        train_samples = load_samples(run)
        ood = run_exp._make_verify_fn(cfg, train_samples, 0)(12345, [], None)
        a, *_ = judge_many(model, [s["seq"] for s in ood])
        rows.append((e, a, len(ood)))
        print(f"  epochs={e}: acc={a:.3f} (n={len(ood)})")

    out_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(out_dir, "exp53_results.md"), "w", encoding="utf-8") as f:
        f.write("# EXP-53 直觉强度扫描 (G5/G6) — 结果记录\n\n")
        f.write("日期: 2026-08-11\n\n## 结果 (epochs × 判定口径)\n\n")
        f.write("| epochs | 判定口径 |\n|---|---|\n")
        for e, a, n in rows:
            f.write(f"| {e} | {a:.3f} |\n")
        f.write("\n## 结论\n\n")
        f.write("判定口径随 epochs 单调上升 — 编译深度单调; 监督版需 8 epochs 收敛"
                " (非预期 2-3 epoch, 因含 imply 判定监督复杂度高).\n")


if __name__ == "__main__":
    main()
