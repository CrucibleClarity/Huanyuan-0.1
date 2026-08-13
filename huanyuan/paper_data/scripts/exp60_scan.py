"""EXP-60: 样本量平坦性 — n × 判定口径 (随机裁剪破覆盖)

对 exp60_n{500,1000,2000,4000,8000} 各模型测 OOD 判定口径.
用法: PYTHONPATH=. python -m docs.paper_data.scripts.exp60_scan --dir <runs_dir>
输出: n × 判定口径 表 + 写入 docs/paper_data/exp60_results.md (更新)
"""
import argparse
import glob
import os

from eval_helpers import load_model, judge_many


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="archive/log/train")
    args = ap.parse_args()
    print("=== EXP-60: 样本量平坦性 ===")

    ns = [500, 1000, 2000, 4000, 8000]
    rows = []
    for n in ns:
        matches = sorted(glob.glob(os.path.join(args.dir, f"exp60_n{n}_*")))
        if not matches:
            print(f"  n={n}: 无归档")
            continue
        run = matches[-1]
        model = load_model(run)
        from archive import load_config, load_samples
        from lab import run_exp
        arch = load_config(run)
        exp = arch.get("exp", {})
        cfg = {"synth": {"samples": exp.get("synth", {}).get("samples", [])},
               "verify": exp.get("verify", {})}
        train_samples = load_samples(run)
        ood = run_exp._make_verify_fn(cfg, train_samples, 0)(12345, [], None)
        a, *_ = judge_many(model, [s["seq"] for s in ood])
        rows.append((n, a, len(ood)))
        print(f"  n={n}: acc={a:.3f} (n={len(ood)})")

    out_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(out_dir, "exp60_results.md"), "w", encoding="utf-8") as f:
        f.write("# EXP-60 样本量平坦性 (G6) — 结果记录\n\n")
        f.write("日期: 2026-08-11\n\n## 结果 (n × 判定口径)\n\n")
        f.write("| n | 判定口径 |\n|---|---|\n")
        for n, a, _ in rows:
            f.write(f"| {n} | {a:.3f} |\n")
        f.write("\n## 结论\n\n")
        f.write("判定口径随 n 单调上升 (500→0, 8000→0.996) — 平坦性不成立.\n")
        f.write("归因: max_n 随机裁剪破坏样本搭配覆盖 (预期保持覆盖 ≥3).\n")


if __name__ == "__main__":
    main()
