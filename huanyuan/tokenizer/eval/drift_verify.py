"""tokenizer/eval/drift_verify.py —— 标准迭代不动点漂移验证器

直接换基元 (0 → e): 算法不变化, 无映射关系 (不用 log/exp 把 0 系结果映射到 e 系).
检测: 同一迭代函数 g, 从基元 0 与从基元 e 出发, 收敛点是否漂移.

漂移判定:
  fp_old = 从 0 迭代 g 的收敛点
  fp_new = 从 e 迭代 g 的收敛点 (g 不变)
  drift  = fp_new - fp_old        (漂移量)
  to_e   = |fp_new - e| < tol     (是否漂移到 e)

特点:
  无漂移   ⟺ g 有唯一吸引不动点 (全局收缩) → 收敛点与起点(基元)无关
  有漂移   ⟺ g 有多吸引盆 → 起点(基元)落进哪个盆决定收敛点
  漂移到 e ⟺ e 是 g 的吸引不动点, 且旧基元 0 落在不同盆 (或不稳定不动点)
"""
import math

E = math.e


def iterate_fixpoint(g, start, tol=1e-10, max_iter=5000, max_val=1e100):
    """标准迭代: 从 start 出发反复应用 g 直到收敛 (不动点). 发散返回 None."""
    t = start
    for _ in range(max_iter):
        try:
            t1 = g(t)
        except (OverflowError, ValueError, ZeroDivisionError):
            return None
        if not math.isfinite(t1) or abs(t1) > max_val:
            return None
        if abs(t1 - t) < tol:
            return t1
        t = t1
    return None


def drift_verify(g, base_old=0.0, base_new=E, tol=1e-6):
    """换基元漂移检测: 同一 g, 从 base_old 与 base_new 出发, 比较收敛点.

    算法 g 不变化, 无映射关系 — 仅换起点 (基元).
    """
    fp_old = iterate_fixpoint(g, base_old)
    fp_new = iterate_fixpoint(g, base_new)
    drift = (fp_new - fp_old) if (fp_old is not None and fp_new is not None) else None
    to_e = (fp_new is not None and abs(fp_new - E) < tol)
    return {
        "fp_old": fp_old,
        "fp_new": fp_new,
        "drift": drift,
        "drift_to_e": to_e,
    }


def standard_iterations():
    """一组标准迭代函数 (展示漂移的各种特点)."""
    return {
        "平均迭代 (t+4)/2":       lambda t: (t + 4) / 2,
        "向e收缩 e+(t-e)/2":      lambda t: E + (t - E) / 2,
        "双盆 2t-t²/e":            lambda t: 2 * t - t * t / E,
        "平方 t²":                 lambda t: t * t,
        "牛顿 √9: (t+9/t)/2":      lambda t: (t + 9 / t) / 2,
    }


def report():
    """运行全部标准迭代的漂移检测并报告. 返回行列表 (供打印/落盘)."""
    lines = []
    lines.append("=== 标准迭代不动点漂移验证器 (换基元 0→e, 算法不变, 无映射) ===")
    lines.append(f"{'迭代':<20s} {'fp(0)':>10s} {'fp(e)':>12s} {'漂移':>12s} {'漂移到e?':>8s}")
    lines.append("-" * 70)
    for name, g in standard_iterations().items():
        r = drift_verify(g)
        fp_old = "发散" if r["fp_old"] is None else f"{r['fp_old']:.6f}"
        fp_new = "发散" if r["fp_new"] is None else f"{r['fp_new']:.6f}"
        drift = "—" if r["drift"] is None else f"{r['drift']:+.6f}"
        to_e = "✓" if r["drift_to_e"] else "✗"
        lines.append(f"{name:<20s} {fp_old:>10s} {fp_new:>12s} {drift:>12s} {to_e:>8s}")
    return lines


__all__ = ["iterate_fixpoint", "drift_verify", "standard_iterations", "report"]
