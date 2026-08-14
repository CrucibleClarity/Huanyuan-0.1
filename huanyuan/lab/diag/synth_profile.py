#!/usr/bin/env python3
"""synth_profile.py —— 样本生成逻辑持久化 profile (覆盖每个合成函数)

对 lab.synth_core 的全部样本生成逻辑做性能剖析:
  - 逐函数计时 (每个 _s_* 注册合成器 + compose_samples 总体)
  - cProfile 全量热点 (cumulative, 持久化到文件)
  - 结果落盘, 不丢 (长任务/中断安全)

用法:
  PYTHONPATH=src python -m lab.diag.synth_profile --config lab/configs/numeral_v4.json
  PYTHONPATH=src python -m lab.diag.synth_profile --config lab/configs/numeral_v4.json --top 40
输出:
  控制台: 逐函数耗时 + 样本量
  归档:   archive/log/profile/synth_profile_<ts>.txt (cProfile 全量)
"""
import argparse
import cProfile
import io
import json
import os
import pstats
import time

from lab import synth_core
from lab.synth_core import _SAMPLE_REGISTRY, compose_samples


def profile_synthetic(config_path, top=30, quick=False):
    """profile 全部样本生成逻辑: 逐合成器计时 + cProfile 总体热点.

    quick=True: 缩小计算量 (只 profile 轻量合成器, 跳过 extrap_2000 等重型),
      hi 参数缩到 5, 只跑代表算子 — profile 看热点, 不需全量.
    """
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    specs = cfg["synth"]["samples"]
    if quick:
        # 只取轻量样本类型 + 缩小 hi (extrap_2000/numeral_split 等重型跳过)
        heavy = {"extrap_2000", "numeral_split", "deep_nest"}
        specs = [dict(s) for s in specs if s["kind"] not in heavy]
        for s in specs:
            if "hi" in s and isinstance(s["hi"], int) and s["hi"] > 5:
                s["hi"] = 5
            if "hi" in s and isinstance(s.get("hi"), int) and s["hi"] > 5:
                s["hi"] = 5
        # 算子去重: 每 kind 只留第一个 (profile 代表, 非全量)
        seen = set()
        specs_dedup = []
        for s in specs:
            k = (s["kind"], s.get("op", s.get("concept", "")))
            if k in seen:
                continue
            seen.add(k)
            specs_dedup.append(s)
        specs = specs_dedup
        print(f"[quick] 轻量样本类型: {len(specs)} 类 (跳过热重样本)")

    # 逐函数计时 (每个注册合成器)
    print(f"=== 逐合成器计时 ({len(specs)} 类样本) ===")
    per_fn = []
    total_samples = 0
    for spec in specs:
        kind = spec["kind"]
        fn = _SAMPLE_REGISTRY.get(kind)
        if fn is None:
            print(f"  ✗ {kind}: 未注册")
            continue
        t0 = time.time()
        ss, p, n = fn(spec, cfg.get("seed", 0))
        dt = time.time() - t0
        total_samples += len(ss)
        per_fn.append((kind, len(ss), dt))
        print(f"  {kind:14s} {spec.get('op', spec.get('concept', '')):12s} "
              f"{len(ss):5d} 样本  {dt*1000:8.1f} ms")
    per_fn.sort(key=lambda x: -x[2])
    print(f"\n  样本总数: {total_samples}")
    print(f"  最慢 3 个: " + ", ".join(f"{k}({dt*1000:.0f}ms)" for k, _, dt in per_fn[:3]))

    # cProfile 总体 (compose_samples, 覆盖所有内部函数)
    print(f"\n=== cProfile 总体 (compose_samples) ===")
    pr = cProfile.Profile()
    pr.enable()
    train, npos, nneg = compose_samples(samples=specs, seed=cfg.get("seed", 0))
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(top)

    # 持久化
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("archive", "log", "profile")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"synth_profile_{ts}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"config: {config_path} quick={quick}\n")
        f.write(f"样本总数: {len(train)} (正{npos} 负{nneg})\n")
        f.write("\n=== 逐合成器耗时 ===\n")
        for kind, ns, dt in per_fn:
            f.write(f"  {kind:14s} {ns:5d} 样本  {dt*1000:8.1f} ms\n")
        f.write("\n=== cProfile cumulative ===\n")
        f.write(s.getvalue())
    print(f"\n  profile 已归档: {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="样本生成 profile (逐函数 + cProfile)")
    ap.add_argument("--config", required=True, help="实验配置 (synth.samples)")
    ap.add_argument("--top", type=int, default=20, help="cProfile 前 N 热点")
    ap.add_argument("--quick", action="store_true", help="轻量模式 (缩小计算量)")
    args = ap.parse_args()
    profile_synthetic(args.config, args.top, quick=args.quick)


if __name__ == "__main__":
    main()
