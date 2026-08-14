"""lab/diag/hc_targets.py —— hardcode_check 目标适配器 (位置参数契约)

hardcode_check 以 target(*args) 调用, args=[config dict]. 合成器为
keyword-only (samples/seed 等). 本模块提供接受位置参数 config 的薄适配,
把实验配置转发到各合成入口 — 供 mask token 观测输出一致性 (硬编码诊断).

用法:
  PYTHONPATH=src python -m lab.diag.hardcode_check \
      --target "lab.diag.hc_targets:compose_from_config" \
      --config lab/configs/numeral_v4.json --scope BCSGPA
"""
from __future__ import annotations

from lab import synth_core
from lab.run_exp import build_samples


def compose_from_config(cfg) -> tuple:
    """配置 → 训练样本 (compose_samples 位置适配). 返回 (samples, npos, nneg)."""
    s = cfg["synth"]
    if "samples" not in s:
        raise ValueError("配置必须提供 synth.samples")
    return synth_core.compose_samples(samples=s["samples"], seed=cfg.get("seed", 0))


def build_from_config(cfg) -> tuple:
    """配置 → (训练样本, ood 配置) (run_exp.build_samples 位置适配)."""
    return build_samples(cfg, cfg.get("seed", 0))


def law_from_config(cfg) -> tuple:
    """配置 → 全部分布式定律样本合并 (law_samples 覆盖层, 位置适配)."""
    s = cfg["synth"]
    if "samples" not in s:
        raise ValueError("配置必须提供 synth.samples")
    all_s, npos, nneg = [], 0, 0
    for spec in s["samples"]:
        if spec.get("kind") != "law":
            continue
        ss, p, n = synth_core.law_samples(op=spec["op"], hi=spec.get("hi", 9),
                                          seed=cfg.get("seed", 0))
        all_s.extend(ss)
        npos += p
        nneg += n
    return all_s, npos, nneg
