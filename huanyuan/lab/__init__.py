"""lab/ —— 组合变化实验模块 (统一接口 + 分析设计闭环)

统一接口调用所有模块 (tokenizer.api / synth / train / infer / verify / archive / head / experiment),
支持组合变化实验 (combos/run_combos) + Token 学习分析-设计闭环。

核心工具:
  judge.py       通用判定器 (统一训练/评估/归档)
  synth_core.py  通用样本合成器 (平衡/全假值/嵌套)
  runner.py      统一编排器 (模块被动, lab 劫持)
  coverage.py    Token 覆盖分析器 (工具1: 样本视角覆盖)
  learncurve.py  Token 学习曲线观测器 (工具2: 分视角学习)
  design.py      实验设计器 (工具3: 覆盖→实验建议)
  analyze.py     闭环编排 (覆盖→学习→设计)

用法:
  from lab import combos, run_combos, api, Judge, analyze, ...
  result = analyze(samples, judge)   # 覆盖+学习+设计闭环
"""
from __future__ import annotations

from itertools import product

# ---- 统一接口: 聚合所有模块 ----
from tokenizer import api
from tokenizer.head import attention
from tokenizer.head.router import pipeline, run as head_run
from tokenizer.head.select import list_selectors, get_selector
from tokenizer.head.compute import list_algorithms, get_algorithm
from synth import build_sample_set
from synth import syntax, digits, base_arith
from train import train, train_seq
from train.data import collate, vocab
from infer import infer
from verify import verify
from archive import save_training, load_training, load_views, run_dir
from experiment import run as experiment_run

# ---- lab 核心工具 ----
from .judge import Judge, judge_sequence
from .synth_core import digits_of, nested_seq, balanced_samples
from .runner import run as runner_run
# ---- lab 诊断工具集 (lab.diag) ----
from .diag import (
    classify_token, roles_of_sequence,
    coverage_report, coverage_summarize,
    token_report, full_report,
    decide, plan,
    analyze, print_report,
    save_report, load_report,
)


def combos(grid: dict) -> list[dict]:
    """参数网格笛卡尔积 → 参数组合列表 (组合变化实验)。"""
    keys = list(grid.keys())
    return [dict(zip(keys, vals)) for vals in product(*[grid[k] for k in keys])]


def run_combos(grid: dict, fn, verbose=True, **kw) -> list:
    """组合实验: 每组合调 fn(**combo, **kw), 收集结果 (可打点)。"""
    results = []
    for c in combos(grid):
        if verbose:
            print(f">>> lab: {c}")
        r = fn(**c, **kw)
        results.append({"combo": c, "result": r})
    return results


__all__ = [
    "combos", "run_combos",
    "api", "attention", "pipeline", "head_run",
    "list_selectors", "get_selector", "list_algorithms", "get_algorithm",
    "build_sample_set", "syntax", "digits", "base_arith",
    "train", "train_seq", "collate", "vocab",
    "infer", "verify",
    "save_training", "load_training", "load_views", "run_dir",
    "experiment_run",
    "Judge", "judge_sequence", "digits_of", "nested_seq", "balanced_samples",
    "runner_run",
    "classify_token", "roles_of_sequence",
    "coverage_report", "coverage_summarize", "token_report", "full_report",
    "decide", "plan", "analyze", "print_report", "save_report", "load_report",
]
