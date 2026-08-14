"""lab/diag/ —— Token 诊断工具集 (分析-设计闭环, 下沉子模块)

统一 io 接口 (第2项):
  save_report(path, data) / load_report(path)  报告读写 (JSON)
  analyze(samples, judge)                       闭环入口
  print_report(result)                          可读输出

零硬编码 token (第3项):
  所有角色识别通过 tokenizer.api 结构化属性 (dtype/arrange/form/references),
  **不硬编码任何 token 名/eid**。观测标签 (角色) 由 api.token_of 动态推导:
    truth:     dtype=bool 且 references 含公设 judgment
    judge_head: dtype=dual 且 references 含 truth 概念
    operator:  arrange ∈ {application, binary_connective, unary_connective, quantified}
    equality:  arrange = equality
    digit:     form=inductive 且 arrange=atom 且 references 链含 digit
    paren:     通过 gtoken/语法结构识别 (呈现层括号)
  自定义观测标签: 工具集允许外部注入自定义标签函数 (label_fn), 不限制死。

工具:
  coverage.py   Token 覆盖分析器 (工具1)
  learncurve.py Token 学习曲线观测器 (工具2)
  design.py     实验设计器 (工具3)
  analyze.py    闭环编排 (覆盖→学习→设计)
"""
from __future__ import annotations

import json
import os

from .roles import classify_token, roles_of_sequence, TOKEN_ROLES
from .coverage import coverage_report, summarize as coverage_summarize
from .learncurve import token_report, full_report
from .design import decide, plan
from .analyze import analyze, print_report


# ---- 统一 io 接口 (第2项) ----
def save_report(path, data) -> str:
    """报告 JSON 落盘 (可复现)。"""
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    return path


def load_report(path) -> dict:
    """报告 JSON 读取。"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


__all__ = [
    "classify_token", "roles_of_sequence", "TOKEN_ROLES",
    "coverage_report", "coverage_summarize",
    "token_report", "full_report",
    "decide", "plan",
    "analyze", "print_report",
    "save_report", "load_report",
]
