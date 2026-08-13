"""tokenizer/eval/itoken_eval.py —— 接口 token (itoken) 语义 (只读计算, 零构造)

itoken 层 (I:) 特性 (用户确立 2026-08-11):
  - **只允许被计算**: itoken 的语义查询 (num value) 是本模块唯一能力 —
    不产出新 token 序列 (不构造别的 token).
  - **不允许构造别的 token**: itoken definition 无 rules/引用 (references 空),
    不参与任何 token 生成.
  - **允许被别的 token 构造**: 普通 token 定义可引用 itoken eid 作参数.

数据源: tokenizer/tokens/itoken.jsonl (I 层). 输出: 语义值.

EXP-90 用途: num 原子 token (20 位数, num 泛化) 的定值语义 — 直觉计算
路径 (短序列) vs 结构计算 (numeral 逐位展开) 对比.
"""
from __future__ import annotations

from ..maintain import core

_ITOKEN_CACHE = None


def _itokens():
    global _ITOKEN_CACHE
    if _ITOKEN_CACHE is None:
        layer = core.load_layer("I") if hasattr(core, "load_layer") else {}
        _ITOKEN_CACHE = {eid: r for eid, r in layer.items()}
    return _ITOKEN_CACHE


def is_itoken(eid: str) -> bool:
    """eid 是否为接口 token."""
    return eid in _itokens()


def itoken_dtype(eid: str) -> str | None:
    """itoken 数据类型: num."""
    t = _itokens().get(eid)
    return t.get("dtype") if t else None


def itoken_value(eid: str) -> int | None:
    """num itoken → 定值 (常数语义)."""
    t = _itokens().get(eid)
    if not t or t.get("dtype") != "num":
        return None
    return t.get("value")


def reset_itoken_cache():
    """清除 itoken 缓存 (数据更新后)."""
    global _ITOKEN_CACHE
    _ITOKEN_CACHE = None


__all__ = ["is_itoken", "itoken_dtype", "itoken_value", "reset_itoken_cache"]
