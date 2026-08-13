"""lab/synth_core_helpers_nobase.py —— EXP-C1a 辅助: 进制表示变体

C1a (radix-fixed): 进制固化 — 训练表示移除进制参数 (cardinality token),
测 base 16/60 外推 → 预期崩 (进制参数化必要).

strip 粒度 (config synth.strip_base):
  "all":         移除 base + cardinality (进制标记全删, 纯 digit — 过度破坏, 连 base10 也崩)
  "cardinality": 只移除 cardinality (进制参数值消失, 保留 base 提示 — C1a 正确粒度)

不修改 synth_core.py 主逻辑 (零侵入); run_exp 训练后对样本做表示层后处理.
"""
from tokenizer import api


def _strip_filter(eid, mode):
    name = None
    try:
        name = api.name(eid)
    except Exception:
        return False
    is_card = name == "cardinality" or name.startswith("cardinality")
    is_base = name == "base" or name.startswith("base_")
    if mode == "all":
        return is_card or is_base
    if mode == "cardinality":
        return is_card
    return False


def strip_base_tokens(samples, mode="cardinality"):
    """移除样本序列中的进制参数 token.

    返回新样本列表 (不修改原样本). mode="cardinality" 保留 base 提示但移除
    进制数值参数 (C1a 正确粒度: 训练无进制参数, 测跨进制崩).
    """
    out = []
    for s in samples:
        new_seq = [e for e in s["seq"] if not _strip_filter(e, mode)]
        ns = dict(s)
        ns["seq"] = new_seq
        out.append(ns)
    return out
