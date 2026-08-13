"""tokenizer/construct/resolve.py —— 符号 → 候选 C token (S 层映射, 不取首项)。

symbol → resolve_derives (S 层 maps_to), 歧义保留全部候选, 由语境收敛。
"""
from .._register import resolve_derives


def resolve(symbol):
    """单个符号 → 候选 C token eid 列表 (不取首项, 歧义保留全部候选)。"""
    return resolve_derives(symbol)


def resolve_seq(symbols):
    """符号序列 → 每符号候选 eid 列表。"""
    return [resolve(s) for s in symbols]
