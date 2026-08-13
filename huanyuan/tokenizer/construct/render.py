"""tokenizer/construct/render.py —— eid 序列 → 形态 (简称 / eid指代 / 字符 / 三形态)。

渲染是记法层的事: 数据库存裸 eid (主键), 展示时 short/char/translate 换壳。
"""
from .._register import token_of, resolve_derives, SYMBOL_REGISTRY, element_symbol
from .expand import brace_logic


def short_of(eid):
    """单个 eid → 简称。歧义符号带 eid 下标指向。"""
    td = token_of(eid)
    syms = [s for s in SYMBOL_REGISTRY.values() if eid in (s.maps_to or {})]
    if syms:
        g = syms[0].glyph
        cands = resolve_derives(g) if g else []
        return f"[{g}]({eid})" if len(cands) > 1 else f"[{g}]"
    return f"[{td.name}]"


def short(eids):
    """eid 序列 → 简称串。"""
    return "".join(short_of(e) for e in eids)


def eid_form(eids):
    """eid 序列 → [D:xxx] 指代串 (中括号直接框 eid)。"""
    return "".join(f"[{e}]" for e in eids)


def char(eids):
    """字符形式: 每 eid 的 S 层 glyph 拼接 (数学记法)。"""
    return "".join(element_symbol(e) for e in eids)


def translate(eids):
    """元素序列 → 三形态翻译: 字符 / 中括号 / 大括号。

    输入: C token eid 序列 (如 5+3=8 的 [D:206,D:250,D:204,D:260,D:209])。
    输出:
      char    : 5+3=8                  (S 层 glyph, 数学记法)
      bracket : [5][+][3][=](D:260)[8] (概念指代, 歧义带 eid 下标)
      brace   : 每元素大括号逻辑展开 (溯源到公设, 元素名序列)
    """
    return {
        "char": char(eids),
        "bracket": short(eids),
        "brace": {element_symbol(e): [token_of(x).name for x in brace_logic(e)] for e in eids},
    }
