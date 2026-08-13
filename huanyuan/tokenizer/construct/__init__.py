"""tokenizer/construct —— 构造器: 自然语言符号 ↔ 四种形态 + NL 定义序列 → 命题。

按关注点分四组 (加新功能进对应模块, 不膨胀单文件):
  resolve  符号 → 候选 C token (resolve, resolve_seq)
  render   eid 序列 → 形态 (short_of, short, eid_form, char, translate)
  expand   eid → 展开向量 (bracket_vec, brace_logic, brace_derived)
  nl       NL/符号序列 → 命题元素序列 (tokenize, tokenize_nl, nl_to_proposition, nl_to_eids)

对外统一 re-export, 既有调用 from tokenizer.construct import X 不变。
"""
from .resolve import resolve, resolve_seq
from .render import short_of, short, eid_form, char, translate
from .expand import bracket_vec, brace_logic, brace_derived
from .nl import tokenize, tokenize_nl, nl_to_proposition, nl_to_eids, nl_define

__all__ = [
    'resolve', 'resolve_seq',
    'short_of', 'short', 'eid_form', 'char', 'translate',
    'bracket_vec', 'brace_logic', 'brace_derived',
    'tokenize', 'tokenize_nl', 'nl_to_proposition', 'nl_to_eids', 'nl_define',
]
