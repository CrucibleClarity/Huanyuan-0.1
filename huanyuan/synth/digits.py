"""synth/digits.py —— 多位数字序列样本 (只调用 tokenizer, G/P 层驱动)

数位符号序列 = assemble(numeral, [digit_n...]) 结构 (gtoken numeral_seq + presentation args),
合成器只调 tokenizer (api.digit_symbols / assemble / print_ast / concepts)。
正例: 合法数位序列; 负例: 混入非数位概念。
"""
from __future__ import annotations

import random

from tokenizer import api

NUMERAL = api.role_token("numeral")


def _num_seq(digit_ids):
    """数位符号序列 → numeral 组装结构 seq (tokenizer assemble)。"""
    ast = api.assemble(NUMERAL, digit_ids)
    return api.concepts(ast)


def digit_seq_samples(n_samples=200, max_digits=3, seed=None, neg_ratio=0.3) -> list[dict]:
    """多位数位符号序列样本 (正负例)。返回 [{seq, valid, notation}]。

    只调 tokenizer: 数位符号取自 api.digit_concepts, 序列由 assemble(numeral) 合成 (G/P 层)。
    """
    v = list(api.digit_concepts().values())
    ops = [e for e in api.all_concepts() if api.arrange_of(e)]
    rng = random.Random(seed)
    out = []
    for _ in range(n_samples):
        nd = rng.randint(1, max_digits)
        digits = [rng.choice(v) for _ in range(nd)]
        if rng.random() < neg_ratio:
            pos = rng.randint(0, len(digits))
            digits.insert(pos, rng.choice(ops))
        seq = _num_seq(digits)
        out.append({"seq": seq, "valid": 1 if len(digits) == nd else 0,
                    "notation": f"{{{nd}}}位"})
    return out


def digit_seq_samples_fixed(nd: int, seed=0) -> dict:
    """nd 位数位符号随机序列 (位数泛化测试用, 20/2000 位)。"""
    v = list(api.digit_concepts().values())
    rng = random.Random(seed)
    digits = [rng.choice(v) for _ in range(nd)]
    return {"seq": _num_seq(digits), "valid": 1, "notation": f"{{{nd}}}位"}
