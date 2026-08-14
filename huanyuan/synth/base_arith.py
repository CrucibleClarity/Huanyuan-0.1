"""synth/base_arith.py —— 不同进制进位运算样本 (全数据驱动, 零硬编码)

进制定义 (用户): 数位符号达到进制基数后归零, 高一位序数符+1 (bool 规则, 无 int token)。
合成: 不同 base (2/8/10) 的 1-3 位加法, 数位符号序列 + 进位规则, 正负例。
数位符号取自 tokenizer (digit_n), 序列由 assemble(numeral) 合成 (G/P 层)。
"""
from __future__ import annotations

import random

from tokenizer import api

NUMERAL = api.role_token("numeral")
ADDITION = api.eid_by_name("addition")
EQUALS = api.role_token("equals")


def _digit_map():
    """数位符号 {n: eid} (tokenizer 数据: S 层符号→数位符号)。"""
    return {n: api.derives_of(str(n))[0] for n in range(10)}


def _repr(n, base):
    """n 的 base 进制数位列表 (进位规则: 数位达到 base 归零, 高位+1)。"""
    digits = []
    while n:
        n, d = divmod(n, base)
        digits.append(d)
    return digits or [0]


def _num_seq(digit_ids):
    return api.concepts(api.assemble(NUMERAL, digit_ids))


def _eq_seq(a_digits, b_digits, c_digits):
    add = api.assemble(ADDITION, [api.assemble(NUMERAL, a_digits), api.assemble(NUMERAL, b_digits)])
    return api.concepts(api.assemble(EQUALS, [add, api.assemble(NUMERAL, c_digits)]))


def base_add_samples(n_samples=200, bases=(2, 8, 10), max_digits=3, seed=None, neg_ratio=0.3):
    """不同进制 1-3 位加法样本 (正负例)。返回 [{seq, valid, base, a, b, c}]。"""
    dm = _digit_map()
    rng = random.Random(seed)
    out = []
    for _ in range(n_samples):
        base = rng.choice(bases)
        hi = base ** max_digits - 1
        a = rng.randint(0, hi)
        b = rng.randint(0, hi - a)
        c = a + b
        if rng.random() < neg_ratio:
            d = rng.randint(0, hi)
            while d == c:
                d = rng.randint(0, hi)
            c_digits = [dm[x] for x in _repr(d, base)]
            valid = 0
        else:
            c_digits = [dm[x] for x in _repr(c, base)]
            valid = 1
        seq = _eq_seq([dm[x] for x in _repr(a, base)], [dm[x] for x in _repr(b, base)], c_digits)
        out.append({"seq": seq, "valid": valid, "base": base, "a": a, "b": b, "c": c,
                    "notation": f"{a}+{b}={c if valid else d} (base{base})"})
    return out
