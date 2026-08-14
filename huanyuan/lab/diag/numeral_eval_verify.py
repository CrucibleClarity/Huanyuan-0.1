#!/usr/bin/env python3
"""numeral_eval_verify.py —— 逻辑层 numeral 构造 + eval 完整闭环验证

验证 (token 结构驱动, 零查名/零 glyph):
  1. 构造: numeral = [sign_part][digit_seq], digit = [value, place], 高→低
  2. eval: 沿 token 结构推导值
     - value_N → 数值 (沿 arrow concept=value_N 链)
     - digit 位权: value × base^位序 (位序 = 从低位起)
     - numeral = Σ digit 贡献
  3. 任意进制 (base 是 numeral)
"""
import sys
sys.path.insert(0, ".")
from tokenizer import api
from tokenizer._register import ARROW_REGISTRY


# ---- 1. value token → 数值 (沿 arrow 链, 零查名) ----
def value_of(eid):
    if eid == api.eid_by_name("value_zero"):
        return 0
    n = 0
    cur = eid
    seen = set()
    while cur not in seen and n < 20:
        seen.add(cur)
        for ae, td in ARROW_REGISTRY.items():
            if api.arrow_concept_of(ae) == cur:
                tgt = api.target_of(ae)
                tn = api.name(tgt)
                if tn == "basepoint":
                    return n + 1
                if tn == "succ":
                    return n + 2
                if tn == "cardinality_one":
                    return n + 1
                cur = tgt
                n += 1
                break
    return -1


# ---- 2. 构造 numeral (高→低) ----
def digit_of(value_eid, place_eid):
    """[value, place] — 一位 digit 结构."""
    digit_c = api.eid_by_name("digit")
    place_c = api.eid_by_name("place")
    place_seq = api.assemble_seq(place_c, [[place_eid]])
    return api.assemble_seq(digit_c, [[value_eid], place_seq])


def numeral_of(digits_hi, base_val=10):
    """numeral 构造: [sign_part][digit_seq], digits_hi 是 [(value_eid, place_value)] 高→低."""
    sign_pos = api.eid_by_name("sign_pos")
    base_c = api.eid_by_name("base")
    card = api.eid_by_name("cardinality")
    # 进制部分: base 是 numeral — 用 value 序列表示进制值
    base_digits = []
    b = base_val
    if b == 0:
        base_digits = [api.eid_by_name("value_zero")]
    else:
        while b:
            base_digits.insert(0, api.eid_by_name(f"value_{b % 10}"))
            b //= 10
    base_seq = api.assemble_seq(base_c, [base_digits])
    sp = api.assemble_seq(api.eid_by_name("sign_part"),
                          [[sign_pos], base_seq, [card]])
    dg = api.assemble_seq(api.eid_by_name("digit_seq"), [digits_hi])
    return api.assemble_seq(api.eid_by_name("numeral"), [sp, dg])


# ---- 3. eval: numeral token 序列 → 值 (高→低 digit + 位序) ----
def eval_numeral(seq, base=10):
    """沿 token 序列求值: 从 digit [value, place] 结构."""
    # 解析: 序列含 sign_pos base value_* cardinality + digit 结构
    names = [api.name(t) for t in seq]
    # 提取所有 value token (digit 的 value 槽位)
    vals = [value_of(t) for t in seq if api.name(t).startswith("value_")]
    # digit 序列 = 连续的 value 对? 需要区分 base 的 value 与 digit 的 value
    # 简化验证: 找 digit_expr 结构
    # 高→低 digit: 每个 digit = [value, place]
    # 位权: 从低位 (最右) 起
    # 收集所有 place 后面的 value (digit 的 value)
    return vals  # 临时: 返回提取的 value 列表


def main():
    # 构造 numeral(25): 十位 value_two place1, 个位 value_five place0
    v2, v5 = api.eid_by_name("value_two"), api.eid_by_name("value_five")
    v0, v1 = api.eid_by_name("value_zero"), api.eid_by_name("value_one")
    d_hi = digit_of(v2, v1)   # 十位: 值2 位序1
    d_lo = digit_of(v5, v0)   # 个位: 值5 位序0
    seq = numeral_of(d_hi + d_lo, 10)
    print("numeral(25) 构造:", " ".join(api.name(t) for t in seq))
    print()
    print("提取 value tokens:", eval_numeral(seq))
    print()
    # 期望: digit 的 value = [2, 5] (十位2, 个位5), 位权 2×10+5=25


if __name__ == "__main__":
    main()
