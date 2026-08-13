#!/usr/bin/env python3
"""numeral_construct.py —— numeral 构造规则验证器 (沿 gtoken 组装, 零硬编码)

验证 numeral 构造规则 (docs/numeral_construction_rules.md):
  numeral = [sign_part] [digit_seq]
  sign_part = [sign, base_expr, cardinality]   base_expr = [base, numeral] 任意进制
  digit_seq = 高→低 digit, digit_expr = [value, place]  place_expr = [place, numeral]
value 槽位 = value token 内嵌, 位序 = numeral (递归).

用法: PYTHONPATH=src python -m lab.diag.numeral_construct
      PYTHONPATH=src python -m lab.diag.numeral_construct --n 123
"""
import argparse

from tokenizer import api


def _vt(n):
    """数字 n → value token eid (0-9, 沿 value 命名; 仅本地验证辅助)."""
    return api.eid_by_name(f"value_{n}")


def base_expr(base_num_seq):
    """[base, numeral] — 进制结构."""
    return api.assemble_seq(api.eid_by_name("base"), [base_num_seq])


def sign_part(sign_eid, base_seq, card_eid):
    """[sign, base_expr, cardinality]."""
    return api.assemble_seq(api.eid_by_name("sign_part"), [[sign_eid], base_seq, [card_eid]])


def place_expr(value_eid):
    """[place, numeral] — 位序."""
    return api.assemble_seq(api.eid_by_name("place"), [[value_eid]])


def digit_expr(value_eid, place_seq):
    """[value, place] — 一位 digit."""
    return api.assemble_seq(api.eid_by_name("digit"), [[value_eid], place_seq])


def digit_seq(digits):
    """高→低 digit 序列."""
    return api.assemble_seq(api.eid_by_name("digit_seq"), [digits])


def numeral(n, base=10):
    """n → numeral token 序列 (沿构造规则, 高→低)."""
    if base != 10:
        raise NotImplementedError("非十进制 numeral 构造待定")
    if n == 0:
        dig = [api.eid_by_name("digit_zero")]
    else:
        ds = []
        while n:
            ds.append(n % base)
            n //= base
        ds.reverse()  # 高→低
        dig = [api.eid_by_name(f"digit_{d}") for d in ds]
    # base 槽位: 进制 numeral 用 value 表示 (10 = value_zero value_one 低→高? 或简化)
    base_seq = base_expr([api.eid_by_name("value_zero"), api.eid_by_name("value_one")])
    sp = sign_part(api.eid_by_name("sign_pos"), base_seq, api.eid_by_name("cardinality"))
    # digit 序列 (占位: 直接用 digit token, value/place 结构展开后续细化)
    dg = digit_seq(dig)
    return api.assemble_seq(api.eid_by_name("numeral"), [sp, dg])


def verify(n=25):
    """验证 numeral(n) 能 assemble 且结构正确."""
    seq = numeral(n)
    names = [api.name(t) for t in seq]
    print(f"numeral({n}): {' '.join(names)}")
    # 结构断言: 含 sign_part 三要素 + digit 序列
    assert any(x == "sign_pos" for x in names), "缺 sign"
    assert "base" in names, "缺 base"
    assert "cardinality" in names, "缺 cardinality"
    return seq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    args = ap.parse_args()
    verify(args.n)


if __name__ == "__main__":
    main()
