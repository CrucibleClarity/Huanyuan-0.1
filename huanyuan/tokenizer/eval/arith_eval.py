"""tokenizer/eval/arith_eval.py —— 算术求值器 (迭代链, 真值由 token 定义提供)

迭代链 (用户确立): 加法 → 乘法 → 幂
  加法: 数位序列逐位 digit_add (进位传递, 读 digit_add 定义)
  乘法: 加法迭代 (a×b = b 次 a 相加, 调用加法求值器)
  幂:   乘法迭代 (base^n = n 次 base 相乘, 调用乘法求值器)

真值全部由 token 定义提供 (digit_add 数位加法表), 零硬编码。
"""
from __future__ import annotations

from ..maintain import core
from .digit_eval import digit_cardinality, numeral_to_digits, digits_to_numeral, _eid_by_name, _name, _digit_eid_for

_ADD = None
_CARRY_ZERO = None
_CARRY_ONE = None


def _eids():
    global _ADD, _CARRY_ZERO, _CARRY_ONE
    if _ADD is None:
        _ADD = _eid_by_name("digit_add")
        _CARRY_ZERO = _eid_by_name("carry_zero")
        _CARRY_ONE = _eid_by_name("carry_one")
    return _ADD, _CARRY_ZERO, _CARRY_ONE


def _zero_digit():
    return _digit_eid_for(0)


def _one_digit():
    return _digit_eid_for(1)


def _digit_add_lookup(a_digit: str, b_digit: str, carry_digit: str):
    """读 digit_add 定义: 两个 digit + 进位 → [结果 digit, 进位 digit]。

    沿 definition.rules 匹配 [equals, [digit_add, a, b, carry], [result, carry_out]]。
    返回 (result_eid, carry_eid); 无匹配 None。
    """
    add, cz, co = _eids()
    d = core.load_all()[add].get("definition") or {}
    for rule in d.get("rules", []) or []:
        term = rule.get("term", [])
        if not (isinstance(term, list) and len(term) == 3):
            continue
        app, result = term[1], term[2]
        if not (isinstance(app, list) and app[:1] == [add]):
            continue
        if list(app[1:]) == [a_digit, b_digit, carry_digit] and isinstance(result, list) and len(result) == 2:
            return result[0], result[1]
    return None


def _add_digit_seq(a_digits: list[str], b_digits: list[str], base: int = 10) -> list[str]:
    """数位序列加法 (逐位 digit_add + 进位传递, 读定义)。

    a_digits/b_digits: 数位序列 (高→低, 首位=最高位)。结果同序 (高→低)。
    """
    add, cz, co = _eids()
    carry = cz
    # 从低位 (序列末尾) 逐位相加, 进位传递
    out_rev = []  # 低→高
    for i in range(max(len(a_digits), len(b_digits))):
        da = a_digits[len(a_digits) - 1 - i] if i < len(a_digits) else _zero_digit()
        db = b_digits[len(b_digits) - 1 - i] if i < len(b_digits) else _zero_digit()
        r = _digit_add_lookup(da, db, carry)
        if r is None:
            raise ValueError(f"digit_add 定义未覆盖: {_name(da)}+{_name(db)} carry={_name(carry)}")
        result_d, carry = r
        out_rev.append(result_d)
    if carry == co:
        out_rev.append(_one_digit())
    out = list(reversed(out_rev))  # 高→低
    # 去除前导零 (最高位零)
    while len(out) > 1 and out[0] == _zero_digit():
        out.pop(0)
    return out


def eval_add(a: int, b: int, base: int = 10) -> int:
    """加法求值: a+b (数位逐位相加, 读 digit_add 定义)。"""
    a_d = numeral_to_digits(a, base)
    b_d = numeral_to_digits(b, base)
    result_d = _add_digit_seq(a_d, b_d, base)
    return digits_to_numeral(result_d, base)


def eval_sub(a: int, b: int, base: int = 10) -> int:
    """减法求值: a-b (数位逐位相减, 结果保留负数位, 不借位)。

    减法 = 加法 + 极性对偶 (用户): a - b = a + neg(b)。
    数位逐位相减, 结果位可为负 (neg 前缀), 位权求值。
    """
    a_d = numeral_to_digits(a, base)
    b_d = numeral_to_digits(b, base)
    neg = _eid_by_name("neg")
    # 逐位相减 (从低位): 结果位 = a_i - b_i (可负), 不做借位
    result_digits_rev = []  # 低→高, 每项 [digit] 或 [neg, digit]
    for i in range(max(len(a_d), len(b_d))):
        da = a_d[len(a_d) - 1 - i] if i < len(a_d) else _zero_digit()
        db = b_d[len(b_d) - 1 - i] if i < len(b_d) else _zero_digit()
        diff = digit_cardinality(da) - digit_cardinality(db)
        abs_diff = abs(diff)
        d = _digit_eid_for(abs_diff)
        if diff < 0:
            result_digits_rev.append([neg, d])
        else:
            result_digits_rev.append([d])
    # 去除最高位零
    while len(result_digits_rev) > 1 and _result_value(result_digits_rev[-1], base) == 0:
        result_digits_rev.pop()
    # 高→低
    result_digits = list(reversed(result_digits_rev))
    flat = []
    for item in result_digits:
        flat.extend(item)
    return digits_to_numeral(flat, base)


def _result_value(digit_item, base):
    """单个结果位 (含 neg 前缀) 的数值。"""
    from .digit_eval import digit_cardinality
    if len(digit_item) == 2 and digit_item[0] == _eid_by_name("neg"):
        return -digit_cardinality(digit_item[1])
    return digit_cardinality(digit_item[0])


def eval_mul(a: int, b: int, base: int = 10) -> int:
    """乘法求值: a×b (加法迭代: b 次 a 相加, 全部经加法求值器)。"""
    if a == 0 or b == 0:
        return 0
    result = 0
    for _ in range(b):
        result = eval_add(result, a, base)
    return result


def eval_pow(base_val: int, exp: int, base: int = 10) -> int:
    """幂求值: base^exp (乘法迭代: exp 次 base 相乘, 全部经乘法求值器)。"""
    if exp == 0:
        return 1
    result = 1
    for _ in range(exp):
        result = eval_mul(result, base_val, base)
    return result


__all__ = ["eval_add", "eval_sub", "eval_mul", "eval_pow", "_add_digit_seq"]
