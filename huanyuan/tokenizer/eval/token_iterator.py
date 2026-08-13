"""tokenizer/eval/token_iterator.py —— 单独迭代器 (纯 token 原生, 无快速替代路径)

输入: 迭代次数 (token digit 序列)
产出: 迭代 n 次后的结果

迭代机制 (全部定义驱动, 零 Python 算术快捷):
  计数器 = 输入迭代次数 n (digit 序列)
  循环: 应用迭代步进 step (digit 序列 → digit 序列) → 计数器递减 1
  递减/递增走 digit_add 定义表 (definition.rules 真值表) 查表/逆推
  计数器归零 → 停止 → 产出结果
"""
from __future__ import annotations

from ..maintain import core
from .digit_eval import _eid_by_name, _digit_eid_for

_Z = None
_O = None
_C0 = None
_C1 = None
_ADD = None


def _eids():
    global _Z, _O, _C0, _C1, _ADD
    if _Z is None:
        _Z = _digit_eid_for(0)
        _O = _digit_eid_for(1)
        _C0 = _eid_by_name("carry_zero")
        _C1 = _eid_by_name("carry_one")
        _ADD = _eid_by_name("digit_add")
    return _Z, _O, _C0, _C1, _ADD


def _digit_add_lookup(d1, d2, carry):
    """查 digit_add 定义表: d1+d2+carry → (result, carry_out). 零 Python 算术."""
    z, o, c0, c1, add = _eids()
    d = (core.load_all().get(add) or {}).get("definition") or {}
    for rule in d.get("rules", []) or []:
        term = rule.get("term")
        if not (isinstance(term, list) and len(term) == 3):
            continue
        app, result = term[1], term[2]
        if not (isinstance(app, list) and app[:1] == [add]):
            continue
        if list(app[1:]) == [d1, d2, carry] and isinstance(result, list) and len(result) == 2:
            return result[0], result[1]
    raise ValueError("digit_add 定义未覆盖")


def _digit_inc_lookup(d):
    """d+1 (digit 级, 无进位): digit_add(d, 1, carry0) → (d+1, carry0). d=9 返回 None (需进位)."""
    z, o, c0, c1, add = _eids()
    r = _digit_add_lookup(d, o, c0)
    return r[0] if r[1] == c0 else None


def _digit_dec_lookup(d):
    """d-1 (digit 级, 无借位): digit_add(d-1, 1, carry0) → (d, carry0). d=0 返回 None (需借位)."""
    z, o, c0, c1, add = _eids()
    for probe in range(10):
        dp = _digit_eid_for(probe)
        r = _digit_add_lookup(dp, o, c0)
        if r[0] == d and r[1] == c0:
            return dp
    return None


def _strip_leading_zero(ds):
    """去除最高位零 (保留至少一位)."""
    z, o, c0, c1, add = _eids()
    out = list(ds)
    while len(out) > 1 and out[0] == z:
        out.pop(0)
    return out


def increment_digits(ds):
    """digit 序列 +1 (纯 token, 进位传递)."""
    z, o, c0, c1, add = _eids()
    out_rev = []
    carry = c0
    for i in range(len(ds)):
        d = ds[len(ds) - 1 - i]
        if i == 0:
            r, carry = _digit_add_lookup(d, o, carry)   # 个位加 1
        else:
            r, carry = _digit_add_lookup(d, z, carry)   # 高位只加进位
        out_rev.append(r)
    if carry == c1:
        out_rev.append(o)
    return _strip_leading_zero(list(reversed(out_rev)))


def decrement_digits(ds):
    """digit 序列 -1 (纯 token, 借位传递). ds 须为正."""
    z, o, c0, c1, add = _eids()
    out = list(ds)
    i = len(out) - 1
    while i >= 0:
        sub = _digit_dec_lookup(out[i])
        if sub is not None:
            out[i] = sub
            break
        out[i] = _digit_eid_for(9)  # 借位: 0-1 = 9
        i -= 1
    return _strip_leading_zero(out)


def is_zero_digits(ds):
    """digit 序列是否全零."""
    z, o, c0, c1, add = _eids()
    return all(d == z for d in ds)


class TokenIterator:
    """单独迭代器: 输入迭代次数, 产出迭代结果. 纯 token 原生.

    start: 初始值 (digit 序列)
    step:  迭代步进 (digit 序列 → digit 序列), 默认 +1 (纯 token)
    run(n_digits): 输入迭代次数 (digit 序列), 产出迭代 n 次后的结果
    """

    def __init__(self, start_digits, step=None):
        self.start = list(start_digits)
        self.step = step if step is not None else increment_digits

    def run(self, n_digits, max_guard=10000):
        result = list(self.start)
        counter = list(n_digits)
        guard = 0
        while not is_zero_digits(counter) and guard < max_guard:
            result = self.step(result)
            counter = decrement_digits(counter)
            guard += 1
        if is_zero_digits(counter) is False and guard >= max_guard:
            raise ValueError("迭代超时: 迭代次数未耗尽")
        return result

    def run_steps(self, n: int):
        """便捷入口: 输入 Python int 迭代次数 (编码为 digit 序列)."""
        from .digit_eval import numeral_to_digits
        return self.run(numeral_to_digits(n))


def _zero_digits():
    from .digit_eval import numeral_to_digits
    return numeral_to_digits(0)


def _one_digits():
    from .digit_eval import numeral_to_digits
    return numeral_to_digits(1)


def token_add(a_digits, b_digits):
    """纯 token 加法: 逐位 digit_add 查表 + 进位传递. 零 Python 算术."""
    z, o, c0, c1, add = _eids()
    carry = c0
    out_rev = []
    for i in range(max(len(a_digits), len(b_digits))):
        da = a_digits[len(a_digits) - 1 - i] if i < len(a_digits) else z
        db = b_digits[len(b_digits) - 1 - i] if i < len(b_digits) else z
        r, carry = _digit_add_lookup(da, db, carry)
        out_rev.append(r)
    if carry == c1:
        out_rev.append(o)
    return _strip_leading_zero(list(reversed(out_rev)))


def token_sub(a_digits, b_digits):
    """纯 token 减法: 递减 b 次 (迭代). 要求 a ≥ b."""
    it = TokenIterator(list(a_digits), step=decrement_digits)
    return it.run(b_digits)


def token_mul(a_digits, b_digits):
    """纯 token 乘法: 加法迭代 b 次 (迭代2 = 迭代的迭代)."""
    it = TokenIterator(_zero_digits(),
                       step=lambda acc: token_add(acc, a_digits))
    return it.run(b_digits)


def token_pow(base_digits, exp_digits):
    """纯 token 幂: 乘法迭代 exp 次 (迭代3 = 乘法的重复)."""
    it = TokenIterator(_one_digits(),
                       step=lambda acc: token_mul(acc, base_digits))
    return it.run(exp_digits)


def token_eval_numeral(digit_seq):
    """纯 token 位权合成: Σ digit × base^pos (用 token 运算, 不经 Python 算术).

    值 = 从 0 开始, 每 pos 加 digit × base^pos (全部走 token 迭代).
    """
    z, o, c0, c1, add = _eids()
    acc = _zero_digits()
    n = len(digit_seq)
    for pos, d in enumerate(reversed(digit_seq)):
        if is_zero_digits([d]):
            continue
        place = token_pow(_digits_of_base(), token_numeral_of(pos))
        contrib = token_mul([d], place)
        acc = token_add(acc, contrib)
    return acc


def _digits_of_base(base: int = 10):
    from .digit_eval import numeral_to_digits
    return numeral_to_digits(base)


def token_numeral_of(n: int):
    from .digit_eval import numeral_to_digits
    return numeral_to_digits(n)


def fixpoint_iterate(step, start_digits, max_iter=1000):
    """自指迭代 (不动点迭代): x_{n+1} = f(x_n) 迭代到 x_{n+1} = x_n (自指稳定).

    自指统一 = 寻找不动点. 这是所有迭代算法的本质:
      Banach 不动点定理 / 牛顿法 / Kripke 最小不动点 / 递归收敛.
    step: token 步进 (digit 序列 → digit 序列), 纯 token.
    收敛判据: 两步结果 digit 序列相等 (x = f(x)).
    """
    x = list(start_digits)
    for _ in range(max_iter):
        x1 = step(x)
        if x1 == x:          # 自指稳定: f(x) = x
            return x
        x = x1
    raise ValueError("自指迭代未收敛到不动点")


def self_apply(f_digits_fn, x_digits):
    """自指应用: 把迭代步进作用到自身 (f(x) = x 的一次自指检查).

    f(x) 是否等于 x (不动点判定, 自指稳定性的检测).
    """
    x1 = f_digits_fn(x_digits)
    return x1 == x_digits


__all__ = [
    "TokenIterator", "increment_digits", "decrement_digits",
    "is_zero_digits", "_digit_add_lookup",
    "token_add", "token_sub", "token_mul", "token_pow",
    "token_eval_numeral", "fixpoint_iterate", "self_apply",
]
