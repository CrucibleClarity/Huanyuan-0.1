"""tokenizer/eval/prime_eval.py —— 素数判定 (基数自身迭代定义, 定义驱动求值)

素数定义 (用户):
  素数 n ⟺ 除"基数自身迭代 n 次"本身外, 任何其他迭代次数组合
         (b,k≥2, 使 b×k=n) 都不能达到 n。

等价: n 无真因子 (无 b∈[2,n-1] 整除 n)。

算法资源 (用户强调):
  - 只需检查 b ≤ √n (对称: b×k=n ⟺ k×b=n, 后半重复)
  - 整除验证: 沿 eval_add 逐次累加 b, 超过 n 提前停 (不完整迭代乘法)
"""
from __future__ import annotations

from .arith_eval import eval_add


def _divides(b: int, n: int, base: int = 10) -> bool:
    """b 是否整除 n (定义驱动: b 自身迭代, 累加达 n 提前停)。"""
    if b > n:
        return False
    acc = 0
    while acc < n:
        acc = eval_add(acc, b, base)
        if acc == n:
            return True
    return False


def is_prime(n: int, base: int = 10) -> bool:
    """素数判定: 无 b,k≥2 使 b×k=n (仅基数自身迭代 n 次可达)。"""
    if n < 2:
        return False
    for b in range(2, n):
        if b > n // b:
            break
        if _divides(b, n, base):
            return False
    return True


def primes_up_to(limit: int, base: int = 10) -> list[int]:
    """返回 [2, limit] 内全部素数。"""
    return [n for n in range(2, limit + 1) if is_prime(n, base)]


__all__ = ["is_prime", "primes_up_to"]
