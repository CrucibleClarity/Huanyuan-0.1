"""tokenizer/eval/ —— 通用求值器 (真值由 token 定义提供, 零硬编码)

三个求值器:
  digit_eval     数字求值器: 数位序列 → 数值 (位权合成: 基数×进制^位序)
  logic_eval     逻辑递归求值器: 逻辑门沿 definition.rules 真值表求值
  compare_eval   比较求值器: 数字域比较沿 definition.rules 真值表求值

对外统一入口 (经 tokenizer.api):
  eval_digit(digit_eids, base) -> int         数字求值
  eval_logic(op, args) -> bool                逻辑求值
  eval_compare(op, a, b, base) -> bool        比较求值
"""
from .digit_eval import (
    digit_cardinality, digits_to_numeral, numeral_to_digits,
    eval_digit, eval_numeral_tokens,
)
from .logic_eval import logic_truth, eval_logic, eval_bool_expr, is_truth_token
from .compare_eval import compare_truth, eval_compare
from .prime_eval import is_prime, primes_up_to
from .numeral_eval import (
    value_number, eval_numeral, eval_digit_value, value_token, valid_digits,
    numeral_of, iterate_from_base, value_iter_path,
)
from .symmetry_eval import (
    eval_reciprocal, eval_division, eval_power, eval_root,
    eval_complement, eval_parallel_sum, verify_laws,
    eval_differential, eval_integral, eval_imaginary, eval_log,
    eval_translation, eval_inversion, eval_exp, eval_iterate, eval_fixpoint,
    eval_rotation, eval_tetration, eval_super_root, eval_super_log,
    eval_coupled_fixpoint, eval_scale, eval_recursion,
)
from .drift_verify import iterate_fixpoint, drift_verify, standard_iterations, report

__all__ = [
    "digit_cardinality", "digits_to_numeral", "numeral_to_digits",
    "eval_digit", "eval_numeral_tokens",
    "logic_truth", "eval_logic", "eval_bool_expr", "is_truth_token",
    "compare_truth", "eval_compare",
    "is_prime", "primes_up_to",
    "value_number", "eval_numeral", "eval_digit_value", "value_token", "valid_digits",
    "numeral_of", "iterate_from_base", "value_iter_path",
    "eval_reciprocal", "eval_division", "eval_power", "eval_root",
    "eval_complement", "eval_parallel_sum", "verify_laws",
    "eval_differential", "eval_integral", "eval_imaginary", "eval_log",
    "eval_translation", "eval_inversion", "eval_exp", "eval_iterate",
    "eval_fixpoint", "eval_rotation", "eval_tetration",
    "eval_super_root", "eval_super_log", "eval_coupled_fixpoint",
    "eval_scale", "eval_recursion",
    "iterate_fixpoint", "drift_verify", "standard_iterations", "report",
]
