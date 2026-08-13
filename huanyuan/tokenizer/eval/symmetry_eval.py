"""tokenizer/eval/symmetry_eval.py —— 对称变换家族求值器 (真值由 token 定义提供)

对称变换家族 (迭代层对偶):
  reciprocal   层极性对偶 (乘法层) (反射@1): a → 1/a        — 分数代数读法
  division     除法 = 乘法∘reciprocal: division(a,b) = a×1/b
  power        正方向 (乘法迭代): a^b
  root         层对偶 (开方/分割): root(a,b) = a^(1/b)
  complement   单位区间对偶 (反射@1/2): a → 1-a           — 分数测度读法
  parallel_sum 加法在乘法层对偶下的 De Morgan 对偶: a∥b = 1/(1/a + 1/b)

值域扩展: reciprocal/division/root/complement 输出有理数 (Fraction),
精确表示分数. 遵循既有约定 (lab 合成器): 算术走 Python, 逻辑/比较走定义.
"""
from __future__ import annotations

from fractions import Fraction

from ..maintain import core
from .digit_eval import _eid_by_name, _name

# 对称家族 name → eid (惰性, 供自引用 token 解析: 规则里 self 即本 token)
_FAMILY = {
    "reciprocal", "power", "root", "division", "complement", "parallel_sum",
}


def _sym_eid(name: str) -> str:
    """对称家族概念 eid (懒加载, 从注册表查 name → eid)。"""
    return _eid_by_name(name)


def eval_reciprocal(a) -> Fraction:
    """reciprocal(a) = 1/a (层对偶). a=0 无倒数 (抛错)。"""
    a = Fraction(a)
    if a == 0:
        raise ZeroDivisionError("reciprocal(0) 未定义 (0 无乘法层对偶)")
    return Fraction(1, a)


def eval_division(a, b) -> Fraction:
    """division(a, b) = a/b = a × reciprocal(b) (分数代数读法)。"""
    a, b = Fraction(a), Fraction(b)
    if b == 0:
        raise ZeroDivisionError("division(a, 0) 未定义 (除以 0)")
    return a / b


def eval_power(a, b: int) -> int:
    """power(a, b) = a^b (正方向 (乘法迭代))。"""
    a, b = int(a), int(b)
    if b < 0:
        raise ValueError("power 指数为负 (幂迭代只定义非负次数)")
    return a ** b


def eval_root(a, b: int) -> Fraction:
    """root(a, b) = a^(1/b) (层对偶, 单位分割).

    精确情况: a 是 b 次完全幂 → 精确分数;
    非精确 (无理, 如 root(2,2)=√2) → 抛错 (新基数轴, 超出 token 值域)。
    """
    a, b = Fraction(a), int(b)
    if b <= 0:
        raise ValueError("root 次数必须为正")
    num, den = a.numerator, a.denominator
    x, ok = _perfect_power(num, b), _perfect_power(den, b)
    if x is None or ok is None:
        raise ValueError(f"root({a},{b}) 非精确 b 次幂 (无理数, 新基数轴, token 值域外)")
    return Fraction(x, ok)


def _perfect_power(n: int, b: int):
    """n 是否恰好是某整数的 b 次幂 → 返回该整数, 否则 None (仅处理非负 n)。"""
    if n == 0 or n == 1:
        return n
    if n < 0 and b % 2 == 1:
        r = _perfect_power(-n, b)
        return -r if r is not None else None
    if n < 0:
        return None
    if b > 1 and 2 ** b > n:
        return None  # 最小非平凡底数 2: 2^b > n 则无整数根 (剪枝防 mid**b 爆炸)
    lo, hi = 1, n
    while lo <= hi:
        mid = (lo + hi) // 2
        p = mid ** b
        if p == n:
            return mid
        if p < n:
            lo = mid + 1
        else:
            hi = mid - 1
    return None


def eval_complement(a) -> Fraction:
    """complement(a) = 1 - a (单位区间对偶, 反射@1/2)。"""
    return Fraction(1) - Fraction(a)


def eval_parallel_sum(a, b) -> Fraction:
    """parallel_sum(a, b) = 1/(1/a + 1/b) = ab/(a+b) (De Morgan 对偶)。"""
    a, b = Fraction(a), Fraction(b)
    if a == 0 or b == 0:
        raise ZeroDivisionError("parallel_sum 分母为 0 (1/a 或 1/b 未定义)")
    return a * b / (a + b)


def eval_differential(x, n) -> Fraction:
    """微分 = 降层算子 (跨层对偶): differentiate(x,n) = n·x^(n-1).

    从 n 层迭代结构提取重复次数 n, 降到 n-1 层。
    """
    x, n = Fraction(x), int(n)
    if n < 1:
        raise ValueError("微分降层需 n ≥ 1 (迭代次数为正)")
    return Fraction(n) * (x ** (n - 1))


def eval_integral(x, n) -> Fraction:
    """积分 = 升层算子 (跨层对偶): integrate(x,n) = x^(n+1)/(n+1).

    结构升到 n+1 层, 除以新次数。
    """
    x, n = Fraction(x), int(n)
    return (x ** (n + 1)) / (n + 1)


def eval_imaginary():
    """复数单位 i = root(neg(1), 2) (命名表达式, 非新基数)。"""
    return 1j


def _is_power_of(x, a):
    """x = a^n (n≥0 整数)? 返回 (n, bool)。"""
    n, cur = 0, Fraction(1)
    while cur < x and n < 1000:
        cur *= a
        n += 1
    if cur == x:
        return n, True
    return None, False


def eval_log(a, x) -> Fraction:
    """log_a(x): 幂的第二逆 (固定底数, 解指数), 测量迭代深度.

    精确情况: x = a^n (n 整数) → 返回 n; 非精确 → 抛错。
    """
    a, x = Fraction(a), Fraction(x)
    if a <= 1:
        raise ValueError("log 底数需 > 1")
    if x <= 0:
        raise ValueError("log 真数需 > 0")
    if x == 1:
        return Fraction(0)
    if x == a:
        return Fraction(1)
    n, ok = _is_power_of(x, a)
    if ok:
        return Fraction(n)
    raise ValueError(f"log_{a}({x}) 非精确 (幂关系外, 超出 token 值域)")


def eval_translation(x) -> Fraction:
    """平移 (模群生成元 T): x → x+1 = complement∘neg。"""
    return Fraction(x) + 1


def eval_inversion(x) -> Fraction:
    """反演 (模群生成元 S): x → -1/x = reciprocal∘neg. 对合。"""
    x = Fraction(x)
    if x == 0:
        raise ZeroDivisionError("inversion(0) 未定义 (0 无倒数)")
    return -Fraction(1, x)


def eval_exp(x) -> float:
    """exp(x): 微分算子不动点 (自指 d/dx exp = exp). e = exp(1)."""
    import math
    return math.exp(float(x))


def eval_iterate(x, n) -> Fraction:
    """iterate(x, n): Church 自指迭代, 加1应用 n 次 = x+n. 数 = 从原点迭代步数."""
    return Fraction(x) + int(n)


def eval_fixpoint(x) -> Fraction:
    """fixpoint(x): 自指迭代到不动点. 平均迭代 g(t)=(t+x)/2 收敛到 x."""
    return Fraction(x)


def eval_rotation(x):
    """rotation(x): 90° 旋转 = 乘 i. 四次旋转 = 恒等 (i⁴=1)."""
    return 1j * complex(x)


def eval_tetration(a, b) -> float:
    """tetration(a, b): 超幂 (幂迭代), a↑↑b = a^(a^...^a) b 次 = 幂的自指迭代."""
    a, b = float(a), int(b)
    if b < 1:
        raise ValueError("tetration 次数需 ≥ 1")
    r = a
    for _ in range(b - 1):
        r = a ** r
    return r


def eval_super_root(x, b) -> float:
    """super_root(x, b): 解 a↑↑b = x 的底数 (层对偶). 二分搜索."""
    b = int(b)
    if b == 1:
        return float(x)
    lo, hi = 0.0, float(x)
    for _ in range(200):
        mid = (lo + hi) / 2
        try:
            v = eval_tetration(mid, b)
        except (OverflowError, ValueError):
            hi = mid
            continue
        if v < x:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def eval_super_log(a, x) -> int:
    """super_log(a, x): 解 a↑↑b = x 的迭代次数 (层对偶)."""
    a, x = float(a), float(x)
    for b in range(1, 12):
        v = eval_tetration(a, b)
        if abs(v - x) < 1e-6:
            return b
        if v > x:
            break
    raise ValueError(f"super_log({a},{x}) 非精确 (非超幂关系)")


def eval_coupled_fixpoint(a, b) -> Fraction:
    """coupled_fixpoint(a, b): 耦合不动点, 解 x = a + b·x → x = a/(1-b)."""
    a, b = Fraction(a), Fraction(b)
    if b == 1:
        raise ValueError("coupled_fixpoint 无解 (b=1, 无不动点)")
    return a / (1 - b)


def eval_scale(x, n) -> int:
    """scale(x, n): 张缩/幂放缩 (对称家族 neg/scale/root), x^n."""
    return eval_power(x, n)


def eval_recursion(x, n) -> Fraction:
    """recursion(x, n): 递归 (结构自指) = iterate(x, n)."""
    return eval_iterate(x, n)


def verify_laws(*args) -> dict:
    """验证 token 规则中编码的定律 (真值由 token 定义提供, 求值校验)。

    沿 definition.rules 逐条: 求值等式两侧, 断言相等。
    返回 {概念名: {passed, total, failures}}。self 自引用解析为对应概念求值器。
    """
    from ..role import role_token
    eq = role_token("equals")
    results = {}
    for name in _LAW_EVAL:
        eid = _sym_eid(name)
        defn = (core.load_all().get(eid) or {}).get("definition") or {}
        rules = defn.get("rules") or []
        total = 0
        failures = []
        for i, rule in enumerate(rules):
            term = rule.get("term")
            if not (isinstance(term, list) and len(term) == 3 and term[0] == eq):
                continue
            total += 1
            if not _law_holds(term[1], term[2], name, args):
                failures.append(i)
        results[name] = {"passed": total - len(failures), "total": total,
                         "failures": failures}
    return results


def _law_holds(lhs, rhs, self_name, args) -> bool:
    """求值一条规则等式 [equals, lhs, rhs], 断言成立。"""
    try:
        l, r = _eval_term(lhs, self_name, args), _eval_term(rhs, self_name, args)
        if l is None or r is None:
            return False
        if isinstance(l, (float, complex)) or isinstance(r, (float, complex)):
            return abs(l - r) < 1e-9
        return l == r
    except Exception:
        return False


_DIGIT_NAMES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def _token_value(eid: str):
    """digit/value token eid → 数值 (Fraction)。值/数符双 token 体系, 按名解析。"""
    n = _name(eid)
    for i, w in enumerate(_DIGIT_NAMES):
        if n.endswith("_" + w) or n == w:
            return Fraction(i)
    return None


def _eval_term(t, self_name, args):
    """求值规则 term (数值/arg:N/self/概念应用), 递归。"""
    if isinstance(t, str):
        if t.startswith("arg:"):
            i = int(t.split(":")[1])
            return Fraction(args[i]) if i < len(args) else None
        if t == "self":
            return eval_imaginary() if self_name == "imaginary" else None
        return _token_value(t)
    if isinstance(t, list) and t:
        head, children = t[0], t[1:]
        if head == "self":
            vals = [_eval_term(x, self_name, args) for x in children]
            if any(v is None for v in vals):
                return None
            return _LAW_EVAL[self_name](*vals)
        name = _name(head)
        vals = [_eval_term(x, self_name, args) for x in children]
        if any(v is None for v in vals):
            return None
        if name == "addition":
            return sum(vals)
        if name == "subtraction":
            return vals[0] - vals[1]
        if name == "multiplication":
            r = vals[0]
            for v in vals[1:]:
                r = r * v
            return r
        if name == "power":
            return vals[0] ** int(vals[1])
        if name == "reciprocal":
            return eval_reciprocal(vals[0])
        if name == "division":
            return eval_division(vals[0], vals[1])
        if name == "neg":
            return -vals[0]
        if name == "complement":
            return 1 - vals[0]
        if name == "translation":
            return vals[0] + 1
        if name == "inversion":
            return -Fraction(1, vals[0])
        if name == "exp":
            return eval_exp(vals[0])
        if name == "iterate":
            return eval_iterate(vals[0], vals[1])
        if name == "fixpoint":
            return eval_fixpoint(vals[0])
        if name == "rotation":
            return eval_rotation(vals[0])
        if name == "tetration":
            return eval_tetration(vals[0], vals[1])
        if name == "super_root":
            return eval_super_root(vals[0], vals[1])
        if name == "super_log":
            return eval_super_log(vals[0], vals[1])
        if name == "coupled_fixpoint":
            return eval_coupled_fixpoint(vals[0], vals[1])
        if name == "scale":
            return eval_scale(vals[0], vals[1])
        if name == "recursion":
            return eval_recursion(vals[0], vals[1])
        if name == "root":
            a, b = vals[0], int(vals[1])
            if isinstance(a, Fraction) and a >= 0:
                try:
                    return eval_root(a, b)
                except ValueError:
                    return None
            import cmath
            return cmath.exp(cmath.log(complex(a)) / b)
        return None
    return None


_LAW_EVAL = {
    "reciprocal": eval_reciprocal,
    "power": eval_power,
    "root": eval_root,
    "division": eval_division,
    "complement": eval_complement,
    "parallel_sum": eval_parallel_sum,
    "differential": eval_differential,
    "integral": eval_integral,
    "imaginary": eval_imaginary,
    "logarithm": eval_log,
    "translation": eval_translation,
    "inversion": eval_inversion,
    "exp": eval_exp,
    "iterate": eval_iterate,
    "fixpoint": eval_fixpoint,
    "rotation": eval_rotation,
    "tetration": eval_tetration,
    "super_root": eval_super_root,
    "super_log": eval_super_log,
    "coupled_fixpoint": eval_coupled_fixpoint,
    "scale": eval_scale,
    "recursion": eval_recursion,
}

_EID_INDEX = None


def _build_eid_index():
    """对称家族 eid → 求值器索引 (一次性, 沿 _LAW_EVAL 结构登记)."""
    global _EID_INDEX
    if _EID_INDEX is not None:
        return _EID_INDEX
    from ..maintain import core
    idx = {}
    for name, fn in _LAW_EVAL.items():
        for eid, f in core.load_layer("C").items():
            if f.get("name") == name:
                idx[eid] = fn
    _EID_INDEX = idx
    return idx


def eval_sym_by_eid(op_eid: str, vals: list) -> float | Fraction | int | None:
    """对称家族求值 (按 eid 分发, 索引缓存, 零每次查名).

    沿 _LAW_EVAL 的 eid 索引定位求值器; 无对应返回 None (非对称家族).
    vals: 数值参数列表 (按算符元数). 供引擎/消费方委托对称语义.
    """
    fn = _build_eid_index().get(op_eid)
    if fn is None:
        return None
    try:
        return fn(*vals)
    except (ZeroDivisionError, ValueError):
        return None


__all__ = [
    "eval_reciprocal", "eval_division", "eval_power", "eval_root",
    "eval_complement", "eval_parallel_sum", "verify_laws",
    "eval_differential", "eval_integral", "eval_imaginary", "eval_log",
    "eval_translation", "eval_inversion", "eval_exp", "eval_iterate",
    "eval_fixpoint", "eval_rotation", "eval_tetration",
    "eval_super_root", "eval_super_log", "eval_coupled_fixpoint",
    "eval_scale", "eval_recursion", "eval_sym_by_eid",
]
