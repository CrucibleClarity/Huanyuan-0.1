"""selfref_calc.py —— 自指 token 衍生计算器 (succ 应用到自身)

数 = succ 的自指嵌套: succ 应用到自身.
  数 1 = succ                          ← 自指 token 本身
  数 2 = succ(succ)                    ← succ 应用到自身
  数 3 = succ(succ(succ))              ← 自指嵌套
  无终止 (无经典 0), 最内层 = succ 本身.

表示: 嵌套结构 ("succ", inner) = succ 应用到 inner; 裸 "succ" = 数 1.
"succ 次 succ 嵌套": 嵌套次数由 succ 应用到自身的结构体现, 无整数计数.
加法 = 自指复合: add(m,n) = 把 m 的 succ 嵌套套到 n 外面.
验证 = 结构等价 (eq), 无整数.

无人类算符 (+ - * / %), 无数字字面量, 无经典计数.
"""


def succ(n, step=None):
    """succ token 应用到 n.

    step=None → 自指嵌套 ("succ", n)  [默认: 数结构, succ 应用到自身]
    step=fn   → 自定义步进 fn(n)      [传参定义 succ: 如 lambda x: x+2]
    """
    if step is None:
        return ("succ", n)
    return step(n)


def one():
    """单位: succ token 本身 (数 1). 非数字 1."""
    return "succ"


def zero():
    """零次应用 (Church 数 0): 不应用任何 succ. 非数字 0, 是自指空结构."""
    return None


def apply_with(n, step, base):
    """数 n 应用到自定义 succ (step) 和 base.

    展开 n 的 succ 自指嵌套, 每层用 step 应用 → 'succ 次' 自定义步进.
    如 succ 定义为 2 (+2): 数 n → 2n;  succ 定义为 ×2: 数 n → 2^n.
    """
    if n is None:          # 数 0: 零次应用
        return base
    if n == "succ":
        return step(base)
    return apply_with(n[1], step, step(base))


def add(m, n):
    """加法 = 自指复合: 把 m 的 succ 嵌套套到 n 外面.

    迭代次数 = m (succ 自指结构); 迭代 'succ 次':
    m = succ(m') → 剥一层, 套一个 succ 到 n 外 → 自指嵌套 m+n 次.
    """
    if m is None:          # 数 0: 零次应用, 0+n = n
        return n
    if m == "succ":
        return ("succ", n)
    return add(m[1], ("succ", n))


def mul(a, b):
    """乘法 = 加法迭代器: a×b = 以 b (succ) 为迭代次数, 迭代'加 a' b 次.

    从零次应用出发: apply_with(b, 加a, 零) = 0 + a + ... (b 次) = a×b.
    """
    return apply_with(b, lambda acc: add(acc, a), zero())


def power(a, b):
    """幂 = 乘法迭代器: a^b = 以 b (succ) 为迭代次数, 迭代'乘 a' b 次.

    从数 1 (单位) 出发: apply_with(b, 乘a, 一) = 1×a×... (b 次) = a^b.
    """
    return apply_with(b, lambda acc: mul(acc, a), one())


def tetration(a, b):
    """超幂 (迭代4) = 幂迭代器: a↑↑b = 以 b (succ) 为迭代次数, 迭代'幂 a' b 次.

    从数 1 出发: apply_with(b, 幂a, 一) = a^(a^(...^a)) (b 次) = a↑↑b.
    """
    return apply_with(b, lambda acc: power(a, acc), one())


def translation(x):
    """平移: x → x+1 = succ 应用一次 (自指 token 一步)."""
    return ("succ", x)


def iterate(n, step, start):
    """Church 迭代: 以 n (succ) 为迭代次数, step 应用 n 次到 start."""
    return apply_with(n, step, start)


def recursion(n, step, start):
    """递归 (结构自指) = 迭代: 递归定义引用自身, 数值上等价于迭代."""
    return apply_with(n, step, start)


def fixpoint(step, start, max_succ=None):
    """不动点迭代: x_{n+1} = step(x_n) 直到 x = step(x) (自指稳定).

    与固定次数迭代不同: 次数由收敛决定 (不动点自指).
    max_succ: 可选迭代次数上限 (succ 结构).
    """
    x = start
    guard = 0
    limit = 1000
    while guard < limit:
        x1 = step(x)
        if eq(x1, x):
            return x
        x = x1
        guard += 1
    raise ValueError("不动点迭代未收敛")


def iterate_succ(n, base):
    """数 n 应用到 succ: 把 n 的 succ 嵌套展开, 每层套一个 succ 到 base.

    'succ 次 succ 嵌套': n 的嵌套层数由 n 的结构 (succ 应用到自身) 决定.
    """
    if n == "succ":
        return ("succ", base)
    return iterate_succ(n[1], ("succ", base))


def eq(a, b):
    """结构等价 (纯结构比较, 无整数)."""
    if a == "succ" or b == "succ":
        return a == b
    if isinstance(a, tuple) and isinstance(b, tuple):
        return eq(a[1], b[1])
    return a == b


__all__ = ["succ", "one", "zero", "add", "mul", "power", "tetration",
           "translation", "iterate", "recursion", "fixpoint",
           "iterate_succ", "apply_with", "eq"]
