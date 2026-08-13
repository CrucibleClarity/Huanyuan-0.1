"""tokenizer/grammar/_ski.py —— SKI 组合子 (G 层归约/抽象)

S/K/I 组合子是 G 层 gtoken (含 reduction 归约规则 pattern→value), 应用逻辑在此实现:
  reduce(expr)    按 gtoken reduction 数据归约 SKI 表达式 (数据驱动)
  abstract(v, e)  bracket abstraction: [var] expr → SKI 组合子表达式
识别 S/K/I 从 gtoken 结构 (reduction pattern/value): I=1参value==pattern, K=2参value=[首], S=3参。
所有规则 (S x y z = x z (y z) 等) 存 gtoken, 代码只读取执行, 不硬编码 token 名。
"""
from ..maintain import core


def _find_g(head):
    """head: gtoken eid (G:7) 或 name (ski_s) → gtoken dict。"""
    all_g = core.load_layer('G')
    if head in all_g:
        return all_g[head]
    for g in all_g.values():
        if g.get('name') == head:
            return g
    return None


def _combinators():
    """从 gtoken 数据识别 S/K/I (按 reduction 结构, 不硬编码名)。

    I: 1 参, value==首参 (恒等)
    K: 2 参, value==首参 (常量)
    S: 3 参 (替换)
    """
    comb = {}
    for g in core.load_layer('G').values():
        red = g.get('reduction')
        if not red:
            continue
        pat, val = red['pattern'], red['value']
        n = len(pat)
        if n == 1 and val == pat[0] and 'I' not in comb:
            comb['I'] = g['eid']
        elif n == 2 and val == pat[0] and 'K' not in comb:
            comb['K'] = g['eid']
        elif n == 3 and 'S' not in comb:
            comb['S'] = g['eid']
    return comb


def _subst(node, bindings):
    """递归替换 node 中的 arg:N (pattern 绑定) → 绑定值; 嵌套列表=应用结构。"""
    if isinstance(node, list):
        return [_subst(x, bindings) for x in node]
    return bindings.get(node, node)


def _flatten_app(node):
    """左结合应用链展平: [[C,a1,a2],b1,b2] → (C, [a1,a2,b1,b2])。返回 (根, 参数列表)。"""
    if not (isinstance(node, list) and node):
        return node, []
    if isinstance(node[0], list):
        root, head_args = _flatten_app(node[0])
        return root, head_args + node[1:]
    return node[0], node[1:]


def _reduce_step(node):
    """一步归约。展平左结合应用链找根组合子, 总参数充足则归约 (K 部分应用+外层参数能合并); 否则递归子项。"""
    if not (isinstance(node, list) and node):
        return node, False
    root, args = _flatten_app(node)
    if isinstance(root, str):
        g = _find_g(root)
        red = g.get('reduction') if g else None
        if red and len(args) >= len(red['pattern']):
            bindings = dict(zip(red['pattern'], args[:len(red['pattern'])]))
            new = _subst(red['value'], bindings)
            for rest in args[len(red['pattern']):]:   # 多参: 左结合, 剩余应用到结果
                new = [new, rest]
            return new, True
    # 无顶层 redex: 递归归约 fn 与每个子项
    new_children = []
    changed = False
    for c in node:
        nc, cc = _reduce_step(c)
        new_children.append(nc)
        changed = changed or cc
    return new_children, changed


def reduce(expr, max_steps=100):
    """按 gtoken reduction 数据归约 SKI 表达式 (到不动点或步数上限)。"""
    steps = 0
    while steps < max_steps:
        expr, changed = _reduce_step(expr)
        if not changed:
            return expr
        steps += 1
    return expr


def _contains(node, var):
    if node == var:
        return True
    if isinstance(node, list):
        return any(_contains(x, var) for x in node)
    return False


def _left_assoc(expr):
    """[fn, a, b, c] → [[[fn,a],b],c] (左结合, SKI 应用)。"""
    if isinstance(expr, list) and len(expr) > 2:
        return _left_assoc([expr[:2]] + expr[2:])
    return expr


def abstract(var, expr):
    """bracket abstraction: [var] expr → SKI 组合子表达式 (读 gtoken S/K/I)。

    [var] var    = I
    [var] M      = K M           (var 不在 M)
    [var] (M N)  = S ([var] M) ([var] N)
    """
    comb = _combinators()
    missing = [k for k in ('S', 'K', 'I') if k not in comb]
    if missing:
        raise RuntimeError(f'缺少 SKI 组合子 gtoken: {missing}')
    expr = _left_assoc(expr)
    if expr == var:
        return [comb['I']]
    if not _contains(expr, var):
        return [comb['K'], expr]
    if isinstance(expr, list) and len(expr) == 2:
        return [comb['S'], abstract(var, expr[0]), abstract(var, expr[1])]
    raise ValueError(f'不支持的结构: {expr!r} (需原子或二元应用 [fn, arg])')
