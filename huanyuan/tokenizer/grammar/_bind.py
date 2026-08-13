"""tokenizer/grammar/_bind.py —— 绑定 contract (调研 B 类: binding signature, 不同构于普通排列)

quantified 等绑定节点在 gtoken definition 声明 binding_arity (FPT binding signature):
  每槽位上下文扩展的变量数。如 quantified [binder, body] → [0,1]: body 槽位在 binder 引入的 1 变量下。
check_scope: 验证表达式作用域 — var:N (De Bruijn 索引) 的 N 必须在当前绑定深度内;
  具名变量 (variable_* 概念, arrange→variable_reference) 引用必须在 binder 引入集内。
★ 数据驱动: binding_arity 从 gtoken definition 读; 变量识别经 arrange 查询; 验证器零硬编码绑定结构。
"""
import re

from ..maintain import core
from ._query import _find

_VAR_RE = re.compile(r'^var:(\d+)$')


def _binding_arity(g):
    """gtoken definition 的 binding_arity (默认全 0: 无绑定)。"""
    defn = g.get('definition') or {}
    return defn.get('binding_arity') or []


def _is_variable(eid):
    """eid 是否变量引用概念: definition.arrange → variable_reference (数据驱动, 不硬编码名)。"""
    f = core.load_all().get(eid, {})
    defn = f.get('definition')
    return bool(isinstance(defn, dict) and defn.get('arrange') == 'variable_reference')


def check_scope(node, depth=0, bound=None, errors=None):
    """验证表达式作用域。

    node: assemble 结果 (dict) 或原子 (str)。
    两种变量表示: var:N (位置编码, 向后兼容, N<绑定深度); 具名变量 (variable_* 概念,
    引用须在 binder 引入集 bound 内)。binder 槽位引入变量; fn 槽位 (概念) 跳过。
    返回错误列表 (空=作用域有效)。
    """
    bound = set() if bound is None else bound
    errors = [] if errors is None else errors
    if isinstance(node, str):
        m = _VAR_RE.match(node)
        if m and int(m.group(1)) >= depth:
            errors.append(f'{node} 超出绑定深度 {depth}')
        elif _is_variable(node) and node not in bound:
            errors.append(f'未绑定变量: {node}')
        return errors
    if isinstance(node, dict):
        try:
            g = _find(node.get('node'))
            arity = _binding_arity(g)
        except KeyError:
            arity = []
        slots = node.get('slots', [])
        children = node.get('children', [])
        ci = 0
        for i, s in enumerate(slots):
            if s == 'fn':
                continue                    # fn 是概念, 非表达式
            if ci < len(children):
                if s == 'binder':
                    child = children[ci]
                    if isinstance(child, str) and _is_variable(child):
                        bound = bound | {child}     # binder 引入具名变量
                    ci += 1
                    continue
                d = depth + (arity[i] if i < len(arity) else 0)
                check_scope(children[ci], d, bound, errors)
                ci += 1
        return errors
    return errors
