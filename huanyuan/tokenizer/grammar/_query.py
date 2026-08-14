"""tokenizer/grammar/_query.py —— 语法查询器

查询 gtoken (ExpressionNode 节点类型) 的排列方法。数据源统一 core.load_layer('G'),
不硬编码任何节点/槽位。node 可用 gtoken eid (G:0) 或 name (equality)。
"""
from ..maintain import core


def _find(node):
    """node: gtoken eid (G:0) 或 name (equality) → gtoken dict。"""
    all_g = core.load_layer('G')
    if node in all_g:
        return all_g[node]
    for g in all_g.values():
        if g.get('name') == node:
            return g
    raise KeyError(f'未知 gtoken: {node!r}')


def _slots(g):
    """从 gtoken definition 读槽位序列 (排列方法, 唯一语法源)。

    definition.rules[0].term = 槽位角色词序列 (arg:N/fn/args/binder/body)。
    """
    defn = g.get('definition') or {}
    for rule in defn.get('rules', []) or []:
        term = rule.get('term')
        if isinstance(term, list):
            return term
    return []


def query(node):
    """查询节点排列方法 (gtoken 全字段, 含 definition 槽位)。"""
    return dict(_find(node))


def scope(node):
    """可组装范围: 槽位布局 (语法源, 供组装器校验)。

    返回 gtoken 的子节点槽位序列; args 槽位 = 可变参数列表, 其余槽位 (arg:N/binder/body) = 固定子项。
    """
    g = _find(node)
    slots = _slots(g)
    return {
        'node': g['eid'],
        'name': g['name'],
        'slots': slots,
        'variable': 'args' in slots,
        'fills': '可变子内容 (args)' if 'args' in slots else f'恰好 {len([s for s in slots if s != "fn"])} 个子内容',
    }
