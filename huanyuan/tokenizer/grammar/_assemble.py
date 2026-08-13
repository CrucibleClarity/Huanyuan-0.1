"""tokenizer/grammar/_assemble.py —— 语法组装器

按 gtoken (排列方法) 定义组装: 从 gtoken definition 读槽位序列, 校验子内容数, 返回结构化节点。
所有排列知识从 gtoken 读 (唯一语法源, 禁止硬编码): 本模块不判断槽位语义
(arg/fn/binder/body 是谁), 只按 definition.rules.term 槽位排布。
node 可为 gtoken (eid/name) 或概念 eid (读 definition.arrange → 节点类型):
  arrange 指向 gtoken; 若该 gtoken 槽位含 fn, 概念作为 fn 前置, 否则概念是节点本身 (不占槽位)。
涉及其他 token (概念/值/嵌套节点) 经接口查询, 不在此硬编码。
"""
from ..maintain import core
from ._query import _find, _slots


def _resolve(node):
    """node: gtoken (eid/name) 或概念 eid → (gtoken dict, 是否概念)。

    概念经 definition.arrange (概念→节点类型映射, 数据驱动) 解析到 gtoken。
    """
    try:
        return _find(node), False
    except KeyError:
        pass
    all_t = core.load_all()
    f = all_t.get(node)
    if f:
        defn = f.get('definition')
        arrange = defn.get('arrange') if isinstance(defn, dict) else None
        if arrange:
            return _find(arrange), True
    raise KeyError(f'无法解析为节点类型: {node!r} (需 gtoken 或带 arrange 的概念)')


def assemble(node, children):
    """按 gtoken 排列方法组装。

    node: gtoken eid/name 或概念 eid (读 arrange); children: 子内容列表。
    槽位含 fn (application): 概念作为 fn 前置; args 槽位 (可变参数列表): children 数量不限;
    否则 (equality/connective/quantified/atom): children 数须匹配子项槽位 (arg:N/binder/body)。
    返回 {node, name, slots, concept, children} — 嵌套 AST 节点。
    """
    g, is_concept = _resolve(node)
    slots = _slots(g)
    children = list(children)
    if 'args' not in slots:
        need = len([s for s in slots if s != 'fn'])
        if len(children) != need:
            raise ValueError(f"[{g['name']}] 需 {need} 个子内容, 收到 {len(children)}: {children!r}")
    seq = ([node] if (is_concept and 'fn' in slots) else []) + children
    return {
        'node': g['eid'],
        'name': g['name'],
        'slots': slots,
        'concept': node if is_concept else None,
        'children': seq,
    }
