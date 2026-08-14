"""tokenizer/construct/expand.py —— eid → 展开向量 (中括号 / 大括号, 沿定义链)。

中括号 = 本层定义引用; 大括号 = 穿透到公设 (form:axiomatic 为展开终点)。
只读结构化 definition (references), 不解析散文。
"""
from collections import Counter

from .._register import token_of, all_eids, definition_refs, is_axiomatic_eid


def bracket_vec(eid):
    """中括号向量: 只展开本层定义。

    [x] = x 的 definition 引用 eid 列表 (一层, 不递归)。
    依据: 结构化 definition 的 references (自动派生), 程序只读结构化定义。
    """
    return definition_refs(token_of(eid).definition)


def _brace_chain(eid, _seen=None):
    """穿透到 base 的展开序列 (不去重, 沿 definition 递归; _seen 防环)。

    遇公理式定义 (form:axiomatic) 终止 —— 公设基是展开终点, 不再展开。
    G 层语法 token (无定义体, 不进 B/C/S 注册表) 同样终止。
    """
    if _seen is None:
        _seen = set()
    if eid in _seen:
        return [eid]
    _seen.add(eid)
    try:
        if is_axiomatic_eid(eid):
            return [eid]
        refs = definition_refs(token_of(eid).definition)
    except KeyError:
        return [eid]
    if not refs:
        return [eid]
    result = [eid]
    for r in refs:
        result.extend(_brace_chain(r, _seen))
    return result


def brace_logic(eid):
    """逻辑大括号: 不去重, 纯按定义展开。"""
    return _brace_chain(eid)


def brace_derived(eid):
    """派生大括号: 按完整 eid 空间重建计数向量。

    穿透展开后, 在全部 eid 空间上计数: 没有=0, 有=1, 重复几次=几。
    返回 {eid: count} (含 0, 空间完整)。
    """
    space = sorted(all_eids())
    cnt = Counter(_brace_chain(eid))
    return {e: cnt.get(e, 0) for e in space}
