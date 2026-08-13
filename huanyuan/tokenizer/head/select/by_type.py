"""选择器: 类型采样 (同类/异类类别, 消费 compute 聚类结果)

ctx: {'types': {eid: 类签名}, 'cluster': {类签名: [eids]}, 'type': 类签名 或 类索引(int)}。
同类/异类识别由 compute 聚类算法 (def_sig/brace_sig) 完成, 本选择器只按类过滤。
"""
from ._registry import register_selector


@register_selector("type")
def select_by_type(sequence, ctx=None):
    ctx = ctx or {}
    types = ctx.get("types")
    want = ctx.get("type")
    if not types or want is None:
        return list(sequence)
    if isinstance(want, int):
        cluster = ctx.get("cluster", {})
        sigs = list(cluster.keys())
        if want >= len(sigs):
            return []
        want = sigs[want]
    return [e for e in sequence if types.get(e) == want]
