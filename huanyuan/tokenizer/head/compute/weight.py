"""算法: 权重计算 (加权采样样本 —— 计算方式的一种, 归计算层)

weight_depth  嵌套深度权重: high_to_low 突出谓词(表层) / low_to_high 突出操作数(深层)
weight_polar  谓词极性权重:  predicate 突出谓词(fn)   / operand 突出其他 token
weight_layer  角色层权重:    按层 (predicate/noun/decorator) 设权重, 高低可配
产物 {eid: weight}, 依赖 attention.depth_map (基础) 与 select/layers.roles_of (分层)。
"""
from ._registry import register_algorithm
from ..attention import depth_map
from ..select.layers import roles_of
from ...construct.expand import brace_logic, bracket_vec


def _jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@register_algorithm("weight_depth")
def weight_depth(eids, ctx=None, mode="high_to_low"):
    """嵌套深度权重变换。mode: high_to_low (表层权重高=突出谓词) / low_to_high (深层高=突出操作数)。"""
    ast = (ctx or {}).get("ast")
    if ast is None:
        return {e: 1.0 for e in eids}
    dm = depth_map(ast)
    if mode == "low_to_high":
        return {e: 1.0 + dm.get(e, 0) for e in eids}
    return {e: 1.0 / (1.0 + dm.get(e, 0)) for e in eids}


@register_algorithm("weight_polar")
def weight_polar(eids, ctx=None, mode="predicate", strength=2.0):
    """谓词/其他加权。mode: predicate (突出谓词) / operand (突出其他 token)。"""
    ast = (ctx or {}).get("ast")
    if ast is None:
        return {e: 1.0 for e in eids}
    rl = roles_of(ast)
    if mode == "operand":
        return {e: strength if rl.get(e) != "predicate" else 1.0 for e in eids}
    return {e: strength if rl.get(e) == "predicate" else 1.0 for e in eids}


@register_algorithm("weight_layer")
def weight_layer(eids, ctx=None, weights=None, mode=None):
    """按角色层 (predicate/noun/decorator) 的注意力权重, 不同层不同权重, 高低可配。

    weights: {'predicate': w, 'noun': w, 'decorator': w} 显式指定 (未指层默认 1.0)。
    mode: 预设快捷 — predicate_high / noun_high / decorator_high。
    """
    ast = (ctx or {}).get("ast")
    if ast is None:
        return {e: 1.0 for e in eids}
    if weights is None:
        if mode is None:
            raise ValueError("weight_layer 需 weights 或 mode 参数")
        weights = {
            "predicate_high": {"predicate": 2.0, "noun": 1.0, "decorator": 1.0},
            "noun_high":      {"predicate": 1.0, "noun": 2.0, "decorator": 1.0},
            "decorator_high": {"predicate": 1.0, "noun": 1.0, "decorator": 2.0},
        }[mode]
    rl = roles_of(ast)
    return {e: weights.get(rl.get(e), 1.0) for e in eids}


@register_algorithm("weight_type")
def weight_type(eids, ctx=None, order=None, weights=None, step=0.5):
    """按类型顺序权重划分 (同类/异类类别, 消费聚类结果)。

    order: 类型顺序列表 (第 0 类权重最高); weights: {类型: 权重} 显式指定;
    step: order 模式下的递减梯度。类型映射取 ctx['types'] (pipeline cluster 动作写入)。
    """
    types = (ctx or {}).get("types", {})
    if weights is None:
        if order is None:
            cluster = (ctx or {}).get("cluster", {})
            order = list(cluster.keys())
        weights = {c: max(1.0 - step * i, 0.0) for i, c in enumerate(order)}
    return {e: weights.get(types.get(e), 1.0) for e in eids}


@register_algorithm("intra_weight")
def intra_weight(eids, ctx=None, mode="typical", feature="brace"):
    """同类内差异权重: 类内成员与类质心的差异度 → 不同权重。

    mode: typical (质心典型成员权重高) / distinct (差异成员权重高)。
    feature: brace (溯源集, 差异体现数值/链长) / def (定义引用集)。
    消费 ctx['cluster'] (pipeline cluster 动作写入); 无聚类时全 1.0。
    """
    cluster = (ctx or {}).get("cluster")
    if not cluster:
        return {e: 1.0 for e in eids}
    members_all = set().union(*cluster.values()) if cluster else set()
    scope = set(eids) | members_all
    if feature == "brace":
        feats = {e: set(brace_logic(e)) for e in scope}
    else:
        feats = {e: set(bracket_vec(e)) for e in scope}
    eids_set = set(eids)
    w = {}
    for sig, members in cluster.items():
        centroid = set().union(*(feats.get(m, set()) for m in members)) if members else set()
        for m in members:
            if m in eids_set:
                sim = _jaccard(feats.get(m, set()), centroid) if centroid else 1.0
                w[m] = sim if mode == "typical" else (1.0 - sim)
    return w
