"""head/router.py —— 注意力路由 (编排采样层与计算层)

接收接口参数 (sequence/selector/algorithm/weight/ctx), 编排:
  select (采样: 哪些参与) → compute (计算: 怎么算 —— 归类 或 权重)

compute 算法两类产物:
  归类算法 (def_sig/brace_sig) → {类签名: [eids]}
  权重算法 (weight_depth/polar/layer/type) → {eid: weight}
路由只负责按参数编排, 不掺算法逻辑。

pipeline: 任意编排 — steps 动作链 (select/weight/cluster), 前步产物进后步;
cluster 动作把聚类结果写入 ctx['cluster'] + ctx['types'] 供 type 采样/类型权重消费。
"""
from __future__ import annotations

from .select import get_selector
from .compute import get_algorithm

DEFAULT_SELECTOR = "all"
DEFAULT_ALGORITHM = "def_sig"


def run(sequence, selector=None, algorithm=None, weight=None, ctx=None, **kw):
    """编排采样层与计算层。

    weight 指定时走权重算法 (返回 {eid: weight}), 否则走归类算法 (返回 {类签名: [eids]})。
    ctx 可选含 ast (语法采样/深度权重所需)。
    """
    ctx = ctx or {}
    sel = get_selector(selector or DEFAULT_SELECTOR)
    selected = sel(sequence, ctx)
    if weight:
        return get_algorithm(weight)(selected, ctx=ctx, **kw)
    return get_algorithm(algorithm or DEFAULT_ALGORITHM)(selected, ctx=ctx, **kw)


def pipeline(sequence, steps, ctx=None):
    """任意编排: steps = 动作列表, 顺序执行, 前步产物进后步。

    动作:
      {'action': 'select',  'name': 选择器, ...}            → [eids]
      {'action': 'cluster', 'name': 归类算法, ...}          → {类签名: [eids]} (写入 ctx cluster/types)
      {'action': 'weight',  'name': 权重算法, ...}          → {eid: weight}
    cluster 后可接 select 'type' / weight 'weight_type' 消费 ctx['types']。
    """
    ctx = dict(ctx or {})
    data = sequence

    def params(st):
        return {k: v for k, v in st.items() if k not in ("action", "name")}

    for st in steps:
        act, name = st.get("action"), st.get("name")
        if act == "select":
            sel_ctx = dict(ctx)
            sel_ctx.update(params(st))
            data = get_selector(name)(data, sel_ctx)
        elif act == "cluster":
            groups = get_algorithm(name)(data, ctx=ctx, **params(st))
            ctx["cluster"] = groups
            ctx["types"] = {e: sig for sig, es in groups.items() for e in es}
        elif act == "weight":
            data = get_algorithm(name)(data, ctx=ctx, **params(st))
        else:
            raise ValueError(f"未知动作: {act!r} (select/cluster/weight)")
    return data


def run_sampled(ast, selector="syntax", algorithm="def_sig", weight=None, ctx=None, **kw):
    """便捷: 从 AST 直接编排 (语法采样 + 计算)。"""
    ctx = dict(ctx or {})
    ctx.setdefault("ast", ast)
    return run([], selector=selector, algorithm=algorithm, weight=weight, ctx=ctx, **kw)
