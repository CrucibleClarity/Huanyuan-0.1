"""synth/aggregator/aggregate.py —— 样本集聚合器 (定义样本 + 合成样本 → 样本集)

样本集 = 定义样本 (待训练 token 及其引用 token 的定义) + 合成样本 (selector 产出),
每样本经 collector 收拢为 transformer 输入向量。
"""
from __future__ import annotations

from tokenizer import api
from ..retriever.retrieve import retrieve
from ..selector.select import select
from ..collector.collapse import collapse, vector_spec


def _definition_samples(retrieval) -> list[dict]:
    """定义样本: 待训练 token + 一层引用 token 的定义 (结构化 + 向量)。"""
    eids = list(dict.fromkeys([retrieval["token"]] + [e for e in retrieval["references"] if api.is_concept(e)]))
    out = []
    for e in eids:
        out.append({
            "kind": "definition",
            "eid": e,
            "name": api.name(e),
            "bracket": api.bracket(e),
            "brace": api.brace(e),
            "counts": api.counts(e),
            "vector": collapse({"vectors": [{"counts": api.counts(e)}]}, "root"),
        })
    return out


def build_sample_set(token_eid, n_synth=10, depth=1, seed=None, vector_method="sum", exclude=None) -> dict:
    """聚合待训练 token 的样本集。

    token_eid      待训练 token
    n_synth        合成样本数
    depth          合成样本嵌套层数
    vector_method  收拢向量方法 (collector.METHODS)
    exclude        排除的 token eid 集合 (训练样本不含, 可解释泛化目标)
    返回 {token, token_name, vector_method, vector_spec, definition_samples, synth_samples}
    """
    retrieval = retrieve(token_eid)
    def_samples = _definition_samples(retrieval)
    synth_samples = select(retrieval, n=n_synth, depth=depth, seed=seed, exclude=exclude)
    for s in synth_samples:
        s["vector"] = collapse(s, vector_method)
    return {
        "token": token_eid,
        "token_name": api.name(token_eid),
        "vector_method": vector_method,
        "vector_spec": vector_spec(),
        "definition_samples": def_samples,
        "synth_samples": synth_samples,
        "exclude": list(exclude) if exclude else [],
    }
