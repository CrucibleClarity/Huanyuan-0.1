"""synth/selector/select.py —— 样本选择器 (范围 → 具体训练样本)

根据检索器范围选择: 语法 (排列方法)、嵌套层数、填入语法位的 token,
构造含待训练 token 的表达式样本。通用 —— 无领域特判 (不假设数字/算术),
候选来自检索结果 (引用概念 + 平行 ctoken)。

沿链覆盖: 轮流以链上每概念 (目标+引用) 为锚, 用其平行兄弟填充样本,
保证每个 ctoken 有归属同一定义的平行 ctoken 同时训练 (可解释泛化前提)。
"""
from __future__ import annotations

import random

from tokenizer import api


def _pick(token, retrieval, rng, depth, exclude):
    """槽位候选: 引用概念 + 平行 ctoken (聚焦目标族), 排除 exclude。"""
    refs = [e for e in retrieval["references"] if api.is_concept(e) and e != token and e not in exclude]
    peers = [e for e in retrieval.get("peers", []) if e != token and e not in exclude]
    cands = list(dict.fromkeys(refs + peers))
    if not cands or depth <= 0:
        return rng.choice(cands) if cands else token
    return _build(rng.choice(cands), retrieval, rng, depth - 1, exclude, None)


def _build(token, retrieval, rng, depth, exclude, focus):
    """构造含 token 的表达式 AST (全正向 assemble)。

    focus: 优先填充到首槽位 (沿链锚定的平行兄弟, 保证其同训)。
    """
    pres = api.presentation_of(token)
    if pres:
        arg_slots = [s for s in pres["grammar"] if s.startswith("arg:")]
    else:
        try:
            g = api.query(token)
        except KeyError:
            arrange = api.arrange_of(token)
            if arrange is None:
                return token
            g = api.query(arrange)
        rules = (g.get("definition") or {}).get("rules") or []
        arg_slots = [s for s in (rules[0].get("term", []) if rules else []) if s != "fn"]
    if not arg_slots:
        return token
    children = []
    for i in range(len(arg_slots)):
        if i == 0 and focus is not None:
            children.append(focus)
        else:
            children.append(_pick(token, retrieval, rng, depth, exclude))
    return api.assemble(token, children)


def select(retrieval, n=1, depth=1, seed=None, exclude=None) -> list[dict]:
    """从检索范围选择 n 个训练样本 (去重保证多样性)。

    每样本 {ast, notation, seq, vectors}: AST、记法、概念序列、注意力向量。
    depth: 嵌套层数 (0=叶子填充, ≥1 允许子表达式嵌套)。
    exclude: 排除的 token eid 集合 (可解释泛化: 训练样本不含目标, 验证组合重建)。
    沿链覆盖: 每样本锚定目标家族平行兄弟; 去重直到 n 个唯一样本 (或尝试上限)。
    """
    exclude = set(exclude or [])
    rng = random.Random(seed)
    fam = [p for p in retrieval.get("peers", []) if p not in exclude]
    out, seen, tries = [], set(), 0
    while len(out) < n and tries < n * 10:
        tries += 1
        focus = rng.choice(fam) if fam else None
        ast = _build(retrieval["token"], retrieval, rng, depth, exclude, focus)
        notation = api.print_ast(ast)
        if notation in seen:
            continue
        seen.add(notation)
        seq = api.concepts(ast)
        out.append({
            "ast": ast,
            "notation": notation,
            "seq": seq,
            "vectors": api.analyze(seq),
            "exclude": sorted(exclude),
        })
    return out
