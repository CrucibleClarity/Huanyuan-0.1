"""synth/retriever/retrieve.py —— 检索器 (接收待训练 token → 可用范围)

通过 tokenizer.api 检索待训练 token 的:
  可用语法范围 (该 token 可用的排列方法/g token)
  可用序列 (token 出现于其定义的组合)
  每语法位可填的样本范围 (槽位候选概念)
"""
from __future__ import annotations

from tokenizer import api


def retrieve(token_eid: str) -> dict:
    """检索待训练 token 的可用范围。

    返回:
      token      待训练 token eid
      references 一层定义引用 (可达概念)
      syntax     可用排列方法列表 [{gtoken, name, slots}]
      slot_ranges {槽位: [候选概念 eid]} (每语法位可填范围)
    """
    refs = api.bracket(token_eid)
    syntax = []
    g = None
    try:
        g = api.query(token_eid)
    except KeyError:
        arrange = api.arrange_of(token_eid)
        if arrange:
            g = api.query(arrange)
    slots = []
    pres = api.presentation_of(token_eid) if g else None
    if pres:
        slots = [s for s in pres["grammar"] if s.startswith("arg:")]
    elif g:
        rules = (g.get("definition") or {}).get("rules") or []
        slots = [s for s in (rules[0].get("term", []) if rules else []) if s != "fn"]
    if g and slots:
        syntax.append({
            "gtoken": g.get("eid"),
            "name": g.get("name"),
            "slots": slots,
            "presentation": pres,
        })

    pool = list(dict.fromkeys([e for e in refs if api.is_concept(e)] + list(api.all_concepts())))
    slot_ranges = {s: pool for s in dict.fromkeys(s for g in syntax for s in g["slots"])}

    # 平行 ctoken 覆盖: 目标家族 (peers(token)) — 保证归属同一定义的 ctoken 同时训练。
    # 引用概念仅作结构子项 (出现在样本), 不扩散其家族 (避免候选爆炸)。
    peers_pool = [p for p in api.peers(token_eid)]
    slot_ranges = {s: list(dict.fromkeys(pool + peers_pool)) for s in slot_ranges}
    return {
        "token": token_eid,
        "references": refs,
        "syntax": syntax,
        "slot_ranges": slot_ranges,
        "peers": list(dict.fromkeys(peers_pool)),
    }
