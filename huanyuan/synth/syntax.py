"""synth/syntax.py —— 语法级正负例合成 (全数据驱动, 零硬编码)

正负例判定 = 语法合规性 (数据驱动, 从 gtoken 排列方法读), 非运算语义。

正例: 语法合规嵌套 (assemble 成功, 多级深度)。
负例: 语法违规 (违反 gtoken 槽位: 子项数错/角色错), 由排列方法逆向构造。
多级嵌套: 深度参数控制递归层数。

奖励正例: 训练损失对正例位置分类, 负例教合法性 (对错区分)。
"""
from __future__ import annotations

import random

from tokenizer import api


def pos_samples(token_eid, n=200, depth=1, seed=None, exclude=None) -> list[dict]:
    """语法正例: 语法合规嵌套样本 (assemble 校验通过, 多级深度, 去重)。"""
    from .selector.select import select
    from .retriever import retrieve
    r = retrieve(token_eid)
    return select(r, n=n, depth=depth, seed=seed, exclude=exclude)


def neg_samples(token_eid, n=100, seed=None, exclude=None) -> list[dict]:
    """语法负例 (多样化, 数据驱动): 
      1) 乱序 (token 在错误位置)
      2) 类型错: 该放数符却错放算符 (digit 位置填运算子)
      3) 子项数错 / 重复
    """
    from .retriever import retrieve
    exclude = set(exclude or [])
    rng = random.Random(seed)
    r = retrieve(token_eid)
    refs = [e for e in r["references"] if api.is_concept(e) and e not in exclude]
    ops = [e for e in api.all_concepts() if api.arrange_of(e) and e not in exclude]
    digits = list(api.digit_concepts().values())
    cands = list(dict.fromkeys(refs + ops + digits))
    out = []
    for _ in range(n):
        k = rng.random()
        if k < 0.33:
            a = rng.choice(cands)
            seq = [a, token_eid, rng.choice(digits)]
        elif k < 0.66:
            # 类型错: 数符位置错放算符 (digit 槽位填运算子)
            seq = [token_eid, rng.choice(ops) if ops else token_eid,
                   rng.choice(digits) if digits else token_eid]
        else:
            seq = [token_eid, rng.choice(cands), rng.choice(cands)]
        out.append({"seq": seq, "valid": 0, "notation": "|".join(api.name(x) for x in seq)})
    return out


def def_samples(tokens) -> list[dict]:
    """定义本身作为训练样本 (语义真值正例)。

    每 token 定义 = [token, 定义引用概念...] 序列, valid=1。
    token 的定义 (引用/结构) 直接进训练, 教模型"这是什么"。
    """
    out = []
    for t in tokens:
        refs = [r for r in api.bracket(t) if api.is_concept(r)]
        seq = [t] + refs
        out.append({"seq": seq, "valid": 1, "kind": "definition",
                    "notation": api.name(t)})
    return out


def samples(token_eid, n=200, depth=1, seed=None, exclude=None, neg_ratio=0.3) -> list[dict]:
    """正负例混合样本 (语法合规判定, 全数据驱动)。"""
    pos = pos_samples(token_eid, n=n, depth=depth, seed=seed, exclude=exclude)
    neg_n = int(n * neg_ratio)
    neg = neg_samples(token_eid, n=neg_n, seed=seed, exclude=exclude)
    for s in pos:
        s["valid"] = 1
    return pos + neg
