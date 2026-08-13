"""lab/diag/roles.py —— Token 角色识别 (沿样本合成逻辑反查, 零硬编码)

核心逻辑: 合成器 judge_sequence(prop, truth) 显式决定序列结构:
    [is_true][命题][truth_true | truth_false]
反查 token 身份 = 逆向合成逻辑 (从样本本身的位置模式归纳, 不做定义层推导):
    judge_head: 序列首 token (is_true, 合成器放置)
    truth:      序列末 token (真值, 合成器放置)
    equality:   排列方法 arrange=equality (tokenizer 数据)
    operator:   排列方法 arrange∈{application,binary_connective,unary_connective,quantified}
    operand:    算子与等号之间的数字
    answer:     等号之后的数字
    digit:      arrange=atom 且链到数字体系的数位符号
    context:    其他
"""
from __future__ import annotations

from tokenizer import api

_OPERATOR_ARRANGES = {"application", "binary_connective", "unary_connective", "quantified"}


def sequence_structure(seq):
    """逆向合成逻辑: 从判定序列抽 (judge_head, 命题, truth)。

    判定序列结构: [is_true][命题...][truth]。返回 (head_eid, prop_eids, truth_eid)。
    零硬编码: 头=首 token, 尾=末 token (合成器放置位置), 中间=命题。
    """
    if not seq:
        return None, [], None
    return seq[0], seq[1:-1], seq[-1]


def roles_of_sequence(seq, label_fn=None):
    """序列 → 每 token 角色 dict {eid: role}。label_fn 可覆盖默认分类。"""
    out = {}
    head, prop, truth = sequence_structure(seq)
    if label_fn is not None:
        for e in seq:
            out[e] = label_fn(e)
        return out
    # 合成器位置: 首=判定头, 末=真值
    if len(seq) >= 1:
        out[seq[0]] = "judge_head"
    if len(seq) >= 2:
        out[seq[-1]] = "truth"
    # 命题内部: 按 arrange 分类 (tokenizer 数据, 零推导)
    after_eq = False
    for e in prop:
        arrange = api.arrange_of(e)
        if arrange == "equality":
            out[e] = "equality"
            after_eq = True
        elif arrange in _OPERATOR_ARRANGES:
            out[e] = "operator"
        elif arrange == "atom":
            out[e] = "answer" if after_eq else "operand"
        elif arrange == "variable_reference":
            out[e] = "variable"
        else:
            out[e] = "context"
    return out


def classify_token(eid, samples):
    """token eid → 角色 (基于样本位置模式, 数据驱动)。

    沿合成逻辑反查: 统计该 token 在样本中出现在判定头/真值/中间位置的频率,
    取最频繁角色。零硬编码。
    """
    roles = {"judge_head": 0, "truth": 0, "interior": 0}
    for s in samples:
        seq = s["seq"]
        if eid not in seq:
            continue
        if seq and seq[0] == eid:
            roles["judge_head"] += 1
        if seq and seq[-1] == eid:
            roles["truth"] += 1
        if len(seq) > 2 and eid in seq[1:-1]:
            roles["interior"] += 1
    if roles["judge_head"] > 0 and roles["judge_head"] >= roles["truth"]:
        return "judge_head"
    if roles["truth"] > 0 and roles["truth"] >= roles["judge_head"]:
        return "truth"
    if roles["interior"]:
        arrange = api.arrange_of(eid)
        if arrange == "equality":
            return "equality"
        if arrange in _OPERATOR_ARRANGES:
            return "operator"
        if arrange == "atom":
            return "digit"
        if arrange == "variable_reference":
            return "variable"
        return "context"
    return "unknown"


TOKEN_ROLES = {
    "truth", "judge_head", "operator", "equality", "digit",
    "operand", "answer", "variable", "context", "unknown",
}
