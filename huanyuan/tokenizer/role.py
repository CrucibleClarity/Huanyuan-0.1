"""tokenizer/role.py —— 组装角色原语 (沿 token 结构识别, 零名字/eid 硬编码)

铁律 (用户确立): 加 token/改 token 字段, 完全不需要外部适配. 组装角色
(judge/equals/bracket/truth/sign/numeral 结构件) 经本模块沿 token 结构
(arrange + P 层呈现定义 + 定义引用) 唯一识别, 消费方只调 role_token(kind),
不写死 token 名. 只有加 token 类别 (B/C/S/G/P/A 架构) 才涉及适配.

角色识别路径 (全部沿结构, 零名字):
  judge    arrange=judgment 的唯一概念
  equals   arrange=equality 且 P 层 grammar 含 '=' 符号 (组装等号)
  bracket  arrange=grouping 的唯一概念
  truth    真值 token 集 (judge 定义 rules 引用的真值)
  is_true  真值判定概念 (judge 组装的首槽位, 沿 judgment gtoken/ptoken)
  numeral  arrange=numeral_expr 的唯一概念
  sign_part/base_expr/digit_seq  arrange 各自唯一的 numeral 结构件
  sign     正负概念 (numeral sign_part 结构的 sign 槽位, 沿 P 层 grammar)
"""
from __future__ import annotations

from .maintain import core
from . import cte

# 角色 → arrange 类别映射 (架构级类别, 改 token 不影响; 改类别才适配)
_ARRANGE_ROLES = {
    "judge": "judgment",
    "bracket": "grouping",
    "numeral": "numeral_expr",
    "sign_part": "sign_part",
    "base": "base_expr",
    "digit_seq": "digit_seq",
    "digit": "digit_expr",
    "place": "place_expr",
}


def role_token(kind: str):
    """组装角色 token eid (沿结构识别, CTE 缓存, 零名字硬编码).

    kind: judge/equals/bracket/truth/is_true/numeral/sign_part/base/
          digit_seq/sign. 未找到 → ValueError.
    """
    return cte.get_or_compile("role:" + kind, cte._version(),
                              lambda: _resolve_role(kind))


def _resolve_role(kind: str):
    if kind == "truth":
        return _truth_role()
    if kind == "is_true":
        return _is_true_role()
    if kind == "equals":
        return _equals_role()
    if kind == "sign":
        return _sign_role()
    if kind == "iteration_layer":
        return _iteration_layer_role()
    arrange = _ARRANGE_ROLES.get(kind)
    if arrange is None:
        raise ValueError(f"未知组装角色: {kind}")
    hits = [e for e in _concept_eids()
            if _arrange_of(e) == arrange]
    if len(hits) != 1:
        raise ValueError(f"角色 {kind}: arrange={arrange} 概念数={len(hits)} (非唯一)")
    return hits[0]


def _iteration_layer_role():
    """迭代阶概念 (iteration_layer): 引 iterate + arrange=atom + 非值 token.

    沿结构: refs 含 iterate, arrange=atom, 且不可作 value 求值 (排除
    value_zero 等值概念) — 唯一命中 iteration_layer."""
    from ._register import token_eid
    from .eval import numeral_eval
    it = token_eid("iterate")
    hits = []
    for e in _concept_eids():
        d = core.load_all().get(e, {}).get("definition") or {}
        refs = d.get("references") or []
        if it not in refs or d.get("arrange") != "atom":
            continue
        # 排除值概念 (value_zero 等可沿 arrow 链求值)
        try:
            numeral_eval.value_number(e)
            continue
        except (ValueError, KeyError):
            pass
        hits.append(e)
    if len(hits) != 1:
        raise ValueError(f"iteration_layer 概念数={len(hits)} (非唯一)")
    return hits[0]


def _truth_role():
    """真值 token 集 (truth_true/false): 沿 logic_eval 真值对原语.

    真值对识别在 logic_eval._truth_eids (tokenizer 原生, 收敛唯一处);
    角色原语经其返回, 消费方不各自查名.
    """
    from .eval.logic_eval import _truth_eids as _te
    T, F = _te()
    return [T, F]


def _is_true_role():
    """真值判定概念 (is_true): judge 组装的首槽位 (沿 judgment gtoken/ptoken).

    judgment ptoken grammar = [⊤, arg:0, arg:1] — ⊤ 符号 (首元素) maps_to
    的概念即 is_true. 经 P 层呈现定义定位.
    """
    judge = _resolve_role("judge")
    for p in core.load_layer("P").values():
        if p.get("concept") == judge:
            grammar = p.get("grammar") or []
            if grammar and isinstance(grammar[0], str) and not grammar[0].startswith("arg:"):
                return _glyph_concept(grammar[0])
    raise ValueError("is_true 角色未定位 (judgment 呈现缺首符号)")


def _equals_role():
    """组装等号: arrange=equality 且 P 层 grammar 含 '=' 符号的概念.

    equality 含 equals_arith/比较算子, 组装等号 = 有 '=' 呈现定义者
    (符号经 glyph maps_to 收敛到该概念), 零名字.
    """
    for e in _concept_eids():
        if _arrange_of(e) != "equality":
            continue
        for p in core.load_layer("P").values():
            if p.get("concept") == e and "=" in (p.get("grammar") or []):
                return e
    raise ValueError("equals 角色未定位 (无 '=' 呈现的 equality 概念)")


def _sign_role():
    """正负概念 (sign): numeral 定义规则的 sign 槽位.

    numeral 规则 [sign, base, cardinality, value_zero] — 首引用 = sign
    概念 (沿定义结构, 零名字).
    """
    num = _resolve_role("numeral")
    d = core.load_all().get(num, {}).get("definition") or {}
    rules = d.get("rules") or []
    for rule in rules:
        term = rule.get("term") if isinstance(rule, dict) else rule
        if isinstance(term, list) and term:
            first = term[0]
            if isinstance(first, str) and first.startswith("D:"):
                return first
    raise ValueError("sign 角色未定位 (numeral 规则缺 sign 槽位)")


def _glyph_concept(glyph: str):
    """符号 → 概念 eid (沿 S 层 maps_to, 取唯一 D: 目标)."""
    from ._register import SYMBOL_REGISTRY
    for sid, td in SYMBOL_REGISTRY.items():
        if getattr(td, "glyph", "") == glyph:
            mt = td.maps_to or {}
            d_targets = [t for t in mt if t.startswith("D:")]
            if len(d_targets) == 1:
                return d_targets[0]
    return None


def _concept_eids():
    from ._register import all_eids
    return [e for e in all_eids() if e.startswith("D:")]


def _arrange_of(eid: str):
    return (core.load_all().get(eid, {}).get("definition") or {}).get("arrange")


__all__ = ["role_token"]
