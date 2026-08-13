"""tokenizer/eval/sample_eval.py —— tokenizer 原生样本基础设施 (全项目共享)

职责 (用户确立):
  tokenizer: 单个样本如何产生 / 每层展开 / 真值 / 序列 token 位的可填范围 (候选域)
  消费方:    在可填范围内选择样本集 (枚举哪些组合/错题比例/展开深度)

原语:
  组装:   judge_seq(prop, truth) → [is_true][命题][truth_true|false]
  真值:   definition_truth(op, arg_tokens) 沿定义 rules 还原; arrow_truth(concept,s,t) A 层存在性
  单样本: sample_definition(op, args) / sample_arrow(concept, s, t) → (seq, truth)|(None,None)
  候选域: logic_candidates() / digit_candidates() / arrow_endpoints(concept)
          (序列 token 位能填哪些 token, tokenizer 职责)
全部零硬编码: 真值从 token 定义/arrow 来, token 从 glyph/结构解析, 无名字正则.
"""
from __future__ import annotations

from .._register import ARROW_REGISTRY, SYMBOL_REGISTRY, token_of, is_derived, resolve_glyph
from ..maintain import core
from .digit_eval import _eid_by_name, _name
from .logic_eval import logic_truth, _truth_eids


def judge_seq(prop, truth):
    """判定序列 (token 原生组装): 沿 judge 概念 (judgment gtoken/ptoken) 组装.

    judgment ptoken grammar = [⊤, arg:0, arg:1] (arg:0=命题, arg:1=真值),
    is_true/truth 符号与槽位来自 P 层, 零手写判定格式.
    """
    from tokenizer import api as _api
    from ..role import role_token
    _T, _F = _truth_eids()
    tr = _T if truth else _F
    return _api.assemble_seq(role_token("judge"), [list(prop), [tr]])


def definition_truth(op, arg_tokens):
    """定义真值: 沿 op 概念 definition.rules 还原 (logic_truth, 零硬编码).

    arg_tokens: 操作数 token 序列 (eid). 匹配 [equals, [op, args...], result].
    返回 bool; 定义未覆盖 → None.
    """
    return logic_truth(op, arg_tokens)


def arrow_truth(concept, source, target):
    """arrow 真值: 是否存在 arrow(concept, source→target) (查 A 层字段, 非 name 正则)."""
    for td in ARROW_REGISTRY.values():
        if td.concept == concept and td.source == source and td.target == target:
            return True
    return False


def sample_definition(op, arg_tokens):
    """单个定义样本: 给定操作数 token → (判定序列, 真值). 定义未覆盖 → (None, None).

    命题沿 op 概念 gtoken/ptoken 组装 (assemble_seq), 判定 judge_seq.
    """
    from tokenizer import api as _api
    op = _resolve_eid(op)
    truth = definition_truth(op, arg_tokens)
    if truth is None:
        return None, None
    prop = _api.assemble_seq(op, [[t] for t in arg_tokens])
    return judge_seq(prop, truth), truth


def sample_arrow(concept, source, target):
    """单个 arrow 样本: 给定概念 + 两端 token → (判定序列, 真值). 真值 = arrow 存在性.

    命题沿 arrow 排列 (source concept target), 判定 judge_seq (token 原生).
    """
    concept = _resolve_eid(concept)
    truth = arrow_truth(concept, source, target)
    return judge_seq([concept] + [source, target], truth), truth


def sample_arrow_token(arrow_eid, source, target):
    """单个 arrow 样本 (含 A 层箭头 token): [is_true][source][arrow_eid][target][truth].

    A 层箭头 token (A:*) 作为序列元素参与训练 (arrow 关系本身可学习),
    真值 = 端点是否匹配该箭头的 source/target (查 A 层字段, 非 name 正则).
    """
    from tokenizer._register import ARROW_REGISTRY
    td = ARROW_REGISTRY[arrow_eid]
    truth = (source == td.source and target == td.target)
    return judge_seq([source, arrow_eid, target], truth), truth


def logic_candidates():
    """逻辑操作数位可填 token: [truth_true, truth_false] (tokenizer 候选域)."""
    _T, _F = _truth_eids()
    return [_T, _F]


def _atom_token(glyph):
    """glyph → 候选 token (inductive+atom 识别, 数据驱动)."""
    for sid in resolve_glyph(glyph):
        for target in SYMBOL_REGISTRY[sid].maps_to:
            if not is_derived(target):
                continue
            defn = token_of(target).definition
            if isinstance(defn, dict) and defn.get("form") == "inductive" and defn.get("arrange") == "atom":
                return target
    return None


def digit_candidates():
    """数字操作数位可填 token 全集 (digit_0..9, 沿 glyph 解析 + inductive atom, tokenizer 候选域).

    消费方在全集内选子集 (如 hi 范围).
    """
    return [_atom_token(str(d)) for d in range(10)]


def value_token(n):
    """数字 n (0-9) → value token eid (glyph 解析 + inductive atom)."""
    return _atom_token(str(n))


def digit_tokens(n):
    """数字 n → digit token 序列 (低→高, 数据驱动)."""
    if n == 0:
        return [_atom_token("0")]
    out = []
    while n:
        n, d = divmod(n, 10)
        out.append(_atom_token(str(d)))
    return out


def arrow_endpoints(concept):
    """arrow 概念 → 端点候选 token 集 (source∪target, 沿 A 层提取, tokenizer 候选域).

    消费方在端点集内枚举组合.
    """
    concept = _resolve_eid(concept)
    pairs = [(td.source, td.target) for td in ARROW_REGISTRY.values()
             if td.concept == concept]
    if not pairs:
        return [], []
    toks = sorted({t for pair in pairs for t in pair})
    return toks, pairs


def _resolve_eid(x):
    """name 或 eid → eid (name 先查, 非名字则视为 eid)."""
    if not isinstance(x, str):
        return x
    try:
        return _eid_by_name(x)
    except KeyError:
        return x


def _op_domain(op):
    """算子域判定: 逻辑 (bool 操作数) / 数字 (0..hi). 沿 arrange 特征, 零硬编码算子名."""
    defn = core.load_all().get(op, {}).get("definition") or {}
    arrange = defn.get("arrange")
    return "logical" if arrange in ("binary_connective", "unary_connective") else "numeric"


op_domain = _op_domain


# ---- 迭代样本 (标准迭代语法: 方向+层数+被迭代算符, tokenizer 原生合成) ----
def iteration_arrow_samples():
    """A 层迭代箭头样本 (沿 ARROW_REGISTRY, 权威迭代链, 零 is_succ 混用).

    iterate 箭头 = 迭代升阶 (succ→addition 等); inverse 箭头 = 迭代对偶
    (power↔root 等). 每箭头生成 [is_true][concept][source][target][truth].
    沿 A 层数据, 加 token/改箭头零适配.
    返回样本列表.
    """
    from tokenizer import api as _api
    samples = []
    for e, td in ARROW_REGISTRY.items():
        cname = _api.name(td.concept)
        if cname not in ("iterate", "inverse"):
            continue
        seq, truth = sample_arrow(td.concept, td.source, td.target)
        samples.append({"seq": seq, "valid": 1, "truth": truth, "depth": 1})
    return samples


def sample_iterate_expression(direction_eid, layer_eid, op_eid):
    """标准迭代语法样本 (用户确立): [is_true][iterate][方向][层数][被迭代算符][truth].

    迭代是针对算符的算符: iterate(方向, 层数, 被迭代算符) — 方向 (increase/
    decrease 升/降阶) + 层数 (numeral: value_one=1层) + 被迭代算符 (succ/
    addition/multiplication...). 沿 iterate_expr gtoken 组装, 可逐层展开.
    真值恒真 (语法声明: 该迭代关系成立).
    """
    from tokenizer import api as _api
    it = _api.eid_by_name("iterate")
    prop = _api.assemble_seq(it, [[direction_eid], [layer_eid], [op_eid]])
    return judge_seq(prop, True), True


def iterate_expression_samples():
    """迭代表达式等式样本 (tokenizer 原生): 迭代(方向,层数,被迭代) = 高阶算符.

    沿 A 层 iterate 箭头 (succ→addition 等), 生成标准迭代语法等式判定:
      [is_true][iterate][方向][层数][被迭代算符][equals][高阶算符][truth]
    例: [is_true][increase][iterate][digit_one][succ][equals][addition][truth]
      = 迭代(正, 1层, succ) = addition (succ 迭代 1 阶 = 加法)
    inverse 对偶箭头 (power→root 等) 生成对称降阶等式:
      [is_true][decrease][iterate][digit_one][power][equals][root][truth]
      = 迭代(负, 1层, power) = root (幂的层对偶 = 开方, 降阶)
    方向: 升阶 iterate 箭头 → increase; inverse 对偶 → decrease.
    零名字/eid 硬编码 (沿 A 层 + 结构).
    """
    from tokenizer import api as _api
    from ..role import role_token
    samples = []
    inc = _api.eid_by_name("increase")
    dec = _api.eid_by_name("decrease")
    one = _api.value_token(1)
    it = _api.eid_by_name("iterate")
    eq = role_token("equals")
    pred = _api.eid_by_name("pred")
    seen = set()
    for e, td in ARROW_REGISTRY.items():
        cname = _api.name(td.concept)
        src, tgt = td.source, td.target
        key = (cname, src, tgt)
        if cname == "iterate":
            direction = inc
        elif cname == "inverse":
            direction = dec
        else:
            continue
        if key in seen:
            continue
        seen.add(key)
        iter_expr = _api.assemble_seq(it, [[direction], [one], [src]])
        prop = _api.assemble_seq(eq, [iter_expr, [tgt]])
        samples.append({"seq": judge_seq(prop, True), "valid": 1, "truth": True, "depth": 1})
    return samples



__all__ = ["judge_seq", "definition_truth", "arrow_truth",
           "sample_definition", "sample_arrow",
           "logic_candidates", "digit_candidates", "value_token", "digit_tokens",
           "arrow_endpoints", "op_domain",
           "iteration_arrow_samples", "sample_iterate_expression",
           "iterate_expression_samples"]
