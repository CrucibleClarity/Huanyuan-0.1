"""lab/synth_core.py —— 通用样本合成器 (纯函数, 无实验设计默认)

职责 (用户确立): 接受实验参数, 确定样本集设计 (枚举范围/错题比例/展开深度).
单个样本如何产生/展开/真值/候选域 = tokenizer 原生 (api.sample_definition/sample_arrow 等).

设计原则:
  - 合成器是纯函数: 所有实验设计参数 (hi/max_depth/op/neg_mode/sample_mode) 显式必传,
    由配置层 (lab/run_exp 的 JSON) 决定, 合成器不拥有任何设计默认。
  - 零硬编码 token: 算子/数字/括号全部经 tokenizer.api 检索, 真值从 token 定义/arrow 来。
  - grid (全组合枚举) 是唯一可归因样本模式; random 禁止 (无法指导实验设计)。

能力:
  digits_of(n)       数字 → 数位符号序列 (负数防护)
  nested_seq(terms, op, result)  交替括号嵌套: ((a op b) op c) op d = result
  balanced_samples(max_depth, hi, op, neg_mode, sample_mode, seed)
                      真假平衡 + 全假值/随机负例 + 等额真样本
  definition_samples(op, hi, neg_mode, ...)  定义样本集设计 (沿 tokenizer 定义原语)
  arrow_samples(concept, ...)                 arrow 样本集设计 (沿 tokenizer arrow 原语)
  compose_samples(samples, seed)              按配置组合多种样本类型
"""
from __future__ import annotations


from tokenizer import api
from lab.judge import judge_sequence


def make_sample(seq, truth, depth=1, **kw):
    """统一样本构造 (tokenizer 原生): seq 沿 gtoken/ptoken 组装 (judge_seq/assemble_seq),
    样本字段 (valid/truth/depth) 统一由本原语构造, 合成器禁手写 dict (零硬编码)."""
    return {"seq": seq, "valid": 1, "truth": truth, "depth": depth, **kw}


def _assemble_logic(op, args, notation="prefix"):
    """逻辑表达式统一组装 (prefix/infix 可切换, 用户确立 2026-08-11).

    prefix: [op][arg0][arg1] — 一阶逻辑前缀 (合成器默认, 全体样本统一).
    infix:  [arg0][op][arg1] — 沿 ptoken 中缀 (imply ptoken [arg:0,→,arg:1]).
    参数可为 token 或子序列 (math 命题嵌套). 语法置换实验教训 (EXP-10):
    模型学会 op 位置 vs 对象位置; 前缀/infix 混用导致 imply 位置预测 base
    (拒绝语法错位). 全部样本须统一 notation.
    """
    if notation == "prefix":
        out = [op]
        for a in args:
            out.extend(a if isinstance(a, list) else [a])
        return out
    if notation == "infix":
        return api.assemble_seq(op, [list(a) if isinstance(a, list) else [a] for a in args])
    raise ValueError(f"未知 notation={notation}: 必须 prefix 或 infix")


def digit_eid(n):
    """数符符号 n → 数符概念 eid。"""
    return api.derives_of(str(n))[0]


def digits_of(n):
    """n 的十进制数位序列 (低→高). 负数 = [neg] + 绝对值数位 (用户确立: 能表示负数).

    neg 经 engine 结构识别 (一元数字算子), 零名字硬编码.
    """
    if n < 0:
        from tokenizer.eval.engine import _neg_eid
        return [_neg_eid()] + digits_of(-n)
    if n == 0:
        return [digit_eid(0)]
    out = []
    while n:
        n, d = divmod(n, 10)
        out.append(digit_eid(d))
    return out


def truth_from_definition(op, arg_tokens):
    """真值从 token 定义还原 (用户铁律: 真值必须由 token 体系提供)。

    op: 算子 eid; arg_tokens: 操作数 token 序列 (digit eid 或 truth eid)。
    沿 definition.rules 匹配 [equals, [op, arg0, arg1...], result] 行,
    返回 result == truth_true。无匹配返回 None (定义未覆盖)。

    零硬编码: 不查算子名, 不写 Python 真值逻辑 — 真值全在 token 定义。
    """
    from tokenizer.maintain import core
    _TRUE = api.role_token("truth")[0]
    d = core.load_all()[op].get("definition") or {}
    for rule in d.get("rules", []) or []:
        term = rule.get("term", [])
        if not (isinstance(term, list) and len(term) == 3):
            continue
        app, result = term[1], term[2]
        if not (isinstance(app, list) and app and app[0] == op):
            continue
        if list(app[1:]) == list(arg_tokens):
            return (result == _TRUE)
    return None


def eval_op(op, a, b, *, max_val=None, base=10):
    """算子求值 (真值由 token 定义还原, 通用求值引擎, 零 Python 算术硬编码).

    经 tokenizer.api.eval_op (通用求值器): 操作数数值 → numeral token 序列
    (numeral 快路径), 求值 (沿定义 rules), 结果 numeral → 数值. 内部数值
    计算唯一允许处 = numeral 快路径; 真值沿 token 定义推导, 零硬编码算子名.
    max_val: 可表示值域上限 (由配置/OOD 位数决定, 禁硬编码; 超运算结果
    超出 token 可表示位数 → ValueError 跳过). None=不限 (训练合成受 hi 控制).
    base: 进制 (数位分解/求值基数, 默认 10).
    """
    _precheck_overflow(op, a, b, max_val)
    arg_toks = [numeral_of(a, base=base), numeral_of(b, base=base)]
    out = api.eval_op(op, arg_toks, base=base)
    result = _numeral_value_of(out, base)
    if max_val is not None and abs(result) > max_val:
        raise ValueError("结果溢出 (超出可表示位数范围)")
    return result


def _precheck_overflow(op, a, b, max_val):
    """超运算值域预判 (迭代语义, 零算子名硬编码): 迭代次数/中间值上界.

    沿引擎迭代语义估计: 层 ≥3 算符 (迭代基础链) 的迭代次数 b 过大致
    结果超 max_val → 提前 ValueError (不经迭代, 防大数慢计算).
    """
    if max_val is None:
        return
    from tokenizer.eval.engine import _iteration_depth, _is_unary
    if _is_unary(op):
        return
    depth = _iteration_depth(op)
    if depth < 3:
        return
    # 递归塔 (tetration 等): a↑↑b 天文数字 — a=2 时 2↑↑4=65536 可行,
    # 2↑↑5 起超出; a≥3 时 a↑↑2 起爆炸 (指数增长)
    from tokenizer.eval.engine import _tower_rule
    if _tower_rule(op) is not None:
        if a > 2 and b > 1:
            raise ValueError("结果溢出 (塔结构爆炸, 超出可表示位数范围)")
        if a == 2 and b > 4:
            raise ValueError("结果溢出 (塔结构爆炸, 超出可表示位数范围)")
    # 迭代 b 次: 结果位数 ≥ b (底数>1 时指数增长, 至少线性累加位数)
    if b > 1 and (b * (a.bit_length() or 1)) > max_val.bit_length():
        raise ValueError("结果溢出 (超出可表示位数范围)")
    # 底数大 + 迭代次数中等: 位数爆炸预判
    if a > 1 and b > 1 and (b.bit_length() > max_val.bit_length()):
        raise ValueError("结果溢出 (超出可表示位数范围)")


def _numeral_value_of(out, base=10):
    """通用求值器输出 (numeral token 序列, 含 neg 前缀) → 数值.

    沿 digit 排序还原位序 (numeral 快路径); neg 前缀 = 负值.
    base: 求值基数 (与生成进制一致).
    """
    from tokenizer.eval.engine import _neg_eid
    seq = list(out)
    if seq and seq[0] == _neg_eid():
        return -api.eval_numeral(seq[1:], base)
    return api.eval_numeral(seq, base)


def eval_rel(op, a, b):
    """关系求值 (真值由 tokenizer 通用求值器提供, 零硬编码)。

    操作数: 逻辑算子=bool → eval_logic (definition.rules 真值表);
            比较算子=数字 → eval_compare (definition.rules 枚举真值表)。
    """
    domain = _op_domain(op)
    if domain == "numeric":
        return api.eval_compare(op, a, b)
    return api.eval_logic(op, [a, b])


_numeral_cache: dict = {}


def numeral_of(n, sign_eid=None, base=10):
    """numeral 表示 (tokenizer 原生构造, 沿 gtoken 组装, 零硬编码).

    委托 api.numeral_of: 整数 n → digit 构造实例序列 (value_token) → 沿
    numeral_expr/sign_part/digit_seq/digit_expr/place_expr 组装.
    位序由 digit 排序还原, 正确路径 (从基点迭代) 见 tokenizer.api.iterate_from_base.
    base: 进制 (数位分解基数, 默认 10). 结果缓存 (同数字重复用于 OOD 大批量生成)."""
    key = (n, sign_eid, base)
    hit = _numeral_cache.get(key)
    if hit is not None:
        return list(hit)
    # 数位分解: n → 高→低 digit 构造实例序列 (沿 base 进制)
    if n == 0:
        dg_spec = [api.value_token(0)]
    else:
        ds = []
        m = abs(n)
        while m:
            m, d = divmod(m, base)
            ds.append(d)
        ds.reverse()
        dg_spec = [api.value_token(d) for d in ds]
    sign = sign_eid or _sign_for(n)
    out = api.numeral_of(dg_spec, sign_eid=sign)
    if len(_numeral_cache) < 10000:
        _numeral_cache[key] = tuple(out)
    return out


def _sign_for(n):
    """数值 n 的符号概念 (沿 A 层 positive/negative 箭头, 零名字).

    正/零 → sign_pos; 负 → sign_neg (用户诊断: 负数表示缺 sign_neg 前缀,
    导致减法负结果样本生成错误). 沿 arrow concept=positive/negative 定位.
    """
    from tokenizer._register import ARROW_REGISTRY
    if n >= 0:
        for td in ARROW_REGISTRY.values():
            if getattr(td, "concept", "") and _name_of(td.concept) == "positive":
                return td.target
    else:
        for td in ARROW_REGISTRY.values():
            if getattr(td, "concept", "") and _name_of(td.concept) == "negative":
                return td.target
    return api.role_token("sign")


def _name_of(eid):
    from tokenizer import api as _a
    try:
        return _a.name(eid)
    except (KeyError, TypeError):
        return ""


def nested_seq(terms, op, result, base=10):
    """嵌套表达式 (沿 gtoken/ptoken 组装, 零硬编码列表): 每层 (expr op t) 由
    op 的 ptoken (如 add_infix [+]) 组装, 括号由 grouping_parens (grouping gtoken)
    包裹, 结果由 equality ptoken (=) 组装. 操作数/结果用 numeral 表示 (两向量折叠).
    base: 数位构造进制 (默认 10).
    所有样本类型共用同一组装逻辑.
    """
    expr = numeral_of(terms[0], base=base)
    for t in terms[1:]:
        expr = api.assemble_seq(op, [expr, numeral_of(t, base=base)])
        expr = api.assemble_seq(api.role_token("bracket"), [expr])
    return api.assemble_seq(api.role_token("equals"), [expr, numeral_of(result, base=base)])


def logic_samples(*, op, seed=0, notation="prefix"):
    """逻辑算子真值表样本 — 沿 tokenizer 定义还原 (非合成器硬编码).

    逻辑算子 (and/or/not/imply...) 的 definition.rules 每条 = 一行真值表:
      {"term": ["D:260", ["D:280", "D:290", "D:290"], "D:290"]}
      = equality( and(truth_true, truth_true), truth_true ) → 判定为真
    逐行抽取 → 判定序列 [is_true][op][T][F...][result]。真值由 rule 结构直接给出。
    notation: prefix=[op][args] / infix=[arg0][op][arg1] (统一语法, 用户确立).

    返回 (样本列表, 真样本数, 假样本数)。
    """
    from tokenizer.maintain import core
    _EQ = api.role_token("equals")
    _TRUTH = api.role_token("truth")
    _TRUE = _TRUTH[0]
    _FALSE = _TRUTH[1]
    if not str(op).startswith("D:"):
        op = api.eid_by_name(op)
    d = core.load_all()[op]["definition"]
    samples = []
    npos = nneg = 0
    for rule in d.get("rules", []) or []:
        term = rule.get("term", [])
        # term = [equals, [op, arg0, arg1...], result]
        if not (isinstance(term, list) and len(term) == 3):
            continue
        app, result = term[1], term[2]
        if not (isinstance(app, list) and len(app) >= 2 and app[0] == op):
            continue
        args = app[1:]
        prop = _assemble_logic(op, list(args), notation)
        truth = (result == _TRUE)
        seq = judge_sequence(prop, truth)
        samples.append(make_sample(seq, truth, 1))
        if truth:
            npos += 1
        else:
            nneg += 1
    return samples, npos, nneg


def logic_nested_samples(*, ops, max_depth=3, seed=0, notation="prefix"):
    """逻辑门深嵌套样本 (用户确立): 多门组合 + 深嵌套判定.

    形式: [is_true][logical_and][T][[logical_or][F][T]][truth] 嵌套组合.
    每门真值表沿 logic_samples (真值定义还原), 嵌套深度 2-3 组合生成.
    量级与 balanced 相当 (多门 × 多组合), 避免被算术样本淹没.
    notation: prefix=[op][args] / infix (统一语法).
    返回 (样本列表, 正例数, 负例数).
    """
    import random
    from itertools import product
    rng = random.Random(seed)
    samples, npos, nneg = [], 0, 0
    _TRUTH = api.role_token("truth")
    _TRUE = _TRUTH[0]
    _FALSE = _TRUTH[1]
    # 每门真值表 (eid → rows: (args_token_list, truth))
    tables = {}
    for op_name in ops:
        e = api.eid_by_name(op_name) if not str(op_name).startswith("D:") else op_name
        rows = []
        for rule in (__import__("tokenizer.maintain.core", fromlist=["load_all"])
                     .load_all()[e]["definition"].get("rules") or []):
            term = rule.get("term", [])
            if not (isinstance(term, list) and len(term) == 3): continue
            app, result = term[1], term[2]
            if not (isinstance(app, list) and app and app[0] == e): continue
            args = [a for a in app[1:]]
            truth = (result == _TRUE)
            rows.append((e, args, truth))
        tables[op_name] = rows
    # 深嵌套: 外层 op (真值沿定义表), 内层 op 先求值
    for outer in ops:
        outer_e = api.eid_by_name(outer) if not outer.startswith("D:") else outer
        inner_ops = [o for o in ops if o != outer]
        for inner in inner_ops:
            inner_e = api.eid_by_name(inner) if not inner.startswith("D:") else inner
            for _ in range(8):
                # 内层真值表行 (参数 token + 真值)
                inner_row = rng.choice(tables[inner])
                inner_args, inner_truth = inner_row[1], inner_row[2]
                inner_tok = _TRUE if inner_truth else _FALSE
                # 外层另一参数随机 bool
                a2 = _TRUE if rng.random() < 0.5 else _FALSE
                # 外层真值: 沿定义表匹配 [outer_e, inner_tok, a2]
                outer_truth = None
                for row in tables[outer]:
                    if row[1] == [inner_tok, a2]:
                        outer_truth = row[2]
                        break
                if outer_truth is None:
                    continue
                # 嵌套 prop: 内层结果 bool token 作外层参数 (深嵌套判定,
                # 与 logic_samples 同形式 [is_true][outer][inner][a2][truth])
                seq = judge_sequence(_assemble_logic(outer_e, [inner_tok, a2], notation), outer_truth)
                samples.append(make_sample(seq, outer_truth, 2))
                if outer_truth:
                    npos += 1
                else:
                    nneg += 1
    return samples, npos, nneg


def logic_structural_samples(*, ops, max_depth=3, n=200, seed=0):
    """完整嵌套逻辑样本 (Level 2 匿名程序基础): 内层门名可见的结构组合.

    与 logic_nested 的区别: 内层是完整子表达式 [and][A][B] (门名可见),
    非化简 bool token. 模型须从语法解析嵌套结构再求值 — 这是
    formation+composition+evaluation 的真正测试 (logic gate 不是独立类别).
    形式: [is_true][outer][inner_expr1][inner_expr2][truth] 递归生成.
    真值沿 eval_bool_expr (token 定义还原, 零硬编码). 返回 (样本, 正, 负).
    """
    import random
    from itertools import product
    from tokenizer.eval.logic_eval import eval_bool_expr
    rng = random.Random(seed)
    samples, npos, nneg = [], 0, 0
    _TRUTH = api.role_token("truth")
    _TRUE = _TRUTH[0]
    _FALSE = _TRUTH[1]
    gate_eids = [api.eid_by_name(o) if not str(o).startswith("D:") else o for o in ops]

    def _expr(depth):
        """随机生成深度 depth 的完整嵌套表达式树 (dict 供 eval_bool_expr)."""
        if depth <= 1:
            return rng.random() < 0.5
        op = rng.choice(gate_eids)
        arity = 1 if api.arrange_of(op) == "unary_connective" else 2
        args = [_expr(depth - 1) if rng.random() < 0.6 else (rng.random() < 0.5)
                for _ in range(arity)]
        return {"op": op, "args": args}

    def _assemble(node, depth):
        """表达式树 → token 序列 (完整嵌套, 内层门名可见)."""
        if not isinstance(node, dict):
            return [_TRUE if node else _FALSE]
        op = node["op"]
        kids = [_assemble(a, depth - 1) if isinstance(a, dict) else
                [_TRUE if a else _FALSE] for a in node["args"]]
        return api.assemble_seq(op, kids)

    for _ in range(n):
        d = rng.randint(2, max_depth)
        node = _expr(d)
        truth = eval_bool_expr(node) if isinstance(node, dict) else node
        prop = _assemble(node, d)
        seq = judge_sequence(prop, truth)
        samples.append(make_sample(seq, truth, d))
        if truth:
            npos += 1
        else:
            nneg += 1
    return samples, npos, nneg


def logic_interdef_samples(*, ops=None, seed=0, imply_judge=True,
                           imply_contrapos=True, imply_proppos=True, imply_truthrow=False,
                           notation="prefix", imply_boost=1):
    """逻辑门互译定义样本 (用户确立): 沿 A 层 inverse 对偶箭头生成对称性判定.

    逻辑门的对称性来源: nand=not(and), nor=not(or), xnor=not(xor) —
    沿 A 层 inverse 箭头 (and↔or, nand↔or, nor↔and, xnor↔xor) 生成
    互译判定: [is_true][and][T][F][=][nor][T][F][truth] 等.
    让模型从对称性学逻辑门 (而非平铺真值表), 与迭代箭头同构.
    imply 特例: 只给 Babel 定义 (imply(A,B)=or(¬A,B) 恒真
    等式 + 逆否), 不给 imply 真值判定监督. imply_judge=True 时额外生成
    深嵌套判定 (imply 结果作外层参数); imply_judge=False 只留定义 (测判定监督必要性).
    EXP-11 删律对照: imply_contrapos=False 删逆否律 / imply_proppos=False 删
    prop 位置定义 / imply_truthrow=True 额外加 1 行 F→F 真值表监督.
    notation: prefix/infix 统一语法 (用户确立, 全部样本须一致).
    返回 (样本列表, 正例数, 负例数).
    """
    from tokenizer._register import ARROW_REGISTRY
    from itertools import product
    samples, npos, nneg = [], 0, 0
    _TRUTH = api.role_token("truth")
    _TRUE = _TRUTH[0]
    _FALSE = _TRUTH[1]
    if ops is None:
        ops = ["logical_and","logical_or","logical_nand","logical_nor","logical_xor","logical_xnor"]
    # 每门真值表 (bool args → truth)
    tables = {}
    for op_name in ops:
        e = api.eid_by_name(op_name) if not op_name.startswith("D:") else op_name
        tb = {}
        for rule in (__import__("tokenizer.maintain.core", fromlist=["load_all"])
                     .load_all()[e]["definition"].get("rules") or []):
            term = rule.get("term", [])
            if not (isinstance(term, list) and len(term) == 3): continue
            app, result = term[1], term[2]
            if not (isinstance(app, list) and app and app[0] == e): continue
            args = [a == _TRUE for a in app[1:]]
            tb[tuple(args)] = (result == _TRUE)
        tables[op_name] = tb
    # 互译: 沿 inverse 箭头对, 验证对偶门真值互补 (De Morgan)
    seen = set()
    for e, td in ARROW_REGISTRY.items():
        if api.name(td.concept) != "inverse":
            continue
        a_name, b_name = api.name(td.source), api.name(td.target)
        a, b = td.source, td.target
        if not (a_name in tables and b_name in tables):
            continue
        if (a, b) in seen:
            continue
        seen.add((a, b))
        # 对偶门: a 与 b 应互补 (a(T,F) 与 b(not T, not F) 相反?)
        # 沿真值表生成互译判定样本
        arity = 1 if api.arrange_of(a) == "unary_connective" else 2
        for args in product([True, False], repeat=arity):
            ta = tables[a_name].get(args)
            tb = tables[b_name].get(args)
            if ta is None or tb is None:
                continue
            # 判定: [is_true][a][args][=][b][inv_args][truth] (De Morgan 对偶)
            inv_args = tuple(not x for x in args)
            tb_inv = tables[b_name].get(inv_args)
            if tb_inv is None:
                continue
            truth = (tb_inv == (not ta))
            prop = _assemble_logic(a, [_TRUE if x else _FALSE for x in args], notation)
            prop_b = _assemble_logic(b, [_TRUE if not x else _FALSE for x in inv_args], notation)
            seq = judge_sequence(prop + [api.role_token("equals")] + prop_b, truth)
            samples.append(make_sample(seq, truth, 2))
            if truth:
                npos += 1
            else:
                nneg += 1
        # 特殊: not 自对合 (not(not A) = A)
        if a_name == "logical_not" and b_name == "logical_not":
            for a_val in (True, False):
                prop = _assemble_logic(a, [_TRUE if a_val else _FALSE], notation)
                seq = judge_sequence(prop + [api.role_token("equals")] + prop, True)
                samples.append(make_sample(seq, True, 2))
                npos += 1
    # imply 定义用中缀语法 (ptoken [arg:0,→,arg:1]) — 用户洞察: 模型学会了
    # op 位置 vs 对象位置, imply 应出现在中间符号位置, 前缀 [imply][A][B] 是
    # 语法错位 (模型在 op 位置预测 base 而非 imply = 拒绝语法错位).
    _IMPLY = api.eid_by_name("logical_imply")
    _OR = api.eid_by_name("logical_or")
    _NOT = api.eid_by_name("logical_not")
    _imply_in_ops = any(o == "logical_imply" for o in (ops or []))
    if _imply_in_ops:
        for _b in range(max(1, imply_boost)):
            for a_val, b_val in product([True, False], repeat=2):
                A = _TRUE if a_val else _FALSE
                B = _TRUE if b_val else _FALSE
                nA = _FALSE if a_val else _TRUE
                nB = _FALSE if b_val else _TRUE
                # 逆否: imply(A,B) = imply(¬B,¬A)  (重言, truth=True)
                if imply_contrapos:
                    prop_cp = _assemble_logic(_IMPLY, [nB, nA], notation)
                    seq_cp = judge_sequence(
                        _assemble_logic(_IMPLY, [A, B], notation) + [api.role_token("equals")] + prop_cp, True)
                    samples.append(make_sample(seq_cp, True, 2))
                    npos += 1
                # 材料蕴含: imply(A,B) = or(¬A, B)  (重言, truth=True)
                prop_or = _assemble_logic(_OR, [_assemble_logic(_NOT, [A], notation), B], notation)
                seq_or = judge_sequence(
                    _assemble_logic(_IMPLY, [A, B], notation) + [api.role_token("equals")] + prop_or, True)
                samples.append(make_sample(seq_or, True, 2))
                npos += 1
        # 额外 1 行 F→F 真值表监督
        if imply_truthrow:
            seq_ff = judge_sequence(_assemble_logic(_IMPLY, [_FALSE, _FALSE], notation), True)
            samples.append(make_sample(seq_ff, True, 1))
            npos += 1
    # imply 深嵌套 (用户洞察: 逐层拆嵌套 — 操作数本身可为复合判定)
    # imply(A, imply(B,C)) / imply(imply(A,B), C) 等: 内层 imply 结果 bool token
    # 作外层操作数 (扁平形式, 与 logic_nested 同构), 让模型看到非对称门在
    # 各位置 (前件/后件) 逐层拆解 — 假判定可出现在任意嵌套深度, 结果独立判定.
    # imply_judge=False 时跳过: 深嵌套是 imply 真值判定监督,
    # 只给 Babel 定义 (imply(A,B)=or(¬A,B) 等式), 不给 imply 判定.
    if imply_judge and _imply_in_ops:
        from tokenizer.eval.logic_eval import logic_truth
        for a_val, b_val, c_val in product([True, False], repeat=3):
            A = _TRUE if a_val else _FALSE
            B = _TRUE if b_val else _FALSE
            C = _TRUE if c_val else _FALSE
            def _tok(b):
                return _TRUE if b else _FALSE
            # 内层 imply(B,C) 求值
            bc = logic_truth(_IMPLY, [B, C])
            if bc is not None:
                # 外层: imply(A, imply(B,C)) — 前件为 A, 后件为内层结果
                ab = logic_truth(_IMPLY, [A, _tok(bc)])
                if ab is not None:
                    seq = judge_sequence(_assemble_logic(_IMPLY, [A, _tok(bc)], notation), ab)
                    samples.append(make_sample(seq, ab, 3))
                    if ab: npos += 1
                    else: nneg += 1
                # 外层: imply(imply(A,B), C) — 前件为内层结果, 后件为 C
                ia = logic_truth(_IMPLY, [A, B])
                if ia is not None:
                    oc = logic_truth(_IMPLY, [_tok(ia), C])
                    if oc is not None:
                        seq = judge_sequence(_assemble_logic(_IMPLY, [_tok(ia), C], notation), oc)
                        samples.append(make_sample(seq, oc, 3))
                        if oc: npos += 1
                        else: nneg += 1
    # imply 定义的位置覆盖 (imply(A,B)=or(¬A,B) 定义用
    # math 命题作操作数 — 让模型从定义学到 imply 可出现在 prop1 末尾 (长边界),
    # 同时无 imply 真值表监督 (定义恒真 truth=True, 模型只学等价结构).
    # imply_proppos=False (EXP-11b) 跳过 — 测 prop 位置定义必要性.
    if imply_proppos and any(o == "logical_imply" for o in (ops or [])):
        _OR = api.eid_by_name("logical_or")
        _NOT = api.eid_by_name("logical_not")
        from lab.synth_core import balanced_samples as _bs
        arith = []
        for op_name, eid in [("addition", api.eid_by_name("addition")),
                             ("subtraction", api.eid_by_name("subtraction"))]:
            ss_, _, _ = _bs(max_depth=2, hi=9, op=eid, neg_mode=1)
            for s_ in ss_:
                seq_ = s_["seq"]
                prop_ = seq_[1:-1] if seq_ and seq_[0] == api.eid_by_name("is_true") else seq_
                arith.append(prop_)
        import random as _r
        rng = _r.Random((seed or 0) + 999)
        if arith:
            for _ in range(8 * max(1, imply_boost)):
                p_a = rng.choice(arith)
                p_b = rng.choice(arith)
                # 定义: imply(p_a, p_b) = or(not p_a, p_b) — 恒真
                prop_l = _assemble_logic(_IMPLY, [p_a, p_b], notation)
                nA = _assemble_logic(_NOT, [p_a], notation)
                prop_r = _assemble_logic(_OR, [nA, p_b], notation)
                seq_d = judge_sequence(prop_l + [api.role_token("equals")] + prop_r, True)
                samples.append(make_sample(seq_d, True, 2))
                npos += 1
    return samples, npos, nneg


def _outer_truth(op_eid, args, _TRUE, _FALSE):
    """外层逻辑门真值 (沿定义真值表匹配, 零名字)."""
    from tokenizer.maintain import core
    from tokenizer.eval.logic_eval import logic_truth
    flat = []
    for a in args:
        if isinstance(a, list):
            # 内层已求值为 bool token
            flat.append(_TRUE if a else _FALSE)
        else:
            flat.append(a)
    t = logic_truth(op_eid, flat)
    return t if t is not None else False


def logic_arith_samples(*, ops, hi=9, seed=0, notation="prefix"):
    """逻辑门 × 数学命题样本 (用户确立): 逻辑门套用数学已知真假的命题.

    数学命题 (balanced 算术判定: 3+4=7 真 / 3+4=8 假) 作为逻辑门操作数:
      [is_true][logical_and][(3+4=7)][(5-2=3)][truth_true]  两个真 → 真
      [is_true][logical_or][(3+4=8)][(5-2=3)][truth_true]   一假一真 → 真
    逻辑门从数学命题学习真假判断 (非裸 bool), 深嵌套多题型.
    notation: prefix/infix 统一语法 (用户确立).
    返回 (样本列表, 正例数, 负例数).
    """
    from itertools import product
    import random
    rng = random.Random(seed)
    samples, npos, nneg = [], 0, 0
    _TRUTH = api.role_token("truth")
    _TRUE = _TRUTH[0]
    _FALSE = _TRUTH[1]
    # 收集数学命题 (balanced 算术): (prop_seq, truth)
    arith_props = []
    for op_name, eid in [("addition", api.eid_by_name("addition")),
                         ("subtraction", api.eid_by_name("subtraction")),
                         ("multiplication", api.eid_by_name("multiplication"))]:
        ss, _, _ = balanced_samples(max_depth=2, hi=hi, op=eid, neg_mode=1)
        for s in ss:
            # 命题 = 去 is_true 和 truth 的中间部分
            seq = s["seq"]
            prop = seq[1:-1] if seq and seq[0] == api.eid_by_name("is_true") else seq
            arith_props.append((prop, s["truth"]))
    # 逻辑门 × 命题组合
    for op_name in ops:
        e = api.eid_by_name(op_name) if not op_name.startswith("D:") else op_name
        n_arity = 1 if api.arrange_of(e) == "unary_connective" else 2
        # imply 特例 (用户洞察): 假的结果 imply 后也可判真 — 显式枚举 4 行
        # 真值表 (TT/TF/FT/FF), 每行用多组数学命题 (含假命题) 配对, 保证反直觉行
        # [F][T]→T / [F][F]→T 覆盖. 模型须认知: 两操作数位置独立, 任一位置可为假,
        # 结果判定只看 前件真且后件假 → 假, 其余全真.
        if op_name == "logical_imply":
            true_props = [p for p, t in arith_props if t]
            false_props = [p for p, t in arith_props if not t]
            rows = [(True, True), (True, False), (False, True), (False, False)]
            for _ in range(8):
                for a_t, b_t in rows:
                    pool_a = true_props if a_t else false_props
                    pool_b = true_props if b_t else false_props
                    if not pool_a or not pool_b:
                        continue
                    p_a = rng.choice(pool_a)
                    p_b = rng.choice(pool_b)
                    prop = _assemble_logic(e, [p_a, p_b], notation)
                    # 真值: imply(T,T)=T, imply(T,F)=F, imply(F,T)=T, imply(F,F)=T
                    truth = not (a_t and not b_t)
                    seq = judge_sequence(prop, truth)
                    samples.append(make_sample(seq, truth, 2))
                    if truth:
                        npos += 1
                    else:
                        nneg += 1
            continue
        for _ in range(12):
            # 随机选 n_arity 个命题
            chosen = rng.sample(arith_props, min(n_arity, len(arith_props)))
            prop_truths = [t for _, t in chosen]
            # 逻辑门真值 (沿定义表)
            arg_toks = [_TRUE if t else _FALSE for t in prop_truths]
            truth = None
            for rule in (__import__("tokenizer.maintain.core", fromlist=["load_all"])
                         .load_all()[e]["definition"].get("rules") or []):
                term = rule.get("term", [])
                if not (isinstance(term, list) and len(term) == 3): continue
                app, result = term[1], term[2]
                if app and app[0] == e and list(app[1:]) == arg_toks:
                    truth = (result == _TRUE)
                    break
            if truth is None:
                continue
            # 判定: [is_true][op][prop1][prop2][truth] — 命题序列作子项直接插入
            # (命题已展平, 经 assemble_seq 的 arg:N 槽位插入, 不递归 AST)
            prop = _assemble_logic(e, [list(p) for p, _ in chosen], notation)
            seq = judge_sequence(prop, truth)
            samples.append(make_sample(seq, truth, 2))
            if truth:
                npos += 1
            else:
                nneg += 1
    return samples, npos, nneg




def balanced_samples(*, max_depth, hi, op, neg_mode="all", sample_mode="grid", seed=0):
    """真假平衡样本 (纯确定性, 零随机, 全枚举可归因)。

    参数 (全为实验设计, 由配置层决定; 无样本数量参数 — 样本量由 hi×max_depth 全枚举决定):
      max_depth  嵌套深度上限
      hi         数字范围 [0, hi]
      op         二元算子 eid (addition/subtraction/multiplication)
      neg_mode   "all"=全假值覆盖 (每组合全部错误答案) / int k=每组合 k 个负例 (确定性前 k 个)
      sample_mode "grid"=全组合枚举 (唯一可归因) — random 禁止
      seed       保留兼容参数 (确定性枚举不使用, 但保留签名)
    返回 (样本列表, 真样本数, 假样本数)。

    平衡策略 (确定性, 无随机补样本):
      真样本 = 全组合枚举 (每组合 1 真); 假样本 = 全组合枚举 × 全错误答案。
      真/假天然 1:hi*d; 若需 1:1 平衡, 由 neg_mode 控制假样本配比 (确定性前 k 个),
      或由 run_exp 在配置层决定 (真样本枚举空间 vs 假样本覆盖范围)。
      subtraction: 可负 (neg token 表达, 用户确立: 能表示负数, 不禁负).
      除法: 跳过除数为 0 (数学约束).
    """
    if sample_mode != "grid":
        raise ValueError(f"禁止 sample_mode={sample_mode}: 必须用 grid (全组合枚举, 可归因)。"
                         "random 样本无法指导实验设计 (不知道哪个 token 序列提升结果)")
    is_div = op == api.eid_by_name("division")
    pos, neg = [], []

    for d in range(2, max_depth + 1):   # 二元运算起 (d=1 单操作数无二元运算, "a op a = a" 假标真 bug 修复)
        combos = [list(x) for x in __import__("itertools").product(range(hi + 1), repeat=d)]
        for terms in combos:
            # 除法: 跳过除数为 0 (0 不可除, value_0 约束)
            if is_div and any(t == 0 for t in terms[1:]):
                continue
            try:
                total = terms[0]
                for t in terms[1:]:
                    total = eval_op(op, total, t)
            except ValueError:
                continue  # 超运算不可求值组合 (无理根/溢出), 跳过 (训练/测试一致)
            pos.append(make_sample(judge_sequence(nested_seq(terms, op, total), True), True, d))
            if neg_mode == "all":
                for bad in range(0, hi * d + 1):
                    if bad != total:
                        neg.append(make_sample(judge_sequence(nested_seq(terms, op, bad), False), False, d))
            else:
                # 确定性前 k 个错误答案 (无随机)
                k = int(neg_mode)
                added = 0
                for bad in range(0, hi * d + 1):
                    if bad != total and added < k:
                        neg.append(make_sample(judge_sequence(nested_seq(terms, op, bad), False), False, d))
                        added += 1
    return pos + neg, len(pos), len(neg)


def dual_token_samples(*, op, dual_token, hi=5, dist="balanced", sample_mode="grid", seed=0):
    """共享操作 + 对偶分类 token 判定样本 (临时 token 注入)。

    序列格式: [is_true][a][rel_op][dual_token][b][truth_true|false]
    - rel_op 为所有比较算子共享的操作 token (临时注入, 名 rel_op)
    - dual_token 为对偶分类 token (区分算子: eq_d/neq_d/gt_d/lt_d 或共享 pol_*/dir_*)
    - 真值由 eval_rel(op, a, b) 计算 (op 决定语义, 但序列中用 dual_token 标识)

    用途: 对偶关系实验 — 不同对偶 token 搭配下, 观察模型能否收敛学习判定。
    对偶 token 定义在实验配置中, 经 tokenizer.token_index.inject_temp 临时注入
    (不污染主 token 数据); 本函数只消费 eid, 不硬编码定义。
    op: 语义算子 eid (equals_arith/not_equals/greater_than/less_than);
    dual_token: 序列中使用的对偶分类 token eid (实验变量)。
    """
    if sample_mode != "grid":
        raise ValueError(f"禁止 sample_mode={sample_mode}: 必须用 grid (全组合枚举, 可归因)。")
    from itertools import product
    _REL = api.eid_by_name("rel_op")
    operands = list(range(hi + 1))
    samples, npos, nneg = [], 0, 0
    for a, b in product(operands, repeat=2):
        truth = bool(eval_rel(op, a, b))
        args = digits_of(a) + [_REL, dual_token] + digits_of(b)
        seq = judge_sequence(args, truth)
        samples.append(make_sample(seq, truth, 1, op=api.name(op)))
        if truth:
            npos += 1
        else:
            nneg += 1
    pos = [s for s in samples if s["truth"]]
    neg = [s for s in samples if not s["truth"]]
    if dist == "balanced":
        k = min(len(pos), len(neg))
        samples = pos[:k] + neg[:k]
        npos = nneg = k
    elif dist == "all":
        samples = pos + neg
        npos, nneg = len(pos), len(neg)
    else:
        raise ValueError(f"未知 dist: {dist!r} (合法: balanced/all)")
    return samples, npos, nneg


def inject_dual_tokens(token_defs):
    """临时注入对偶 token (C 层), 实验开始前调用。

    token_defs: {token名: 定义 dict} 来自实验配置 (如 rel_op/eq_d 等的 definition),
    不硬编码定义 — 实验配置数据决定 token 形态。
    经 token_index.inject_temp 注入, 不写 jsonl; 实验结束 clear_cache 清除。
    返回 {token名: eid} 映射。
    """
    from tokenizer import token_index, _register, api as _api
    rows = []
    for i, (name, defn) in enumerate(token_defs.items()):
        rows.append({"eid": f"D:dual{i}", "name": name, "dtype": "bool", "definition": defn})
    token_index.inject_temp("C", rows)
    _register.load_derive()
    return {n: _api.eid_by_name(n) for n in token_defs}


def _op_domain(op):
    """算子域判定: "logical" (布尔操作数) / "numeric" (数字操作数) / None。

    域判定沿 token 定义结构 (engine op_meta), 零算子名硬编码:
      binary_connective/unary_connective → logical (真值表操作数)
      equality                    → numeric (数字比较)
      application                 → numeric (算术)
    不再维护写死的逻辑/比较算子名集合.
    """
    from tokenizer.eval.engine import op_meta as _opm
    arrange = _opm(op).get("arrange")
    if arrange in ("binary_connective", "unary_connective"):
        return "logical"
    if arrange in ("equality", "application"):
        return "numeric"
    return None


def rel_samples(*, op, hi=5, neg_mode="all", dist="balanced", sample_mode="grid", seed=0):
    """二元关系判定样本 (零结果项, 真值=关系成立与否)。

    格式: [is_true][a][op][b][truth_true|false]
    - 数字关系 (比较算子 =/≠/>/</≥/≤): 操作数全枚举 [0..hi]², 真值 = eval_rel。
    - 布尔关系 (逻辑门 and/or/not/imply/iff/xor/nand/nor/xnor): 操作数全枚举 bool²,
      真值 = eval_rel。not 为一元 (单操作数)。
    纯确定性全枚举 (grid), 零随机, 可归因。

    分布 (dist, 用户方法论: 平衡/偏真/偏假三分布训练):
      dist="balanced"  → 真:假 = 1:1 (截断假样本到等额真)
      dist="bias_true" → 真样本多 (假样本截断到真样本的 1/3, 真占 ~75%)
      dist="bias_false"→ 假样本多 (保留全部假, 真样本截断, 假占 ~75%)
      neg_mode 仅兼容参数 (dist 取代), 默认由 dist 决定。

    返回 (样本列表, 真样本数, 假样本数)。
    """
    if sample_mode != "grid":
        raise ValueError(f"禁止 sample_mode={sample_mode}: 必须用 grid (全组合枚举, 可归因)。")
    domain = _op_domain(op)
    if domain is None:
        raise ValueError(f"未支持关系算子: {api.name(op)} (非比较/逻辑算子)")
    from itertools import product
    samples = []
    npos = nneg = 0
    if domain == "numeric":
        operands = list(range(hi + 1))
    else:
        operands = [True, False]
    # 一元算子 (logical_not): 单操作数; 二元: 双操作数 (沿 engine 结构原语)
    from tokenizer.eval.engine import op_meta as _opm
    n_arity = 1 if _opm(op).get("is_unary") else 2
    for combo in product(operands, repeat=n_arity):
        truth = bool(eval_rel(op, *combo))
        if domain == "numeric":
            args = []
            for v in combo:
                args += digits_of(v)
        else:
            _TRUTH = api.role_token("truth")
            args = [_TRUTH[0] if v else _TRUTH[1]
                    for v in combo]
        prop = api.assemble_seq(op, [args]) if domain != "numeric" else api.assemble_seq(op, [args])
        seq = judge_sequence(prop, truth)
        if truth:
            samples.append(make_sample(seq, True, 1))
            npos += 1
        else:
            samples.append(make_sample(seq, False, 1))
            nneg += 1
    # 分布控制 (用户方法论: 平衡/偏真/偏假)
    pos = [s for s in samples if s["truth"]]
    neg = [s for s in samples if not s["truth"]]
    if dist == "balanced":
        k = min(len(pos), len(neg))
        samples = pos[:k] + neg[:k]
        npos = nneg = k
    elif dist == "bias_true":
        k = max(len(pos) // 3, 1)
        samples = pos + neg[:k]
        npos, nneg = len(pos), len(neg[:k])
    elif dist == "bias_false":
        k = max(len(neg) // 3, 1)
        samples = pos[:k] + neg
        npos, nneg = len(pos[:k]), len(neg)
    elif dist == "all":
        samples = pos + neg
        npos, nneg = len(pos), len(neg)
    else:
        raise ValueError(f"未知 dist: {dist!r} (合法: balanced/bias_true/bias_false/all)")
    return samples, npos, nneg


def definition_samples(*, op, hi=9, neg_mode="all", sample_mode="grid", seed=0):
    """定义样本集设计 (消费方): 在 tokenizer 候选域内选择样本.

    tokenizer 原生提供: digit_candidates (候选域) + sample_definition (单样本+真值).
    本函数只做样本集设计: 枚举操作数组合 (hi 范围), 错题比例 (neg_mode).
    真值完全来自 tokenizer 定义还原, 零硬编码.
    返回 (样本列表, 正例数, 负例数).
    """
    if sample_mode != "grid":
        raise ValueError(f"禁止 sample_mode={sample_mode}: 必须用 grid (全组合枚举, 可归因)")
    op_eid = api.eid_by_name(op)
    if api.op_domain(op_eid) == "logical":
        cands = api.logic_candidates()
        if api.eid_by_name("logical_not") == op_eid:
            pairs = [(a,) for a in cands]          # 一元: 单操作数
        else:
            pairs = [(a, b) for a in cands for b in cands]
    else:
        cands = api.digit_candidates()[:hi + 1]
        pairs = [(a, b) for a in cands for b in cands]
    samples, npos, nneg = [], 0, 0
    for combo in pairs:
        seq, truth = api.sample_definition(op_eid, list(combo))
        if seq is None:
            continue
        samples.append(make_sample(seq, truth, 1))
        if truth:
            npos += 1
        else:
            nneg += 1
    return samples, npos, nneg


_ARG = type("_ARG", (), {"__slots__": ("idx",)})


def _arg(idx):
    """绑定位哨兵 (模板编译: 占位, 实例化替换)."""
    a = _ARG.__new__(_ARG)
    a.idx = idx
    return a


def _compile_law(node, op_eid):
    """term 结构 → 模板 (CTE: 固定子结构一次编译, 绑定位用 _ARG 哨兵占位).

    与 _inst 同结构 (self→op, arg:N→绑定), 但 arg 位置用哨兵标记 —
    实例化只做 O(n) 复制+替换, 不走递归 assemble_seq (组合量不变).
    """
    if isinstance(node, str):
        if node == "self":
            return [op_eid]
        if node.startswith("arg:"):
            return [_arg(int(node[4:]))]
        return [node]
    if isinstance(node, list) and node:
        head = node[0]
        parts = [_compile_law(ch, op_eid) for ch in node[1:]]
        if isinstance(head, str) and (head == "self" or head.startswith("arg:")):
            fn = op_eid if head == "self" else _arg(int(head[4:]))
            return api.assemble_seq(fn, parts)
        return api.assemble_seq(head, parts)
    return [node]


def _instantiate_law_template(template, bindings):
    """模板实例化: 哨兵替换为绑定 token (O(n) 复制, 无递归)."""
    out = []
    for t in template:
        if isinstance(t, _ARG):
            out.append(bindings[t.idx])
        else:
            out.append(t)
    return out


_LAW_TEMPLATE_CACHE = {}


def _law_term_signature(term, op_eid):
    """term 数据签名 (校验模板是否失效): 基于全局数据版本 + term 结构.

    全局数据版本 (cte.DATA_VERSION) 在 maintain 写入时递增 — term/refs 变化
    由 invalidate 覆盖 (数据版本变 → 签名变 → 幂等重编译). term 结构本地标识.
    零硬编码: 无字段名/数值硬编码.
    """
    from tokenizer import cte
    return (cte._version(), str(term))


def _get_law_template(op_eid, term):
    """获取 term 模板 (统一 CTE 预编译 + 签名校验): 一致复用, 变了幂等重编译.

    经 tokenizer.cte 统一缓存 (与 gtoken 派生同机制), 数据签名 (term+refs)
    一致 → 零重编译复用; 变化 → 幂等重编译替换.
    """
    from tokenizer import cte
    sig = _law_term_signature(term, op_eid)
    key = "law:" + op_eid + ":" + str(term)
    return cte.get_or_compile(key, sig, lambda: _compile_law(term, op_eid))


def precompile_law(op_eid):
    """预编译阶段: 编译 op 全部规则的 term 模板 (幂等, 数据变化才重编译).

    每次运行前调用, 先校验 token 数据是否有变化 — 变了幂等重编译, 没变复用.
    返回编译的模板数.
    """
    from tokenizer.maintain import core
    d = core.load_all().get(op_eid, {}).get("definition") or {}
    n = 0
    for rule in (d.get("rules") or []):
        term = rule.get("term")
        if isinstance(term, list) and len(term) == 3:
            _get_law_template(op_eid, term)
            n += 1
    return n


def _law_terms(op_eid, reps):
    """规则 term → 实例化判定样本 (CTE: 模板编译, 组合量不变).

    每条 rule 的 term 结构固定 (self→op, arg:N 绑定位), 预编译为模板
    (校验签名, 数据变化才重编译), product 组合只做 O(n) 替换 —
    固定子结构不重复 assemble_seq (公共子表达式消除).
    reps: 每 arg 的代表值 digit token 列表.
    返回 [(seq, True), ...].
    """
    from tokenizer.maintain import core
    from itertools import product as _prod
    d = core.load_all()[op_eid].get("definition") or {}
    out = []
    for rule in (d.get("rules") or []):
        term = rule.get("term")
        if not (isinstance(term, list) and len(term) == 3):
            continue
        args = _collect_args(term)
        n_args = len(args)
        template = _get_law_template(op_eid, term)   # 预编译/复用
        for combo in _prod(reps, repeat=n_args):
            bindings = {a: combo[i] for i, a in enumerate(args)}
            seq = _instantiate_law_template(template, bindings)
            out.append((seq, True))
    return out


def _collect_args(node):
    """递归收集 term 中出现的 arg:N 索引 (去重保序)."""
    args = []
    seen = set()

    def _walk(n):
        if isinstance(n, str):
            if n.startswith("arg:"):
                i = int(n[4:])
                if i not in seen:
                    seen.add(i)
                    args.append(i)
            return
        if isinstance(n, list):
            for ch in n:
                _walk(ch)

    _walk(node)
    return args


def law_samples(*, op, hi=9, sample_mode="grid", seed=0, neg_mode=1, mode="correct"):
    """定义方程样本 (方程即样本 + 少量精取负例): 定义方程 + 轻量负例.

    正例 (定义方程): 沿 definition.rules 实例化 (arg:N 绑定代表值), 方程恒真.
    负例 (少量, 有逻辑):
      1. 同构结果错: 正例方程仅结果 value 换错 (与正例完全相同只结果不同)
      2. 判定型:     [is_true][numeral(a)][op][numeral(b)] = [total] [truth]
                     仅少量代表搭配 (边界 0/1/hi + 中间), 每搭配 1 负例 (total±1)
      (一元算子如 neg 无二元判定, 只做同构结果错)
    负例量 ~每正例 1 + 判定代表搭配少量 — 少而充分, 不爆炸.
    mode (matched-control, 用户确立): 破坏 law relation 但保持样本量/token数/
    操作数分布一致 —
      correct          定义方程原样 (可解释 law)
      shuffle          结果 value 随机重排 (输入输出随机配对, 破坏关系)
      wrong-symmetry   一致但错误的关系 (result→a+1 固定错误规则)
    三组对照证明: 是 specific relational structure ⟹ OOD, 非 extra data.
    返回 (样本列表, 正例数, 负例数).
    """
    if sample_mode != "grid":
        raise ValueError(f"禁止 sample_mode={sample_mode}: 必须用 grid")
    op_eid = api.eid_by_name(op)
    # 代表值: 0 / 1 / hi (边界) + 中间 (各角度对比; 少而充分) — 限单 digit
    # (方程实例化用单 digit reps; 一元判定的多位数覆盖在 _meta 分支单独处理)
    reps = []
    for v in sorted({0, 1, hi, hi // 2}):
        if 0 <= v <= 9:
            reps.append(api.digit_candidates()[v])
    rows = _law_terms(op_eid, reps)
    samples, npos, nneg = [], 0, 0
    import random
    rng = random.Random(seed)

    # 1. 方程正例 + 同构结果错负例 (仅结果 value 换错, 三 mode 统一构造)
    for seq, _ in rows:
        if mode == "correct":
            seq_final = seq
        else:
            seq_final = _law_apply_mode(seq, op_eid, mode, rng)
        samples.append(make_sample(seq_final, True, 1))
        npos += 1
        neg = _law_result_neg(seq_final if mode == "correct" else seq)
        if neg is not None:
            samples.append(make_sample(neg, False, 1))
            nneg += 1

    # 2. 判定型负例 (安全二元算子 + 小值域超运算 + neg 一元)
    # 分类沿 engine 结构原语 (迭代深度/一元), 零算子名硬编码
    from tokenizer.eval.engine import op_meta as _opm
    _meta = _opm(op_eid)
    if _meta.get("is_unary"):
        # 一元算符 (neg/translation/inversion): 一元判定 [op][a][=][op(a)]
        # 与二元判定同构: 单操作数 a, 结果 op(a), equals 连接
        # 代表值含多位数 (10/99 等), 覆盖 OOD 位数 (单 digit 判定学不会
        # 多位数 neg(71)=-71 — 用户诊断: neg 0-acc 根因是位数覆盖不足)
        # 代表值含多位数 (10/99 等), 覆盖 OOD 位数 (单 digit 判定学不会
        # 多位数 neg(71)=-71 — 用户诊断: neg 0-acc 根因是位数覆盖不足)
        reps = sorted({0, 1, hi, hi // 2, 10, 99,
                       10 ** (len(str(max(hi, 99))) - 1)})
        for a in [v for v in reps if 0 <= v <= max(hi, 99)]:
            try:
                total = _unary_eval(op_eid, a)
            except ValueError:
                continue
            prop = api.assemble_seq(op_eid, [digits_of(a)])
            seq = api.assemble_seq(api.role_token("equals"),
                                   [prop, digits_of(total)])
            samples.append(make_sample(judge_sequence(seq, True), True, 1))
            npos += 1
            try:
                bad = total + 1 if total + 1 != total else total - 1
                seq_b = api.assemble_seq(api.role_token("equals"),
                                         [prop, digits_of(bad)])
                samples.append(make_sample(judge_sequence(seq_b, False), False, 1))
                nneg += 1
            except ValueError:
                continue
    elif _is_safe_op(op_eid):
        # 安全算子: 边界 + 中间搭配
        pairs = [(0, 0), (0, 1), (1, 1), (1, hi), (hi, 1), (hi // 2, hi // 2),
                 (hi, hi)]
        pairs = [(a, b) for a, b in pairs if 0 <= a <= hi and 0 <= b <= hi]
        for a, b in pairs:
            try:
                total = eval_op(op_eid, a, b)
            except ValueError:
                continue
            samples.append(make_sample(judge_sequence(
                nested_seq([a, b], op_eid, total), True), True, 1))
            npos += 1
            wrong = total + 1
            try:
                samples.append(make_sample(judge_sequence(
                    nested_seq([a, b], op_eid, wrong), False), False, 1))
                nneg += 1
            except ValueError:
                continue
    elif _is_small_op(op_eid):
        # 超运算 (root/tetration): 小值域判定 (a,b ≤3, max_val 防溢出)
        small = [s for s in range(min(hi, 3) + 1)]
        for a in small:
            for b in small:
                try:
                    total = eval_op(op_eid, a, b, max_val=999)
                except ValueError:
                    continue
                samples.append(make_sample(judge_sequence(
                    nested_seq([a, b], op_eid, total), True), True, 1))
                npos += 1
                wrong = total + 1
                try:
                    samples.append(make_sample(judge_sequence(
                        nested_seq([a, b], op_eid, wrong), False), False, 1))
                    nneg += 1
                except ValueError:
                    continue
    return samples, npos, nneg


def _law_apply_mode(seq, op_eid, mode, rng):
    """law 正例结果变换 (matched-control): 破坏关系, 保持 token 结构.

    定位 equals 后的结果 value token (value 概念, 沿 arrow concept), 按 mode 替换:
      shuffle          换成随机其他 value 概念 (0-9 中非当前值)
      wrong-symmetry   一致错误: 结果 → 第一个操作数 + 1 (固定错误规则)
    保持: 样本量/token 数/操作数分布/base10/single-digit 完全一致.
    """
    eq = api.role_token("equals")
    for i in range(len(seq) - 1):
        if seq[i] == eq and i + 1 < len(seq) and _is_value_token(seq[i + 1]):
            cur = seq[i + 1]
            if mode == "shuffle":
                cands = [v for v in _value_concepts() if v != cur]
                seq[i + 1] = rng.choice(cands) if cands else cur
            elif mode == "wrong-symmetry":
                # 一致错误: result = a + 1 (a = 方程第一个操作数 value)
                a_tok = _first_operand(seq, op_eid)
                try:
                    a = _value_number_of(a_tok) if a_tok else 0
                    seq[i + 1] = _value_concept_for(min(a + 1, 9))
                except (ValueError, KeyError):
                    pass
            return list(seq)
    return list(seq)


def _value_concepts():
    """value 概念 token 全集 (沿 arrow 链可求值 + value_zero, 零名字)."""
    from tokenizer.eval import numeral_eval
    out = []
    for td in _all_arrows():
        try:
            numeral_eval.value_number(td.concept)
            out.append(td.concept)
        except (ValueError, KeyError):
            continue
    # value_zero (零基准, 无 concept 箭头)
    from tokenizer.maintain import core
    for e in core.load_layer("C"):
        try:
            if numeral_eval.value_number(e) == 0:
                out.append(e)
        except (ValueError, KeyError):
            continue
    return sorted(set(out))


def _value_concept_for(n):
    """数值 n → value 概念 token (沿 arrow 链反向)."""
    from tokenizer.eval import numeral_eval
    for e in _value_concepts():
        try:
            if numeral_eval.value_number(e) == n:
                return e
        except (ValueError, KeyError):
            continue
    raise ValueError(f"value 概念未找到: {n}")


def _all_arrows():
    from tokenizer._register import ARROW_REGISTRY
    return list(ARROW_REGISTRY.values())


def _value_number_of(eid):
    """value 概念 → 数值 (沿 arrow 链, 零名字)."""
    from tokenizer.eval import numeral_eval
    return numeral_eval.value_number(eid)


def _first_operand(seq, op_eid):
    """方程序列中 op 的第一个操作数 value token."""
    for i, e in enumerate(seq):
        if e == op_eid and i + 1 < len(seq) and _is_value_token(seq[i + 1]):
            return seq[i + 1]
        if e == op_eid:
            for j in range(i + 1, len(seq)):
                if _is_value_token(seq[j]):
                    return seq[j]
    return None


def _is_safe_op(op_eid):
    """安全二元算子 (判定型负例可用): 结果可表示, 无超运算溢出.

    沿 engine 结构原语 (零算子名硬编码): 一元 (neg) 或迭代深度 ≤3
    (addition/subtraction/multiplication/division) = 结果有界可表示.
    深度 ≥4 (root/tetration 超运算) = 值域爆炸, 非安全.
    """
    from tokenizer.eval.engine import op_meta as _opm
    m = _opm(op_eid)
    if m.get("is_unary"):
        return True
    return m.get("depth", 99) <= 3


def _is_small_op(op_eid):
    """超运算 (小值域判定可用): 迭代深度 ≥4 (root/tetration), 值域受限防溢出."""
    from tokenizer.eval.engine import op_meta as _opm
    return _opm(op_eid).get("depth", 0) >= 4


def _law_result_neg(seq):
    """正例方程 → 同构结果错负例 (仅 equals 后结果 value 换错), 无则 None."""
    eq = api.role_token("equals")
    for i in range(len(seq) - 1):
        # 结果 value token 识别沿 arrow 链 (value_number 可求值), 非名字前缀
        if seq[i] == eq and i + 1 < len(seq) and _is_value_token(seq[i + 1]):
            cur = seq[i + 1]
            alt = None
            for v in api.valid_digits(10):
                if v != cur:
                    alt = v
                    break
            if alt is None:
                return None
            neg = list(seq)
            neg[i + 1] = alt
            return neg
    return None


def _is_value_token(eid: str) -> bool:
    """eid 是否为 value token (value 概念, 沿 arrow 链或零基准, 零名字).

    value token = 沿 arrow 链可求值 (value_number 成功) — 涵盖 value_zero
    (零基准, 无 concept 箭头) 与 value_one..nine (arrow concept 引用).
    digit 构造实例沿 arrow 链不可求值 (排除).
    """
    from tokenizer.eval import numeral_eval
    try:
        numeral_eval.value_number(eid)
        return True
    except (ValueError, KeyError):
        return False


def arrow_samples(*, concept, sample_mode="grid", seed=0):
    """arrow 样本集设计 (消费方): 端点全组合枚举.

    tokenizer 原生提供: arrow_endpoints (候选域) + sample_arrow (单样本+真值).
    真值 = arrow 存在性 (tokenizer 查 A 层字段), 零硬编码.
    全组合枚举 (grid): 端点集内全部 source×target 组合, 真=存在 arrow, 假=不存在.
    (coercion/succ 等端点集小的 concept 全枚举安全: 假样本提供"不提升/不后继"负例正则;
    inverse 端点集大 (22²=484) 会假样本爆炸 — 由配置决定是否纳入, 合成器自身保持 grid.)
    返回 (样本列表, 正例数, 负例数).
    """
    if sample_mode != "grid":
        raise ValueError(f"禁止 sample_mode={sample_mode}: 必须用 grid (全组合枚举, 可归因)")
    concept_eid = api.eid_by_name(concept)
    toks, pairs = api.arrow_endpoints(concept_eid)
    if not toks:
        return [], 0, 0
    samples, npos, nneg = [], 0, 0
    for s in toks:
        for t in toks:
            seq, truth = api.sample_arrow(concept_eid, s, t)
            samples.append(make_sample(seq, truth, 1))
            if truth:
                npos += 1
            else:
                nneg += 1
    return samples, npos, nneg


def coercion_samples(*, hi=20, sample_mode="grid", seed=0):
    """数域提升样本 (消费方): 值 n 沿 coercion 箭头提升 (tokenizer 原生 sample_coercion).

    枚举 n ∈ [0, hi] × 全部 coercion 箭头, 真样本 (正确提升) + 假样本 (错误提升,
    目标表示某位错) 1:1 — 判定口径有真有假, 防无脑猜真.
    返回 (样本列表, 正例数, 负例数).
    """
    if sample_mode != "grid":
        raise ValueError(f"禁止 sample_mode={sample_mode}: 必须用 grid (全组合枚举, 可归因)")
    arrows = api.coercion_arrows()
    samples, npos, nneg = [], 0, 0
    for a in arrows:
        for n in range(0, hi + 1):
            samples.append(api.sample_coercion(n, a["eid"], truth=True))
            npos += 1
            samples.append(api.sample_coercion(n, a["eid"], truth=False))
            nneg += 1
    return samples, npos, nneg


# ---- 样本类型注册表 (原生适配: 新增题型/领域只需注册合成器, 禁改 compose_samples) ----
_SAMPLE_REGISTRY: dict = {}


def register_sample(kind):
    """注册样本类型: @register_sample('xxx') 装饰合成器 (spec, seed) -> (ss, pos, neg)。"""
    def deco(fn):
        _SAMPLE_REGISTRY[kind] = fn
        return fn
    return deco


@register_sample("logic_nested")
def _s_logic_nested(spec, seed):
    return logic_nested_samples(ops=spec["ops"], max_depth=spec.get("max_depth", 3),
                                seed=seed, notation=spec.get("notation", "prefix"))


@register_sample("logic_arith")
def _s_logic_arith(spec, seed):
    return logic_arith_samples(ops=spec["ops"], hi=spec.get("hi", 9), seed=seed,
                               notation=spec.get("notation", "prefix"))


@register_sample("logic_interdef")
def _s_logic_interdef(spec, seed):
    return logic_interdef_samples(ops=spec.get("ops"), seed=seed,
                                  imply_judge=spec.get("imply_judge", True),
                                  imply_contrapos=spec.get("imply_contrapos", True),
                                  imply_proppos=spec.get("imply_proppos", True),
                                  imply_truthrow=spec.get("imply_truthrow", False),
                                  notation=spec.get("notation", "prefix"),
                                  imply_boost=spec.get("imply_boost", 1))


@register_sample("logic_structural")
def _s_logic_structural(spec, seed):
    return logic_structural_samples(ops=spec.get("ops"), seed=seed,
                                    max_depth=spec.get("max_depth", 3),
                                    n=spec.get("n", 200))


@register_sample("balanced")
def _s_balanced(spec, seed):
    return balanced_samples(max_depth=spec["max_depth"], hi=spec["hi"],
                            op=api.eid_by_name(spec["op"]),
                            neg_mode=spec.get("neg_mode", "all"),
                            sample_mode=spec.get("sample_mode", "grid"), seed=seed)


@register_sample("definition")
def _s_definition(spec, seed):
    return definition_samples(op=spec["op"], hi=spec.get("hi", 9),
                              neg_mode=spec.get("neg_mode", "all"),
                              sample_mode=spec.get("sample_mode", "grid"), seed=seed)


@register_sample("law")
def _s_law(spec, seed):
    return law_samples(op=spec["op"], hi=spec.get("hi", 9),
                       sample_mode=spec.get("sample_mode", "grid"), seed=seed,
                       mode=spec.get("mode", "correct"))


@register_sample("arrow")
def _s_arrow(spec, seed):
    return arrow_samples(concept=spec["concept"],
                         sample_mode=spec.get("sample_mode", "grid"), seed=seed)


@register_sample("fill")
def _s_fill(spec, seed):
    return fill_samples(op=spec["op"], hi=spec.get("hi", 9),
                        sample_mode=spec.get("sample_mode", "grid"), seed=seed)


@register_sample("choose")
def _s_choose(spec, seed):
    return choose_samples(op=spec["op"], hi=spec.get("hi", 9), k=spec.get("k", 2),
                          sample_mode=spec.get("sample_mode", "grid"), seed=seed)


@register_sample("coercion")
def _s_coercion(spec, seed):
    return coercion_samples(hi=spec.get("hi", 20),
                            sample_mode=spec.get("sample_mode", "grid"), seed=seed)


@register_sample("stepwise")
def _s_stepwise(spec, seed):
    return stepwise_samples(op=spec["op"], hi=spec.get("hi", 9),
                            max_depth=spec.get("max_depth", 3),
                            sample_mode=spec.get("sample_mode", "grid"), seed=seed)


@register_sample("deep_nest")
def _s_deep_nest(spec, seed):
    return deep_nest_samples(op=spec["op"], hi=spec.get("hi", 9),
                             depths=spec["depths"], n=spec.get("n", 100),
                             neg_mode=spec.get("neg_mode", 1), seed=seed)


@register_sample("extrap_2000")
def _s_extrap_2000(spec, seed):
    return extrap_2000_samples(op=spec["op"], ndigits=spec.get("ndigits", 2000),
                               seed=seed)


def trace_samples(*, op, hi=9, max_depth=4, neg_mode=1, sample_step=1, seed=0):
    """逐层教学轨迹样本 (零硬编码): 深嵌套每层中间结果显式, 教"先算内层再算外层".

    序列: is_true (a op b) = r1 (r1 op c) = r2 ... = total truth_true
      - 每步 (a op t) = r 沿 nested_seq 组装 (assemble_seq → op ptoken + grouping + equality)
      - 判定沿 judge_sequence (judge 概念), 序列零手写 (token 定义驱动)
      - 中间结果沿 eval_op (token 定义真值)
    组合 = 数字候选域 [0,hi] 枚举; sample_step>1 均匀步进采样 (控制样本量, 迭代加速).
    返回 (样本列表, 正例数, 负例数).
    """
    op_eid = api.eid_by_name(op) if not str(op).startswith("D:") else op
    samples, npos, nneg = [], 0, 0
    for d in range(2, max_depth + 1):
        for idx, terms in enumerate(__import__("itertools").product(range(hi + 1), repeat=d)):
            if sample_step > 1 and idx % sample_step:
                continue
            r = terms[0]
            try:
                steps = []
                for t in terms[1:]:
                    r_new = eval_op(op_eid, r, t)
                    steps.append((r, t, r_new))
                    r = r_new
            except ValueError:
                continue
            prop = []
            for a, t, res in steps:
                prop.extend(nested_seq([a, t], op_eid, res))
            samples.append(make_sample(judge_sequence(prop, True), True, d))
            npos += 1
            if neg_mode:
                bad = steps[-1][2] + 1
                prop_bad = []
                for a, t, res in steps[:-1]:
                    prop_bad.extend(nested_seq([a, t], op_eid, res))
                prop_bad.extend(nested_seq([steps[-1][0], steps[-1][1]], op_eid, bad))
                samples.append(make_sample(judge_sequence(prop_bad, False), False, d))
                nneg += 1
    return samples, npos, nneg


@register_sample("trace")
def _s_trace(spec, seed):
    return trace_samples(op=spec["op"], hi=spec.get("hi", 9),
                         max_depth=spec.get("max_depth", 4),
                         neg_mode=spec.get("neg_mode", 1),
                         sample_step=spec.get("sample_step", 1), seed=seed)


def numeral_split_samples(*, lo=100, hi=999, min_token_count=3, neg_mode=1, seed=0):
    """numeral 拆解教学样本 (3 位整数拆分判定, 表示法 + 加减法, bool 口径):
      is_true ( [100] addition [20] addition [6] ) = [126] truth_true
    覆盖 3 位内所有划分形态 (均匀穿梭):
      [126] = [100]+[20]+[6]  (按位) / [120]+[6] / [100]+[26] / [106]+[20]
    样本量 = token 搭配充分驱动: min_token_count>0 时, 所有 digit 相邻组合 (搭配对)
      覆盖 ≥ 该值即停 (token 搭配训练充分就够, 非枚举/采样全空间).
    组装沿 numeral_of/nested_seq (零硬编码). 返回 (样本, 正, 负).
    """
    from collections import Counter
    op_eid = api.eid_by_name("addition")
    # 全 token 搭配充分: 所有参与训练 token 的相邻搭配对覆盖 ≥ min_token_count (不只 digit)
    pair_cover = Counter()
    samples, npos, nneg = [], 0, 0
    all_pairs_seen = set()
    for n in range(lo, hi + 1):
        a, rem = divmod(n, 100)
        b, c = divmod(rem, 10)
        units = [a * 100, b * 10, c]
        splits = set()
        for mask in range(1, 8):            # 位权子集 → 2 部分划分 (子集 vs 补集)
            s = sum(units[i] for i in range(3) if mask >> i & 1)
            comp = n - s
            if s <= comp:
                splits.add((s, comp))
        splits.add(tuple(units))            # 3 部分按位
        for parts in sorted(splits):
            seq = judge_sequence(nested_seq(parts, op_eid, n), True)
            samples.append(make_sample(seq, True, len(parts)))
            npos += 1
            for p in zip(seq, seq[1:]):
                pair_cover[p] += 1
                all_pairs_seen.add(p)
            if neg_mode:
                seq_bad = judge_sequence(nested_seq(parts, op_eid, n + 1), False)
                samples.append(make_sample(seq_bad, False, len(parts)))
                nneg += 1
        if min_token_count and all_pairs_seen \
                and all(pair_cover.get(p, 0) >= min_token_count for p in all_pairs_seen):
            break
    return samples, npos, nneg


@register_sample("numeral_split")
def _s_numeral_split(spec, seed):
    return numeral_split_samples(lo=spec.get("lo", 100), hi=spec.get("hi", 999),
                                 min_token_count=spec.get("min_token_count"),
                                 neg_mode=spec.get("neg_mode", 1), seed=seed)


@register_sample("cat")
def _s_cat(spec, seed):
    return cat_samples(seed=seed)


def cat_samples(*, seed=0):
    """范畴论定律样本 (tokenizer 原生, token 定义驱动): 沿 identity/inverse/composition
    概念定义 + A 层态射生成定律判定 (单位元律/对偶律), 零硬编码定律列表."""
    samples, npos, nneg = [], 0, 0
    _TRUE = api.role_token("truth")[0]
    for s in api.category_law_samples():
        samples.append(make_sample(s, s[-1] == _TRUE, 1))
        if s[-1] == _TRUE:
            npos += 1
        else:
            nneg += 1
    return samples, npos, nneg


def deep_nest_samples(*, op, hi, depths, n=100, neg_mode=1, seed=0):
    """深嵌套泛化样本 (采样, 防枚举爆炸): 对每个嵌套深度随机 n 组合.

    depth = terms 数 (括号层 = depth-1); 训练 (1-3 层) 全枚举可归因,
    外推 (5/10 层) 采样 (全枚举 6^11 天文数字). 样本 = nested_seq 判定序列.
    返回 (样本列表, 总样本, 0).
    """
    import random
    rng = random.Random(seed)
    op_eid = api.eid_by_name(op) if not str(op).startswith("D:") else op
    samples, npos = [], 0
    for d in depths:
        for _ in range(n):
            terms = [rng.randint(0, hi) for _ in range(d)]
            try:
                total = terms[0]
                for t in terms[1:]:
                    total = eval_op(op_eid, total, t)
            except ValueError:
                continue
            seq = judge_sequence(nested_seq(terms, op_eid, total), True)
            samples.append(make_sample(seq, True, d))
            npos += 1
            if neg_mode:
                seq_f = judge_sequence(nested_seq(terms, op_eid, total + 1), False)
                samples.append(make_sample(seq_f, False, d))
    return samples, npos, len(samples) - npos


@register_sample("logic")
def _s_logic(spec, seed):
    return logic_samples(op=spec["op"], seed=seed, notation=spec.get("notation", "prefix"))


def stepwise_samples(*, op, hi=9, max_depth=3, sample_mode="grid", seed=0):
    """逐步计算教学样本 (操作顺序 = 内层优先, 用户确立): 每层嵌套计算显式中间结果.

    ((a+b)+c): [is_true] [(][a][op][b][)][=][r1] [(][r1][op][c][)][=][r2] [truth_true]
    与 nested_seq (仅最终判定) 的区别: 教学"先算内层再算外层"的计算顺序,
    中间结果 r1/r2 显式展示 — 模型从判定样本学不到操作顺序 (用户诊断).
    返回 (样本列表, 正例数, 0).
    """
    if sample_mode != "grid":
        raise ValueError(f"禁止 sample_mode={sample_mode}: 必须用 grid")
    op_eid = api.eid_by_name(op) if not str(op).startswith("D:") else op
    samples, npos = [], 0
    for d in range(2, max_depth + 1):
        for terms in __import__("itertools").product(range(hi + 1), repeat=d):
            steps, r = [], terms[0]
            try:
                for t in terms[1:]:
                    r_new = eval_op(op_eid, r, t)
                    steps.append((r, t, r_new))
                    r = r_new
            except ValueError:
                continue
            prop = []
            for a, t, res in steps:
                prop.extend(nested_seq([a, t], op_eid, res))
            samples.append(make_sample(judge_sequence(prop, True), True, d))
            npos += 1
    return samples, npos, 0


def compose_samples(*, samples, seed=0):
    """样本集组合 (消费方): 按配置列表依次合成并合并.

    samples: [{kind: ..., 各类型参数}] — 分发走 _SAMPLE_REGISTRY (注册表原生适配,
    新增题型/领域只需 @register_sample, 禁改本函数). 返回 (合并样本, 总正例, 总负例).
    """
    all_s, npos, nneg = [], 0, 0
    for spec in samples:
        kind = spec["kind"]
        fn = _SAMPLE_REGISTRY.get(kind)
        if fn is None:
            raise ValueError(f"未知样本类型: {kind} (须经 @register_sample 注册)")
        ss, p, n = fn(spec, seed)
        # 注入样本溯源: 整个 spec 配置作为 raw 溯源 (供归档诊断按类型/参数对比),
        # 零字段枚举 (不硬编码 op/kind 等字段名 — 配置数据整体投影到样本)
        for s in ss:
            s.setdefault("spec", spec)
        all_s.extend(ss)
        npos += p
        nneg += n
    return all_s, npos, nneg


def token_category(eid) -> str:
    """token 语义类别 (同类/异类判定基础): 沿 token 命名体系 (token 内分类).

    digit=数字位 / value=数值 / operator=算子 / sign=正负 / representation=表示包裹
    feature=内禀特征(base/cardinality) / truth=真值 / bracket=运算括号 / judge=判定 / arrow=箭头.
    """
    n = api.name(eid)
    if n.startswith("digit_") or n.startswith("value_"):
        return "digit"
    if n in ("sign", "sign_pos", "sign_neg", "neg"):
        return "sign"
    if n.startswith("representation"):
        return "representation"
    if n in ("base", "cardinality"):
        return "feature"
    if n.startswith("truth_"):
        return "truth"
    if n in ("left_bracket", "right_bracket"):
        return "bracket"
    if n in ("is_true", "equals_arith"):
        return "judge"
    if eid.startswith("A:"):
        return "arrow"
    return "operator"


def cooccurrence_view(samples, n=2, min_count=3, top=40) -> dict:
    """token 搭配视图 (训练样本 n-gram 频次): 找搭配量少的 n-token 序列, 分同类/异类.

    训练完整性 = token 与平行概念在其他 token 搭配中的充分性 (非 loss/acc);
    n=2 相邻对 / n=3 三连 / n=4 四连搭配; 低频 = 模型未见够的上下文 (泛化崩候选).
    同类 = n-gram 全部同语义类别; 异类 = 跨类别 (含异类 token).
    目标: 整体搭配量 ≥ min_count (好训练约 3). 返回 {"same": [...], "cross": [...]} 升序 top.
    """
    from collections import Counter
    gram = Counter()
    for s in samples:
        seq = s["seq"]
        for i in range(len(seq) - n + 1):
            gram[tuple(seq[i:i + n])] += 1
    same, cross = [], []
    for tup, c in gram.items():
        if c >= min_count:
            continue
        cats = {token_category(e) for e in tup}
        names = " ".join(api.name(e) for e in tup)
        rec = (names, c, len(cats))
        if len(cats) == 1:
            same.append(rec)
        else:
            cross.append(rec)
    return {"same": sorted(same, key=lambda x: x[1])[:top],
            "cross": sorted(cross, key=lambda x: x[1])[:top]}
    all_s, npos, nneg = [], 0, 0
    for spec in samples:
        kind = spec["kind"]
        fn = _SAMPLE_REGISTRY.get(kind)
        if fn is None:
            raise ValueError(f"未知样本类型: {kind} (须经 @register_sample 注册)")
        ss, p, n = fn(spec, seed)
        all_s.extend(ss)
        npos += p
        nneg += n
    return all_s, npos, nneg


def ood_samples(*, op, digits, n=1000, mode="mixed", neg_mode=1, seed=0, result_digits=None):
    """泛化样本采样 (消费方): 多位数 (digits 位) 判定样本, 多样化泛化测试.

    result_digits: 运算**结果**位数上限 (用户确立: 20 位是结果范围非输入;
    默认 = digits 兼容旧行为). 结果超上限 → ValueError 跳过 (防溢出).

    mode:
      random         随机 n 组操作数 (真题+错题)
      representative 代表性子集 (进位边界: 999..9+1, 全9, 10^k-1)
      carry          进位链 (跨位进位: 99..9+1 → 100..0, 大数+小数, 相同数)
      mixed          全部混合 (随机+边界+进位, 用户要求"各种乱七八糟泛化都测")
    每组合生成: 真题 (正例) + neg_mode 个错题 (负例, 错误结果).
    真值暂用 eval_op (多位数无 arrow/定义覆盖, 标注待 tokenizer 求值能力完善).
    返回 (样本列表, 正例数, 负例数).
    """
    op_eid = api.eid_by_name(op)
    # 一元算子 (arrange=unary_connective, 如 neg): 判定 [is_true][a][op][op(a)][truth]
    # 沿 op 的 ptoken/gtoken 组装, 两侧各一个 numeral — 与二元判定同构 (无 OOD 0 判定).
    if api.arrange_of(op_eid) == "unary_connective":
        return _ood_unary_samples(op_eid, digits, n, mode, neg_mode, seed)
    hi = 10 ** digits
    combos = []
    if mode in ("random", "mixed"):
        import random as _r
        _r.seed(seed)
        for _ in range(n):
            combos.append((_r.randint(0, hi - 1), _r.randint(0, hi - 1)))
    if mode in ("representative", "mixed"):
        for k in range(1, digits + 1):
            full9 = 10 ** k - 1
            combos.append((full9, 1))       # 999..9 + 1 (进位链)
            combos.append((1, full9))
            combos.append((full9, full9))   # 999..9 + 999..9 (多进位)
            combos.append((full9, 10 ** (k - 1)))
    if mode in ("carry", "mixed"):
        for k in range(1, digits + 1):
            full9 = 10 ** k - 1
            combos.append((full9, 1))       # 全位进位
            combos.append((10 ** (k - 1), full9))
            combos.append((full9, full9))
            combos.append((full9, 10 ** (k - 1) - 1))  # 进位边界
    if mode not in ("random", "representative", "carry", "mixed"):
        raise ValueError(f"未知泛化模式: {mode} (合法: random/representative/carry/mixed)")
    combos = list(dict.fromkeys(combos))
    # 过滤非法组合 (除法除数为 0, value_0 不可除)
    if op_eid == api.eid_by_name("division"):
        combos = [(a, b) for a, b in combos if b != 0]
    samples, npos, nneg = [], 0, 0
    for a, b in combos:
        try:
            total = eval_op(op_eid, a, b,
                            max_val=10 ** (result_digits or digits) - 1)
        except ValueError:
            continue  # 超运算不可求值组合, 跳过 (训练/测试一致)
        samples.append(make_sample(judge_sequence(nested_seq([a, b], op_eid, total), True), True))
        npos += 1
        for bad in range(total + 1, total + 1 + int(neg_mode)):
            samples.append(make_sample(judge_sequence(nested_seq([a, b], op_eid, bad), False), False))
            nneg += 1
    return samples, npos, nneg


def _ood_unary_samples(op_eid, digits, n, mode, neg_mode, seed):
    """一元算子 OOD 判定样本 (arrange=unary_connective, 如 neg): [is_true][a][op][op(a)][truth].

    一元算子无二元 `a op b` 形式, OOD 判定序列 = 输入值 a 与输出 op(a) 并置,
    op 作为中缀判定符 (与二元判定同构) — 消除"一元算子 OOD 0 判定"现象.
    求值沿一元语义: neg(a) = -a (eval_op 二元签名不支持, 一元单独处理).
    外推: digits 位输入 → 输出同位数 (neg 不改变位数), 支持 2000 位输出外推.
    """
    import random
    rng = random.Random(seed)
    hi = 10 ** digits - 1
    combos = []
    if mode in ("random", "mixed"):
        for _ in range(n):
            combos.append(rng.randint(0, hi))
    if mode in ("representative", "mixed", "carry"):
        combos += [0, 1, hi, 10 ** (digits - 1), hi - 1, 10 ** (digits - 1) - 1]
    combos = list(dict.fromkeys(combos))
    samples, npos, nneg = [], 0, 0
    for a in combos:
        try:
            total = _unary_eval(op_eid, a)
        except ValueError:
            continue
        # 判定序列: [is_true][op][numeral(a)][=][numeral(total)][truth]
        # 与训练 law 一元判定同构 (equals 关系, 零手写列表)
        prop = api.assemble_seq(op_eid, [digits_of(a)])
        seq = api.assemble_seq(api.role_token("equals"), [prop, digits_of(total)])
        samples.append(make_sample(judge_sequence(seq, True), True))
        npos += 1
        for bad in range(1, 1 + int(neg_mode)):
            bad_total = total + bad if total + bad != total else total - 1
            seq_b = api.assemble_seq(api.role_token("equals"),
                                     [prop, digits_of(bad_total)])
            samples.append(make_sample(judge_sequence(seq_b, False), False))
            nneg += 1
    return samples, npos, nneg


def _unary_eval(op_eid, a):
    """一元算子求值 (通用求值引擎, 真值沿 token 定义, 零硬编码).

    输入一元算子 eid (neg) + 数值 a → 结果数值. 经 api.eval_op:
    numeral token 序列 (numeral 快路径) → 求值 → numeral → 数值.
    """
    out = api.eval_op(op_eid, [numeral_of(a)])
    return _numeral_value_of(out)


def extrap_2000_samples(*, op, ndigits=2000, seed=0):
    """2000 位输出外推样本 (输出真为 ndigits 位, 无嵌套 + 深嵌套).

    构造操作数使计算结果恰为 ndigits 位 (如 10^(ndigits-1) + 10^(ndigits-1)
    = 2×10^(ndigits-1) — ndigits 位; 乘法 10^(n-1) × 10 递增; 嵌套累加保持位宽).
    无嵌套: [a op b = total] (total 真为 ndigits 位).
    深嵌套: ((a op b) op c) ... 逐层, 中间/最终结果保持 ndigits 位.
    用途: 验证模型外推到超长输出位 (2000 位 digit 序列泛化).
    返回 (样本列表, 正例数, 0).
    """
    op_eid = api.eid_by_name(op) if not str(op).startswith("D:") else op
    base = 10 ** (ndigits - 1)          # 1 后 ndigits-1 个 0 (ndigits 位)
    half = base // 2                    # 加法: base/2 + base/2 = base (ndigits 位)
    small = 10 ** (ndigits // 2)        # 乘法因子 (半宽, 乘积 ~1.5 宽)
    samples, npos = [], 0
    terms_sets = []
    if op in ("addition", "subtraction"):
        terms_sets.append((half, half))          # 无嵌套: 2×half = base (ndigits 位)
        terms_sets.append((base, 1))             # base+1 (仍 ndigits 位)
        terms_sets.append((base - 1, 1))         # 边界进位: base-1+1 = base
    elif op in ("multiplication", "power"):
        terms_sets.append((small, small))        # small² (约 ndigits+2 位, 真外推)
        terms_sets.append((small, 2))
    elif op == "division":
        terms_sets.append((base * 2, 2))         # 2base/2 = base
    elif op == "root":
        terms_sets.append((base * base, 2))      # sqrt(base²) = base
    elif op == "neg":
        for a in (base, base + 1):
            total = _unary_eval(op_eid, a)
            prop = api.assemble_seq(op_eid, [digits_of(a), digits_of(total)])
            samples.append(make_sample(judge_sequence(prop, True), True))
            npos += 1
        return samples, npos, 0
    else:
        raise ValueError(f"2000 位外推未支持算子: {op}")
    for terms in terms_sets:
        total = terms[0]
        for t in terms[1:]:
            total = eval_op(op_eid, total, t)
        seq = judge_sequence(nested_seq(list(terms), op_eid, total), True)
        samples.append(make_sample(seq, True, len(terms)))
        npos += 1
        # 深嵌套: 同算子逐层叠加保持位宽 ((total op t) op t...)
        deep = terms[0]
        steps = []
        for t in (terms[1:] or (terms[1],)):
            deep = eval_op(op_eid, deep, t)
            steps.append(deep)
        if len(terms) == 2:
            nested_terms = [terms[0]] + [t for _ in range(1) for t in terms[1:]]
            try:
                t_total = eval_op(op_eid, base, 1)
                seq_d = judge_sequence(nested_seq([base, 1, 0] if op in ("addition", "subtraction") else [base, 1], op_eid, t_total), True)
                samples.append(make_sample(seq_d, True, 3))
                npos += 1
            except ValueError:
                pass
    return samples, npos, 0


# ---- 多种题型样本 (question token 体系) ----
def fill_samples(*, op, hi=9, sample_mode="grid", seed=0):
    """填空题型样本: [question][a][op][gap][equals][c] → 填 gap 位置.

    gap 位置 = 待填箭头位 (question token 体系). 答案 = b (填 gap 的 digit 序列).
    样本含 gap_pos (答案起始位置) + answer (答案 token 序列), 供 collate 位置监督.
    零硬编码: question/gap/equals 经 tokenizer 检索.
    返回 (样本列表, 正例数, 0).
    """
    if sample_mode != "grid":
        raise ValueError(f"禁止 sample_mode={sample_mode}: 必须用 grid")
    op_eid = api.eid_by_name(op)
    fill_eid = api.eid_by_name("fill")
    gap_eid = api.eid_by_name("gap")
    nums = api.digit_concepts()          # 候选域: token 定义派生 (inductive+atom 引用链)
    vals = sorted(n for n in nums if n <= hi)
    samples = []
    for a in vals:
        for c in vals:
            answers = []
            for x in vals:
                try:
                    if eval_op(op_eid, a, x) == c:
                        answers.append(x)
                except ValueError:
                    continue
            if not answers:
                continue
            a_tok, c_tok = digits_of(a), digits_of(c)
            seq = api.assemble_seq(fill_eid, [a_tok, [op_eid], c_tok])
            gap_pos = seq.index(gap_eid)
            for b in answers:
                samples.append(make_sample(seq, True, 1, gap_pos=gap_pos, answer=digits_of(b)))
    return samples, len(samples), 0


def choose_samples(*, op, hi=9, k=2, sample_mode="grid", seed=0):
    """选择题型样本: [question][a][op][gap][equals][c][candidate][候选...] → 选正确答案.

    候选 = 正确答案 + k 个错误候选 (确定性). gap 位置填正确答案.
    样本含 gap_pos + answer (正确答案 token 序列).
    返回 (样本列表, 正例数, 0).
    """
    if sample_mode != "grid":
        raise ValueError(f"禁止 sample_mode={sample_mode}: 必须用 grid")
    op_eid = api.eid_by_name(op)
    choose_eid = api.eid_by_name("choose")
    gap_eid = api.eid_by_name("gap")
    nums = api.digit_concepts()          # 候选域: token 定义派生
    vals = sorted(n for n in nums if n <= hi)
    samples = []
    for a in vals:
        for c in vals:
            answers = []
            for x in vals:
                try:
                    if eval_op(op_eid, a, x) == c:
                        answers.append(x)
                except ValueError:
                    continue
            if not answers:
                continue
            a_tok, c_tok = digits_of(a), digits_of(c)
            wrongs = [x for x in vals if x not in answers]
            cands = answers[:] + wrongs[:max(0, k + 1 - len(answers))]
            cand_tok = []
            for v in cands:
                cand_tok += digits_of(v)
            seq = api.assemble_seq(choose_eid, [a_tok, [op_eid], c_tok, cand_tok])
            gap_pos = seq.index(gap_eid)
            for b in answers:
                samples.append(make_sample(seq, True, 1, gap_pos=gap_pos, answer=digits_of(b)))
    return samples, len(samples), 0


def _invalid_combo(op_eid, a, b):
    """非法组合 (跳过): 除法除数为 0 (0 不可除, value_0 约束). 减法可负 (neg 表达)."""
    if op_eid == api.eid_by_name("division") and b == 0:
        return True
    return False


def iteration_staircase_samples(*, hi=999, min_token_count=3, neg_mode=1, seed=0):
    """数的迭代展开样本 (用户确立): 同一数沿位权/加法逐步分解的等式链.

    形式: [123][=][120][+][3][=][103][+][20][=][100][+][23]
      — 同一总数 n 沿不同位权组合逐步展开 (加法迭代到数值的可视链条),
        模型看到"数 = 低阶迭代组合" (迭代升阶的反向: 高阶合成回低阶).
      多步链式 (stepwise 组装): 每步 (a op b) = r 显式中间结果.
    迭代升阶箭头由 arrow concept=iterate 单独提供 (A 层), 本样本只做
    数的逐层展开 — 不混用 is_succ 等一阶迭代标记 (用户纠正).

    样本量: 代表值少量精取 (同 law_samples 模式, 非全量枚举):
      进位边界 (199/109/190), 全 9, 中间值, hi — 每值 1 正例 + neg_mode 负例.
      组合覆盖 token 搭配充分即停 (min_token_count 控制, 同 numeral_split).
    返回 (样本列表, 正例数, 负例数).
    """
    from tokenizer.eval.engine import op_meta as _opm
    add_eid = next(e for e in _all_c_eids()
                   if _opm(e).get("is_succ") and _opm(e).get("base") is None)
    from collections import Counter
    pair_cover = Counter()
    samples, npos, nneg = [], 0, 0
    all_pairs_seen = set()
    # 代表值: 进位边界 + 全9 + 中间 (各角度对比; 少而充分)
    reps = sorted({199, 109, 190, 110, 101, 191, 999, hi, hi // 2,
                   10 * (hi // 100) + (hi % 10) if hi >= 100 else hi})
    for n in reps:
        if n < 100 or n > hi:
            continue
        a, rem = divmod(n, 100)
        b, c = divmod(rem, 10)
        units = [a * 100, b * 10, c]
        # 3 步链: n = (a00+b0)+c → 先合并 a00+b0, 再 +c
        try:
            s1 = eval_op(add_eid, units[0], units[1])
            s2 = eval_op(add_eid, s1, units[2])
        except ValueError:
            continue
        prop = []
        prop.extend(nested_seq([units[0], units[1]], add_eid, s1))
        prop.extend(nested_seq([s1, units[2]], add_eid, s2))
        seq = judge_sequence(prop, True)
        samples.append(make_sample(seq, True, 2))
        npos += 1
        for p in zip(seq, seq[1:]):
            pair_cover[p] += 1
            all_pairs_seen.add(p)
        if neg_mode:
            seq_bad = judge_sequence(nested_seq([units[0], units[1]], add_eid, s1 + 1), False)
            samples.append(make_sample(seq_bad, False, 2))
            nneg += 1
        if min_token_count and all_pairs_seen \
                and all(pair_cover.get(p, 0) >= min_token_count for p in all_pairs_seen):
            break
    return samples, npos, nneg


def _all_c_eids():
    from tokenizer import api as _api
    from tokenizer._register import all_eids
    return [e for e in all_eids() if e.startswith("D:")]


@register_sample("iteration_staircase")
def _s_iteration_staircase(spec, seed):
    return iteration_staircase_samples(hi=spec.get("hi", 999),
                                       min_token_count=spec.get("min_token_count", 3),
                                       neg_mode=spec.get("neg_mode", 1), seed=seed)


def iteration_expression_samples(*, seed=0):
    """标准迭代语法样本 (tokenizer 原生合成, 用户确立).

    委托 tokenizer.eval.sample_eval.iterate_expression_samples:
      沿 A 层 iterate 箭头 (权威迭代链) 生成 迭代(方向, 层数, 被迭代算符)
      判定序列 — 方向 (increase/decimal) + 层数 (value_one) + 被迭代算符,
      标准 iterate_expr gtoken 组装, 可逐层展开. 合成器零硬编码.
    返回 (样本列表, 正例数, 负例数).
    """
    from tokenizer.eval.sample_eval import iterate_expression_samples as _nat
    samples, npos, nneg = [], 0, 0
    for s in _nat():
        samples.append(make_sample(s["seq"], s["truth"], 1))
        npos += 1
    return samples, npos, nneg


@register_sample("iteration_expression")
def _s_iteration_expression(spec, seed):
    return iteration_expression_samples(seed=seed)


def unary_judge_samples(*, op, hi=99, neg_mode=1, seed=0):
    """一元算子判定样本 (neg/translation/inversion 独立判定, 平衡 is_true 分布).

    问题 (用户诊断): neg 一元判定样本被 bracket 主导 (is_true 后 bracket 3408
    vs neg 12) — 模型学"is_true 后→bracket"捷径, 从不输出 neg.
    本样本: 一元算子的多位数判定 ([is_true][op][a][=][op(a)]), 量级与
    balanced 相当 (~200), 平衡类别分布.
    代表值: 单 digit (0/1/hi/中间) + 多位数 (10/99/进位边界), 覆盖 OOD 位数.
    返回 (样本列表, 正例数, 负例数).
    """
    import random
    rng = random.Random(seed)
    op_eid = api.eid_by_name(op) if not str(op).startswith("D:") else op
    samples, npos, nneg = [], 0, 0
    reps = sorted({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 19, 20, 49, 50,
                   90, 91, 99, 101, 109, 110, 999,
                   10 ** (len(str(max(hi, 99))) - 1)})
    for a in [v for v in reps if 0 <= v <= max(hi, 99)]:
        try:
            total = _unary_eval(op_eid, a)
        except ValueError:
            continue
        prop = api.assemble_seq(op_eid, [digits_of(a)])
        seq = api.assemble_seq(api.role_token("equals"), [prop, digits_of(total)])
        samples.append(make_sample(judge_sequence(seq, True), True, 1))
        npos += 1
        for _ in range(int(neg_mode)):
            bad = total + rng.randint(1, 3) if total + rng.randint(1, 3) != total else total - 1
            seq_b = api.assemble_seq(api.role_token("equals"),
                                     [prop, digits_of(bad)])
            samples.append(make_sample(judge_sequence(seq_b, False), False, 1))
            nneg += 1
    return samples, npos, nneg


@register_sample("unary_judge")
def _s_unary_judge(spec, seed):
    return unary_judge_samples(op=spec["op"], hi=spec.get("hi", 99),
                               neg_mode=spec.get("neg_mode", 1), seed=seed)


def cartesian_ood_samples(*, ops, ndigits=2, n=20, neg_mode=1, seed=0):
    """联合笛卡尔 OOD 样本 (用户确立): 高阶算符 × 多位数联合未见组合.

    训练只用个位数 (hi≤9); 联合 OOD = 高阶算符 (tetration/root/super_log)
    × 2位数操作数 (11-99) — 模型从未见"高阶算符+多位数"组合.
    判定序列沿 nested_seq 组装 (tokenizer 原生), 真值沿引擎求值.
    返回 (样本列表, 正例数, 负例数).
    """
    import random
    rng = random.Random(seed)
    samples, npos, nneg = [], 0, 0
    for op in ops:
        op_eid = api.eid_by_name(op) if not str(op).startswith("D:") else op
        for _ in range(n):
            a = rng.randint(10 ** (ndigits - 1), 10 ** ndigits - 1)
            b = rng.randint(10 ** (ndigits - 1), 10 ** ndigits - 1)
            try:
                total = eval_op(op_eid, a, b)
            except ValueError:
                continue
            seq = judge_sequence(nested_seq([a, b], op_eid, total), True)
            samples.append(make_sample(seq, True, 1))
            npos += 1
            if neg_mode:
                seq_b = judge_sequence(nested_seq([a, b], op_eid, total + 1), False)
                samples.append(make_sample(seq_b, False, 1))
                nneg += 1
    return samples, npos, nneg


@register_sample("cartesian_ood")
def _s_cartesian_ood(spec, seed):
    return cartesian_ood_samples(ops=spec["ops"], ndigits=spec.get("ndigits", 2),
                                 n=spec.get("n", 20), neg_mode=spec.get("neg_mode", 1),
                                 seed=seed)


def radix_ood_samples(*, ops, bases=(3, 4, 5, 6, 7, 8, 9), max_digits=20,
                      n=20, neg_mode=1, seed=0):
    """进制 × 结果位数联合 OOD (用户确立): 进制 3-9 分别测试, 结果 ≤ max_digits 位.

    用户约束: max_digits 是运算**结果**位数上限 (非输入), 防高阶算子溢出.
    操作数在 base 进制下采样 (数位 ∈ [0, base-1]), 结果超 max_digits 位
    → ValueError 跳过 (溢出). 判定序列沿 nested_seq (base 进制组装).
    返回 (样本列表, 正例数, 负例数).
    """
    import random
    rng = random.Random(seed)
    samples, npos, nneg = [], 0, 0
    for base in bases:
        max_val = base ** max_digits - 1
        # 操作数限制在 5 位内 (结果才可能达 max_digits 位; 高阶算子防溢出)
        hi = base ** 5 - 1
        for op in ops:
            op_eid = api.eid_by_name(op) if not str(op).startswith("D:") else op
            for _ in range(n):
                a = rng.randint(0, hi)
                b = rng.randint(0, hi)
                try:
                    total = eval_op(op_eid, a, b, max_val=max_val, base=base)
                except ValueError:
                    continue
                seq = judge_sequence(nested_seq([a, b], op_eid, total, base=base), True)
                samples.append(make_sample(seq, True, 1))
                npos += 1
                if neg_mode:
                    try:
                        seq_b = judge_sequence(nested_seq([a, b], op_eid, total + 1, base=base), False)
                    except ValueError:
                        continue
                    samples.append(make_sample(seq_b, False, 1))
                    nneg += 1
    return samples, npos, nneg


@register_sample("radix_ood")
def _s_radix_ood(spec, seed):
    return radix_ood_samples(ops=spec["ops"], bases=spec.get("bases", (3, 4, 5, 6, 7, 8, 9)),
                             max_digits=spec.get("max_digits", 20), n=spec.get("n", 20),
                             neg_mode=spec.get("neg_mode", 1), seed=seed)

