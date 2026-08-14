"""tokenizer/api.py —— tokenizer 公共接口封装 (外部模块唯一入口)

样本合成器等 tokenizer 外模块只经本层, 不碰内部 (私有注册表/子模块)。
能力:
  元数据   name(eid), eid_by_name, derives_of(glyph), all_concepts
  组装     assemble(concept, children) → AST; parse(notation); print_ast(ast)
  概念     concepts(ast) → 概念 eid 序列 (去重保序)
  向量     bracket(eid), brace(eid), counts(eid)
  注意力   analyze(seq), head_run/pipeline, Head/MultiHead/batch_run
"""
from __future__ import annotations

from collections import Counter

from ._register import token_of, token_eid, resolve_derives, all_eids, is_derived
from .grammar import (
    assemble as _assemble, parse as _parse, print_ast as _print_ast,
    query as _query, scope as _scope,
)
from .construct.expand import bracket_vec, brace_logic, brace_derived
from .head.attention import analyze as _analyze, depth_map, depth_weight
from .head.router import run as head_run, pipeline as head_pipeline
from .head.heads import Head, MultiHead, batch_run
from .eval import (
    eval_digit as _eval_digit, eval_logic as _eval_logic,
    eval_bool_expr as _eval_bool_expr, logic_truth as _logic_truth,
    digit_cardinality as _digit_cardinality,
    digits_to_numeral as _digits_to_numeral, numeral_to_digits as _numeral_to_digits,
    eval_compare as _eval_compare, compare_truth as _compare_truth,
    value_number as _value_number, eval_numeral as _eval_numeral,
    eval_digit_value as _eval_digit_value, value_token as _value_token,
    valid_digits as _valid_digits, numeral_of as _numeral_of,
    iterate_from_base as _iterate_from_base,
    is_prime as _is_prime, primes_up_to as _primes_up_to,
    eval_reciprocal as _eval_reciprocal, eval_division as _eval_division,
    eval_power as _eval_power, eval_root as _eval_root,
    eval_complement as _eval_complement, eval_parallel_sum as _eval_parallel_sum,
    verify_laws as _verify_laws,
    eval_differential as _eval_differential, eval_integral as _eval_integral,
    eval_imaginary as _eval_imaginary, eval_log as _eval_log,
    eval_translation as _eval_translation, eval_inversion as _eval_inversion,
    eval_exp as _eval_exp, eval_iterate as _eval_iterate, eval_fixpoint as _eval_fixpoint,
    eval_rotation as _eval_rotation, eval_tetration as _eval_tetration,
    eval_super_root as _eval_super_root, eval_super_log as _eval_super_log,
    eval_coupled_fixpoint as _eval_coupled_fixpoint,
    eval_scale as _eval_scale, eval_recursion as _eval_recursion,
)
from .eval.drift_verify import (
    iterate_fixpoint as _iterate_fixpoint, drift_verify as _drift_verify,
    report as _drift_report,
)
from .eval.engine import (
    eval_op as _engine_eval_op,
    load_ops as _engine_load_ops,
    load_all_ops as _engine_load_all_ops,
)
from .role import role_token as _role_token
from .eval.sample_eval import (
    judge_seq as _judge_seq,
    definition_truth as _definition_truth,
    arrow_truth as _arrow_truth,
    sample_definition as _sample_definition,
    sample_arrow as _sample_arrow,
    logic_candidates as _logic_candidates,
    digit_candidates as _digit_candidates,
    value_token as _value_token,
    digit_tokens as _digit_tokens,
    arrow_endpoints as _arrow_endpoints,
    op_domain as _op_domain,
)


# ---- 通用求值器 (真值由 token 定义提供, 零硬编码) ----
def eval_digit(digit_eids, base=10):
    """数字求值: 数位序列 (digit eid, 低→高) → 数值 (位权合成)。"""
    return _eval_digit(digit_eids, base)


def eval_logic(op, args):
    """逻辑求值: 算子 eid + bool 参数 → bool (沿 definition.rules 真值表)。"""
    return _eval_logic(op, args)


def logic_truth(op, arg_tokens):
    """逻辑真值查询: 算子 eid + 操作数 truth token → bool|None (定义匹配)。"""
    return _logic_truth(op, arg_tokens)


def eval_bool_expr(expr):
    """递归求值逻辑表达式树 (嵌套运算)。"""
    return _eval_bool_expr(expr)


def digit_cardinality(digit_eid):
    """数符 → 基数 (0-9, 沿 symbol maps_to)。"""
    return _digit_cardinality(digit_eid)


def digits_to_numeral(digit_eids, base=10):
    """数位序列 → 数值 (位权合成)。"""
    return _digits_to_numeral(digit_eids, base)


def numeral_to_digits(value, base=10):
    """数值 → 数位序列 (digit eid, 低→高)。"""
    return _numeral_to_digits(value, base)


def eval_compare(op, a, b, base=10):
    """比较求值: 算子 eid + 数值 → bool (真值表定义驱动)。"""
    return _eval_compare(op, a, b, base)


def compare_truth(op, digit_a, digit_b):
    """比较真值查询: 算子 eid + 两位 digit token → bool|None。"""
    return _compare_truth(op, digit_a, digit_b)


# ---- 逻辑层 numeral 查询/生成原语 (单流向下, 下游零 tokenizer 逻辑) ----
def value_number(eid: str) -> int:
    """value token → 数值 (沿 arrow 链, 零查名). 查询原语 (核验)."""
    return _value_number(eid)


def value_token(n: int) -> str:
    """数值 n → value token eid (沿 arrow 链反向, 零查名). 查询原语 (采样/构造)."""
    return _value_token(n)


def valid_digits(base: int) -> list[str]:
    """某进制下可用 value token 列表 (候选域, 查询原语)."""
    return _valid_digits(base)


def eval_numeral(numeral_seq, base=10) -> int:
    """numeral 结构序列 → 值 (沿 arrow 链 + 位权, 零查名). 查询原语 (核验)."""
    return _eval_numeral(numeral_seq, base)


def eval_digit_value(digit_eid: str) -> int:
    """digit token → 数值 (沿 symbol maps_to → arrow 链). 查询原语 (核验)."""
    return _eval_digit_value(digit_eid)


def numeral_of(digits_spec, base_spec=None, sign_eid=None):
    """numeral 生成 (声明 → 序列, 沿 gtoken 组装). 生成原语.

    digits_spec: value token 列表 (高→低). base_spec: 进制 value token 列表或 None=10."""
    return _numeral_of(digits_spec, base_spec, sign_eid)


def iterate_from_base(k: int):
    """数字 k 的正确计算路径: 从基点迭代 succ 应用 k 次 (λ 演算数字).

    这是正确路径 (完整计算语义), 不是快路径 — 快路径 (digit 排序还原位序)
    是它的简写/投影, 语义由本路径定义. 返回 succ 迭代链 AST."""
    return _iterate_from_base(k)


def is_prime(n, base=10):
    """素数判定 (基数自身迭代定义): 无 b,k≥2 使 b×k=n。"""
    return _is_prime(n, base)


def primes_up_to(limit, base=10):
    """[2, limit] 内全部素数。"""
    return _primes_up_to(limit, base)


# ---- 通用求值引擎 (输入/输出纯原生 token 序列, 零硬编码) ----
def eval_op(op_eid: str, arg_tokens: list[list[str]], *, base: int = 10) -> list[str]:
    """通用求值: 算子 + 操作数 token 序列 → 结果 token 序列.

    铁律 (用户确立): 输入是 token 序列, 输出是 token 序列 — 接口绝不输出
    非 token 值; 唯一允许数字参与计算处 = numeral 快路径 (digit 排序还原
    位序); 真值沿 token 定义推导 (definition.rules), 零硬编码算子名/eid.
    arg_tokens: 每操作数的 numeral token 序列 或 truth token 序列.
    """
    return _engine_eval_op(op_eid, arg_tokens, base=base)


def load_ops(op_eids: list[str]) -> int:
    """统一加载器 batch load: 一次预编译多个算符定义 (幂等, 数据变化重编译)."""
    return _engine_load_ops(op_eids)


def load_all_ops() -> int:
    """统一加载器: 预编译全部可求值算子."""
    return _engine_load_all_ops()


# ---- 组装角色原语 (沿 token 结构识别, 零名字硬编码) ----
def role_token(kind: str):
    """组装角色 token eid: judge/equals/bracket/truth/is_true/numeral/
    sign_part/base/digit_seq/sign. 沿 token 结构 (arrange/P 层/定义) 识别,
    加 token/改字段零适配; 只有加 token 类别 (架构) 才适配."""
    return _role_token(kind)


# ---- 对称变换家族 (迭代层对偶, 分数表达) ----
def eval_reciprocal(a):
    """reciprocal(a) = 1/a (层极性对偶 (乘法层), 反射@1)。"""
    return _eval_reciprocal(a)


def eval_division(a, b):
    """division(a, b) = a/b = a × reciprocal(b) (分数代数读法)。"""
    return _eval_division(a, b)


def eval_power(a, b):
    """power(a, b) = a^b (正方向 (乘法迭代))。"""
    return _eval_power(a, b)


def eval_root(a, b):
    """root(a, b) = a^(1/b) (层对偶, 单位分割/新基数轴)。"""
    return _eval_root(a, b)


def eval_complement(a):
    """complement(a) = 1 - a (单位区间对偶, 反射@1/2, 测度读法)。"""
    return _eval_complement(a)


def eval_parallel_sum(a, b):
    """parallel_sum(a, b) = 1/(1/a + 1/b) (加法在乘法层对偶下的 De Morgan 对偶)。"""
    return _eval_parallel_sum(a, b)


def verify_laws(*args):
    """沿 definition.rules 逐条验证对称家族定律 (真值由 token 定义提供)。"""
    return _verify_laws(*args)


def eval_differential(x, n):
    """微分 = 降层算子: differentiate(x,n) = n·x^(n-1) (提取次数, 降层)。"""
    return _eval_differential(x, n)


def eval_integral(x, n):
    """积分 = 升层算子: integrate(x,n) = x^(n+1)/(n+1) (升层, 除以新次数)。"""
    return _eval_integral(x, n)


def eval_imaginary():
    """复数单位 i = root(neg(1), 2) (命名表达式, 非新基数)。"""
    return _eval_imaginary()


def eval_log(a, x):
    """log_a(x): 幂的第二逆 (固定底数, 解指数), 测量迭代深度. 降层律 log(x·y)=log(x)+log(y)。"""
    return _eval_log(a, x)


def eval_translation(x):
    """平移 (模群生成元 T): x → x+1 = complement∘neg。"""
    return _eval_translation(x)


def eval_inversion(x):
    """反演 (模群生成元 S): x → -1/x = reciprocal∘neg, 对合。"""
    return _eval_inversion(x)


def eval_exp(x):
    """exp(x): 微分算子不动点 (自指 d/dx exp = exp). e = exp(1)。"""
    return _eval_exp(x)


def eval_iterate(x, n):
    """iterate(x, n): Church 自指迭代, 加1应用 n 次 = x+n (数 = 迭代步数)。"""
    return _eval_iterate(x, n)


def eval_fixpoint(x):
    """fixpoint(x): 自指迭代到不动点 (自指平均迭代收敛到 x)。"""
    return _eval_fixpoint(x)


def eval_rotation(x):
    """rotation(x): 90° 旋转 = 乘 i (imaginary), 四次旋转恒等 (i⁴=1)。"""
    return _eval_rotation(x)


def eval_tetration(a, b):
    """tetration(a, b): 超幂 (幂迭代), a↑↑b = 幂的自指迭代。"""
    return _eval_tetration(a, b)


def eval_super_root(x, b):
    """super_root(x, b): 解 a↑↑b = x 的底数 (层对偶)。"""
    return _eval_super_root(x, b)


def eval_super_log(a, x):
    """super_log(a, x): 解 a↑↑b = x 的迭代次数 (层对偶)。"""
    return _eval_super_log(a, x)


def eval_coupled_fixpoint(a, b):
    """coupled_fixpoint(a, b): 耦合不动点, 解 x = a + b·x (多自指交互稳定态)。"""
    return _eval_coupled_fixpoint(a, b)


def eval_scale(x, n):
    """scale(x, n): 张缩/幂放缩 (对称家族 neg/scale/root), x^n。"""
    return _eval_scale(x, n)


def eval_recursion(x, n):
    """recursion(x, n): 递归 (结构自指) = iterate(x, n)。"""
    return _eval_recursion(x, n)


# ---- 标准迭代不动点漂移验证 (直接换基元, 算法不变, 无映射) ----
def drift_verify(g, base_old=0.0, base_new=None):
    """换基元漂移检测: 同一迭代 g, 从基元 0 与 e 出发, 比较收敛点漂移。"""
    if base_new is None:
        from math import e as _e
        base_new = _e
    return _drift_verify(g, base_old, base_new)


def fixpoint_drift_report():
    """运行全部标准迭代的漂移检测, 返回报告行。"""
    return _drift_report()


# ---- 原生样本基础设施 (真值从 token 来, 零硬编码) ----
def judge_seq(prop, truth):
    """判定序列: [is_true][命题][truth_true|false] (token 原生组装)。"""
    return _judge_seq(prop, truth)


def definition_truth(op, arg_tokens):
    """定义真值: 沿 op 定义 rules 还原 (真值从 token 定义来)。"""
    return _definition_truth(op, arg_tokens)


def arrow_truth(concept, source, target):
    """arrow 真值: 存在 arrow(concept, source→target)? (查 A 层字段, 非 name 正则)。"""
    return _arrow_truth(concept, source, target)


def sample_definition(op, arg_tokens):
    """单个定义样本: 操作数 token → (判定序列, 真值); 定义未覆盖 → (None, None)。"""
    return _sample_definition(op, arg_tokens)


def sample_arrow(concept, source, target):
    """单个 arrow 样本: 概念+两端 → (判定序列, 真值); 真值 = arrow 存在性。"""
    return _sample_arrow(concept, source, target)


def logic_candidates():
    """逻辑操作数位可填 token: [truth_true, truth_false] (tokenizer 候选域)。"""
    return _logic_candidates()


def digit_candidates():
    """数字操作数位可填 token 全集 (digit_0..9, tokenizer 候选域)。"""
    return _digit_candidates()


def value_token(n):
    """数字 n (0-9) → value token eid (glyph 解析, 数据驱动)。"""
    return _value_token(n)


def digit_tokens(n):
    """数字 n → digit token 序列 (低→高, 数据驱动)。"""
    return _digit_tokens(n)


def arrow_endpoints(concept):
    """arrow 概念 → 端点候选 token 集 + 现有箭头对 (tokenizer 候选域)。"""
    return _arrow_endpoints(concept)


def op_domain(op):
    """算子域判定: 逻辑 (bool 操作数) / 数字 (沿 arrange 特征, 零硬编码算子名)。"""
    return _op_domain(op)


def upper_closure(eid) -> set:
    """上层闭包 (tokenizer 原生): 被实验 token 按定义 + arrow 完全展开后的上层集合.

    沿 definition references (bracket) + A 层箭头关联 (concept/source/target)
    递归展开, 得该 token 的全部上层 token — 实验配置据此自动推导需充分训练的
    token 集 (用户: 被实验 token 按定义和 arrow 展开的上层都要充分训练).
    """
    from ._register import ARROW_REGISTRY
    seen, stack = set(), [eid]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        for r in bracket(cur):
            if r not in seen:
                stack.append(r)
        for td in ARROW_REGISTRY.values():
            if td.concept == cur:
                for x in (td.source, td.target):
                    if x not in seen:
                        stack.append(x)
            elif td.source == cur or td.target == cur:
                if td.concept not in seen:
                    stack.append(td.concept)
    return seen


def assemble_seq(concept, children):
    """概念 → token 序列 (tokenizer 原生, 沿 arrange→gtoken→ptoken 组装, 零硬编码列表).

    优先沿 ptoken grammar (符号层+槽位): 槽位 arg:N 填 children[N] (可为子序列),
    符号 glyph 解析为 token (resolve_glyph); 无 ptoken 沿 gtoken 排列 (arg:N 槽位);
    atom (无槽位) 直接展平 children. 样本生成据此组装, 不手写序列.
    """
    from ._register import resolve_glyph

    def _tok(x, prefer=None):
        if isinstance(x, str) and x.startswith(("B:", "D:", "S:", "G:", "P:", "A:")):
            return x
        hits = resolve_glyph(x) if isinstance(x, str) else []
        if hits:
            s_tok = hits[0]
            maps = token_of(s_tok).maps_to if hasattr(token_of(s_tok), "maps_to") else {}
            if maps:
                if prefer in maps:
                    return prefer   # 多映射歧义: 优先 ptoken 的 concept
                return max(maps, key=maps.get)
            return s_tok
        return x

    def _children(idx):
        return children[idx] if idx < len(children) else []

    ptok = presentation_of(concept)
    if ptok:
        out = []
        for x in ptok["grammar"]:
            if isinstance(x, str) and x.startswith("arg:"):
                out.extend(_children(int(x[4:])))
            else:
                out.append(_tok(x, concept))
        return out
    arrange = arrange_of(concept)
    if arrange and arrange not in ("atom",):
        # gtoken 排列 (槽位序列): fn=函数(概念自身), args=参数列表(全 children),
        # arg:N=第N child, eid 字面量=排列固定 token (如 question/gap/equals), 其他槽位名不输出
        out = []
        for x in arrange_slots(arrange):
            if x == "fn":
                out.append(concept)
            elif x == "args":
                for ch in children:
                    out.extend(ch)
            elif isinstance(x, str) and x.startswith("args:"):
                for ch in children[int(x[5:]):]:
                    out.extend(ch)
            elif isinstance(x, str) and x.startswith("arg:"):
                out.extend(_children(int(x[4:])))
            elif isinstance(x, str) and x.startswith(("B:", "D:", "S:", "G:", "P:", "A:")):
                out.append(x)
        return out
    out = []
    for ch in children:
        out.extend(ch)
    return out


def arrange_slots(arrange):
    """gtoken 名 → 排列槽位序列 (读 gtoken definition.rules.term). CTE 缓存."""
    from . import cte
    from .maintain import core

    def _sig():
        # 签名: 全局数据版本 (maintain 写入递增, 无需每项 stat)
        return cte._version()

    def _compile():
        out = []
        for e, f in core.load_all().items():
            if f.get("_layer") == "G" and f.get("name") == arrange:
                rules = (f.get("definition") or {}).get("rules") or []
                out = rules[0].get("term") if rules else []
                break
        return out

    return cte.get_or_compile("slots:" + arrange, _sig(), _compile)


def assemble_ast(node):
    """嵌套 AST → token 序列 (tokenizer 原生, 沿 gtoken/ptoken 递归组装, 零手动拼装).

    AST: [概念eid, child...] (child 为 AST 或单 token eid); token 序列 list 直接展开.
    拼接结构 (应用/复合/等式嵌套) 完全由各概念的 gtoken/ptoken 排列递归完成 —
    代码只提供概念结构数据, 不写嵌套拼装.
    """
    if isinstance(node, str):
        return [node]
    if isinstance(node, list) and node and isinstance(node[0], str) and node[0].startswith(("B:", "D:", "A:")):
        return assemble_seq(node[0], [assemble_ast(ch) for ch in node[1:]])
    out = []
    for ch in node:
        out.extend(assemble_ast(ch))
    return out


def _instantiate_law(node, bindings):
    """定律规则实例化 (tokenizer 原生): proposition 结构槽位 arg:N → bindings[N],
    概念应用沿 gtoken/ptoken 组装 (assemble_seq), 零手动拼装. 返回 token 序列."""
    if isinstance(node, str):
        if node.startswith("arg:"):
            b = bindings.get(int(node[4:]))
            return list(b) if isinstance(b, (list, tuple)) else [b]
        return [node]
    if isinstance(node, list) and node:
        head = node[0]
        if isinstance(head, str) and head.startswith("arg:"):
            return _instantiate_law(head, bindings)
        parts = [_instantiate_law(ch, bindings) for ch in node[1:]]
        return assemble_seq(head, parts)
    return [node]


def category_law_samples() -> list[list[str]]:
    """范畴论定律样本 (token 定义驱动, 零硬编码): 遍历全部 C 概念 definition.rules
    的 proposition (定律结构在 token: identity 单位元律 / inverse 对偶律 / composition
    关联律 / functor 保复合 / isomorphism 可逆 / category 封闭 / natural_transformation
    自然性), 槽位 arg:N → 态射 (A 层箭头 concept), 判定 = judge_seq.
    返回判定序列列表.
    """
    from ._register import ARROW_REGISTRY
    from .eval.sample_eval import judge_seq as _judge_seq
    morphs = sorted({td.concept for td in ARROW_REGISTRY.values()})
    out = []
    for e in all_eids():
        if not e.startswith("D:"):
            continue
        d = token_of(e).definition
        if not isinstance(d, dict):
            continue
        for rule in (d.get("rules") or []):
            prop = rule.get("proposition")
            if not prop:
                continue
            nargs = max(_props_args(prop), default=-1) + 1
            combos = __import__("itertools").product(morphs, repeat=min(nargs, 2))
            for b in combos:
                bindings = dict(enumerate(b))
                seq = _instantiate_law(prop, bindings)
                out.append(_judge_seq(seq, True))
    return out


def _props_args(node):
    """命题结构中的 arg:N 槽位 (递归收集)."""
    if isinstance(node, str):
        if node.startswith("arg:"):
            yield int(node[4:])
        return
    if isinstance(node, list):
        for ch in node:
            yield from _props_args(ch)


def iteration_depth(op_eid) -> int:
    """迭代层数 (tokenizer 原生, 定义驱动): 层数 = 沿定义引用链的迭代基础深度.

    层数由 token 定义结构推导, 零硬编码数值: 沿 references 递归找"算符迭代算符"
    链 (succ/pred/iterate 为迭代原子基, 非算符), 取**最深**迭代基础 (防定义规则
    里混入辅助算符, 如 tetration 规则含 addition 但迭代基础是 power):
    addition 基于 succ (层1), multiplication 基于 addition (层2),
    power 基于 multiplication (层3), tetration 基于 power (层4).
    """
    succ = eid_by_name("succ"); pred = eid_by_name("pred")
    it = eid_by_name("iterate")
    atoms = {succ, pred, it}

    def _depth(cur, seen):
        if cur in seen:
            return 1
        bases = [r for r in bracket(cur)
                 if r not in atoms and arrange_of(r) == "application"]
        if not bases:
            return 1
        return 1 + max(_depth(b, seen | {cur}) for b in bases)

    return _depth(op_eid, set())


def direction_ops() -> list[dict]:
    """全方向算子 (候选域原语): 沿方向体系自动发现, 零硬编码算子名.

    语义 (用户确立): 数学算符 ≠ 方向 token。算符定义 = 数学计算 token +
    方向 token + grammar (arrange) 三元组。方向 token 是纯维度 (positive 是
    "正方向"维度, 不 claim 加法或乘法; 各迭代层由算符的完整定义区分:
    addition=succ迭代+positive, multiplication=addition迭代+positive+increase)。
    歧义防范: direction_ops 只发现"完整定义 (含计算+方向) 的算符", 方向 token
    (arrange=atom) 本身永不作为算子候选。

    方向集: 种子 {direction, relation, iteration_layer} 上行闭包, 方向 token =
    arrange=atom 且 references 含方向集的 C 概念 (positive/increase 引用
    direction; negative/decrease 引用其逆元; successor/predecessor 引用 relation)。
    算子 = arrange∈{application,unary_connective,binary_connective} 且 references
    直含方向集 (不沿算子间引用传递 — 避免 power→multiplication 膨胀)。
    返回 [{eid, name, directions: [方向token名]}], 按 eid 排序。
    """
    seeds = {eid_by_name(n) for n in ("direction", "relation", "iteration_layer")}
    dir_set = set(seeds)
    changed = True
    while changed:
        changed = False
        for e in all_eids():
            if e in dir_set:
                continue
            d = token_of(e).definition
            if not isinstance(d, dict) or d.get("arrange") != "atom":
                continue
            refs = d.get("references") or []
            if refs and any(r in dir_set for r in refs):
                dir_set.add(e)
                changed = True
    ops = {"application", "unary_connective", "binary_connective"}
    out = []
    for e in sorted(all_eids()):
        d = token_of(e).definition
        if not isinstance(d, dict) or d.get("arrange") not in ops:
            continue
        dir_refs = [token_of(r).name for r in (d.get("references") or []) if r in dir_set]
        if dir_refs:
            out.append({"eid": e, "name": token_of(e).name, "directions": dir_refs})
    return out


# ---- 元数据 ----
def name(eid: str) -> str:
    try:
        return token_of(eid).name
    except KeyError:
        from .maintain import core
        fields = core.load_all().get(eid)
        return fields.get("name", eid) if fields else eid


def eid_by_name(n: str) -> str:
    try:
        return token_eid(n)
    except KeyError:
        # itoken (I 层): 接口 token 不进 _register 索引, 按名查 itoken 层 (只读计算)
        from .eval.itoken_eval import _itokens
        for eid, t in _itokens().items():
            if t.get("name") == n:
                return eid
        raise


def derives_of(glyph: str) -> list[str]:
    """glyph → 候选概念 eid (多义候选, 不取首项)。"""
    return resolve_derives(glyph)


def all_concepts() -> dict:
    """全部 C 层概念 {eid: name} (供合成器枚举可用概念)。"""
    return {e: token_of(e).name for e in all_eids() if e.startswith("D:")}


def eid_space() -> list[str]:
    """全 eid 空间 (B/D/S/G, 排序) — 向量维度依据。"""
    return sorted(all_eids())


_ARRANGE_CACHE: dict = {}


def arrange_of(eid: str) -> str | None:
    """概念 eid → arrange 指向的 gtoken 名 (概念→节点类型映射, 无则 None). 结果缓存."""
    hit = _ARRANGE_CACHE.get(eid)
    if hit is not None:
        return hit
    defn = token_of(eid).definition
    out = defn.get("arrange") if isinstance(defn, dict) else None
    if len(_ARRANGE_CACHE) < 10000:
        _ARRANGE_CACHE[eid] = out
    return out


def is_concept(eid: str) -> bool:
    """是否为 C 层概念 (derive)。"""
    return is_derived(eid)


def digit_concepts() -> dict:
    """数值概念 → {n: eid}, 全数据驱动 (无概念名硬编码)。

    识别: C 层 inductive + atom 概念, 且被单字符数字符号 maps_to (数字集 = 符号→概念).
    排序: 沿符号 glyph 的数字值 (字符属性, 非 token 名).
    """
    from ._register import SYMBOL_REGISTRY, is_derived
    out = {}
    for sid, sym in SYMBOL_REGISTRY.items():
        g = getattr(sym, "glyph", "") or ""
        if len(g) == 1 and g.isdigit() and g.isascii():
            for target in (getattr(sym, "maps_to", None) or {}):
                if not is_derived(target):
                    continue
                defn = token_of(target).definition
                if isinstance(defn, dict) and defn.get("form") == "inductive" and defn.get("arrange") == "atom":
                    out[int(g)] = target
    return {n: out[n] for n in sorted(out)}


def peers(eid: str, threshold=0.3) -> list[str]:
    """平行 ctoken: 与 eid 归属同一定义的兄弟概念 (定义引用 Jaccard ≥ 阈值)。

    充分训练前提: 每个 ctoken 须有归属于同一定义的平行 ctoken 同时训练,
    沿定义链平行训练后组合泛化 (如训 = < > ≥ → 泛化 ≤)。
    threshold 过滤: 共享引用比率 (同族概念高相似, 无关概念排除)。
    """
    refs = set(bracket(eid))

    def _sim(r):
        if not refs and not r:
            return 0.0
        return len(refs & r) / len(refs | r)

    return [e for e in all_concepts() if e != eid and _sim(set(bracket(e))) >= threshold]


# ---- 箭头 (A 层: 两个被抽象 ctoken 的有向关系, concept = atoken 等价 ctoken) ----
def all_arrows() -> dict:
    """全部箭头 {eid: name} (A 层)。"""
    from ._register import ARROW_REGISTRY
    return {e: td.name for e, td in ARROW_REGISTRY.items()}


def is_arrow(eid: str) -> bool:
    """eid 是否为 A 层箭头 (两个 ctoken 的有向关系)。"""
    from ._register import ARROW_REGISTRY
    return eid in ARROW_REGISTRY


def source_of(eid: str) -> str:
    """箭头 eid → source ctoken (被抽象的 source 材料)。"""
    from ._register import ARROW_REGISTRY
    return ARROW_REGISTRY[eid].source


def target_of(eid: str) -> str:
    """箭头 eid → target ctoken (被抽象的 target 材料)。"""
    from ._register import ARROW_REGISTRY
    return ARROW_REGISTRY[eid].target


def arrow_concept_of(eid: str) -> str:
    """箭头 eid → atoken 等价 ctoken (箭头概念化)。"""
    from ._register import ARROW_REGISTRY
    return ARROW_REGISTRY[eid].concept


def arrows_of(concept: str) -> list[str]:
    """概念 eid → 参与的全部箭头 (concept/source/target 任一), 数据驱动。"""
    from ._register import ARROW_REGISTRY
    return [e for e, td in ARROW_REGISTRY.items()
            if td.concept == concept or td.source == concept or td.target == concept]


def arrows_by_concept(concept: str) -> list[str]:
    """概念 eid → 以该概念为 atoken 等价 ctoken 的箭头 (箭头概念化检索)。"""
    from ._register import ARROW_REGISTRY
    return [e for e, td in ARROW_REGISTRY.items() if td.concept == concept]


def coercion_arrows() -> list[dict]:
    """数域提升箭头 (原生原语, token 驱动零硬编码): A 层 concept=coercion 的单态射.

    沿箭头概念化检索: 提升箭头 = concept 指向 coercion 概念 (D:216) 的 arrow,
    即范畴论数域提升链 Nat↪Int↪Rat↪Real↪Complex 的 A 层实例.
    返回 [{eid, source, target}] (source=低域, target=高域), 按 eid 排序.
    """
    c = eid_by_name("coercion")
    from ._register import ARROW_REGISTRY
    out = [{"eid": e, "source": td.source, "target": td.target}
           for e, td in ARROW_REGISTRY.items() if td.concept == c]
    return sorted(out, key=lambda d: d["eid"])


def _digit_digits(n):
    """数值 n → 数符序列 (digit token, numeral 表示逻辑, 低→高)."""
    if n == 0:
        return [digit_candidates()[0]]
    out = []
    while n:
        n, d = divmod(n, 10)
        out.append(_digit_candidates()[d])
    return out


def sample_coercion(n, arrow_eid, truth=True):
    """数域提升判定样本 (tokenizer 原生, 零硬编码): 值 n 从低域提升到高域.

    序列: [is_true][source 数符][coercion 箭头][target 表示][truth].
    target 表示法沿数域表示 token (token 驱动): integer=同数符; rational=[n][∕][1];
    real=[√][n²] (n 的平方是完美平方, 可表示); complex=[n][+][0][i].
    truth=False 时生成假样本 (错误提升): 目标表示某位错 (rational 分子 n+1,
    real 非完美平方, complex 实部 n+1) — 判定有真假的样例, 防无脑猜真.
    返回 {seq, valid, depth, truth}.
    """
    from ._register import ARROW_REGISTRY
    from .role import role_token
    arrow = ARROW_REGISTRY[arrow_eid]
    src, tgt = arrow.source, arrow.target
    is_true = role_token("is_true")
    _TRUTH = role_token("truth")
    tr = _TRUTH[0]
    fa = _TRUTH[1]
    digits = _digit_digits(n)
    m = n + 1  # 错误值 (假样本用)
    if tgt == eid_by_name("integer_domain"):
        rep = digits if truth else _digit_digits(m)
    elif tgt == eid_by_name("rational_domain"):
        rep = (digits + [eid_by_name("fraction"), eid_by_name("value_one")] if truth
               else digits + [eid_by_name("fraction"), _digit_digits(2)[0]])
    elif tgt == eid_by_name("real_domain"):
        rep = [eid_by_name("radical")] + (_digit_digits(n * n) if truth else _digit_digits(n * n + 1))
    elif tgt == eid_by_name("complex_domain"):
        rep = (digits + [eid_by_name("addition"), eid_by_name("value_zero"), eid_by_name("imaginary_unit")]
               if truth else _digit_digits(m) + [eid_by_name("addition"), eid_by_name("value_zero"), eid_by_name("imaginary_unit")])
    else:
        raise ValueError(f"未知目标数域: {tgt}")
    seq = [is_true] + digits + [arrow_eid] + rep + [tr if truth else fa]
    return {"seq": seq, "valid": 1, "depth": 1, "truth": truth}


# ---- 组装 / 渲染 / 解析 ----
def assemble(concept, children):
    """概念 eid → 嵌套 AST (全正向)。"""
    return _assemble(concept, children)


def parse(notation: str):
    """记法 → AST (呈现层驱动, 反向调试用)。"""
    return _parse(notation)


def print_ast(ast) -> str:
    """AST → 记法字符串。"""
    return _print_ast(ast)


def query(node) -> dict:
    """查询节点排列方法 (gtoken 全字段, 含槽位 definition)。"""
    return _query(node)


def scope(node) -> dict:
    """可组装范围: 槽位布局 (arg:N/fn/args/binder/body)。"""
    return _scope(node)


def presentation_of(eid: str) -> dict | None:
    """概念 eid → 呈现定义 {concept, grammar, precedence, associativity} (无呈现则 None). CTE 缓存."""
    from . import cte
    from .maintain import core

    def _sig():
        return cte._version()

    def _compile():
        out = None
        for p in core.load_layer("P").values():
            if p.get("concept") == eid:
                out = {
                    "concept": eid,
                    "grammar": p.get("grammar", []),
                    "precedence": p.get("precedence"),
                    "associativity": p.get("associativity", "left"),
                }
                break
        return out

    return cte.get_or_compile("pres:" + eid, _sig(), _compile)


# ---- 概念序列 ----
def concepts(ast) -> list[str]:
    """AST → 概念 eid 序列 (去重保序; fn 前置冗余跳过; 原子 AST=str 直接返回)。"""
    if isinstance(ast, str):
        return [ast] if is_concept(ast) else []
    out = []

    def walk(node):
        if node.get("concept"):
            out.append(node["concept"])
        children = node.get("children", [])
        if "fn" in node.get("slots", []):
            children = children[1:]
        for c in children:
            if isinstance(c, dict):
                walk(c)
            elif isinstance(c, str) and c.startswith("D:"):
                out.append(c)

    walk(ast)
    seen, seq = set(), []
    for e in out:
        if e not in seen:
            seen.add(e)
            seq.append(e)
    return seq


# ---- 向量 ----
def bracket(eid: str) -> list[str]:
    """[x] 中括号向量: 一层定义引用。"""
    return bracket_vec(eid)


def brace(eid: str) -> list[str]:
    """{x} 大括号向量: 穿透到公设的溯源链。"""
    return brace_logic(eid)


def counts(eid: str) -> dict:
    """{x} 派生计数 (非零槽)。"""
    return {k: v for k, v in brace_derived(eid).items() if v > 0}


# ---- 注意力 ----
def analyze(sequence) -> list[dict]:
    """概念序列 → 每概念注意力分析向量。"""
    return _analyze(sequence)


__all__ = [
    "name", "eid_by_name", "derives_of", "all_concepts", "eid_space",
    "arrange_of", "is_concept", "digit_concepts", "peers",
    "all_arrows", "is_arrow", "source_of", "target_of", "arrow_concept_of",
    "arrows_of", "arrows_by_concept",
    "assemble", "parse", "print_ast", "concepts", "query", "scope", "presentation_of",
    "bracket", "brace", "counts",
    "analyze", "depth_map", "depth_weight",
    "head_run", "head_pipeline", "Head", "MultiHead", "batch_run",
    "eval_digit", "eval_logic", "logic_truth", "eval_bool_expr",
    "digit_cardinality", "digits_to_numeral", "numeral_to_digits",
    "eval_compare", "compare_truth",
    "value_number", "value_token", "valid_digits", "eval_numeral",
    "eval_digit_value", "numeral_of", "iterate_from_base",
    "is_prime", "primes_up_to",
    "eval_reciprocal", "eval_division", "eval_power", "eval_root",
    "eval_complement", "eval_parallel_sum", "verify_laws",
    "eval_differential", "eval_integral", "eval_imaginary", "eval_log",
    "eval_translation", "eval_inversion", "eval_exp", "eval_iterate",
    "eval_fixpoint", "eval_rotation", "eval_tetration",
    "eval_super_root", "eval_super_log", "eval_coupled_fixpoint",
    "eval_scale", "eval_recursion",
    "eval_op", "load_ops", "load_all_ops",
    "drift_verify", "fixpoint_drift_report",
    "judge_seq", "definition_truth", "arrow_truth",
    "sample_definition", "sample_arrow", "sample_arrow_token",
    "logic_candidates", "digit_candidates",
    "value_token", "digit_tokens", "arrow_endpoints", "op_domain",
    "direction_ops", "coercion_arrows", "sample_coercion",
    "iteration_depth", "upper_closure", "category_law_samples",
    "assemble_seq", "arrange_slots", "assemble_ast",
]
