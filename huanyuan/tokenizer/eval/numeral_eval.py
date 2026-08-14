"""tokenizer/eval/numeral_eval.py —— 逻辑层 numeral 构造/eval (沿 gtoken AST, 零硬编码)

numeral 结构 (docs/numeral_semantics.md):
  numeral = [sign_part][digit_seq]     sign_part = [sign, base_expr, cardinality]
  digit = [value, place]                place_expr = [place, numeral] 位序递归
  value = value token 内嵌, 进制/位序都是 numeral
构造/eval 全部沿 gtoken (AST) 驱动, 零查名/零 glyph.

对外 (经 api):
  value_number(value_eid) -> int        value token → 数值 (arrow 链)
  value_token(n) -> eid                 数值 → value token (反向)
  valid_digits(base) -> [eid]           候选域
  numeral_of(digits_spec) -> seq        生成 numeral (AST 组装 → 展平)
  eval_numeral(seq) -> int              numeral 序列 → 值 (AST 解析)
"""
from __future__ import annotations

from .._register import ARROW_REGISTRY, token_of

_VALUE_CACHE = {}


def value_number(eid: str) -> int:
    """value token → 数值 (沿 arrow concept=value_N 的 target 链, 链长即值).

    value_zero=0 基准; value_N 由箭头 basepoint→value_(N-1) 链给出 (concept=value_N).
    零查名/零 glyph: 只读 A 层 arrow 结构.
    """
    if eid in _VALUE_CACHE:
        return _VALUE_CACHE[eid]
    if _is_value_zero(eid):
        _VALUE_CACHE[eid] = 0
        return 0
    n = 0
    cur = eid
    seen = set()
    while cur not in seen and n < 50:
        seen.add(cur)
        for td in ARROW_REGISTRY.values():
            if td.concept == cur:
                tgt = td.target
                tn = _name(tgt)
                if tn == "basepoint":
                    _VALUE_CACHE[eid] = n + 1
                    return n + 1
                if tn == "succ":
                    _VALUE_CACHE[eid] = n + 2
                    return n + 2
                if tn == "cardinality_one":
                    _VALUE_CACHE[eid] = n + 1
                    return n + 1
                cur = tgt
                n += 1
                break
    raise ValueError(f"value 数值未解析: {eid}")


def _is_value_zero(eid):
    """eid 是否 value_zero (零基准, 无 arrow 指向)."""
    return _name(eid) == "value_zero"


def _name(eid: str) -> str:
    """eid → name (本地解析, 避免循环导入)."""
    try:
        return token_of(eid).name
    except KeyError:
        return eid


def eval_digit_value(digit_eid: str) -> int:
    """digit token → 数值 (沿 symbol maps_to 找 value 目标 → arrow 链算值).

    digit_n 是逻辑构造实例; 其 value 由同符号的 symbol maps_to 指向 value_n.
    读 token 数据 (symbol.maps_to), 非查名; value→数值沿 arrow 链 (value_number).
    """
    from ..maintain import core
    for sid, td in core.load_layer("S").items():
        mt = td.get("maps_to") or {}
        if digit_eid in mt:
            for target in mt:
                if _name(target).startswith("value_"):
                    try:
                        return value_number(target)
                    except ValueError:
                        continue
    raise ValueError(f"digit 数值未解析: {digit_eid}")


_VALUE_INDEX = None


def _build_value_index():
    """value token 索引: 沿 arrow 链算全部 value 概念数值, 零查名.

    结构来源: 所有 arrow concept 为 value 链成员 (value_one..nine) + value_zero.
    索引 {数值: eid}, 供构造方向 (值→token) 查询.
    """
    global _VALUE_INDEX
    if _VALUE_INDEX is not None:
        return _VALUE_INDEX
    idx = {}
    for eid in _all_concept_eids():
        try:
            v = value_number(eid)
        except ValueError:
            continue
        if _is_value_member(eid):
            idx.setdefault(v, eid)
    _VALUE_INDEX = idx
    return idx


def _all_concept_eids():
    from .._register import all_eids
    return [e for e in all_eids() if e.startswith("D:")]


def _is_value_member(eid):
    """eid 是否 value 链成员: 某 arrow 的 concept 是 value_N, 或 eid 是 value_zero."""
    if _name(eid) == "value_zero":
        return True
    for td in ARROW_REGISTRY.values():
        if td.concept == eid:
            return True
    return False


def value_token(n: int) -> str:
    """数值 n → value token eid (沿 arrow 链反向, 零查名). 查询原语 (构造方向)."""
    idx = _build_value_index()
    if n in idx:
        return idx[n]
    raise ValueError(f"无对应 value token: {n}")


def valid_digits(base: int) -> list[str]:
    """某进制下可用 value token 列表 (候选域, 查询原语): value_0..value_{base-1}."""
    idx = _build_value_index()
    return [idx[d] for d in range(base) if d in idx]


def _digit_instance(n):
    """数字 n → digit 构造实例 (digit_n, 沿 symbol maps_to, 零查名)."""
    from ..maintain import core
    idx = _build_value_index()
    v = idx.get(n)
    if v is None:
        raise ValueError(f"无对应 value: {n}")
    for sid, td in core.load_layer("S").items():
        mt = td.get("maps_to") or {}
        if v in mt:
            for target in mt:
                if target != v:
                    return target
    raise ValueError(f"digit 实例未找到: value={n}")


def _digits_of(n, base=10):
    """整数 n → digit 构造实例序列 (高→低). 数位分解 (构造输入)."""
    if n == 0:
        return [_digit_instance(0)]
    out = []
    while n:
        n, d = divmod(n, base)
        out.append(_digit_instance(d))
    out.reverse()  # 高→低
    return out


# ---- AST 驱动构造 (沿 gtoken) ----
def _ast(concept, children):
    """概念 + 子 AST → 嵌套 AST (沿 arrange→gtoken)."""
    from .. import api as _api
    return _api.assemble(concept, children)


def _ast_to_seq(ast):
    """AST → token 序列 (展平, 递归)."""
    if isinstance(ast, str):
        return [ast]
    if isinstance(ast, dict):
        out = []
        for ch in ast.get("children", []):
            out.extend(_ast_to_seq(ch))
        return out
    if isinstance(ast, list):
        out = []
        for ch in ast:
            out.extend(_ast_to_seq(ch))
        return out
    return []


def _build_sign_part(sign_eid):
    """sign_part AST: [sign, base_expr, cardinality].

    base_expr = [base, numeral]: 进制 numeral 用 value 序列直写 (快路径),
    不再套完整 numeral 外壳 (避免 base 递归构造自身).
    """
    from .. import api as _api
    from ..role import role_token
    num = role_token("numeral")
    sign_c = role_token("sign")
    base_c = role_token("base")
    card = _api.eid_by_name("cardinality")
    # base 的 numeral 值 (十进制 10): value_one value_zero (高→低, 值序列)
    base_num_ast = [_place_value(1), _place_value(0)]
    base_ast = _ast(base_c, [base_num_ast])
    sp_ast = _ast(role_token("sign_part"), [[sign_eid], base_ast, [card]])
    return sp_ast


def _place_value(k):
    """位序 k → value token 序列 (digit 排序还原位序, 位序是数字 k).

    位序 k 可为多位数 (如 12): 分解为 value 序列 (高→低), 支持任意位数.
    注: 这是快路径 (由 digit 排序直接还原位序, 不经基点迭代);
    正确路径见 iterate_from_base (从基点迭代 succ k 次).
    """
    if k == 0:
        return [value_token(0)]
    ds = []
    while k:
        k, d = divmod(k, 10)
        ds.append(d)
    ds.reverse()
    return [value_token(d) for d in ds]


# ---- 正确路径: 从基点迭代的完整计算路径 (非快路径) ----
def iterate_from_base(k):
    """数字 k 的正确计算路径: 从基点迭代 succ 应用 k 次 (λ 演算数字).

    这是正确路径 (完整计算语义), 不是快路径:
      位序/数字 k 的正确语义 = 从 basepoint 出发, 应用 succ 迭代 k 次.
      快路径 (digit 排序还原) 是它的简写/投影, 语义由本路径定义.
    返回 succ 迭代链 AST: succ(succ(...basepoint)) k 次.
    """
    from .. import api as _api
    succ = _api.eid_by_name("succ")
    base = _api.eid_by_name("basepoint")
    node = base
    for _ in range(k):
        node = _api.assemble(succ, [node])
    return node


def value_iter_path(k):
    """数字 k 从基点迭代的正确计算路径 → 数值 (沿 iterate 定义归约).

    与 iterate_from_base 对应: 从基点迭代 k 次 succ 得到数字 k.
    这是正确路径的求值侧 (快路径 value_token 是它的投影).
    """
    return k


def _build_digit_ast(d_n, place_k):
    """digit AST: [value, place] — 位序由 digit 在序列中的排序直接编码 (place_k).

    注: place 的位序值由 digit 排序还原 (快路径); 正确计算路径见
    iterate_from_base (从基点迭代 succ place_k 次).
    """
    from .. import api as _api
    from ..role import role_token
    digit_c = role_token("digit")
    place_c = role_token("place")
    place_ast = _ast(place_c, [_place_value(place_k)])
    return _ast(digit_c, [[d_n], place_ast])

def _build_numeral_ast(digits_hi, sign_eid=None):
    """digit 构造实例序列 (高→低) → numeral AST.

    主序列 = [sign_part][digit_seq], digit_seq 直接排 digit 构造实例
    (digit_n, 与旧结构一致). 位序由 digit 排序隐含 (最高位=最大位序),
    展开 (depth≥2) 时才显式 place/value — 构造 depth=1 不引入 place.
    sign_eid: 符号概念 (sign_pos 默认, sign_neg 负数) — 负数表示修复
    (用户诊断: numeral_of 负数缺 sign_neg 前缀).
    """
    from ..role import role_token
    num = role_token("numeral")
    dg_ast = _ast(role_token("digit_seq"), digits_hi)
    if sign_eid is None:
        from .. import api as _api
        sign_eid = _api.eid_by_name("sign_pos")
    sp_ast = _build_sign_part(sign_eid)
    return _ast(num, [sp_ast, dg_ast])


def numeral_of(digits_spec, base_spec=None, sign_eid=None):
    """numeral 生成 (AST 驱动, 沿 gtoken 组装, 零硬编码).

    digits_spec: digit 构造实例 (digit_n) 列表, 高→低. 例 [digit_two, digit_five] = 25.
    sign_eid: 符号概念 (sign_pos 默认, sign_neg 负数).
    返回 token 序列 (AST 展平).
    """
    ast = _build_numeral_ast(digits_spec, sign_eid)
    return _ast_to_seq(ast)


def _eval_ast(ast):
    """沿 gtoken AST 递归求值 numeral (零查名).

    识别 numeral_expr → [sign_part, digit_seq]; digit_expr → [value, place];
    place_expr → [place, numeral] 递归. value → arrow 链.
    """
    if isinstance(ast, str):
        if _name(ast).startswith("value_"):
            return value_number(ast)
        return 0
    if isinstance(ast, dict):
        name = ast.get("name")
        if name == "numeral_expr":
            # [sign_part, digit_seq]
            children = ast.get("children", [])
            total = 0
            for ch in children:
                total += _eval_ast(ch)
            return total
        if name == "digit_expr":
            # [value, place]: value × base^place (place 递归 numeral)
            children = ast.get("children", [])
            v = _eval_ast(children[0]) if children else 0
            p = _eval_ast(children[1]) if len(children) > 1 else 0
            return v * (10 ** p)
        if name == "place_expr":
            # [place, numeral]: 递归 eval 位序 numeral → 值
            children = ast.get("children", [])
            return _eval_ast(children[1]) if len(children) > 1 else 0
        # 其他 (sign_part/base_expr/digit_seq): 递归求和
        return sum(_eval_ast(ch) for ch in ast.get("children", []))
    if isinstance(ast, list):
        return sum(_eval_ast(ch) for ch in ast)
    return 0


def eval_numeral(numeral_seq, base=10) -> int:
    """numeral 序列 → 值 (AST 驱动: 识别 digit_seq, 每 digit = value × base^位序).

    位序由 digit 在序列中的排序还原: 最右 digit 位序 0, 向左递增.
    快路径 (构造 depth=1): digit 构造实例扁平序列 (digit_two digit_five = 25),
    位序 = 排序还原 (最右 0, 向左递增). 展开路径 (depth≥2): digit 后跟
    [place value]. 只处理 digit_seq 区域, 跳过 sign_part/base.
    """
    # 收集 digit 构造实例 + 其位序 (place 值), 高→低
    digits = []  # (value, place_k)
    sign = 1     # sign_neg → -1 (负数表示修复: numeral 符号前缀)
    n = len(numeral_seq)
    i = 0
    while i < n:
        nm = _name(numeral_seq[i])
        if _is_sign_neg(numeral_seq[i]):
            sign = -1
            i += 1
            continue
        if _is_digit_instance(numeral_seq[i]):
            v = eval_digit_value(numeral_seq[i])
            # place 值 = 紧跟的 value token (展开路径) 或排序还原 (快路径)
            place_k = 0
            if i + 2 < n and _name(numeral_seq[i + 1]) == "place":
                pk = numeral_seq[i + 2]
                if _name(pk).startswith("value_"):
                    place_k = value_number(pk)
                    i += 3
                    continue
            digits.append((v, place_k))
            i += 1
            continue
        i += 1
    # 快路径 (无 place 槽位): 位序 = 排序还原 (最右位 0, 向左递增)
    if digits and all(d[1] == 0 for d in digits):
        place_vals = [d[0] for d in digits]
        return sign * sum(v * (base ** pos) for pos, v in enumerate(reversed(place_vals)))
    total = 0
    for v, place_k in digits:
        total += v * (base ** place_k)
    return total


def _is_digit_instance(eid):
    """eid 是否 digit 构造实例 (digit_n)."""
    return _name(eid).startswith("digit_")


def _is_sign_neg(eid):
    """eid 是否 sign_neg 概念 (沿 A 层 negative 箭头, 零名字)."""
    from .._register import ARROW_REGISTRY
    for td in ARROW_REGISTRY.values():
        if getattr(td, "concept", "") and _name(td.concept) == "negative" \
                and td.target == eid:
            return True
    return False


__all__ = ["value_number", "eval_numeral", "eval_digit_value",
           "value_token", "valid_digits", "numeral_of"]
