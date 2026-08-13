"""maintain/core.py —— 维护/诊断抽象基础设施

token 是一, 多 json 是投影。统一读写/校验/诊断全部基于投影声明,
加新维护工具不重写, token 架构更新 (加层/加字段/加文件) 只改 PROJECTIONS。

设计:
  PROJECTIONS  投影声明: 每层 → {file, eid前缀, 字段列表}  ← 架构更新的唯一落点
  load_all()   所有层 → 统一索引 {eid: 合并字段}             ← 诊断/校验的数据源
  write_token() 按投影声明分发字段到各文件                   ← 维护写入
   layer_of()   层推断 (glyph/maps_to→S, definition 槽位→G, grammar→P, form:axiomatic→B, 否则→C)
  validate()   通用校验 (name 必填, definition 引用已注册)
  ref_of()     简称自动生成 (有 glyph→[glyph], 无→[name], 歧义下标 eid)
"""
import json
import os
import re

_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tokens")  # tokenizer/tokens/

# 投影声明: 架构更新的唯一落点。加层/加文件/加字段 = 改这里。
PROJECTIONS = {
    'B': {'file': 'baseloop.jsonl',        'prefix': 'B:', 'fields': ['eid', 'name', 'dtype', 'definition']},
    'C': {'file': 'concept_token.jsonl',   'prefix': 'D:', 'fields': ['eid', 'name', 'dtype', 'definition', 'maps_to']},
    'S': {'file': 'symbol_tokens.jsonl',   'prefix': 'S:', 'fields': ['eid', 'name', 'dtype', 'glyph', 'maps_to']},
    'G': {'file': 'grammar.jsonl',         'prefix': 'G:', 'fields': ['eid', 'name', 'definition', 'reduction']},
    'P': {'file': 'presentation.jsonl',    'prefix': 'P:', 'fields': ['eid', 'name', 'concept', 'grammar', 'precedence', 'associativity']},
    'X': {'file': 'explain.jsonl',         'prefix': None,  'fields': ['eid', 'name', 'dtype', 'ref', 'intension']},
    'A': {'file': 'arrow_tokens.jsonl',    'prefix': 'A:', 'fields': ['eid', 'name', 'dtype', 'source', 'target', 'concept']},
    'I': {'file': 'itoken.jsonl',          'prefix': 'I:', 'fields': ['eid', 'name', 'dtype', 'interface', 'value', 'lo', 'hi', 'op', 'definition']},
}

# 定义形态 (definition schema 词汇, 见 _research_brief/定义schema设计草案.md §1)
VALID_FORMS = {'explicit', 'inductive', 'recursive', 'axiomatic', 'implicit'}
# 语义政策 (definition schema, 调研: 定义方法的命题化 §10/§11.3): 定义如何获得语义
# BodyForm 与 Logic 多对多, policy 标注语义机制; 与 obligations (证明义务) 配合校验
VALID_POLICIES = {
    'expandable',        # 显式: 定义体可展开替换被定义项 (保守扩展, 调研 §2.1)
    'least_fixpoint',    # 归纳: 构造规则的最小闭包/最小不动点 (调研 §4.1)
    'greatest_fixpoint', # 余归纳: 最大不动点 (调研 §4.1 Tarski)
    'equations',         # 递归: 全称方程族 (基例+递归步) (调研 §5.1)
    'unique_expansion',  # 隐式: 约束在基础结构上唯一扩张 (调研 §1.3)
    'axiomatic',         # 公理化: 约束模型类的公式集 (调研 §1.5)
}
# 元层证明义务 (definition schema, 调研 §11.5 原样): 定义校验的证明目标, 非定义体连接词
VALID_OBLIGATIONS = {
    'freshness', 'well_formedness',
    'conservativeness', 'eliminability',
    'existence_uniqueness',
    'monotonicity_or_positivity', 'leastness',
    'coverage_and_nonoverlap',
    'termination_or_productivity',
    'consistency_or_model_existence',
}
# 绑定变量伪参: var:N = De Bruijn 索引 (向外第 N 层量词绑定的变量, 无引用边)
VAR_RE = re.compile(r'^var:\d+$')
# 排列槽位角色词 (gtoken 定义用, 类伪参, 程序透传, 无引用边):
#   arg:N=第N子项, fn=函数槽位(概念), args=参数列表(可变), binder=绑定变量, body=体
SLOT_RE = re.compile(r'^(?:fn|args(?::\d+)?|binder|body|arg:\d+)$')
_DEFAULT_EID = {'B': 0, 'C': 100, 'S': 300, 'G': 0, 'P': 0, 'A': 0}


class MaintainError(Exception):
    """编译风格错误: code(错误类型) + message + diag(诊断信息)。"""

    def __init__(self, code, message, diag=None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.diag = diag


def _path(layer):
    return os.path.join(_DIR, PROJECTIONS[layer]['file'])


_LAYER_CACHE: dict = {}
_ALL_CACHE = None


def invalidate():
    """注册表缓存失效: maintain 写入后调用, 保证读到最新数据."""
    _LAYER_CACHE.clear()
    global _ALL_CACHE
    _ALL_CACHE = None


def read_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            rows.append(json.loads(s))
    return rows


def write_jsonl(path, rows):
    """写回 jsonl, 保留原文件顶部注释行 (文件级文档说明)。写后缓存失效."""
    header = []
    try:
        with open(path, encoding='utf-8') as f:
            for line in f:
                if line.startswith('#'):
                    header.append(line)
                else:
                    break
    except FileNotFoundError:
        pass
    with open(path, 'w', encoding='utf-8') as f:
        for line in header:
            f.write(line)
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    invalidate()


def load_layer(layer):
    """读某层 → {eid: 字段dict} (缓存: 仅首次读盘, maintain 写入后失效)."""
    if layer not in _LAYER_CACHE:
        _LAYER_CACHE[layer] = read_jsonl(_path(layer))
    return {r['eid']: r for r in _LAYER_CACHE[layer]}


def load_all():
    """所有层 → 统一索引 {eid: 合并字段 + _layer(主层)} (缓存).

    解释层 (X) 不主导: 先并入 X (ref/intension), 再由 B/C/S 数据层覆盖共享字段 (name/dtype),
    保证数据层是权威 (X 只贡献解释性字段)。
    """
    global _ALL_CACHE
    if _ALL_CACHE is None:
        merged = {}
        for layer in ('X', 'B', 'C', 'S', 'G', 'P', 'A', 'I'):
            for eid, fields in load_layer(layer).items():
                merged.setdefault(eid, {})
                merged[eid].update(fields)
                if layer != 'X':
                    merged[eid]['_layer'] = layer
        _ALL_CACHE = merged
    return _ALL_CACHE


def all_eids(all_tokens=None):
    return set((all_tokens if all_tokens is not None else load_all()))


def _is_eid_literal(x):
    """gtoken 排列中的固定 token 字面量 (eid, 如 question/gap/equals)."""
    return bool(re.match(r'^[BDSGPA]:\d+$', x))


def _is_slot_definition(defn):
    """排列定义 (gtoken): definition 有 rules, 且每个 rules.term 元素全是槽位角色词
    (arg:N/fn/args/binder/body) 或 eid 字面量 (排列中的固定 token).

    语义定义 (C) 的 term 引用 eid/伪参, 非纯槽位; 公理 (B, 无 rules) 也不是。
    atom/eos (无子结构) 用 form=explicit + rules=[{term:[]}], 空 term 也判排列定义。
    """
    if not isinstance(defn, dict):
        return False
    rules = defn.get('rules')
    if not rules:
        return False
    for rule in rules:
        if not isinstance(rule, dict):
            return False
        term = rule.get('term')
        if not isinstance(term, list):
            return False
        if not all(isinstance(x, str) and (SLOT_RE.match(x) or _is_eid_literal(x)) for x in term):
            return False
        if not any(isinstance(x, str) and SLOT_RE.match(x) for x in term):
            return False   # gtoken 排列必须含槽位 (纯 eid term = C 概念引用)
    return True


def layer_of(fields, eid=None):
    """层推断: source/target → A (箭头, 两个被抽象 ctoken 的关系);
    glyph/maps_to → S; 排列定义 (term 全槽位角色词) → G; form:axiomatic (公设自指声明) → B; 其余默认 → C (含未完善/空定义)。

    默认 C: 概念先落 C 层, 未完善 (空 definition) 也留 C 等补定义。
    排列定义是 G 层特征: gtoken 的 definition.rules.term 是槽位角色词序列 (排列方法), 落 G 不落 C。
    仅自指升 B: axiomatic 是 token 自我声明为公设 (定义自指为其根基), 其余形态 (含 recursive 自引/self 直觉)
    都不是公设, 留在 C; 真正的接地环由 --diagnose 检出后经 C↔B 移动落位。
    A 层: arrow token 有 source/target (两个被抽象 ctoken) + concept (atoken 等价的 ctoken)。
    """
    if 'source' in fields and 'target' in fields:
        return 'A'
    if 'glyph' in fields or 'maps_to' in fields:
        return 'S'
    if _is_slot_definition(fields.get('definition')):
        return 'G'
    if 'grammar' in fields:
        return 'P'   # 呈现定义 (concrete syntax: 具体记法)
    defn = fields.get('definition')
    if isinstance(defn, dict) and defn.get('form') == 'axiomatic':
        return 'B'
    return 'C'


def walk_strings(defn):
    """递归收集 definition 结构全部叶子字符串 (dict 键除外)。"""
    out = []
    def _w(x):
        if isinstance(x, dict):
            for v in x.values():
                _w(v)
        elif isinstance(x, list):
            for v in x:
                _w(v)
        elif isinstance(x, str):
            out.append(x)
    _w(defn)
    return out


def walk_refs(defn, all_tokens):
    """definition 结构中所有已注册 eid 引用 (按注册表判定, 数据驱动, 无正则)。"""
    return [s for s in walk_strings(defn) if s in all_tokens]


def _schema_refs(defn):
    """按定义 schema 收集引用位置叶子 (signature.ref / rules term/pattern/value / constructor / constraints.term)。

    form/kind 等是标签非引用; 只有引用位置才校验注册 + 进 references。
    """
    found = []

    def _collect_term(t):
        if isinstance(t, str):
            if SLOT_RE.match(t):
                return  # gtoken 槽位角色词 (arg:N/fn/args/binder/body): 排列方法, 非引用
            found.append(t)
        elif isinstance(t, list):
            for x in t:
                _collect_term(x)
        elif isinstance(t, dict):
            for x in t.values():
                _collect_term(x)

    sig = defn.get('signature')
    if isinstance(sig, dict):
        for p in sig.get('params', []) or []:
            if isinstance(p, dict) and 'ref' in p:
                found.append(p['ref'])
        res = sig.get('result')
        if isinstance(res, dict) and 'ref' in res:
            found.append(res['ref'])
    for rule in defn.get('rules', []) or []:
        if isinstance(rule, dict):
            # 双序列: term=语言序列, proposition=命题序列 (命题等价连接); 括号=嵌套列表分组
            for key in ('term', 'pattern', 'value', 'proposition'):
                if key in rule:
                    _collect_term(rule[key])
            eq = rule.get('equiv')
            if isinstance(eq, str):
                found.append(eq)
            c = rule.get('constructor')
            if isinstance(c, dict):
                if 'ref' in c:
                    found.append(c['ref'])
                if 'arg' in c:
                    _collect_term(c['arg'])
    for cons in defn.get('constraints', []) or []:
        if isinstance(cons, dict) and 'term' in cons:
            _collect_term(cons['term'])
    return found


def derive_references(defn, all_tokens):
    """references 自动派生: 引用位置的全部已注册 eid, 去重保序 (rules 与 references 不漂移)。"""
    seen, out = set(), []
    for r in _schema_refs(defn):
        if r in all_tokens and r not in seen:
            seen.add(r)
            out.append(r)
    return out


def definition_refs(fields):
    """从 load_all 字段 dict 提取定义引用列表 (兼容双格式)。"""
    defn = fields.get('definition')
    if isinstance(defn, dict):
        return list(defn.get('references', []))
    return list(defn) if defn else []


def resolve_eid(name, layer, layer_tokens):
    """按 name 定位已有 eid, 无则自动分配 (本层空间下一个空闲编号)。"""
    for eid, fields in layer_tokens.items():
        if fields.get('name') == name:
            return eid, False
    prefix = PROJECTIONS[layer]['prefix']
    nums = [int(e.split(':')[1]) for e in layer_tokens if e.startswith(prefix)]
    return f'{prefix}{max(nums) + 1 if nums else _DEFAULT_EID[layer]}', True


def _validate_gtoken(defn):
    """gtoken 排列定义: form explicit (排列方法 = 槽位序列显式定义), rules.term 全槽位角色词 (_is_slot_definition 已保证)。

    binding_arity (可选, 调研 FPT binding signature): 每槽位上下文扩展的变量数 (非负整数列表)。
    如 quantified [binder, body] → binding_arity [0,1]: body 槽位在 binder 引入的 1 变量下 (作用域)。
    """
    form = defn.get('form')
    if form != 'explicit':
        raise MaintainError('GTKIND_INVALID', f'gtoken 定义形态必须是 explicit (排列方法), 收到: {form!r}',
                            '排列方法 = 槽位序列显式定义: {"form":"explicit","rules":[{"term":["arg:0","arg:1"]}]}')
    binding_arity = defn.get('binding_arity')
    if binding_arity is not None:
        if not isinstance(binding_arity, list) or not all(isinstance(n, int) and n >= 0 for n in binding_arity):
            raise MaintainError('BINDING_INVALID', 'binding_arity 必须是非负整数列表 (每槽位上下文扩展的变量数)',
                                '如 quantified [binder, body] → [0,1]: body 在 binder 引入的 1 变量下')


def validate(tok, all_tokens, flat_ok=False):
    """通用校验 + 诊断 (definition 兼容旧列表/结构化; 排列定义 gtoken 走 _validate_gtoken; reduction 归约规则)。

    flat_ok: 幂等重录放行 TERM_FLAT (输入与现有同名 token 定义一致时, 现有平铺残渣允许确认/重同步)。
    """
    if not isinstance(tok, dict):
        raise MaintainError('TYPE_INVALID', '输入必须是 JSON 对象', f'实际类型: {type(tok).__name__}')
    if not tok.get('name'):
        raise MaintainError('NAME_MISSING', 'name 字段必填',
                            '固定格式: {"name": "...", "definition": {...}, "intension": "..."}')
    # A 层 (arrow token): source/target/concept 必须已注册 (两个被抽象 ctoken + atoken 等价 ctoken)
    if 'source' in tok or 'target' in tok:
        for k in ('source', 'target', 'concept'):
            v = tok.get(k)
            if v is not None and v not in all_tokens:
                raise MaintainError('REF_UNREGISTERED', f'arrow {k} 未注册: {v}',
                                    f'A 层箭头须引用已注册 token: source/target=被抽象 ctoken, concept=atoken 等价 ctoken')
        if tok.get('concept') is not None and tok['concept'] == tok.get('source'):
            raise MaintainError('ARROW_CONCEPT_SAME', f'arrow concept 不能等于 source: {tok["concept"]}',
                                'atoken 等价 ctoken 是箭头概念化, 独立于被抽象的 source/target')
    defn = tok.get('definition')
    if isinstance(defn, dict):
        if _is_slot_definition(defn):
            _validate_gtoken(defn)          # 排列定义 (gtoken)
        else:
            _validate_structured(defn, all_tokens, flat_ok)   # 语义定义 (B/C)
    else:
        for ref in (defn or []):
            if ref not in all_tokens:
                raise MaintainError('REF_UNREGISTERED', f'definition 引用未注册: {ref}',
                                    f'引用了不存在的 eid, 需先完善 [{ref}] 再引用')
    # gtoken reduction (归约规则, 可选, 如 SKI)
    reduction = tok.get('reduction')
    if reduction is not None:
        if not isinstance(reduction, dict) or 'pattern' not in reduction or 'value' not in reduction:
            raise MaintainError('REDUCTION_INVALID', 'reduction 必须是 {pattern, value}',
                                '归约规则: pattern 匹配参数 (arg:N), value 是替换结构 (arg:N 原子或嵌套应用)')
        if not isinstance(reduction['pattern'], list):
            raise MaintainError('REDUCTION_INVALID', 'reduction.pattern 必须是列表 (arg:N 参数模式)')
        if not isinstance(reduction['value'], (str, list)):
            raise MaintainError('REDUCTION_INVALID', 'reduction.value 必须是字符串(arg:N 原子)或列表(嵌套应用)')
    # 呈现定义 (P): concept 已注册 + grammar 字符串列表 (arg:N 槽位/符号字面量) + precedence/associativity 合法
    if 'grammar' in tok:
        concept = tok.get('concept')
        if concept is not None and concept not in all_tokens:
            raise MaintainError('REF_UNREGISTERED', f'呈现定义 concept 未注册: {concept}')
        grammar = tok['grammar']
        if not isinstance(grammar, list) or not all(isinstance(x, str) for x in grammar):
            raise MaintainError('GRAMMAR_INVALID', 'grammar 必须是字符串列表 (arg:N 槽位 或 符号字面量)')
        precedence = tok.get('precedence')
        if precedence is not None and not isinstance(precedence, int):
            raise MaintainError('PRECEDENCE_INVALID', f'precedence 必须是整数, 收到: {precedence!r}')
        associativity = tok.get('associativity')
        if associativity is not None and associativity not in ('left', 'right'):
            raise MaintainError('ASSOC_INVALID', f'associativity 必须是 left/right, 收到: {associativity!r}')


def _check_nested_term(term, all_tokens, params, top=False, skip_flat=False):
    """递归校验嵌套 term (type term = ref|param|application(ref, term*), 定义schema设计草案 §2)。

    原子 str: 已注册 eid 或签名伪参 (params/self/result/var:N) — 由引用检查负责。
    节点 list: [head, 子项...], head 是字符串 (eid/概念), 子项递归为 term。
    ★ 平铺残渣检测 (仅顶层 top=True): 顶层列表长度≥2 且全为 str 且无伪参 → v2 平铺 eid 序列 (TERM_FLAT)。
      识别断言 [D:343, self] 含伪参 self → 豁免 (合法); 全 eid 无结构 → 判残渣要求嵌套。
      嵌套 children (top=False) 不检测: application 节点 [fn, arg, arg] 全原子合法。
    ★ skip_flat (幂等重录): 输入与现有同名 token 定义一致时放行 TERM_FLAT —
      现有平铺残渣 (v2 残留) 允许维护工具幂等确认/重同步; 新定义/修改仍强制嵌套。
    """
    if isinstance(term, str):
        return
    if not isinstance(term, list):
        raise MaintainError('TERM_INVALID', f'term 元素必须是字符串或嵌套列表, 收到: {term!r}',
                            '嵌套格式: [头节点, 子项...], 子项可再嵌套 (ExpressionNode)')
    if not term:
        return                                    # 空列表 = 无子结构 (atom/eos)
    if not isinstance(term[0], str):
        raise MaintainError('TERM_HEAD_INVALID', f'节点头必须是字符串 (eid/概念), 收到: {term[0]!r}',
                            '嵌套节点格式: [头, 子项...]')
    if top and not skip_flat and len(term) >= 2 and all(isinstance(x, str) and x not in params and not VAR_RE.match(x) for x in term):
        # 顶层首元素是节点头 (gtoken 或有 arrange 的概念) → 合法 ExpressionNode (如 application(succ, four) = [D:200, D:205]);
        # 无 arrange 的 eid 序列 (如 [B:4, B:4, D:250, ...]) 才是 v2 平铺残渣
        head = term[0]
        headf = all_tokens.get(head)
        is_node_head = False
        if headf:
            if headf.get('_layer') == 'G':
                is_node_head = True
            elif isinstance(headf.get('definition'), dict) and headf['definition'].get('arrange'):
                is_node_head = True
        if not is_node_head:
            raise MaintainError('TERM_FLAT', '定义体是平铺 eid 序列 (v2 残渣), 要求嵌套 ExpressionNode 结构',
                                '嵌套格式: [头, 子项...], 子项可再嵌套; 例 equality(application(addition,5,3),8) = ["D:260",["D:250","D:206","D:204"],"D:209"]')
    for c in term[1:]:
        _check_nested_term(c, all_tokens, params, top=False, skip_flat=skip_flat)


def _validate_structured(defn, all_tokens, flat_ok=False):
    """结构化定义校验: form/policy/obligations/arrange 合法; 引用位置每个叶子串是已注册 eid 或签名参数。"""
    form = defn.get('form')
    if form not in VALID_FORMS:
        raise MaintainError('FORM_INVALID', f'未知定义形态: {form!r}',
                            f'合法形态: {", ".join(sorted(VALID_FORMS))}')
    arrange = defn.get('arrange')
    if arrange is not None:
        # arrange = 概念→节点类型映射: 指向已注册 gtoken name (排列方法, G 层)
        if not isinstance(arrange, str):
            raise MaintainError('ARRANGE_INVALID', f'arrange 必须是 gtoken 名, 收到: {arrange!r}')
        g = next((f for f in all_tokens.values()
                  if f.get('_layer') == 'G' and f.get('name') == arrange), None)
        if g is None:
            raise MaintainError('ARRANGE_INVALID', f'arrange 引用未注册 gtoken: {arrange!r}',
                                f'合法 gtoken: {sorted(f["name"] for f in all_tokens.values() if f.get("_layer") == "G")}')
    policy = defn.get('policy')
    if policy is not None and policy not in VALID_POLICIES:
        raise MaintainError('POLICY_INVALID', f'未知语义政策: {policy!r}',
                            f'合法政策: {", ".join(sorted(VALID_POLICIES))}')
    obligations = defn.get('obligations')
    if obligations is not None:
        if not isinstance(obligations, list) or not all(isinstance(o, str) for o in obligations):
            raise MaintainError('OBLIGATIONS_INVALID', 'obligations 必须是字符串列表')
        bad = sorted(o for o in obligations if o not in VALID_OBLIGATIONS)
        if bad:
            raise MaintainError('OBLIGATIONS_INVALID', f'未知证明义务: {bad}',
                                f'合法义务: {", ".join(sorted(VALID_OBLIGATIONS))}')
    params = {'result', 'self'}   # result=签名结果伪参; self=正在定义的这个 token (直觉/类型自指, 无引用边)
    sig = defn.get('signature') or {}
    for i, _ in enumerate(sig.get('params') or []):
        params.add(f'arg:{i}')
    for rule in defn.get('rules', []) or []:
        if isinstance(rule, dict) and 'term' in rule and isinstance(rule['term'], list):
            _check_nested_term(rule['term'], all_tokens, params, top=True, skip_flat=flat_ok)   # 嵌套 ExpressionNode 结构 (v2 平铺残渣拦截; 幂等重录放行)
    for s in _schema_refs(defn):
        if s in all_tokens or s in params or VAR_RE.match(s):
            continue
        raise MaintainError('REF_UNREGISTERED', f'定义中未注册引用/非法叶子串: {s!r}',
                            f'叶子串必须是已注册 eid、签名参数或绑定变量 ({", ".join(sorted(params))}, var:N)')


def ref_of(eid, all_tokens):
    """简称自动生成: 有 glyph → [glyph]; 无 → [name]; 歧义下标 eid。"""
    fields = all_tokens.get(eid, {})
    syms = [s for s in all_tokens.values()
            if s.get('_layer') == 'S' and eid in (s.get('maps_to') or {})]
    if syms:
        g = syms[0].get('glyph') or syms[0].get('name', '')
        cands = set(t for s in syms for t in (s.get('maps_to') or {}))
        return f'[{g}]({eid})' if len(cands) > 1 else f'[{g}]'
    return f'[{fields.get("name", eid)}]'
