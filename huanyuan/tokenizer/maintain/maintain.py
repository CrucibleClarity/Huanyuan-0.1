#!/usr/bin/env python
"""maintain/maintain.py —— token 维护工具 (基于 core 抽象基础设施)

录入: 输入一个 token (固定格式 JSON 行) → 自动 eid, 按投影分发到所有 json 文件。
诊断: --diagnose → token 体系覆盖度 + 环检测。
移动: --move <eid> --to <B|C> → C↔B 移动 (文件迁移 + eid 前缀变更 + 引用级联)。
批量: --batch <file> → 批量录入 (JSONL 或 JSON 数组), 编译风格逐条报告, 任一失败 exit 1。
分歧: --conflict keep-both|keep-one → 同名冲突显式消解 (全保留重写 / 保留一迁移他)。
移动: --move <eid> --to <B|C> → C↔B 移动 (文件迁移 + eid 前缀变更 + 引用级联)。
全部基于 core.PROJECTIONS, 不硬编码文件/字段。

★ 无删除: token 只有"未完善", 没有"需删除" — 维护工具关注完善度, 不增删。
  幂等: 相同内容重复提交 → 确认; 空→有 → 填充; 列表→结构化同引用 → 格式迁移;
        结构化新增 rules/constraints (不改旧) → 完善; 改动已有项 → 冲突报错 + 诊断。
  分歧: 同名冲突由 --conflict 显式消解; merge 的退役是"重复实体"收尾, 非原则性删除。

用法:
  echo '{"name":"xxx","definition":[...],"intension":"..."}' | python -m tokenizer.maintain.maintain
  python -m tokenizer.maintain.maintain --batch batch.json       # JSONL 或 JSON 数组
  python -m tokenizer.maintain.maintain --rewrite-defs defs.jsonl  # 批量改写已存在定义 (定义微调)
  python -m tokenizer.maintain.maintain --diagnose
  python -m tokenizer.maintain.maintain --move D:250 --to B
  python -m tokenizer.maintain.maintain --conflict keep-both --target D:250 --name addition_op
  python -m tokenizer.maintain.maintain --conflict keep-both --target S:411 --maps-to '{"D:323":1}'   # S 层: 改符号所指 (maps_to/glyph 均可)
  python -m tokenizer.maintain.maintain --conflict keep-one --keep D:109 --other D:342 --merge
  python -m tokenizer.maintain.maintain --conflict keep-one --keep D:109 --other D:342 --to B
"""
import json
import sys

from . import core
from .core import MaintainError


def parse(line):
    """解析输入行 (JSON)。"""
    try:
        return json.loads(line)
    except json.JSONDecodeError as e:
        raise MaintainError('JSON_INVALID', f'JSON 解析错误: {e.msg}',
                            f'第 {e.lineno} 列 {e.colno}: 原始行 = {line.strip()[:80]}')


def load_batch(path):
    """读批量文件 → JSON 行列表: JSONL (每行一个 token) 或 JSON 数组。"""
    try:
        text = open(path, encoding='utf-8').read().strip()
    except OSError as e:
        raise MaintainError('BATCH_FILE', f'无法读取批量文件: {e}')
    if not text:
        raise MaintainError('BATCH_EMPTY', f'批量文件为空: {path}')
    if text.startswith('['):
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise MaintainError('JSON_INVALID', f'JSON 数组解析错误: {e.msg}',
                                f'第 {e.lineno} 列 {e.colno}')
        if not isinstance(data, list):
            raise MaintainError('BATCH_TYPE', 'JSON 数组必须是对象列表')
        if not data:
            raise MaintainError('BATCH_EMPTY', f'批量文件无有效条目: {path}')
        return [json.dumps(x, ensure_ascii=False) for x in data]
    lines = [l for l in text.split('\n') if l.strip() and not l.strip().startswith('#')]
    if not lines:
        raise MaintainError('BATCH_EMPTY', f'批量文件无有效条目: {path}')
    return lines


def process_entries(lines):
    """批量处理 JSON 行, 编译风格逐条报告 (错误不中止), 任一失败 exit 1。"""
    all_tokens = core.load_all()
    code = 0
    for i, line in enumerate(lines, 1):
        try:
            tok = parse(line)
            flat_ok = _is_flat_idempotent(tok, all_tokens)
            core.validate(tok, all_tokens, flat_ok=flat_ok)
            layer = core.layer_of(tok)
            eid, is_new = core.resolve_eid(tok['name'], layer, core.load_layer(layer))
            action, eid, data_file = write_token(tok, eid, is_new)
            print(f'  ✓ {action} [{eid}] {tok["name"]} ({layer}) → {data_file.split("/")[-1]}')
            all_tokens = core.load_all()   # 刷新索引 (后续行可能引用新 eid)
        except MaintainError as err:
            code = 1
            print(f'  ✗ 错误 [{err.code}] (第 {i} 条 {line.strip()[:60]}): {err.message}')
            if err.diag:
                print(f'    诊断: {err.diag}')
    return code


def _is_empty(v):
    return v in (None, "", []) or v == {}


def rewrite_defs_batch(lines):
    """批量改写已存在定义 (定义微调)。

    输入 (JSONL/JSON 数组), 每行一条:
      {"eid": "D:105", "definition": {"form": "explicit", "rules": [...]}, ...}

    - definition 必需语义覆盖; name/dtype/intension 为可选覆盖 (同步 explain + 重算 ref)
    - 幂等: 目标定义与现有一致 → 跳过不报错 (可安全重跑同一文件)
    - 每批走 core.validate + references 自动派生; 任一失败 exit 1
    """
    all_tokens = core.load_all()
    code = 0
    for i, line in enumerate(lines, 1):
        try:
            tok = parse(line)
            eid = tok.get('eid')
            if not eid:
                raise MaintainError('REWRITE_NO_EID', f'第 {i} 条缺少 eid (定义微调以 eid 定位)')
            if eid not in all_tokens:
                raise MaintainError('REWRITE_UNKNOWN', f'未知 eid: {eid}')
            fields = all_tokens[eid]
            overrides = {}
            for k in ('definition', 'name', 'dtype', 'intension'):
                if k in tok:
                    overrides[k] = tok[k]
            if not overrides:
                raise MaintainError('REWRITE_NO_FIELDS', f'第 {i} 条 [{eid}] 无改写字段')
            cur_def = fields.get('definition')
            if 'definition' in overrides and isinstance(cur_def, dict):
                if _norm_definition(cur_def) == _norm_definition(overrides['definition']):
                    print(f'  ✓ 已一致 [{eid}] {fields["name"]} → 跳过 (幂等)')
                    continue
            msg = _rewrite_fields(eid, overrides)
            print(f'  ✓ {msg}')
            all_tokens = core.load_all()   # 刷新 (后续行可能引用新定义)
        except MaintainError as err:
            code = 1
            print(f'  ✗ 错误 [{err.code}] (第 {i} 条 {line.strip()[:60]}): {err.message}')
            if err.diag:
                print(f'    诊断: {err.diag}')
    return code
def _definition_equal(a, b):
    """定义语义比较: 提取引用集 (旧列表 vs 新结构化 dict = 同一语义, 格式迁移非冲突)。"""
    def refs(x):
        if isinstance(x, dict):
            return list(x.get('references', []))
        return list(x) if x else []
    return set(refs(a)) == set(refs(b))


def _norm_definition(d):
    """规范化定义: 去掉自动派生的 references (幂等重录判定用, 输入不含 references)。"""
    if isinstance(d, dict):
        return {k: v for k, v in d.items() if k != 'references'}
    return d


def _is_flat_idempotent(tok, all_tokens):
    """输入与现有同名 token 定义一致 (幂等重录) → 放行 TERM_FLAT。

    现有 v2 平铺残渣允许维护工具幂等确认/重同步 (不重新强制嵌套);
    新定义/内容变更仍走 TERM_FLAT 拦截 (嵌套是强制格式)。
    """
    name = tok.get('name')
    if not name:
        return False
    defn_in = tok.get('definition')
    for f in all_tokens.values():
        if f.get('name') == name and f.get('_layer') in ('B', 'C', 'S'):
            if _norm_definition(f.get('definition')) == _norm_definition(defn_in):
                return True
    return False


def _definition_completes(a, b):
    """b 是 a 的完善: form 同, 旧 rules/constraints 是新的前缀子集, signature 不退化。

    只有未完善没有需删除 → 允许"新增不删改" (加规则/加约束), 禁止改动已有项。
    """
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    if a.get('form') != b.get('form'):
        return False
    for k in ('signature',):
        if a.get(k) and a.get(k) != b.get(k):
            return False
    for k in ('rules', 'constraints'):
        old, new = a.get(k) or [], b.get(k) or []
        if len(new) < len(old):
            return False
        for item in old:
            if item not in new:
                return False
    return True


def _diff(existing, data_row, ex_explain, explain_row):
    """(fills, diffs): 空→有=首次填充; 非空→不同=冲突; 列表→结构化同引用=格式迁移填充。"""
    fills, diffs = [], []
    if existing:
        for k, v in data_row.items():
            if k in ('eid', 'name'):
                continue
            if k == 'definition':
                ex = existing.get(k)
                if ex == v:
                    continue  # 完全一致 (references 已自动派生, 重跑相等 → 幂等)
                if isinstance(ex, list) and isinstance(v, dict) and _definition_equal(ex, v):
                    fills.append('definition: 格式迁移 列表→结构化 (引用集一致)')
                    continue
                if isinstance(ex, dict) and isinstance(v, dict) and _definition_completes(ex, v):
                    fills.append('definition: 完善 (新增 rules/constraints, 未改动已有项)')
                    continue
            if existing.get(k) != v:
                (fills if _is_empty(existing.get(k)) and v else diffs).append(f'{k}: {existing.get(k)!r} → {v!r}')
    if ex_explain:
        for k, v in explain_row.items():
            if ex_explain.get(k) != v and k not in ('eid', 'name', 'ref'):
                (fills if _is_empty(ex_explain.get(k)) and v else diffs).append(f'{k}: {ex_explain.get(k)!r} → {v!r}')
    return fills, diffs


def write_token(tok, eid, is_new):
    """按投影声明分发 token 字段到各文件。返回 (action, eid, 数据文件)。"""
    layer = core.layer_of(tok)
    data_file = core._path(layer)
    rows = core.read_jsonl(data_file)

    # 主数据行: 只含该层声明字段 (投影分发)
    data_row = {'eid': eid, 'name': tok['name']}
    for f in core.PROJECTIONS[layer]['fields']:
        if f in tok and f not in ('eid', 'name'):
            data_row[f] = tok[f]

    # references 自动派生: 结构化定义按引用位置重建, rules 与 references 不漂移
    if isinstance(data_row.get('definition'), dict):
        data_row['definition']['references'] = core.derive_references(
            data_row['definition'], core.load_all())

    # explain 行 (intension + ref 自动生成; 新建 token 未落盘, 先并入内存索引供 ref 取自身 name)
    # G 层 (排列方法) / P 层 (呈现记法) 的投影即其结构本身, 无散文解释/intension/dtype, 不建 explain 条目
    explain_row = None
    if layer not in ('G', 'P'):
        explain_rows = core.read_jsonl(core._path('X'))
        ref_all = dict(core.load_all())
        ref_all.setdefault(eid, {}).update(data_row)
        ref = core.ref_of(eid, ref_all)
        explain_row = {'eid': eid, 'name': tok['name'], 'dtype': tok.get('dtype', 'bool'),
                       'ref': ref, 'intension': tok.get('intension', '')}

    # 幂等/填充检查 (无删除: 只完善, 不冲突覆盖)
    action = '新建'
    if not is_new:
        existing = next((r for r in rows if r.get('eid') == eid), None)
        ex_exp = None
        if explain_row is not None:
            ex_exp = next((r for r in explain_rows if r.get('eid') == eid), None)
        fills, diffs = _diff(existing, data_row, ex_exp, explain_row)
        if diffs:
            raise MaintainError('CONFLICT', f'同名 token 内容冲突: [{eid}] {tok["name"]}',
                                '字段差异: ' + ', '.join(diffs))
        if not fills:
            return '已存在且一致', eid, data_file
        action = '已填充'

    # 写入主数据行
    found = False
    for i, r in enumerate(rows):
        if r.get('eid') == eid:
            rows[i] = data_row
            found = True
            break
    if not found:
        rows.append(data_row)
    core.write_jsonl(data_file, rows)

    # 写 explain (G 层无 explain 投影, 跳过)
    if explain_row is not None:
        found = False
        for i, r in enumerate(explain_rows):
            if r.get('eid') == eid:
                explain_rows[i] = explain_row
                found = True
                break
        if not found:
            explain_rows.append(explain_row)
        core.write_jsonl(core._path('X'), explain_rows)

    # token 数据已变化: 统一 CTE 缓存失效 (gtoken 派生/law 模板, 幂等重编译)
    core.invalidate()
    try:
        from .. import cte
        cte.invalidate_all()
    except ImportError:
        pass

    return action, eid, data_file


def _rewrite_eid(x, old, new):
    """递归改写结构 (dict/list/str, 含 dict 键) 中所有 old→new。"""
    if isinstance(x, dict):
        return {(_rewrite_eid(k, old, new) if isinstance(k, str) else k): _rewrite_eid(v, old, new)
                for k, v in x.items()}
    if isinstance(x, list):
        return [_rewrite_eid(v, old, new) for v in x]
    if isinstance(x, str) and x == old:
        return new
    return x


def move_token(eid, target):
    """C↔B 移动: 文件迁移 + eid 前缀变更 + 引用级联 + (C→B) 标 axiomatic。

    副作用由 maintain 承担, 移动后不悬空:
      - B/C 层所有 definition 结构递归改写 old eid → new eid
      - S 层 maps_to 键改写 (S→C 指代跟随)
      - explain 行 eid 变更 + ref 重算
      - C→B: definition 置 {form:axiomatic, references:[]} (公设 = 展开终点)
      - B→C: definition 清空 (变为未完善派生, 由 maintain 补定义)
    """
    all_tokens = core.load_all()
    cur = all_tokens.get(eid)
    if not cur:
        raise MaintainError('MOVE_UNKNOWN', f'未知 eid: {eid}')
    src = cur.get('_layer')
    if src not in ('B', 'C'):
        raise MaintainError('MOVE_LAYER', f'仅支持 B↔C 移动, {eid} 当前在 {src} 层')
    if target not in ('B', 'C'):
        raise MaintainError('MOVE_LAYER', f'目标层必须为 B 或 C, 收到: {target!r}')
    if src == target:
        raise MaintainError('MOVE_SAME', f'{eid} 已在 {target} 层, 无需移动')

    new_eid = core.PROJECTIONS[target]['prefix'] + eid.split(':')[1]
    src_file, tgt_file = core._path(src), core._path(target)
    src_rows, tgt_rows = core.read_jsonl(src_file), core.read_jsonl(tgt_file)
    row = next((r for r in src_rows if r.get('eid') == eid), None)
    if row is None:
        raise MaintainError('MOVE_UNKNOWN', f'{eid} 不在 {src} 层文件')
    if any(r.get('eid') == new_eid for r in tgt_rows):
        raise MaintainError('MOVE_EID_CONFLICT', f'{target} 层已存在 {new_eid}')
    if any(r.get('name') == row['name'] for r in tgt_rows):
        raise MaintainError('MOVE_NAME_CONFLICT', f'{target} 层已存在同名 token: {row["name"]}')

    # 引用级联: 改写所有 B/C 行 definition + S 行 maps_to (跳过被移动行自身)
    cascade = 0
    for f in (src_rows, tgt_rows):
        for r in f:
            if r.get('eid') == eid or 'definition' not in r:
                continue
            new_def = _rewrite_eid(r['definition'], eid, new_eid)
            if new_def != r['definition']:
                r['definition'] = new_def
                cascade += 1
    s_rows = core.read_jsonl(core._path('S'))
    for r in s_rows:
        if r.get('maps_to') and eid in r['maps_to']:
            r['maps_to'] = _rewrite_eid(r['maps_to'], eid, new_eid)
            cascade += 1

    # 移动行转换
    row['eid'] = new_eid
    if target == 'B':
        row['definition'] = {'form': 'axiomatic', 'references': []}
    else:
        row.pop('definition', None)

    # 校验级联完整性: 不得残留旧 eid (写盘前检查, 失败不落盘)
    remaining = [r.get('eid') for f in (src_rows, tgt_rows) for r in f
                 if eid in core.walk_strings(r.get('definition', {}))]
    if remaining:
        raise MaintainError('MOVE_ORPHAN', f'级联后仍有引用指向旧 eid {eid}',
                            f'位置: {remaining}')

    src_rows = [r for r in src_rows if r is not row]
    tgt_rows.append(row)
    core.write_jsonl(src_file, src_rows)
    core.write_jsonl(tgt_file, tgt_rows)
    core.write_jsonl(core._path('S'), s_rows)

    # explain: eid 变更 + ref 重算 (glyph/maps_to 已跟随, ref_of 正确)
    x_rows = core.read_jsonl(core._path('X'))
    for r in x_rows:
        if r.get('eid') == eid:
            r['eid'] = new_eid
            r['ref'] = core.ref_of(new_eid, core.load_all())
    core.write_jsonl(core._path('X'), x_rows)

    return new_eid, src, target, cascade


def _cascade_rewrite(old, new):
    """级联改写 old→new: 所有 B/C 行 definition + S 行 maps_to (引用重指向)。"""
    n = 0
    for layer in ('B', 'C'):
        rows = core.read_jsonl(core._path(layer))
        for r in rows:
            if 'definition' not in r:
                continue
            d = _rewrite_eid(r['definition'], old, new)
            if d != r['definition']:
                r['definition'] = d
                n += 1
        core.write_jsonl(core._path(layer), rows)
    s_rows = core.read_jsonl(core._path('S'))
    for r in s_rows:
        if r.get('maps_to') and old in r['maps_to']:
            r['maps_to'] = _rewrite_eid(r['maps_to'], old, new)
            n += 1
    core.write_jsonl(core._path('S'), s_rows)
    return n


def _orphans(eid):
    """残留引用: 数据层 definition 与 S 层 maps_to 中仍指向 eid 的位置。"""
    refs = []
    for layer in ('B', 'C'):
        for r in core.read_jsonl(core._path(layer)):
            if eid in core.walk_strings(r.get('definition', {})):
                refs.append(r.get('eid'))
    for r in core.read_jsonl(core._path('S')):
        if r.get('maps_to') and eid in r['maps_to']:
            refs.append(r.get('eid'))
    return refs


def _set_row_field(eid, field, value):
    """改写某 eid 行的字段 (数据层 + 解释层); definition 结构化时自动派生 references。"""
    all_tokens = core.load_all()
    layer = all_tokens[eid]['_layer']
    rows = core.read_jsonl(core._path(layer))
    for r in rows:
        if r['eid'] == eid:
            if field == 'definition' and isinstance(value, dict):
                value['references'] = core.derive_references(value, all_tokens)
            r[field] = value
    core.write_jsonl(core._path(layer), rows)
    if field in ('intension', 'dtype', 'name'):
        x_rows = core.read_jsonl(core._path('X'))
        for r in x_rows:
            if r['eid'] == eid:
                r[field] = value
                if field in ('name', 'dtype'):
                    r['ref'] = core.ref_of(eid, core.load_all())
        core.write_jsonl(core._path('X'), x_rows)


def _retire(eid):
    """退役: 移除该 eid 的数据行与解释行 (合并/消歧后的重复实体), 并检查无残留引用。"""
    all_tokens = core.load_all()
    layer = all_tokens[eid]['_layer']
    rows = core.read_jsonl(core._path(layer))
    core.write_jsonl(core._path(layer), [r for r in rows if r['eid'] != eid])
    x_rows = core.read_jsonl(core._path('X'))
    core.write_jsonl(core._path('X'), [r for r in x_rows if r['eid'] != eid])
    left = _orphans(eid)
    if left:
        raise MaintainError('CONFLICT_ORPHAN', f'退役后仍有引用指向 {eid}: {left}')


def _rewrite_fields(eid, overrides):
    """全保留重写: 改写指定字段 (任意部分/全部), 两者共存, 不删任何 token。"""
    all_tokens = core.load_all()
    fields = all_tokens.get(eid)
    if not fields:
        raise MaintainError('CONFLICT_UNKNOWN', f'未知 eid: {eid}')
    layer = fields['_layer']
    if 'name' in overrides:
        name = overrides['name']
        others = [r['eid'] for L in ('B', 'C', 'S', 'G') for r in core.read_jsonl(core._path(L))
                  if r.get('name') == name and r.get('eid') != eid]
        if others:
            raise MaintainError('NAME_UNIQUE', f'名字已存在: {name} @ {others}')
    if 'arrange' in overrides:
        # C 层概念→节点类型映射: 合并进 definition (保留旧字段, 只加/改 arrange)
        arrange = overrides.pop('arrange')
        cur_def = dict(fields.get('definition') or {'form': 'explicit'})
        cur_def['arrange'] = arrange
        core.validate({'name': fields['name'], 'definition': cur_def}, all_tokens)
        cur_def['references'] = core.derive_references(cur_def, all_tokens)
        overrides['definition'] = cur_def
    if 'definition' in overrides:
        core.validate({'name': fields['name'], 'definition': overrides['definition']}, all_tokens)
        overrides['definition'] = dict(overrides['definition'])
        overrides['definition']['references'] = core.derive_references(
            overrides['definition'], all_tokens)
    if 'maps_to' in overrides:
        if not isinstance(overrides['maps_to'], dict):
            raise MaintainError('MAPS_TO_INVALID', 'maps_to 必须是 {eid: 权重} 对象')
        for k in overrides['maps_to']:
            if k not in all_tokens:
                raise MaintainError('REF_UNREGISTERED', f'maps_to 目标未注册: {k}')
    if 'reduction' in overrides:
        # G 层 gtoken 归约规则改写: 构造完整 token 走 validate (reduction 合法)
        tok = {'name': fields['name'], 'reduction': overrides['reduction']}
        core.validate(tok, all_tokens)
    rows = core.read_jsonl(core._path(layer))
    old_maps = []
    for r in rows:
        if r['eid'] == eid:
            if layer == 'S':
                old_maps = list((r.get('maps_to') or {}))
            for k, v in overrides.items():
                r[k] = v
    core.write_jsonl(core._path(layer), rows)
    # ref 重算: 改名/改类型/改glyph/改maps_to 会改变指代 → 重算本 token + maps_to 影响到的 C token
    if any(k in overrides for k in ('name', 'dtype', 'intension', 'glyph', 'maps_to')):
        affected = {eid} | (set(old_maps) | set(overrides['maps_to'])) if 'maps_to' in overrides else {eid}
        x_rows = core.read_jsonl(core._path('X'))
        for r in x_rows:
            if r['eid'] in affected:
                if r['eid'] == eid:
                    for k in ('name', 'dtype', 'intension'):
                        if k in overrides:
                            r[k] = overrides[k]
                r['ref'] = core.ref_of(r['eid'], core.load_all())
        core.write_jsonl(core._path('X'), x_rows)
    return f'重写 [{eid}] 字段: {", ".join(overrides)} (两者共存)'


def _merge_tokens(keep, other, fields):
    """保留某一个: 另一个并入保留者后退役 (merge); 引用重指向 other→keep。"""
    all_tokens = core.load_all()
    if keep not in all_tokens or other not in all_tokens:
        raise MaintainError('CONFLICT_UNKNOWN', f'未知 eid: keep={keep} other={other}')
    if keep == other:
        raise MaintainError('CONFLICT_ARGS', 'keep 与 other 必须不同')
    kf, of = all_tokens[keep], all_tokens[other]
    fields = fields or ['definition', 'intension', 'dtype']
    merged = []
    for f in fields:
        kv, ov = kf.get(f), of.get(f)
        if _is_empty(kv) and not _is_empty(ov):
            _set_row_field(keep, f, ov)
            merged.append(f)
        elif not _is_empty(kv) and not _is_empty(ov) and kv != ov:
            print(f'  提示: [{keep}] 的 {f} 已有内容且与 [{other}] 不同, 未并入 (可用 keep-both 重写)')
    n = _cascade_rewrite(other, keep)
    _retire(other)
    return f'保留 [{keep}], 并入 [{", ".join(merged) or "无"}], 引用重指向 {n} 处, [{other}] 退役'


def _keep_and_move(keep, other, layer):
    """保留某一个: 保留者原位, 另一个迁至目标层 (复用 move_token 级联)。"""
    if keep not in core.load_all() or other not in core.load_all():
        raise MaintainError('CONFLICT_UNKNOWN', f'未知 eid: keep={keep} other={other}')
    new_eid, src, tgt, n = move_token(other, layer)
    return f'保留 [{keep}], 迁移 [{other}] {src}→{tgt} → [{new_eid}], 级联 {n} 处'


def conflict_resolve(argv):
    """分歧处理 (同名冲突显式消解)。

      keep-both 全保留重写: --conflict keep-both --target <eid> [--name/--dtype/--definition/--intension]
                           改写指定字段 (任意部分/全部), 两者共存, 不删任何 token
      keep-one  保留一迁移他: --conflict keep-one --keep <eid> --other <eid>
                            (--merge [--fields a,b] | --to <B|C>)
                           merge=内容并入保留者后退役; --to=迁移至目标层
    """
    def opt(name):
        if name in argv:
            j = argv.index(name)
            if j + 1 < len(argv):
                return argv[j + 1]
        return None

    if len(argv) < 1:
        raise MaintainError('CONFLICT_ARGS', '--conflict 需模式 keep-both|keep-one')
    mode = argv[0]
    if mode == 'keep-both':
        target = opt('--target')
        if not target:
            raise MaintainError('CONFLICT_ARGS', 'keep-both 需 --target <eid>')
        overrides = {}
        for flag, key in (('--name', 'name'), ('--dtype', 'dtype')):
            v = opt(flag)
            if v:
                overrides[key] = v
        dv = opt('--definition')
        if dv is not None:
            try:
                overrides['definition'] = json.loads(dv)
            except json.JSONDecodeError as e:
                raise MaintainError('JSON_INVALID', f'--definition JSON 解析错误: {e.msg}')
        iv = opt('--intension')
        if iv is not None:
            overrides['intension'] = iv
        gv = opt('--glyph')            # S 层: 改符号本体 (glyph/maps_to = S token 的定义)
        if gv is not None:
            overrides['glyph'] = gv
        mv = opt('--maps-to')
        if mv is not None:
            try:
                overrides['maps_to'] = json.loads(mv)
            except json.JSONDecodeError as e:
                raise MaintainError('JSON_INVALID', f'--maps-to JSON 解析错误: {e.msg}')
        rv = opt('--reduction')        # G 层: 归约规则改写 (如 SKI)
        if rv is not None:
            try:
                overrides['reduction'] = json.loads(rv)
            except json.JSONDecodeError as e:
                raise MaintainError('JSON_INVALID', f'--reduction JSON 解析错误: {e.msg}')
        pv = opt('--grammar')          # P 层: 呈现 grammar 改写 (concrete syntax)
        if pv is not None:
            try:
                overrides['grammar'] = json.loads(pv)
            except json.JSONDecodeError as e:
                raise MaintainError('JSON_INVALID', f'--grammar JSON 解析错误: {e.msg}')
        ar = opt('--arrange')          # C 层: 概念→节点类型映射改写 (definition.arrange)
        if ar is not None:
            overrides['arrange'] = ar
        if not overrides:
            raise MaintainError('CONFLICT_ARGS', 'keep-both 需至少一个重写字段')
        return _rewrite_fields(target, overrides)
    if mode == 'keep-one':
        keep, other = opt('--keep'), opt('--other')
        if not keep or not other:
            raise MaintainError('CONFLICT_ARGS', 'keep-one 需 --keep <eid> --other <eid>')
        if '--merge' in argv:
            fl = opt('--fields')
            fields = [s.strip() for s in fl.split(',')] if fl else None
            return _merge_tokens(keep, other, fields)
        layer = opt('--to')
        if layer in ('B', 'C'):
            return _keep_and_move(keep, other, layer)
        raise MaintainError('CONFLICT_ARGS', 'keep-one 需 --merge 或 --to <B|C>')
    raise MaintainError('CONFLICT_ARGS', f'未知分歧模式: {mode!r}, 需 keep-both|keep-one')


def main(argv):
    if '--diagnose' in argv:
        from .diagnose import diagnose, render
        print(render(diagnose()))
        return 0

    if '--move' in argv:
        i = argv.index('--move')
        if i + 1 >= len(argv):
            print('错误 [MOVE_ARGS]: --move 缺少 eid 参数')
            print('  用法: python -m tokenizer.maintain.maintain --move <eid> --to <B|C>')
            return 1
        eid = argv[i + 1]
        target = None
        if '--to' in argv:
            j = argv.index('--to')
            if j + 1 < len(argv):
                target = argv[j + 1].upper()
        if target not in ('B', 'C'):
            print(f'错误 [MOVE_ARGS]: 目标层必须为 B 或 C, 收到: {target!r}')
            return 1
        try:
            new_eid, src, tgt, n = move_token(eid, target)
            print(f'  ✓ 移动 [{eid}] {src}→{tgt} → [{new_eid}], 引用级联更新 {n} 处')
            return 0
        except MaintainError as err:
            print(f'  ✗ 错误 [{err.code}]: {err.message}')
            if err.diag:
                print(f'    诊断: {err.diag}')
            return 1

    if '--batch' in argv:
        i = argv.index('--batch')
        if i + 1 >= len(argv):
            print('错误 [BATCH_ARGS]: --batch 缺少文件路径参数')
            print('  用法: python -m tokenizer.maintain.maintain --batch <file.json>')
            return 1
        try:
            lines = load_batch(argv[i + 1])
        except MaintainError as err:
            print(f'  ✗ 错误 [{err.code}]: {err.message}')
            if err.diag:
                print(f'    诊断: {err.diag}')
            return 1
        return process_entries(lines)

    if '--rewrite-defs' in argv:
        i = argv.index('--rewrite-defs')
        if i + 1 >= len(argv):
            print('错误 [REWRITE_ARGS]: --rewrite-defs 缺少文件路径参数')
            print('  用法: python -m tokenizer.maintain.maintain --rewrite-defs <defs.jsonl>')
            return 1
        try:
            lines = load_batch(argv[i + 1])
        except MaintainError as err:
            print(f'  ✗ 错误 [{err.code}]: {err.message}')
            if err.diag:
                print(f'    诊断: {err.diag}')
            return 1
        return rewrite_defs_batch(lines)

    if '--conflict' in argv:
        i = argv.index('--conflict')
        try:
            print(f'  ✓ {conflict_resolve(argv[i + 1:])}')
            return 0
        except MaintainError as err:
            print(f'  ✗ 错误 [{err.code}]: {err.message}')
            if err.diag:
                print(f'    诊断: {err.diag}')
            return 1

    lines = argv if argv else [l for l in sys.stdin.read().split('\n') if l.strip()]
    if not lines:
        print('错误 [NO_INPUT]: 未提供 token 输入')
        print('  诊断: echo \'{"name":"..."}\' | python -m tokenizer.maintain.maintain')
        return 1
    return process_entries(lines)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
