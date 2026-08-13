"""maintain/diagnose.py —— token 体系诊断 (基于 core 抽象基础设施)

从统一索引 (core.load_all) 自动推导覆盖度, 不手写判断:
  definition 覆盖 (有/缺 definition)
  glyph 覆盖      (有/缺 glyph)
  环检测           (definition 引用环, 展示 eid 链供分析 C↔B 移动)
★ head 已废除 (硬编码纪律), 不再检查。

架构更新只改 core.PROJECTIONS, 诊断自动适配。
"""
from collections import defaultdict

from . import core


def _form_of(fields):
    """定义形态: 结构化 dict→form; 旧格式/未完善→None。"""
    defn = fields.get('definition')
    return defn.get('form') if isinstance(defn, dict) else None


def _cycle_chain(comp, adj):
    """在强连通分量内找一条环链 (A→…→A), 供人分析。"""
    cset = set(comp)
    for e in comp:
        if e in adj.get(e, []):
            return [e, e]
    for start in comp:
        path, seen = [start], {start}
        def dfs(u):
            for w in adj.get(u, []):
                if w not in cset or w in seen:
                    continue
                if w == start:
                    return True
                seen.add(w)
                path.append(w)
                if dfs(w):
                    return True
                path.pop()
                seen.discard(w)
            return False
        if dfs(start):
            path.append(start)
            return path
    return sorted(comp)


def _cycles(c_tokens):
    """C 层 definition 引用图找环 (Tarjan SCC; 沿引用只走 C 层, B/S 是终点)。"""
    c_ids = set(c_tokens)
    adj = {e: [r for r in core.definition_refs(f) if r in c_ids]
           for e, f in c_tokens.items()}

    index, low, on_stack, stack, sccs, counter = {}, {}, {}, [], [], [0]
    def strongconnect(v):
        index[v] = low[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        on_stack[v] = True
        for w in adj.get(v, []):
            if w not in index:
                strongconnect(w)
                low[v] = min(low[v], low[w])
            elif on_stack.get(w):
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                comp.append(w)
                if w == v:
                    break
            sccs.append(comp)

    for v in c_ids:
        if v not in index:
            strongconnect(v)

    result = []
    for comp in sccs:
        has_self = any(e in adj.get(e, []) for e in comp)
        if len(comp) > 1 or has_self:
            result.append(_cycle_chain(comp, adj))
    return result


def _has_content(fields):
    """有实质定义内容 (非占位): 结构化有 rules/constraints/references; 旧格式列表非空。
    仅 form+arrange (如 forall 占位) = 未完善, 仍计缺定义。"""
    defn = fields.get('definition')
    if isinstance(defn, dict):
        return bool(defn.get('rules') or defn.get('constraints') or defn.get('references'))
    return bool(defn)


def _has_recursive_call(rules, self_eid):
    """任一规则的右端 (equality 第二参数) 引用自身 → 递归调用存在。

    equality 结构 [eq, lhs, rhs]: 真值表/枚举定义 rhs 是原子 (不含自身, 无递归);
    结构递归 rhs 含自身调用 (如 addition 递归步 S(add(a,b)) 含 D:250)。
    调研 §3.5/§8.3: 真值表=有限函数表 (显式, 非环); 结构递归=良基递归 (环合法)。
    """
    def walk(x):
        if isinstance(x, str):
            return x == self_eid
        if isinstance(x, list):
            return any(walk(c) for c in x)
        return False
    for rule in rules or []:
        term = rule.get('term') if isinstance(rule, dict) else None
        if isinstance(term, list) and len(term) >= 2:
            if walk(term[-1]):
                return True
    return False


def _cycle_status(members, c_tokens):
    """环状态标注 (依据调研 §8.3 机械可判定部分, 证书级判定留人工/后续):
      recursive 且有递归调用 -> 结构递归 (良基候选, 需下降证书);
      explicit 且无递归调用   -> 显式枚举 (有限函数表, 非环);
      explicit 且有递归调用   -> 显式自引用 (疑, 需证书);
      recursive 但无调用      -> 标注异常;
      axiomatic                -> 公设 (非问题);
      2+ 元互指环              -> 互指结构 (高维网络投影边, 需证书判定真实/欠定);
      其余                     -> UNKNOWN。
    ★ 环是树状投影高维网络的痕迹: 真实结构环 (对偶/互定义) 保留, 不机械打破
      (打破除尽 = 数学失真); 只有欠定循环才需证书判定。
    """
    uniq = len({e for e, _, _ in members})
    out = []
    for eid, name, form in members:
        f = c_tokens[eid]
        defn = f.get('definition')
        rules = defn.get('rules') if isinstance(defn, dict) else None
        if uniq >= 2:
            out.append((eid, name, form, '互指结构 (高维网络投影边, 需证书判定真实/欠定)'))
            continue
        if form == 'recursive':
            tag = '结构递归 (良基候选, 需下降证书)' if _has_recursive_call(rules, eid) \
                else 'recursive 但无递归调用 (异常)'
            out.append((eid, name, form, tag))
        elif form == 'explicit':
            tag = '显式枚举 (有限函数表, 非环)' if not _has_recursive_call(rules, eid) \
                else '显式定义自引用 (疑, 需证书)'
            out.append((eid, name, form, tag))
        elif form == 'axiomatic':
            out.append((eid, name, form, '公设 (非问题)'))
        else:
            out.append((eid, name, form or '—', 'UNKNOWN (需证书)'))
    return out


def diagnose():
    all_tokens = core.load_all()
    # C 层 token (主数据层为 C, 排除解释层残留)
    c_tokens = {e: f for e, f in all_tokens.items() if f.get('_layer') == 'C'}
    # G 层 gtoken (排列方法); definition 缺失 = 未完善
    g_tokens = {e: f for e, f in all_tokens.items() if f.get('_layer') == 'G'}

    with_def = [(e, f['name']) for e, f in c_tokens.items() if _has_content(f)]
    missing_def = [(e, f['name']) for e, f in c_tokens.items() if not _has_content(f)]

    # glyph 覆盖: C token 被 S 层 symbol maps_to 反查
    eid_to_glyph = defaultdict(list)
    for e, f in all_tokens.items():
        if f.get('_layer') == 'S' and f.get('maps_to'):
            eid_to_glyph[list(f['maps_to'].keys())[0]].append(
                f.get('glyph') or f.get('name'))
    # 残差豁免: 残差机制 token (name=residual) 与其实例 (references 含 residual)
    # 固有无符号 (学科具象投影丢失的不可符号化信息), 缺 glyph 非缺陷, 豁免诊断。
    residual_eid = next((e for e, f in c_tokens.items() if f.get('name') == 'residual'), None)

    def _residual_exempt(e, f):
        if e == residual_eid:
            return True
        if residual_eid is None:
            return False
        return residual_eid in core.definition_refs(f)

    with_glyph = [(e, f['name']) for e, f in c_tokens.items() if eid_to_glyph[e]]
    missing_glyph_all = [(e, f['name']) for e, f in c_tokens.items() if not eid_to_glyph[e]]
    residual_exempt = [x for x in missing_glyph_all if _residual_exempt(x[0], c_tokens[x[0]])]
    missing_glyph = [x for x in missing_glyph_all if not _residual_exempt(x[0], c_tokens[x[0]])]

    cycles = [{
        "chain": chain,
        "members": [(e, c_tokens[e]['name'], _form_of(c_tokens[e])) for e in chain],
        "status": _cycle_status([(e, c_tokens[e]['name'], _form_of(c_tokens[e])) for e in chain], c_tokens),
    } for chain in _cycles(c_tokens)]

    return {
        "total": len(c_tokens),
        "with_def": with_def, "missing_def": missing_def,
        "with_glyph": with_glyph, "missing_glyph": missing_glyph,
        "residual_exempt": residual_exempt,
        "g_total": len(g_tokens),
        "g_missing_def": [(e, f['name']) for e, f in g_tokens.items() if not f.get('definition')],
        "cycles": cycles,
    }


def render(r):
    lines = ["=" * 60, "token 体系诊断", "=" * 60]
    lines.append(f"总计: {r['total']}")
    lines.append("")

    for label, have, miss in [
        ("definition", r["with_def"], r["missing_def"]),
        ("glyph", r["with_glyph"], r["missing_glyph"]),
    ]:
        lines.append(f"{label}: 有 {len(have)}, 缺 {len(miss)}")
        if label == 'glyph' and r.get('residual_exempt'):
            lines.append(f"  固有无符号豁免: {len(r['residual_exempt'])} (残差 token, 缺 glyph 非缺陷)")
        if miss:
            groups = defaultdict(list)
            for e, n in miss:
                prefix = n.split("_")[0] if "_" in n else n[:6]
                groups[prefix].append((e, n))
            for g, items in sorted(groups.items()):
                lines.append(f"  [{g}] ({len(items)})")
                for e, n in items[:3]:
                    lines.append(f"    {e} {n}")
                if len(items) > 3:
                    lines.append(f"    ... ({len(items)-3} more)")
        lines.append("")

    has_all = not r["missing_def"] and not r["missing_glyph"]
    if has_all:
        lines.append("✓ 全通过")
    else:
        parts = []
        if r["missing_def"]:
            parts.append(f"{len(r['missing_def'])} 缺定义")
        if r["missing_glyph"]:
            parts.append(f"{len(r['missing_glyph'])} 缺 glyph")
        lines.append(f"◐ {' + '.join(parts)}")

    # G 层 (gtoken: 排列方法), 独立于概念覆盖度
    lines.append("")
    g_miss = r["g_missing_def"]
    if g_miss:
        lines.append(f"G 层: {r['g_total']} 个 gtoken, {len(g_miss)} 缺定义: "
                     + ", ".join(f"{e} {n}" for e, n in g_miss))
    else:
        lines.append(f"G 层: {r['g_total']} 个 gtoken (定义齐全)")

    # 环检测: 引用环是信号 (环中某 token 实为公理, 经 C↔B 移动落位), 不静默
    lines.append("")
    if r["cycles"]:
        lines.append(f"环检测: {len(r['cycles'])} 个引用环")
        for cyc in r["cycles"]:
            lines.append(f"  {' → '.join(cyc['chain'])}")
            for e, name, form, tag in cyc.get("status", []):
                lines.append(f"      {e} {name}  form={form or '—'}  [{tag}]")
    else:
        lines.append("环检测: 无引用环")
    return "\n".join(lines)


if __name__ == "__main__":
    print(render(diagnose()))
