"""tokenizer/grammar/_present.py —— 呈现层 (concrete syntax, 读呈现定义)

呈现定义 (P 层) = 具体记法: {concept, grammar, precedence, associativity}
  grammar: arg:N=子项槽位, 其他=符号字面量 (Terminal)
printer(ast) → 文本: 按呈现定义 grammar 输出符号 + 递归打印子项; 原子输出 S 层 glyph
parser(text) → AST: operator-precedence parsing (按呈现定义 precedence 分层结合)
★ 零硬编码: 符号/优先级/结合性全从呈现定义读; 概念→glyph 从 S 层读; 数据驱动。
"""
from ..maintain import core


def _presentations():
    """P 层呈现定义 → {concept_eid: {grammar, precedence, associativity}}。"""
    out = {}
    for p in core.load_layer('P').values():
        concept = p.get('concept')
        if concept:
            out[concept] = {'grammar': p.get('grammar', []),
                            'precedence': p.get('precedence'),
                            'associativity': p.get('associativity', 'left')}
    return out


def _eid_to_glyph():
    """概念 eid → 首个 S 层 glyph (符号材料, 数据驱动)。"""
    out = {}
    for f in core.load_all().values():
        if f.get('_layer') == 'S' and f.get('maps_to') and f.get('glyph'):
            for eid in f['maps_to']:
                out.setdefault(eid, f['glyph'])
    return out


def _glyph_to_eids():
    """glyph → 候选概念 eid 列表 (S 层 maps_to, 歧义保留全部)。"""
    out = {}
    for f in core.load_all().values():
        if f.get('_layer') == 'S' and f.get('maps_to') and f.get('glyph'):
            for eid in f['maps_to']:
                out.setdefault(f['glyph'], []).append(eid)
    return out


def _print_node(concept, children, presentations, glyphs):
    """按呈现定义 grammar 打印一个节点: arg:N 填子项, 符号字面量输出。"""
    pres = presentations.get(concept)
    if not pres:
        return glyphs.get(concept, concept)      # 无呈现 (atom): 输出概念 glyph
    grammar = pres['grammar']
    child_items = children
    if child_items and child_items[0] == concept:   # 去掉 fn (概念前置)
        child_items = child_items[1:]
    out, ci = [], 0
    for g in grammar:
        if g == 'args':                      # 可变子项序列 (如多位数字串)
            for c in child_items:
                out.append(_print_ast(c, presentations, glyphs))
        elif g.startswith('arg:'):
            out.append(_print_ast(child_items[ci], presentations, glyphs))
            ci += 1
        else:
            out.append(g)                          # 符号字面量 (Terminal)
    return ''.join(out)


def _print_ast(node, presentations, glyphs):
    """AST → 文本。node = assemble 结果 (dict) 或原子 eid (str)。"""
    if isinstance(node, dict):
        return _print_node(node.get('concept'), node.get('children', []), presentations, glyphs)
    return glyphs.get(node, node)


def print_ast(node):
    """AST → 文本 (读呈现定义, 零硬编码)。"""
    return _print_ast(node, _presentations(), _eid_to_glyph())


def _tokenize(text, symbols):
    """文本 → token 列表: 符号字面量 (最长匹配) 或 原子字符。"""
    syms = sorted(symbols, key=len, reverse=True)
    toks, i, n = [], 0, len(text)
    while i < n:
        for s in syms:
            if text.startswith(s, i):
                toks.append(('sym', s))
                i += len(s)
                break
        else:
            toks.append(('atom', text[i]))
            i += 1
    return toks


def _parse_atom(toks, pos, glyph_to_eids):
    """解析原子 (glyph → 概念 eid), 返回 (concept_eid 或 None, 新位置)。"""
    if pos < len(toks) and toks[pos][0] == 'atom':
        cands = glyph_to_eids.get(toks[pos][1], [])
        if len(cands) == 1:
            return cands[0], pos + 1
    return None, pos


def parse(text):
    """文本 → AST (operator-precedence parsing, 读呈现定义 precedence/associativity)。

    符号字面量 → 呈现定义 concept; 原子 glyph → 概念 (歧义保留候选, 不取首项)。
    """
    from ._assemble import assemble
    presentations = _presentations()
    glyph_to_eids = _glyph_to_eids()
    # 符号 → 呈现定义 (符号字面量 → 对应 concept)
    symbols, symbol_concept = set(), {}
    for concept, p in presentations.items():
        for g in p['grammar']:
            if not g.startswith('arg:') and g not in symbol_concept:
                symbols.add(g)
                symbol_concept[g] = concept
    toks = _tokenize(text, symbols)

    def parse_expr(min_prec, pos):
        lhs, pos = _parse_atom(toks, pos, glyph_to_eids)
        while pos < len(toks):
            if toks[pos][0] != 'sym':
                break
            op = toks[pos][0]
            concept = symbol_concept.get(toks[pos][1])
            if not concept:
                break
            prec = presentations[concept].get('precedence')
            if prec is None or prec < min_prec:
                break
            assoc = presentations[concept].get('associativity', 'left')
            next_min = prec if assoc == 'left' else prec + 1
            rhs, pos2 = parse_expr(next_min, pos + 1)
            if rhs is None:
                break
            lhs = assemble(concept, [lhs, rhs])
            pos = pos2
        return lhs, pos

    ast, _ = parse_expr(0, 0)
    return ast
