"""tokenizer/construct/nl.py —— NL/符号序列 → 命题元素序列 (分词 + 解析 + 消歧)。

tokenize: 裸记法 (最长匹配 glyph/名, 未知字符合并为词) 与括号简写 ([x](eid)) 自动识别。
nl_to_proposition / nl_to_eids: 每 token → 候选 eid; 伪参透传; 歧义/未解析显式报告。
数据驱动: glyph 集来自 S 层, 名称集来自注册表, 不硬编码 token 名。
"""
import re

from .._register import (token_eid, resolve_derives, SYMBOL_BY_GLYPH,
                         TOKEN_BY_NAME, DERIVE_BY_NAME, SYMBOL_BY_NAME)

_PSEUDO_RE = re.compile(r'^(?:self|result|arg:\d+|var:\d+)$')   # 定义语法伪参/绑定变量 (无引用边, 透传)

_match_cache = {}


def _match_lists():
    """glyph/名称 匹配表 (按长度降序, 模块级缓存, 数据驱动不硬编码)。"""
    if 'lists' not in _match_cache:
        glyphs = sorted(SYMBOL_BY_GLYPH.keys(), key=len, reverse=True)
        names = sorted(set(TOKEN_BY_NAME) | set(DERIVE_BY_NAME) | set(SYMBOL_BY_NAME), key=len, reverse=True)
        _match_cache['lists'] = (glyphs, names)
    return _match_cache['lists']


def _tokenize_short(s):
    """括号简写 [x](eid)? → [(token, 消歧eid)]。括号分界, (eid) 消歧, 空格忽略。"""
    tokens, i, n = [], 0, len(s)
    while i < n:
        if s[i] in ' \t':
            i += 1
            continue
        if s[i] != '[':
            raise ValueError(f'简写解析错误: 位置 {i} 期望 [, 收到 {s[i]!r} (片段 ...{s[max(0,i-12):i+15]}...)')
        j = s.index(']', i)
        content = s[i + 1:j].strip()
        if not content:
            raise ValueError(f'简写解析错误: 空 [] 于位置 {i}')
        i = j + 1
        dis = None
        if i < n and s[i] == '(':
            k = s.index(')', i)
            dis = s[i + 1:k].strip()
            i = k + 1
        tokens.append((content, dis))
    return tokens


def _tokenize_nl_raw(s):
    """裸 NL/数学记法 → [(token, None)]。最长匹配 glyph/名; 未匹配字符合并为词 (不逐字碎)。"""
    glyphs, names = _match_lists()
    tokens, i, n, word = [], 0, len(s), []

    def flush():
        nonlocal word
        if word:
            tokens.append((''.join(word), None))
            word = []

    while i < n:
        c = s[i]
        if c.isspace():
            flush()
            i += 1
            continue
        for cand in glyphs + names:
            if s.startswith(cand, i):
                flush()
                tokens.append((cand, None))
                i += len(cand)
                break
        else:
            word.append(c)      # 未匹配 → 累积成词 (未知/中文词整体保留)
            i += 1
    flush()
    return tokens


def tokenize(s):
    """NL 定义序列 → [(token, 消歧eid)]。自动识别: [ 开头 = 括号简写, 否则裸记法。"""
    if s.lstrip()[:1] == '[':
        return _tokenize_short(s)
    return _tokenize_nl_raw(s)


def tokenize_nl(s):
    """裸 NL/数学记法 → token 列表 (无消歧信息)。"""
    return [t for t, _ in _tokenize_nl_raw(s)]


def nl_to_proposition(s, disambiguation=None):
    """自然语言定义序列 → 命题元素序列 (每元素: token + 候选 eids)。

    token 解析: glyph → 候选 eid (resolve_derives, 不取首项); 名 → 注册表唯一 eid;
    括号简写的 (eid) 或 disambiguation 参数 = 显式消歧;
    歧义未消歧 → 保留候选集; 未解析 → 保留原文 (resolved=False)。
    """
    disambiguation = disambiguation or {}
    out = []
    for t, dis in tokenize(s):
        if _PSEUDO_RE.match(t):
            out.append({'token': t, 'eids': [t], 'resolved': True})   # 伪参透传
            continue
        eid = dis or disambiguation.get(t)
        if eid:
            out.append({'token': t, 'eids': [eid], 'resolved': True})
            continue
        cands = resolve_derives(t)
        if not cands:
            try:
                cands = [token_eid(t)]
            except KeyError:
                cands = []
        out.append({'token': t, 'eids': cands, 'resolved': len(cands) == 1})
    return out


def nl_to_eids(s, disambiguation=None):
    """NL 定义序列 → 干净 eid 序列。全部消歧成功才返回, 否则报错列出未解析/歧义项。"""
    parts = nl_to_proposition(s, disambiguation)
    bad = [p for p in parts if not p['resolved']]
    if bad:
        detail = '; '.join(f"{p['token']} → {p['eids'] or '未解析'}" for p in bad)
        raise ValueError(f'NL 序列未完全消歧: {detail}')
    return [p['eids'][0] for p in parts]


def nl_define(s, disambiguation=None):
    """NL 定义 → 简写序列 + 歧义诊断报告 (AI 可读, 供决策消歧)。

    不自动消歧 (纪律: 歧义不取首项): 歧义项在主序列标 [glyph](D:?),
    附 ambiguous 报告列出全部候选 eid + name + intension + definition 简写,
    由 AI/人阅读 intension 语义按上下文 (语境) 收敛决策。
    """
    from ..maintain import core
    from .render import short_of
    parts = nl_to_proposition(s, disambiguation)
    all_t = core.load_all()
    seq, ambiguous = [], {}
    for p in parts:
        if not p['resolved']:
            seq.append(f"[{p['token']}](D:?)")
            cands = []
            for eid in p['eids']:
                f = all_t.get(eid, {})
                defn = f.get('definition')
                if isinstance(defn, dict):
                    refs = [all_t.get(r, {}).get('name', r) for r in defn.get('references', [])]
                    ds = f"{defn.get('form')}[{', '.join(refs)}]" if refs else str(defn.get('form'))
                else:
                    ds = '旧格式引用' if defn else '未定义'
                cands.append({'eid': eid, 'name': f.get('name', eid),
                              'intension': f.get('intension', ''),
                              'definition': ds})
            ambiguous[p['token']] = cands
        else:
            seq.append(short_of(p['eids'][0]))
    return {'sequence': ''.join(seq), 'ambiguous': ambiguous}
