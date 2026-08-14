"""_register.py —— 三层 token 注册器 (tokenizer 内部实现)

数据源: tokenizer/tokens/ (经 token_index.py 统一加载)。
加载 baseloop.jsonl (B:公设基) + concept_token.jsonl (D:派生) + symbol_tokens.jsonl (S:符号)。
eid 统一为字符串: B:0, D:100, S:300。
★ derived_from/explained_by 已删除 (归属由定义/引用承载); head 已废除 (注意算子在 head/ 模块, 非 token 属性)。
"""
import json
import os
from dataclasses import dataclass, field

from . import token_index
from .token_index import read_rows


@dataclass
class TokenDef:
    eid: str
    name: str
    dtype: str = "bool"
    definition: list = field(default_factory=list)   # 定义: 旧格式=引用 eid 列表; 新格式={form,signature,rules,constraints,references} 结构化
    maps_to: dict = field(default_factory=dict)  # symbol → derive
    # ★ intension 已迁移至解释层 explain.jsonl (禁止程序调用, 供 AI/人类阅读)

# 三层注册表
TOKEN_REGISTRY:  dict[str, TokenDef] = {}   # baseloop
TOKEN_BY_NAME:   dict[str, str] = {}        # baseloop name → eid
DERIVE_REGISTRY: dict[str, TokenDef] = {}   # derive
DERIVE_BY_NAME:  dict[str, str] = {}        # derive name → eid
SYMBOL_REGISTRY: dict[str, TokenDef] = {}   # symbol
SYMBOL_BY_NAME:  dict[str, str] = {}        # symbol name → eid
SYMBOL_BY_GLYPH: dict[str, list[str]] = {}  # glyph → [eid]
ARROW_REGISTRY:  dict[str, TokenDef] = {}   # arrow (A 层: 两个被抽象 ctoken 的关系)
ARROW_BY_NAME:   dict[str, str] = {}        # arrow name → eid


def _this_dir():
    return token_index.TOKENS_DIR


def load_baseloop(path=None):
    if path is None:
        path = token_index._path("B")
    TOKEN_REGISTRY.clear(); TOKEN_BY_NAME.clear()
    _load_jsonl(path, TOKEN_REGISTRY, TOKEN_BY_NAME)
    return TOKEN_REGISTRY


def load_derive(path=None):
    if path is None:
        path = token_index._path("C")
    DERIVE_REGISTRY.clear(); DERIVE_BY_NAME.clear()
    _load_jsonl(path, DERIVE_REGISTRY, DERIVE_BY_NAME)
    return DERIVE_REGISTRY


def load_symbols(path=None):
    if path is None:
        path = token_index._path("S")
    SYMBOL_REGISTRY.clear(); SYMBOL_BY_NAME.clear(); SYMBOL_BY_GLYPH.clear()
    _load_jsonl(path, SYMBOL_REGISTRY, SYMBOL_BY_NAME)
    for td in SYMBOL_REGISTRY.values():
        g = td.glyph
        SYMBOL_BY_GLYPH.setdefault(g, []).append(td.eid)
    return SYMBOL_REGISTRY


def load_arrows(path=None):
    if path is None:
        path = token_index._path("A")
    ARROW_REGISTRY.clear(); ARROW_BY_NAME.clear()
    _load_jsonl(path, ARROW_REGISTRY, ARROW_BY_NAME)
    return ARROW_REGISTRY


def _load_jsonl(path, registry, name_index):
    # 优先经 token_index 读 (含临时注入; 缓存命中则跳过文件)
    layer = token_index._TOKEN_FILES_LAYER.get(os.path.basename(path))
    if layer is not None and layer in token_index._TEMP_INJECT:
        rows = token_index.read_rows(layer)
        for d in rows:
            _register_row(d, registry, name_index)
        return
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            d = json.loads(line)
            _register_row(d, registry, name_index)


def _register_row(d, registry, name_index):
    td = TokenDef(
        eid=str(d["eid"]),
        name=d["name"],
        dtype=d.get("dtype", "bool"),
        definition=d.get("definition", []) or [],
        maps_to=d.get("maps_to", {}),
    )
    if 'glyph' in d: td.glyph = d['glyph']
    for k, v in d.items():
        if k not in {'eid','name','dtype','intension','definition','maps_to','glyph'}:
            setattr(td, k, v)
    if td.eid in registry:
        raise ValueError(f"{d.get('eid')} eid 重复: {td.eid}")
    registry[td.eid] = td
    name_index[td.name] = td.eid


def token_of(eid: str) -> TokenDef:
    for reg in [TOKEN_REGISTRY, DERIVE_REGISTRY, SYMBOL_REGISTRY, ARROW_REGISTRY]:
        if eid in reg: return reg[eid]
    raise KeyError(f"未注册: {eid}")


def definition_refs(defn):
    """结构化定义提取引用: dict→references 列表; list→自身; 空→[]。兼容迁移过渡期双格式。"""
    if isinstance(defn, dict):
        return list(defn.get('references', []))
    return list(defn) if defn else []


def definition_form(defn):
    """定义形态: dict→form 字段; 非 dict (旧格式/未完善)→None。"""
    if isinstance(defn, dict):
        return defn.get('form')
    return None


def is_axiomatic_eid(eid: str) -> bool:
    """eid 是否为公理式定义 (form:axiomatic, brace 穿透展开终点)。"""
    return definition_form(token_of(eid).definition) == 'axiomatic'


def token_name(eid: str) -> str:
    return token_of(eid).name


def is_postulate(eid: str) -> bool: return eid in TOKEN_REGISTRY
def is_derived(eid: str) -> bool:   return eid in DERIVE_REGISTRY


def token_eid(name: str) -> str:
    for idx in [TOKEN_BY_NAME, DERIVE_BY_NAME, SYMBOL_BY_NAME, ARROW_BY_NAME]:
        if name in idx: return idx[name]
    raise KeyError(f"未注册 token: {name!r}")


def resolve_glyph(glyph: str) -> list[str]:
    return SYMBOL_BY_GLYPH.get(glyph, [])


def resolve_derives(glyph: str) -> list[str]:
    """glyph → 候选 C token (derive) eid 列表, 不取首项。

    S 是 C 的集合体: 一个符号可对应多个概念, 由语境收敛。
    调研(符号与概念映射)明确: 取首项 = 结果依注册顺序, 静默错误, 禁止。
    """
    result = []
    for sid in resolve_glyph(glyph):
        for target in SYMBOL_REGISTRY[sid].maps_to:
            if target not in result:
                result.append(target)
    return result


def _maps_to_targets(sid: str) -> list[str]:
    """symbol → maps_to 的全部候选 target (不取首项)。"""
    return list(SYMBOL_REGISTRY[sid].maps_to.keys())


def element_symbol(eid: str) -> str:
    td = token_of(eid)
    if hasattr(td, 'glyph'): return td.glyph
    for g, sids in SYMBOL_BY_GLYPH.items():
        for sid in sids:
            mt = SYMBOL_REGISTRY[sid].maps_to
            if mt and eid in _maps_to_targets(sid): return g
    return td.name


def all_eids() -> dict:
    result = {}
    for reg in [TOKEN_REGISTRY, DERIVE_REGISTRY, SYMBOL_REGISTRY, ARROW_REGISTRY]:
        result.update(reg)
    return result


ELEMENT_REGISTRY = None

# 模块加载
load_baseloop()
load_derive()
load_symbols()
load_arrows()
ELEMENT_REGISTRY = all_eids()


if __name__ == "__main__":
    print(f"baseloop: {len(TOKEN_REGISTRY)}, derive: {len(DERIVE_REGISTRY)}, symbol: {len(SYMBOL_REGISTRY)}")
    print(f"glyph 索引: {len(SYMBOL_BY_GLYPH)}, '=' → {len(resolve_glyph('='))} 个 eid")
