"""tokenizer/token_index.py —— token 数据统一索引 (唯一加载入口)

标准化 token 数据存储:
  tokenizer/tokens/
    baseloop.jsonl         B 层: 公设基 (axiomatic)
    concept_token.jsonl    C 层: 派生概念 (显式/归纳/递归/隐式定义)
    explain.jsonl          X 层: 解释层 (intension, 供 AI/人类阅读, 程序禁止调用)
    symbol_tokens.jsonl    S 层: 符号 (glyph → maps_to 候选概念)
    grammar.jsonl          G 层: gtoken (排列方法, 槽位序列)
    presentation.jsonl     P 层: 呈现 (concrete syntax, 优先级/结合性)

所有 token 数据只经本模块加载/查询。消费方禁止直写文件路径。
"""
import json
import os

TOKENS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokens")

# 标准文件名映射: layer → 文件名
TOKEN_FILES = {
    "B": "baseloop.jsonl",
    "C": "concept_token.jsonl",
    "X": "explain.jsonl",
    "S": "symbol_tokens.jsonl",
    "G": "grammar.jsonl",
    "P": "presentation.jsonl",
    "A": "arrow_tokens.jsonl",
    "I": "itoken.jsonl",
}
# 反向: 文件名 → layer
_TOKEN_FILES_LAYER = {v: k for k, v in TOKEN_FILES.items()}

_LAYER_CACHE: dict[str, list[dict]] = {}
# 临时 token 注入 (实验用, 不写 jsonl): layer → [行 dict]
_TEMP_INJECT: dict[str, list[dict]] = {}


def _path(layer: str) -> str:
    if layer not in TOKEN_FILES:
        raise KeyError(f"未知层: {layer!r} (合法: {sorted(TOKEN_FILES)})")
    return os.path.join(TOKENS_DIR, TOKEN_FILES[layer])


def inject_temp(layer: str, rows: list[dict]) -> None:
    """注入临时 token (实验用, 不进 jsonl 文件)。

    tokenizer 对外能力: 实验 (lab) 需要自定义 token 搭配分析时,
    经此注入, 不污染主 token 数据; 实验结束 clear_cache 清除。
    rows: [{eid, name, definition, ...}] (与 jsonl 行同结构)。
    """
    if layer not in TOKEN_FILES:
        raise KeyError(f"未知层: {layer!r}")
    _TEMP_INJECT.setdefault(layer, []).extend([dict(r) for r in rows])
    _LAYER_CACHE.pop(layer, None)


def read_rows(layer: str) -> list[dict]:
    """读某层全部行 (缺文件返回空列表; 含临时注入)。"""
    if layer in _LAYER_CACHE:
        return _LAYER_CACHE[layer]
    rows = []
    path = _path(layer)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                rows.append(json.loads(s))
    for tr in _TEMP_INJECT.get(layer, []):
        rows.append(dict(tr))
    _LAYER_CACHE[layer] = rows
    return rows


def write_rows(layer: str, rows: list[dict]) -> None:
    """写某层全部行 (保留原文件顶部注释行)。"""
    _LAYER_CACHE[layer] = [dict(r) for r in rows]
    path = _path(layer)
    header = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("#"):
                    header.append(line)
                else:
                    break
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in header:
            f.write(line)
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def clear_cache() -> None:
    """清空层缓存 + 临时注入 (实验结束刷新)。"""
    _LAYER_CACHE.clear()
    _TEMP_INJECT.clear()


def index(layer: str) -> dict[str, dict]:
    """层 → {eid: 字段dict} 索引。"""
    return {r["eid"]: r for r in read_rows(layer)}


def all_index() -> dict[str, dict]:
    """全层合并索引 {eid: 合并字段 + _layer(主层)}。解释层 (X) 不主导。"""
    merged = {}
    for layer in ("X", "B", "C", "S", "G", "P", "A"):
        for eid, fields in index(layer).items():
            merged.setdefault(eid, {})
            merged[eid].update(fields)
            if layer != "X":
                merged[eid]["_layer"] = layer
    return merged


def by_name(name: str) -> dict | None:
    """name → 字段dict (全层, 无则 None)。"""
    for row in all_index().values():
        if row.get("name") == name:
            return row
    return None


def by_eid(eid: str) -> dict | None:
    """eid → 字段dict (全层, 无则 None)。"""
    return all_index().get(eid)
