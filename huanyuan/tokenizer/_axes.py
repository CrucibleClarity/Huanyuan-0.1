"""_axes.py —— 从三层 token 生成训练用轴空间 (tokenizer 内部实现)

全部 token (baseloop B: + derive D: + symbol S:) 构成轴空间, 按 eid 字符串排序。
"""
from ._register import TOKEN_REGISTRY, DERIVE_REGISTRY, SYMBOL_REGISTRY

_all = list(TOKEN_REGISTRY.values()) + list(DERIVE_REGISTRY.values()) + list(SYMBOL_REGISTRY.values())
AXES = sorted(
    [{"eid": td.eid, "name": td.name, "dtype": td.dtype} for td in _all],
    key=lambda a: a["eid"],
)

CONCEPT_DIM = len(AXES)
AXIS_INDEX = {a["eid"]: i for i, a in enumerate(AXES)}
AXIS_BY_NAME = {a["name"]: i for i, a in enumerate(AXES)}
AXIS_NAME = [a["name"] for a in AXES]
AXIS_DTYPES = [a["dtype"] for a in AXES]

if __name__ == "__main__":
    print(f"=== tokenizer 轴空间 (CONCEPT_DIM={CONCEPT_DIM}) ===")
    for i, a in enumerate(AXES[:20]):
        print(f"  [{i:2d}] {a['eid']:8} {a['name']:22} ({a['dtype']})")
    print(f"  ... ({len(AXES)-20} more)")
