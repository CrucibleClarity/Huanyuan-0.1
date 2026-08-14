"""_vector.py —— token → 坐标 / 向量

★ 变换只依赖 C token 本身 (name / intension / 关系)。
  head/digit 等旁路字段已删除, 不作为变换依据。
  tokens_of(eid) = {token.name: 1}  (identity only)
  vector_of(S:xxx) = symbol自身 + maps_to 直接值
"""
from functools import lru_cache
import torch
from ._axes import CONCEPT_DIM, AXIS_BY_NAME
from ._register import (
    TOKEN_REGISTRY, DERIVE_REGISTRY, SYMBOL_REGISTRY, token_of,
)


@lru_cache(maxsize=512)
def tokens_of(eid: str) -> dict:
    """eid → token 坐标。只含自身 identity。
    返回 {token_name: 1} 用于:
      - 向量构建 (identity)
      - 后续 head 注意算子的输入 (同类/异类归类)
    """
    td = token_of(eid)
    result = {td.name: 1}

    # 符号 token (S 层定义特征 glyph): maps_to 直接值
    if hasattr(td, 'glyph') and td.maps_to:
        for target_eid, val in td.maps_to.items():
            result[token_of(target_eid).name] = float(val)

    return result


def vector_of(eid: str) -> torch.Tensor:
    v = torch.zeros(CONCEPT_DIM)
    for name, val in tokens_of(eid).items():
        idx = AXIS_BY_NAME.get(name)
        if idx is not None:
            v[idx] = float(val)
    return v


def vectors_of(eids: list[str]) -> list:
    return [vector_of(e) for e in eids]


if __name__ == "__main__":
    from ._axes import AXIS_NAME

    # 符号 → 向量
    from ._register import SYMBOL_BY_GLYPH
    for g in ["5", "+", ">"]:
        eids = SYMBOL_BY_GLYPH.get(g, [])
        if eids:
            v = vector_of(eids[0])
            active = {AXIS_NAME[i]: v[i].item() for i in range(CONCEPT_DIM) if v[i] != 0}
            print(f"{g}: {active}")

    # derive 层只返回 identity
    print(f"\naddition tokens_of: {list(tokens_of('D:250').keys())}")
    print(f"value_five tokens_of: {list(tokens_of('D:206').keys())}")
