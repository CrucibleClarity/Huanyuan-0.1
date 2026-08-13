"""synth/collector/ —— 收拢/整形 (嵌套向量 → transformer 输入向量)"""
from .collapse import collapse, vector_spec, METHODS
from .shaper import shape, shape_spec

__all__ = ["collapse", "vector_spec", "METHODS", "shape", "shape_spec"]
