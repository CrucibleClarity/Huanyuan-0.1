"""shaper/ —— 向量整形器 (样本 → transformer 输入形态, 可插拔)

output 三选: sequence (多向量, token 级) / layer (多向量, 逐层) / collapse (单向量)
order 展平顺序 (可插拔): preorder / postorder / level / infix
encode 节点编码 (可插拔): counts / depth / role
"""
from ._registry import register_order, register_encode, get_order, get_encode, list_orders, list_encodes
from . import orders, encode as encode_impl
from .shape import shape, shape_spec
from .unshape import unshape, prototypes, reset

__all__ = [
    "shape", "shape_spec",
    "unshape", "prototypes", "reset",
    "register_order", "register_encode", "get_order", "get_encode", "list_orders", "list_encodes",
]
