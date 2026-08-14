"""选择器: 全部 token 参与 (默认)

head 是注意算子: 默认注意序列中的全部 token。后续可加"按属性/按角色/按类型"
等过滤选择器, 每算法一脚本, @register_selector 注册。
"""
from ._registry import register_selector


@register_selector("all")
def select_all(eids, ctx=None):
    """全部参与 head 计算。"""
    return list(eids)
