"""select/layers/ —— 分层采样 (谓词/名词/装饰)

每层一个选择器脚本, @register_selector 注册; roles_of 提供层映射 (数据驱动槽位)。
"""
from . import predicate, noun, decorator
from ._roles import roles_of

__all__ = ["roles_of", "predicate", "noun", "decorator"]
