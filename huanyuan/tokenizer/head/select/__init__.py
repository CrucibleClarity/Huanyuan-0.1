"""select/ —— head 选择层 (哪些 token 参与, 可插拔)

每个选择算法一个脚本, @register_selector 注册, get_selector(name) 取用。
"""
from . import by_all
from . import by_syntax
from . import by_type
from . import by_intra
from . import layers
from ._registry import register_selector, get_selector, list_selectors
