"""select/_registry.py —— 选择器注册 (可插拔, 每算法一脚本)

@register_selector("名字") 装饰器注册, get_selector(name) 取用。
"""
_registry = {}


def register_selector(name):
    def decorator(fn):
        _registry[name] = fn
        return fn
    return decorator


def get_selector(name):
    if name not in _registry:
        raise KeyError(f"未注册选择器: {name!r}, 可用: {list(_registry)}")
    return _registry[name]


def list_selectors():
    return sorted(_registry)
