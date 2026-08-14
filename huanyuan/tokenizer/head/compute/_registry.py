"""compute/_registry.py —— 算法注册 (可插拔, 每算法一脚本)

@register_algorithm("名字") 装饰器注册, get_algorithm(name) 取用。
"""
_registry = {}


def register_algorithm(name):
    def decorator(fn):
        _registry[name] = fn
        return fn
    return decorator


def get_algorithm(name):
    if name not in _registry:
        raise KeyError(f"未注册算法: {name!r}, 可用: {list(_registry)}")
    return _registry[name]


def list_algorithms():
    return sorted(_registry)
