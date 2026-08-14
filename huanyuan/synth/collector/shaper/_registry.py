"""shaper/_registry.py —— 展平顺序/节点编码策略注册 (可插拔)"""
_orders = {}
_encodes = {}


def register_order(name):
    def deco(fn):
        _orders[name] = fn
        return fn
    return deco


def register_encode(name):
    def deco(fn):
        _encodes[name] = fn
        return fn
    return deco


def get_order(name):
    if name not in _orders:
        raise KeyError(f"未知展平顺序: {name!r}, 可用: {list_orders()}")
    return _orders[name]


def get_encode(name):
    if name not in _encodes:
        raise KeyError(f"未知编码策略: {name!r}, 可用: {list_encodes()}")
    return _encodes[name]


def list_orders():
    return sorted(_orders)


def list_encodes():
    return sorted(_encodes)
