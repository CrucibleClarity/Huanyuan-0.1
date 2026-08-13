"""tokenizer/cte.py —— 统一 CTE 预编译缓存 (公共子表达式消除, 幂等重编译)

所有 gtoken 派生 (arrange/presentation/slots/law 模板等) 注册到统一缓存:
  - 预编译阶段: 首次调用或数据签名变化时编译一次
  - 签名校验: 缓存项带数据签名, 数据 (token 定义) 变化 → 幂等重编译
  - 统一失效: maintain 写入 / 实验注入后调 invalidate_all, 全部缓存失效
     (幂等: 下次访问重编译, 无脏状态)

用法:
  from tokenizer import cte
  cte.get_or_compile('slots', 'atom', sig_fn, compile_fn)
  cte.invalidate_all()   # maintain 写入后 / 实验结束
"""
from __future__ import annotations

import threading

# 缓存项: {key: (signature, value)}; signature 由调用方数据签名决定
_CACHE: dict = {}
# RLock: compile_fn 内部可能再调 cte (重入), 需可重入锁
_LOCK = threading.RLock()
# 全局数据版本: maintain 写入/实验注入时 +1, 签名基于它 (无需每项 stat)
DATA_VERSION = 0


def _version():
    """当前数据版本 (签名基准): maintain 写入/cte 失效时递增."""
    return DATA_VERSION


def get_or_compile(key, signature, compile_fn):
    """CTE 读取: 签名一致返回缓存, 变化幂等重编译 (锁保护).

    key:       缓存项标识 (如 'slots:atom')
    signature: 当前数据签名 (token 定义变化则变) — 由调用方算
    compile_fn: () -> value  编译函数 (纯函数, 幂等可重入)
    """
    global _CACHE
    hit = _CACHE.get(key)
    if hit is not None and hit[0] == signature:
        return hit[1]
    with _LOCK:
        # 双重检查 (并发下另一线程可能已编译)
        hit = _CACHE.get(key)
        if hit is not None and hit[0] == signature:
            return hit[1]
        value = compile_fn()
        if len(_CACHE) < 20000:
            _CACHE[key] = (signature, value)
        return value


def invalidate_all():
    """全部 CTE 缓存失效 (幂等: 无脏状态, 下次访问重编译).

    maintain 写入后 / 实验临时注入后 / 数据文件变化时调用.
    数据版本递增 (签名基准), 旧缓存项签名不匹配 → 下次访问幂等重编译.
    """
    global _CACHE, DATA_VERSION
    with _LOCK:
        _CACHE.clear()
        DATA_VERSION += 1


def invalidate_prefix(prefix: str):
    """按前缀失效 (如 'slots:' 只清 gtoken 槽位缓存). 数据版本递增."""
    global _CACHE, DATA_VERSION
    with _LOCK:
        _CACHE = {k: v for k, v in _CACHE.items() if not k.startswith(prefix)}
        DATA_VERSION += 1


def cache_stats() -> dict:
    """缓存统计 (诊断用): {key: 命中类型}."""
    return {k: "cached" for k in _CACHE}


__all__ = ["get_or_compile", "invalidate_all", "invalidate_prefix", "cache_stats"]
