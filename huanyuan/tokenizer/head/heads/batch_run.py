"""heads/batch_run.py —— 批量运行 (多样本 × 头配置)

对一批样本执行同一 runner (Head.run / MultiHead.run), 结果与样本一一对应。
每样本可携带独立 ctx (不同 AST), 或共享一个 ctx。
"""
from __future__ import annotations


def batch_run(samples, runner, ctx=None):
    """samples: 序列列表 或 (sequence, ctx) 列表; runner: 可调用 run(sequence, ctx)。

    返回 [结果], 与 samples 同序。
    """
    results = []
    for item in samples:
        if isinstance(item, tuple) and len(item) == 2:
            seq, c = item
            results.append(runner(seq, c))
        else:
            results.append(runner(item, ctx))
    return results
