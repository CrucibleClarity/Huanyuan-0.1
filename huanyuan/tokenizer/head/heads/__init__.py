"""heads/ —— 多头机制 (头实例 + 多头编排 + 批量运行)

Head      头实例: (采样器, 计算配置, 参数) 组合, 独立关注一个维度
MultiHead 多头: 多头并行 run + 归并 (mean/weighted/concat/vote)
batch_run 批量: 多样本 × 头配置执行

复用 select/compute registry 与 router, 不新增层。
"""
from .head import Head
from .multihead import MultiHead, combine_results
from .batch_run import batch_run

__all__ = ["Head", "MultiHead", "combine_results", "batch_run"]
