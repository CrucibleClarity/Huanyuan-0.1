"""synth/ —— 样本合成器 (tokenizer 外首个模块)

输入一个待训练 token, 合成该 token 的训练样本集。只经 tokenizer.api。
四个子模块:
  retriever  检索器   待训练 token → 可用语法范围/序列/每语法位可填范围
  selector   样本选择器 范围 → 选语法/嵌套层数/槽位 token, 确定训练样本
  collector  收拢器   嵌套向量 → transformer 输入向量 (方法可插拔)
  aggregator 样本集聚合器 定义样本 + 合成样本 → 样本集

主接口:
  from synth import build_sample_set, shape
  build_sample_set(token_eid, n_synth=..., depth=..., vector_method=...)
  shape(sample, output=..., order=..., encode=...)  → 样本 → 整形向量 (格式转换)
"""
from __future__ import annotations

from .aggregator.aggregate import build_sample_set
from .collector.collapse import METHODS as VECTOR_METHODS
from .collector.shaper.shape import shape as _shape

shape = _shape

__all__ = ["build_sample_set", "VECTOR_METHODS", "shape"]
