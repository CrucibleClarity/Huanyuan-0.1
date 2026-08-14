"""compute/ —— head 算法层 (怎么算, 可插拔)

每个算法一个脚本, @register_algorithm 注册, get_algorithm(name) 取用。
算法:
  def_sig      — 内涵: definition 引用集相似度聚类
  brace_sig    — 内涵: 溯源展开 {x} 相似度聚类 (对比更本质)
  ext_sig      — 外延: 样本中共现上下文相似度聚类
  weight_depth — 嵌套深度权重变换 (突出谓词/操作数)
  weight_polar — 谓词/其他极性权重
  weight_layer — 角色层权重 (谓词/名词/装饰, 高低可配)
  weight_type  — 按类型顺序权重划分
"""
from . import def_sig, brace_sig, ext_sig, weight, dedup
from ._registry import register_algorithm, get_algorithm, list_algorithms
