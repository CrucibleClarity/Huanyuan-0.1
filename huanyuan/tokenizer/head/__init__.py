"""head/ —— 注意算子 (同类/异类归类, 不是 token 属性)

head 不属于逻辑: token 数据无 head 字段, 也不该有。head 是观察者从
token 序列/体系中注意到的类别结构 (同类/异类)。

两层 (均可插拔, 每算法一脚本):
  select/  选择层: 哪些 token 参与
  compute/ 算法层: 怎么归类

  用法:
  from tokenizer.head import head_of, group_of
  head_of(seq)   # {eid: 类签名}
  group_of(seq)  # {类签名: [eids]}
"""
from .select import get_selector, list_selectors
from .compute import get_algorithm, list_algorithms

DEFAULT_SELECTOR = "all"
DEFAULT_ALGORITHM = "def_sig"   # definition 引用签名聚类 (同类/异类)


def group_of(sequence, selector=None, algorithm=None, **kw):
    """注意: 序列 → 同类/异类分组 {类签名: [eids]}。"""
    sel = get_selector(selector or DEFAULT_SELECTOR)
    alg = get_algorithm(algorithm or DEFAULT_ALGORITHM)
    selected = sel(sequence)
    return alg(selected, **kw)


def head_of(sequence, selector=None, algorithm=None, **kw):
    """注意: 序列 → {eid: 类签名}。"""
    groups = group_of(sequence, selector, algorithm, **kw)
    return {e: sig for sig, eids in groups.items() for e in eids}
