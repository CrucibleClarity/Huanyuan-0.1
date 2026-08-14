"""tokenizer/grammar/_schema.py —— G 层只读访问 (数据源)

G 层 (gtoken: token 的排列方法) 数据由 maintain/core.PROJECTIONS 单一管理:
注册/校验/诊断走 maintain 单一入口 (G: eid), 本模块只暴露只读查询,
数据源统一 core.load_layer('G'), 不另起加载 (不建平行系统)。

★ gtoken 用 definition 定义排列 (form=explicit, rules.term 槽位序列): 槽位角色词
  arg:N/fn/args/binder/body; 排列知识在 definition (唯一语法源), 无 arity/slots 旁路字段。

公共接口:
  discover()   → 全部 gtoken (供消费方遍历)
组装/查询/批量走统一入口 tokenizer.grammar (query/scope/assemble/execute)。
"""
from ..maintain import core


def discover():
    """列出全部 gtoken (排列方法), 数据源统一 core.load_layer('G')。"""
    return [dict(r) for r in core.load_layer('G').values()]
