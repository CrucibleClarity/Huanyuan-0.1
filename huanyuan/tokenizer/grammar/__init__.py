"""grammar/ —— 语法层 (G 层: gtoken, token 的排列方法)

G 层存 token 的排列方法 (怎么用), 数据由 maintain 单一入口管理 (G: eid, 已进 PROJECTIONS)。
gtoken 用 definition 定义排列 {eid, name, definition, reduction}: form=explicit +
rules.term 槽位序列 (槽位角色词 arg:N=子项/fn=函数/args=参数列表/binder=绑定/body=体)。
节点类型 atom/application/equality/binary_connective/unary_connective/quantified/eos;
reduction=归约规则 (如 SKI)。中缀/优先级/符号=呈现层 (P 层), 不进 gtoken。

★ 统一 IO 接口: 消费方只经本包入口 (query/scope/assemble/execute/discover),
  禁止直调子模块 (_query/_assemble/_execute/_schema)。
★ gtoken 是唯一语法源: 所有排列知识从 gtoken 读, 组装器不硬编码节点/槽位语义。

公共接口:
  discover()              → 全部 gtoken
  query(node)             → 查节点排列方法 (node = G:eid 或 name)
  scope(node)             → 可组装范围 (arity 要求 + 槽位布局)
  assemble(node, children)→ 按 gtoken 组装 (校验槽位, 返回嵌套 AST 节点)
  execute(tasks)          → 批量执行 query/scope/assemble
"""
from ._schema import discover
from ._query import query, scope
from ._assemble import assemble
from ._execute import execute
from ._ski import reduce, abstract
from ._present import print_ast, parse
from ._bind import check_scope

__all__ = ['discover', 'query', 'scope', 'assemble', 'execute', 'reduce', 'abstract', 'print_ast', 'parse', 'check_scope']
