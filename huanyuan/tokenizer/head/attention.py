"""head/attention.py —— 注意力分析工具 (接口脚本)

对样本概念序列输出注意力分析基础向量 (同类性/嵌套蕴含/类别分离的结构信号):
  [x] 中括号:  一层定义引用            (construct.expand.bracket_vec)
  {x} 大括号:  穿透到公设的溯源链      (construct.expand.brace_logic)
  {x} 派生计数: eid 空间计数 (非零槽)  (construct.expand.brace_derived)
  嵌套层级:   AST 深度 → 每 token 深度与权重 (呈现层 P 层 parse 产物驱动)

只经权威接口 (construct/expand + _register), 不建平行访问。
采样 (哪些 token 参与) 走 head/select 选择器 (all 全量 / syntax 语法角色)。
"""
from __future__ import annotations

from .._register import token_of
from ..construct.expand import bracket_vec, brace_logic, brace_derived


def analyze(sequence) -> list[dict]:
    """概念 eid 序列 → 每概念 {eid, name, bracket, brace, counts} (注意力分析基础)。

    sequence: 概念 eid 列表 (如 ['D:206', 'D:250', 'D:204', 'D:260', 'D:209'])。
    同类概念共享 brace 溯源链 / 中括号引用; 派生计数含结构编码 (successor 计数=数值)。
    """
    rows, seen = [], set()
    for e in sequence:
        if e in seen:
            continue
        seen.add(e)
        bd = brace_derived(e)
        rows.append({
            "eid": e,
            "name": token_of(e).name,
            "bracket": bracket_vec(e),
            "brace": brace_logic(e),
            "counts": {k: v for k, v in bd.items() if v > 0},
        })
    return rows


def depth_map(ast) -> dict:
    """语法嵌套层级识别: AST → {概念 eid: 嵌套深度} (根=0, 子层+1)。

    ast: grammar.assemble / parse 产物 (呈现层 P 层 precedence/associativity 决定嵌套结构)。
    """
    out = {}

    def walk(node, d):
        if node.get("concept"):
            out[node["concept"]] = d
        children = node.get("children", [])
        if "fn" in node.get("slots", []):
            children = children[1:]
        for c in children:
            if isinstance(c, dict):
                walk(c, d + 1)
            elif isinstance(c, str) and c.startswith("D:"):
                out[c] = d + 1

    walk(ast, 0)
    return out


def depth_weight(ast, mode="inverse") -> dict:
    """按嵌套深度加权采样样本: {概念 eid: 权重}。

    mode:
      inverse     w = 1/(1+d)           (越深越弱, 线性衰减)
      inverse_log w = 1/log2(2+d)       (越深越弱, 对数衰减, 深层保留更多)
    """
    dm = depth_map(ast)
    if mode == "inverse":
        return {e: 1.0 / (1.0 + d) for e, d in dm.items()}
    if mode == "inverse_log":
        import math
        return {e: 1.0 / math.log2(2.0 + d) for e, d in dm.items()}
    raise ValueError(f"未知加权模式: {mode!r} (inverse / inverse_log)")
