#!/usr/bin/env python3
"""ns_profile.py —— 样本生成单循环逐次纳秒级 profile

对核心函数做单循环测量: 每次调用独立计时 (time.perf_counter_ns),
观测单次调用的耗时分布 (哪些次慢, 是否逐次读写/缓存未命中导致的抖动).

与完整 profile 的区别: 不跑全量, 只跑少量循环 (如 20 次), 逐次记录
单次耗时 — 找到 单次慢调用 (缓存冷启动/逐次读写/嵌套展开路径) 而非平均.

用法:
  PYTHONPATH=src python -m lab.diag.ns_profile
"""
import json
import os
import sys
import time

from tokenizer import api
from lab import synth_core


def per_call(name, fn, n=20):
    """单循环逐次测量: 每次调用独立计时, 输出单次耗时 (ns)."""
    print(f"=== {name} ({n} 次单循环) ===")
    times = []
    for i in range(n):
        t = time.perf_counter_ns()
        fn()
        dt = time.perf_counter_ns() - t
        times.append(dt)
    print(f"  单次耗时 (ns): {times}")
    print(f"  min={min(times)} median={sorted(times)[len(times)//2]} max={max(times)} "
          f"avg={sum(times)//len(times)}")
    print()


def main():
    e5 = api.eid_by_name("digit_five")
    op = api.eid_by_name("addition")

    # tokenizer 原语: 观测缓存冷启动 vs 命中
    per_call("api.assemble_seq(atom)", lambda: api.assemble_seq(e5, []))
    per_call("api.arrange_slots(atom)", lambda: api.arrange_slots("atom"))
    per_call("api.presentation_of", lambda: api.presentation_of(e5))

    # 样本组合: 观测嵌套展开路径
    per_call("numeral_of(25)", lambda: synth_core.numeral_of(25))
    per_call("nested_seq([2,3],+,5)", lambda: synth_core.nested_seq([2, 3], op, 5))
    per_call("judge_sequence", lambda: synth_core.judge_sequence([e5], True))

    # law 负例采样: 观测每次调用
    per_call("law_samples(division)", lambda: synth_core.law_samples(op="division", hi=9), 5)

    # 持久化
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("archive", "log", "profile")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"ns_profile_{ts}.txt")
    print(f"  已归档: {out_path}")


# ================= 穿透模式 (--chain): 单次调用调用树 =================
# 单次调用入口函数, sys.settrace 动态追踪调用树: 每个函数 次数/耗时/占父比例,
# 热节点 (占父 >= 50% 或 self >= 100µs) 机械显影 — 瓶颈层一眼可见, 模型只判定。
# 与清单模式分工: 穿透找层 (结构显影), 清单纯测 (perf_counter_ns 无污染)。
def _qualname(code_or_frame):
    code = code_or_frame.f_code if hasattr(code_or_frame, "f_code") else code_or_frame
    return getattr(code, "co_qualname", code.co_name)


def chain_profile(entry, kwargs, max_nodes=2000, warmup=1):
    # warmup: 先跑 N 次不计时 (模块加载/缓存冷启动不入调用树) — 穿透稳态热路径
    for _ in range(warmup):
        entry(**kwargs)
    root = {"name": _qualname(entry.__code__) if hasattr(entry, "__code__") else entry.__qualname__,
            "count": 1, "total": 0, "self": 0, "children": [], "n_nodes": 0, "truncated": False}
    stack = []  # [node, t0]
    events = [0]

    def tracer(frame, event, arg):
        events[0] += 1
        if event == "call":
            if root["n_nodes"] >= max_nodes:
                root["truncated"] = True
                return None
            if not stack:
                # 入口自身调用: 压哨兵不建节点 (root 已含入口)
                stack.append([None, time.perf_counter_ns()])
                return tracer
            node = {"name": _qualname(frame), "count": 1, "total": 0,
                    "self": 0, "children": []}
            root["n_nodes"] += 1
            if stack[-1][0] is None:
                root["children"].append(node)   # 入口内第一层: 挂 root
            else:
                stack[-1][0]["children"].append(node)
            stack.append([node, time.perf_counter_ns()])
        elif event == "return":
            if stack:
                node, t0 = stack.pop()
                if node is not None:
                    node["total"] += time.perf_counter_ns() - t0
        return tracer

    sys.setprofile(tracer)   # setprofile: 只发 call/return 事件, 比 settrace 省 ~25%
    try:
        t0 = time.perf_counter_ns()
        entry(**kwargs)
        root["total"] = time.perf_counter_ns() - t0
    finally:
        sys.setprofile(None)
    root["events"] = events[0]

    # self = total - children total (叶子 self = total)
    def calc(node):
        cs = sum(calc(c) for c in node["children"])
        node["self"] = max(0, node["total"] - cs)
        return node["total"]
    calc(root)
    return root


def _print_tree(node, parent_total, depth=0, prefix="", is_last=True):
    """调用树缩进显影: 每行 函数 (次数, 耗时, 占父%) + 热标记; 微叶子折叠."""
    if depth == 0:
        pct = 100.0
        branch = ""
    else:
        pct = node["total"] / parent_total * 100 if parent_total else 0
        branch = "└─ " if is_last else "├─ "
    hot = " ← 热" if (depth > 0 and pct >= 50) or node["self"] >= 100_000 else ""
    print(f"{prefix}{branch}{node['name']} ({node['count']}次, {node['total']//1000}µs, "
          f"self {node['self']//1000}µs, {pct:.0f}%){hot}")
    child_prefix = prefix + ("" if depth == 0 else ("    " if is_last else "│   "))
    kids = node["children"]
    micro = [c for c in kids if not c["children"] and c["self"] < 2000]
    kids = [c for c in kids if c not in micro]
    for i, c in enumerate(kids):
        _print_tree(c, node["total"], depth + 1, child_prefix, i == len(kids) - 1)
    if micro:
        ms = sum(c["self"] for c in micro)
        print(f"{child_prefix}└─ +{len(micro)} 微调用 (共 {ms//1000}µs, self < 2µs 折叠)")


def chain_main():
    """--chain <module.func> [k=v ...]: 入口单次调用穿透调用链. 例:
       python -m lab.diag.ns_profile --chain lab.synth_core.law_samples op=division hi=9"""
    import importlib
    if len(sys.argv) < 3:
        print("用法: --chain <module.func> [k=v ...]")
        sys.exit(1)
    mod_name, _, fn_name = sys.argv[2].rpartition(".")
    kwargs = {}
    for a in sys.argv[3:]:
        k, _, v = a.partition("=")
        kwargs[k] = int(v) if v.lstrip("-").isdigit() else v
    mod = importlib.import_module(mod_name)
    entry = getattr(mod, fn_name)
    print(f"=== 穿透: {sys.argv[2]}({kwargs}) 单次调用 ===")
    print("  注: 穿透模式含 trace 开销 (~70×), 看【占比】找层; 绝对耗时用清单模式纯测")
    tree = chain_profile(entry, kwargs)
    _print_tree(tree, tree["total"])
    print(f"\n  入口总耗时 {tree['total']//1000}µs (含 trace 开销); 节点 {tree['n_nodes']}, 事件 {tree.get('events', '?')}"
          f"{' [截断: 深处未显影, 对热节点单独 --chain]' if tree.get('truncated') else ''}")
    print("  用法: 沿 ← 热 节点向下看瓶颈层; 对热节点再穿透: --chain <热节点模块.函数> ...")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--chain":
        chain_main()
    else:
        main()
