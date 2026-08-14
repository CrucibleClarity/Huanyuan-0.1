#!/usr/bin/env python3
"""全项目架构调查 — 结构化解析 arch-req-research 输出 (非 sed hack)。

用法: python tokenizer/arch_survey.py [v5根] [--depth N] [--chains] [--bypass]
对根下每层模块目录跑 arch-req-research 三阶段 (全感知/边界双向/旁路),
汇总: 边界表 (module | inbound | outbound | bypasses) + 模块依赖方向 + 旁路清单。
输出为工具原生伪 JSON 行的结构化解析 (splitTop 同工具口径), 零文本替换。
"""
import subprocess
import sys
import os
import re

TOOL = os.path.expanduser("~/.agents/tools/arch-req-research/index.js")
EXCLUDE = re.compile(r"(node_modules|\.venv|venv|site-packages|\.git|dist|build|__pycache__|\.cache)")


def split_top(s):
    """顶层逗号切分 (跟踪 [] {} 深度), 同 arch-req-research splitTop。
    保留给嵌套行; parse_line 快路径直接 str.split (消费行无嵌套)。"""
    out, cur, depth = [], "", 0
    for ch in s:
        if ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    return out


def parse_line(line):
    if not line.startswith("["):
        return None
    # 快路径: C 层 split (arch_survey 消费的 meta/chain/bypass 行语义无嵌套,
    # 字段无逗号; 嵌套行 (s/f) 由其他消费方走 split_top)
    return line[1:-1].split(",")


def meta_fields(meta):
    """meta 伪 JSON 行 → dict (inbound/outbound/bypasses)。"""
    d = {}
    for f in meta:
        if ":" in f and not f.startswith(("target:", "module:", "parent:")):
            k, _, v = f.partition(":")
            d[k] = int(v)
    return d


def module_dirs(root, depth):
    """根下 depth 层生产代码目录 (每目录=独立模块)。"""
    mods = []
    stack = [(root, 1)]
    while stack:
        d, lvl = stack.pop()
        if lvl > depth:
            continue
        try:
            entries = sorted(os.listdir(d))
        except OSError:
            continue
        for n in entries:
            if EXCLUDE.search(n):
                continue
            p = os.path.join(d, n)
            if os.path.isdir(p):
                rel = os.path.relpath(p, root)
                mods.append((rel, p))
                stack.append((p, lvl + 1))
    return sorted(mods, key=lambda x: x[0])


def run_module(mod_dir):
    """对单模块目录跑 arch-req-research, 返回 (meta, chains, bypasses, full)。
    full = code-analysis 透传段全量行 (s/f/u/run); ARCH_REQ_FULL=0 环境变量可关闭。"""
    r = subprocess.run(["node", TOOL, mod_dir], capture_output=True,
                       text=True, timeout=180)
    out = r.stdout.splitlines() if r.stdout else []
    meta = next((parse_line(l) for l in out if l.startswith("[meta,arch_req_research")), None)
    chains = [parse_line(l) for l in out if l.startswith("[chain")]
    bypasses = [parse_line(l) for l in out if l.startswith("[bypass")]
    full = []
    in_full = False
    for l in out:
        if l.startswith("[meta,code-analysis-full"):
            in_full = True
        elif in_full and l.startswith("["):
            full.append(l)
    return meta_fields(meta) if meta else {}, chains, bypasses, full


def run_all(root, depth=1):
    """一次 spawn arch-req-research 穿透全部模块 (目录递归, 默认带透传段),
    返回 {模块绝对路径: (meta, chains, bypasses, full)} — N 次子进程 -> 1 次.
    ARCH_REQ_FULL=0 环境变量可关透传 (解析减量)."""
    args = ["node", TOOL, root]
    if depth and depth != float("inf"):
        args += ["--depth", str(depth)]
    r = subprocess.run(args, capture_output=True, text=True, timeout=600)
    out = r.stdout.splitlines() if r.stdout else []
    result, cur, meta, chains, bypasses, full = {}, None, {}, [], [], []
    in_full = False
    for l in out:
        if l.startswith("[meta,module-start"):
            if cur is not None:
                result[cur] = (meta, chains, bypasses, full)
            cur = l[len("[meta,module-start,"):].rstrip("]")   # 伪 JSON 行收尾 ]
            meta, chains, bypasses, full = {}, [], [], []
            in_full = False
        elif l.startswith("[meta,code-analysis-full"):
            in_full = True
        elif l.startswith(("[meta,arch_req_research", "[chain", "[bypass")):
            # 单次 tuple 前缀匹配 (C 层), 类型分发用 split 首字段
            f = parse_line(l)
            t = f[0]
            if t == "meta":
                meta = meta_fields(f)
            elif t == "chain":
                chains.append(f)
            elif t == "bypass":
                bypasses.append(f)
        elif in_full and l.startswith("["):
            full.append(l)
    if cur is not None:
        result[cur] = (meta, chains, bypasses, full)
    return result


def main():
    root = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else ".")
    depth = 1
    show_chains = "--chains" in sys.argv
    show_bypass = "--bypass" in sys.argv
    show_full = "--no-full" not in sys.argv   # 默认展示 code-analysis 完整输出
    if "--depth" in sys.argv:
        depth = int(sys.argv[sys.argv.index("--depth") + 1])

    mods = module_dirs(root, depth)
    # 一次 spawn 拿全部模块结果 (各视图共用, 避免多遍全量子进程)
    results = run_all(root, depth)
    print("===== 边界表 (module | inbound | outbound | bypasses) =====")
    for rel, p in mods:
        meta, _, _, _ = results.get(os.path.abspath(p), ({}, [], [], []))
        print(f"[{rel} | in:{meta.get('inbound', 0)} | out:{meta.get('outbound', 0)} | bypass:{meta.get('bypasses', 0)}")

    if show_chains:
        print("\n===== 模块依赖方向 (X→Y, 去重) =====")
        seen = set()
        for rel, p in mods:
            _, chains, _, _ = results.get(os.path.abspath(p), ({}, [], [], []))
            for c in chains:
                if len(c) > 2 and "→" in c[2] and not c[2].startswith("lib:"):
                    edge = c[2]
                    if edge not in seen:
                        seen.add(edge)
                        print(edge)

    if show_bypass:
        print("\n===== 旁路清单 (绕过模块直接调用外部, 去重计数) =====")
        agg = {}
        for rel, p in mods:
            _, _, bypasses, _ = results.get(os.path.abspath(p), ({}, [], [], []))
            for b in bypasses:
                key = (b[1], b[2])
                agg[key] = agg.get(key, 0) + 1
        for (target, note), n in sorted(agg.items()):
            print(f"x{n} {target} ({note})")

    if show_full:
        # code-analysis 完整输出透传 (s 依赖链 / f 函数调用链+代码 / u 悬空引用), 每模块一段
        print("\n===== code-analysis 完整输出 (s/f/u, 每模块) =====")
        for rel, p in mods:
            _, _, _, full = results.get(os.path.abspath(p), ({}, [], [], []))
            if full:
                print(f"[module {rel}]")
                for l in full:
                    print(l)


if __name__ == "__main__":
    main()
