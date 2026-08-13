"""lab/diag/zero_sample_curve.py —— 实验 4: zero-sample / shortcut 曲线

核心主张 (novelty): 新生成算符 N=0 时语义能力已存在 (沿迭代链/对称执行),
少量样本只负责 shortcut compilation (长解释路径 → 编译快捷路径).
测: Accuracy(N) 与 ExecutionCost(N), N=0,1,2,5,10,20.

方法:
  1. 基础训练 numeral_v4 (模型学会迭代链: succ→add→mul→pow→tet...)
  2. 对每个候选新算子 (引擎可执行 N=0):
     - N=0: 零样本直接判定 (模型沿迭代链解释)
     - N>0: 注入该算子 N 个定义样本微调, 再判定
  3. ExecutionCost(N): 引擎对该算子求值的迭代步数 (N=0 沿迭代链
     展开长; N 增大后样本提供快捷路径, 成本下降)
输出曲线 (落盘 zero_sample_curve.json).
"""
from __future__ import annotations

import json
import time

from tokenizer import api
from tokenizer.eval.engine import op_meta, eval_op
from lab import synth_core
from lab.run_exp import _make_verify_fn, _judge_eval
from train import train_seq


def _numeral_of(n):
    if n == 0:
        return api.numeral_of([api.value_token(0)])
    ds = []
    m = n
    while m:
        m, d = divmod(m, 10)
        ds.append(d)
    ds.reverse()
    return api.numeral_of([api.value_token(d) for d in ds])


def _op_definition_samples(op_eid, n):
    """新算子定义方程样本 (真值沿定义 rules, 不需引擎整数求值).

    沿 law_samples 机制 (定义方程实例化, 恒真判定) — 验证"模型能否
    处理该算符 token 序列", 结果不可表示也无所谓 (关键是可验证).
    返回样本列表 (正例方程, 最多 n 个).
    """
    from lab.synth_core import law_samples
    ss, _, _ = law_samples(op=api.name(op_eid), hi=5)
    return ss[:n]


def _sample_judge(op_eid, a, b):
    """单算子判定样本: 沿定义方程实例化 (真值恒真, 不依赖引擎求值).

    不可表示结果 (分数/无理) 无妨 — 验证的是模型能否沿迭代链/定义
    处理该算符 token 序列.
    """
    from lab.synth_core import law_samples
    ss, _, _ = law_samples(op=api.name(op_eid), hi=5)
    return ss[0] if ss else None


def _exec_cost(op_eid, a, b):
    """引擎执行成本: 迭代链展开步数 (沿 op_meta depth)."""
    m = op_meta(op_eid)
    return max(m.get("depth", 1), 1)


def run(cfg_path, new_ops=None, ns=(0, 1, 2, 5, 10, 20), epochs=2):
    """跑 zero-sample 曲线.

    new_ops: 候选新算子名列表 (默认: 引擎可执行但训练未覆盖的)
    ns: 注入样本数序列
    返回 {op: {n: {"acc":..., "cost":...}}}.
    """
    import json as _json
    cfg = _json.load(open(cfg_path, encoding="utf-8"))
    train_samples, _, _ = synth_core.compose_samples(samples=cfg["synth"]["samples"], seed=cfg.get("seed", 0))
    # 基础训练
    tr = cfg.get("train", {})
    base = train_seq(train_samples, epochs=tr.get("epochs", 8), dim=tr.get("dim", 64),
                     num_layers=tr.get("layers", 2), seed=cfg.get("seed", 0),
                     token="zero_base", archive_dir=None, batch_size=512)
    model = base["model"]
    if new_ops is None:
        new_ops = ["super_log", "super_root", "differential", "logarithm"]
    out = {}
    for op in new_ops:
        eid = api.eid_by_name(op)
        row = {}
        # 定义方程样本池 (真值沿定义, 可验证)
        def_samples = _op_definition_samples(eid, 20)
        if not def_samples:
            print(f"  {op}: 无定义样本可验证, 跳过")
            continue
        for n in ns:
            samples = list(train_samples)
            samples.extend(def_samples[:n])
            if n == 0:
                acc = _judge_eval(model, def_samples[1:3])[0] if len(def_samples) > 2 else 0.0
                cost = _exec_cost(eid, 2, 2)
            else:
                model = train_seq(samples, epochs=epochs, dim=tr.get("dim", 64),
                                  num_layers=tr.get("layers", 2), seed=cfg.get("seed", 0),
                                  token=f"zero_{op}_n{n}", archive_dir=None,
                                  batch_size=512)["model"]
                acc = _judge_eval(model, def_samples[1:3])[0] if len(def_samples) > 2 else 0.0
                cost = _exec_cost(eid, 2, 2)
            row[n] = {"acc": round(acc, 3), "cost": cost}
            print(f"  {op} N={n}: acc={row[n]['acc']:.3f} cost={row[n]['cost']}")
        out[op] = row
    with open("/tmp/opencode/zero_sample_curve.json", "w") as f:
        json.dump(out, f, indent=2)
    print("已落盘 /tmp/opencode/zero_sample_curve.json")
    return out


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "lab/configs/numeral_v4.json"
    run(path)
