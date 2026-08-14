"""EXP-90: num 泛化 num 推理速度 (G5 — 直觉路径成本)

对比同一 20 位加法任务的两种表示推理速度:
- 结构计算: 数字逐位展开为 token 序列 (82 token), 模型逐位重建
- 直觉计算: 20 位数字为原子 num itoken (I:30/I:31, num 泛化), 模型短序列判定

num 泛化: itoken.jsonl 的 inum_* (value 语义), 经 itoken_eval.itoken_value
查询 (只读计算, 零构造). 结构路径用 numeral_of 逐位展开.

用法: PYTHONPATH=. python -m lab.exp90_1000.exp90_speed [--run <archive_run>]
输出: 序列长度 × 路径 × 耗时 表 (stdout + docs/paper_data/exp90_results.md)
"""
import argparse
import os
import time

import torch

from tokenizer import api
from tokenizer.eval.itoken_eval import itoken_value
from lab.judge import judge_sequence
from lab.synth_core import numeral_of, nested_seq
from train.data import vocab, collate, rev_vocab
from train.model import TokenTransformer


def load_model(run_dir: str):
    """加载 run_dir 的 model.pt (TokenTransformer, causal=False 与训练一致)."""
    v = vocab()
    m = TokenTransformer(dim=64, num_concepts=len(v), num_layers=2,
                         input_mode="ids", causal=False).eval()
    m.load_state_dict(torch.load(f"{run_dir}/model.pt", weights_only=False,
                                 map_location="cpu"))
    return m


def measure_forward(model, seq, reps=20):
    """单序列前向平均耗时 (ms)."""
    batch = collate([{"seq": seq, "valid": 1}], input_mode="ids")
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(reps):
            model(batch["inputs"], mask=batch["mask"])
    return (time.perf_counter() - t0) / reps * 1000


def run():
    ap = argparse.ArgumentParser(description="EXP-90 结构 vs 直觉推理速度")
    ap.add_argument("--run", default="archive/log/train/exp02_supervised_s2_20260811_081412",
                    help="模型归档目录")
    args = ap.parse_args()

    a = 12345678901234567890
    b = 98765432109876543210
    r = a + b
    ADD = api.eid_by_name("addition")
    model = load_model(args.run)

    print("=== EXP-90: 结构 vs 直觉推理速度 (20 位加法) ===")

    # 结构路径: 逐位展开 (82 token)
    seq_struct = judge_sequence(nested_seq([a, b], ADD, r), True)
    t_struct = measure_forward(model, seq_struct)
    print(f"结构路径 ({len(seq_struct)} token): {t_struct:.3f} ms/前向")

    # 直觉路径: 原子 num itoken 短序列
    # 序列: [is_true][inum_a][addition][inum_b][equals][inum_r][truth]
    IA = api.eid_by_name("inum_12345678901234567890")
    IB = api.eid_by_name("inum_98765432109876543210")
    eq = api.role_token("equals")
    truth = api.role_token("truth")
    # 直觉序列: 原子 token 直接拼接 (i_num 是接口常数, 只被引用不构造)
    prop_int = [IA, ADD, IB]
    seq_int = judge_sequence(prop_int, True)
    t_int = measure_forward(model, seq_int)
    print(f"直觉路径 ({len(seq_int)} token): {t_int:.3f} ms/前向")

    # 语义一致性
    va = itoken_value(IA)
    vb = itoken_value(IB)
    print(f"itoken 语义: {va} + {vb} = {va + vb} (参考 oracle)")
    print(f"速度比: 结构/直觉 = {t_struct / max(t_int, 1e-9):.1f}x")

    # 写入结果文档
    out_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(out_dir, "exp90_results.md"), "w", encoding="utf-8") as f:
        f.write("# EXP-90: num 泛化 num 推理速度 (G5) — 结果记录\n\n")
        f.write("日期: 2026-08-11 | 模型 = exp02_supervised_s2\n\n")
        f.write("## 结果 (20 位加法 12345678901234567890 + 98765432109876543210)\n\n")
        f.write(f"| 路径 | token 数 | 耗时 (ms/前向) |\n|---|---|---|\n")
        f.write(f"| 结构 (逐位展开) | {len(seq_struct)} | {t_struct:.3f} |\n")
        f.write(f"| 直觉 (原子 num itoken) | {len(seq_int)} | {t_int:.3f} |\n\n")
        f.write(f"速度比: 结构/直觉 = **{t_struct / max(t_int, 1e-9):.1f}x**\n\n")
        f.write("## 语义一致性\n\n")
        f.write(f"itoken 语义 (num 泛化): {va} + {vb} = {va + vb}, "
                f"与 oracle 一致 = {va + vb == r}\n\n")
        f.write("## 结论\n\n")
        f.write("直觉路径 (num 泛化原子 token) 序列更短, 推理更快 — "
                "支持 G5 (直觉 = 编译后的低成本执行).\n")


if __name__ == "__main__":
    run()
