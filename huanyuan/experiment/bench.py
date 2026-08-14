"""experiment/bench.py —— 实验批跑脚本 (0 硬编码, 归档可复现)

跑完整管线 (合成 → 训练 → 验证 → 归档), 不拼装, 全部经 experiment.run。
参数命令行给定, seed 固定 → 每次运行归档可复现 (log/train/<run>/ 含 config/samples/model/metrics/views)。

用法:
  python -m experiment.bench --tokens D:250 D:251 --exclude D:260 \
      --shape defexpand --n 150 --epochs 12 --seed 3 --depth 1
"""
from __future__ import annotations

import argparse

from experiment import run
from tokenizer import api


def main() -> None:
    ap = argparse.ArgumentParser(description="实验批跑 (0 硬编码, 归档可复现)")
    ap.add_argument("--tokens", nargs="+", required=True, help="待训练 ctoken (首个为主 token)")
    ap.add_argument("--exclude", nargs="*", default=[], help="排除 token (拼装泛化目标)")
    ap.add_argument("--shape", default="sequence_counts", help="shaper 输出形态")
    ap.add_argument("--ids", action="store_true", help="token id 序列输入 (嵌入+位置, 非向量)")
    ap.add_argument("--expand", type=int, default=None, help="ids 模式语法展开深度 (None=平铺)")
    ap.add_argument("--n", type=int, default=100, help="每 token 样本数")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    main_tok = args.tokens[0]
    res = run(main_tok, n_samples=args.n, epochs=args.epochs, depth=args.depth,
              seed=args.seed, shape_method=args.shape, tokens=args.tokens,
              exclude=args.exclude,
              input_mode="ids" if args.ids else "vector", expand=args.expand)
    v = res["views"]
    print(f"token={main_tok}({api.name(main_tok)}) run={res['run_dir']}")
    print(f"  mode={'ids-expand'+str(args.expand) if args.ids else 'vector'} "
          f"overall={v['overall_acc']:.3f} ood={v['ood_acc']:.3f} "
          f"gen={v['gen_acc']:.3f} gen_success={v['gen_success']:.3f}")
    print(f"  多层嵌套泛化: {v.get('depth_gen', {})}")
    print(f"  curve={[round(x, 3) for x in v['curve']['losses']]}")
    print(f"  exclude={[api.name(e) for e in args.exclude]}")
    print(f"  归档: {res['run_dir']} (config/samples/model/metrics/views.json)")


if __name__ == "__main__":
    main()
