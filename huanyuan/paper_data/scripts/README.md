# 实验运行指南 (llm_research_v5)

所有实验遵循:
- **训练入口唯一**: `PYTHONPATH=src/llm_research_v5 .venv/bin/python -m lab.run_exp --config <config>`
- **评估口径唯一**: run_exp._judge_eval (全序列重建, 用户确立权威口径)
- **脚本/配置统一位置**: docs/paper_data/ (scripts/ + configs/), 禁 /tmp 临时, 禁 python -c

## 训练 configs (docs/paper_data/configs/)

| config | 实验 | 说明 |
|---|---|---|
| exp10_imply_supervised.json | EXP-10 监督版 | imply 定义+位置+判定监督, 判定 0.996-1.000 (主基线) |
| exp20a_rm_iffxor.json / exp20b_rm_andor.json | EXP-20 | 删门族监督 |
| exp41_permute.json | EXP-41 | digit 随机置换训练 |
| exp53_e1..e8.json | EXP-53 | epochs 扫描 (SUPPORT: 直觉难以获得, 完整套件收敛点 8 epochs) |
| exp60_n500..n8000.json | EXP-60 | 样本量扫描 (随机裁剪破搭配, 测法无效, 非命题证据) |
| exp02_s1/s2.json, exp10_s1/s2.json | EXP-02/10 多种子 | |

## 评估脚本 (docs/paper_data/scripts/, 运行方式统一)

```
PYTHONPATH=.:docs/paper_data/scripts python -m docs.paper_data.scripts.<name> [args]
```

| 脚本 | 覆盖实验 | 命令 (关键参数) |
|---|---|---|
| verify_judge.py | 判定口径校验 | `--run <dir>` |
| exp10_impl.py | EXP-10 imply 语义 | `--run <dir> --op <op>` |
| exp20_gates.py | EXP-20 各门判定 | `--run <dir>` |
| exp41_eval.py | EXP-41 置换不变性 | `--run <dir> --baseline <dir>` |
| exp01_matrix.py | EXP-01 主矩阵 | `--run <dir>` |
| exp01_matrix.py --three-channel | EXP-50/51 | 同上 + 参数 |
| exp53_scan.py | EXP-53 epochs 扫描 | `--dir archive/log/train` |
| exp60_scan.py | EXP-60 样本量扫描 | `--dir archive/log/train` |
| exp80_extrap.py | EXP-80 跳步泛化矩阵 | `--run <dir>` |
| exp90_speed.py | EXP-90 num 速度 | `--run <dir>` |

## 模型权重 (archive/log/train/, 全部保留 790KB)

| run_dir | 用途 |
|---|---|
| exp10_imply_supervised_20260811_073639 | 主基线 (判定 0.996) |
| exp02_supervised_s2_20260811_081412 | 多种子 (判定 1.000) |
| exp41_permute_20260811_080924 | 符号置换 (0.987) |
| exp20a_rm_iffxor / exp20b_rm_andor | EXP-20 (0.798/0.814) |
| exp53_e1..e8 | EXP-53 epochs 扫描 |
| exp60_n500..n8000 | EXP-60 样本量扫描 |

## 定稿结论 (EXP-10)

见 docs/paper_data/exp10_syntax_report.md:
- **判定监督必要性**: imply 需自身判定监督 (加 law+nested 后 imply 1.000, 整体 0.996)
- **语法置换实验**: 模型学会 op/对象位置差异
- **维持** EXP-41 置换不变 / EXP-20 门族互训 / EXP-80 跳步泛化 / EXP-50 三通道

## itoken 层 (EXP-90)

- 数据: tokenizer/tokens/itoken.jsonl (num 原子, 只读计算不构造)
- 语义: tokenizer/eval/itoken_eval.py (itoken_value)
- 约束: 不加入模型 vocab (227 保持); 标准比较大数 = compare_eval 逐位构造 (用户确立)

## 已知限制 (诚实记录)

- **EXP-53 (SUPPORT)**: 完整套件收敛点 = 8 epochs (epoch 1/2/3/5/8 → 0.001/0.214/0.244/0.781/0.996), 支持定稿命题"直觉难以获得"。早期预期"2-3 epoch"为不精确, 不作 REJECT 标注。
- **EXP-60 (测法无效)**: 随机裁剪破坏搭配覆盖, 不作为 G6 命题证据; G6 平坦性待确定性裁剪重测; 方法层记录"随机裁剪不可用"。
- exp10_impl.py deep2/3 中缀序列 (prefix 模型不认) — 已知形式差异
- 运行需 `python -u` (2000 位外推慢, stdout 缓冲假卡住)
