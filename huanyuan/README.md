# 幻元 (Huanyuan)

可解释泛化的最小可复现包 — 极小 Transformer 通过因子化 token 体系实现系统化外推。

## 授权

**保留所有权利 (All Rights Reserved)**。非开源许可 — 本包**禁止** MIT/Apache/BSD 等
宽松授权。任何使用、修改、部署、商业应用均需**事先申请书面授权**。详见 LICENSE。

## 环境

- Python 3.14 + PyTorch 2.x (纯 PyTorch, 无其他依赖)
- 无 CUDA 可跑 (CPU 推理; 2000 位外推建议 GPU)

## 结构

```
huanyuan/
├── tokenizer/          token 体系 (数据 tokenizer/tokens/ + 求值引擎)
├── train/              训练 (train_seq + int8 QAT)
├── verify/             验证 (run_exp._judge_eval 权威判定口径)
├── lab/                实验编排 (run_exp) + 样本合成 (synth_core)
├── synth/ infer/       合成/推理辅助
├── experiment/         统一接口
├── archive/            模型权重 (6 个关键 checkpoint)
└── paper_data/
    ├── configs/        实验配置
    ├── scripts/        评估脚本
    └── *.md, results.csv  结果表
```

## 快速验证 (复现全部核心结论)

```bash
cd huanyuan
export PYTHONPATH=.:paper_data/scripts

# 1. 判定口径校验
python -m paper_data.scripts.verify_judge --run archive/log/train/exp02_supervised_s2_20260811_081412

# 2. imply 语义 (主基线)
python -m paper_data.scripts.exp10_impl --run archive/log/train/exp10_imply_supervised_20260811_073639
#   预期: imply math OOD acc=1.000

# 3. 逻辑门家族判定
python -m paper_data.scripts.exp20_gates --run archive/log/train/exp10_imply_supervised_20260811_073639
#   预期: 9 门全 1.000

# 4. 符号置换不变性 (学关系非符号)
python -m paper_data.scripts.exp41_eval --run archive/log/train/exp41_permute_20260811_080924 --baseline archive/log/train/exp10_imply_supervised_20260811_073639
#   预期: 0.987 vs 0.996 (差 ≤0.02)

# 5. 主 OOD 矩阵 + 三通道一致
python -m paper_data.scripts.exp01_matrix --run archive/log/train/exp10_imply_supervised_20260811_073639
python -m paper_data.scripts.exp01_matrix --run archive/log/train/exp10_imply_supervised_20260811_073639 --three-channel
#   预期: 矩阵 ≥0.996, 三通道 1.000

# 6. 外推零衰减 (跳步泛化)
python -m paper_data.scripts.exp80_extrap --run archive/log/train/exp02_supervised_s2_20260811_081412
#   预期: 2000位 / 进制60 / 100%置换 全 1.000

# 7. 去因子化对照 (迭代隐藏崩)
python -m paper_data.scripts.expc1_eval --run archive/log/train/expc1b_iter_hidden_20260811_115723
#   预期: root 0.000 / tetration 0.000, addition 1.000

# 8. int8 QAT 无损压缩
python -m paper_data.scripts.quant_q12 --run archive/log/train/qat_int8_20260811_140441
```

## 重新训练

```bash
python -m lab.run_exp --config paper_data/configs/exp10_imply_supervised.json
python -m lab.run_exp --config paper_data/configs/exp02_s2.json
python -m lab.run_exp --config paper_data/configs/qat_int8.json   # int8 QAT
```

## 权威口径

**判定口径 (run_exp._judge_eval) = 全序列重建** (每位置预测与真实 seq 一致,
全对才计正确)。位置级 acc 不可信 (会掩盖结构错误)。所有数值均用判定口径。

## 关键结果一览

| 实验 | 结果 | 文件 |
|---|---|---|
| EXP-10 imply 监督 | imply OOD 1.000, 整体 0.996 | exp10_syntax_report.md |
| EXP-20 门族互训 | 9 门全 1.000 | exp20_results.md |
| EXP-41 符号置换 | 0.987 (Δ0.009) | exp41_results.md |
| EXP-50 三通道 | 构造=形式=直觉 1.000 | exp50_51_results.md |
| EXP-80 外推 | 2000位/进制60/100%置换全 1.000 | exp80_results.md |
| EXP-C1b 迭代 | root/tetration 0.000 (崩) | expc1_results.md |
| int8 QAT | 判定 1.000, 3.68x 压缩 | qat_int8_results.md |

## 已知限制

- 195K 超小模型, q1/q2 事后量化精度差 (q2 0.91, q1 崩); int8 QAT 是无损路径
- llama.cpp/vllm GGUF 仅支持 decoder-only, 本项目 encoder-only 架构不可转
