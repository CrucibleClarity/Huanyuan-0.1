# int8 QAT 训练体积测试 (2026-08-11)

模型 = exp02_supervised_s2 套件 (qat_bits=8) | 训练: qat_int8_20260811_140441

## 方法

量化感知训练 (QAT): train_seq 每 epoch 后 `_qat_round_weights` 把权重量化
回 int8 格点 (straight-through) — 训练全程权重约束到 int8 可表示值.

## 结果

| 指标 | fp32 基线 | int8 QAT |
|---|---|---|
| 判定口径 (run_exp _judge_eval) | 1.000 | **1.000** (零损失) |
| 权重量化误差 | — | **0.00e+00** (在 int8 格点) |
| model.pt 体积 (fp32 存储) | 790,790B | 790,790B (torch.save 仍 fp32) |
| **int8 打包体积** | — | **214,894B (27.2%)** |
| 压缩比 | — | **3.68x** |

## 结论

1. **int8 QAT 训练判定口径 1.000 零损失** — 权重约束到 int8 格点后模型
   精度完全保持 (195K 模型对 int8 量化鲁棒).
2. **真实体积压缩 3.68x** (790KB → 215KB) — int8 打包存储, 判定无损.
3. 对比方案 1 (事后 q1/q2 块量化): int8 QAT **训练时**量化, 精度远优于
   事后 1/2bit (q2 0.91, q1 崩) — 量化感知训练是超小模型量化的正确路径.

## 实现

- train/__init__.py: train_seq 加 `qat_bits` 参数, `_qat_round_weights` 每 epoch 量化格点
- run_exp.py: config `train.qat_bits` 透传
- config: docs/paper_data/configs/qat_int8.json
- 模型: archive/log/train/qat_int8_20260811_140441
