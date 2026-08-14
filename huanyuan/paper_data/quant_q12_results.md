# 现状最优模型 q1/q2 量化评估 (方案 1: PyTorch 块量化)

日期: 2026-08-11 | 模型 = exp02_supervised_s2 (最优, fp32 判定 1.000)
脚本: docs/paper_data/scripts/quant_q12.py

## 背景 (架构限制)

TokenTransformer (195K 参数, Encoder-only 双向注意力, 无位置编码, 双 head)
**不兼容 llama.cpp/vllm 的 GGUF decoder-only 架构** — 无法直接 q1/q2 量化。
用户选择: ①PyTorch 块量化 ②标准 decoder 重训复现。

## 方案 1: PyTorch 对称块量化 (零外部依赖)

- q1: 每块 1bit 对称量化 (qmax=1, ±1 两级)
- q2: 每块 2bit 对称量化 (qmax=3, ±3 四阶)
- block: 块大小 (scale 粒度); 只量化 ≥1000 参数权重层 (线性/注意力), norm/bias 保留 fp32

## 结果 (run_exp _judge_eval, 256 样本: logic 算术 + addition)

| 配置 | 判定口径 | Δ (vs fp32) | 量化后大小 |
|---|---|---|---|
| fp32 基线 | 1.0000 | — | 790KB |
| q1 (block=64) | 0.0000 | -1.00 | 821KB |
| q1 (block=32) | 0.0000 | -1.00 | 441KB |
| q1 (block=16) | 0.0000 | -1.00 | 465KB |
| **q2 (block=32)** | **0.9141** | **-0.086** | **441KB** |
| q2 (block=16) | 0.6836 | -0.32 | 465KB |

## 结论

1. **q1 (1bit) 完全崩** (0.0000) — 1bit 两级信息对 195K 超小模型致命,
   权重精度不足以维持判定链。
2. **q2 (2bit) 最佳 (block=32): 0.9141** — 精度保持 91%, 大小压缩 44%
   (790KB → 441KB)。2bit 块量化对超小模型是极限 (仍损失 9% 判定)。
3. **block=32 是甜点** (block=16 反而 0.68 — 小块 scale 粒度反而引入
   更多局部误差或特定层过量化)。

## 启示

- 195K 超小模型 q1 不可行, q2 勉强 (0.91) — 极端量化收益低 (仅 44% 压缩,
  精度损 9%)。这与大型 LLM (7B+, 冗余高) 的 q2 行为不同 — 超小模型无冗余。
- 标准 decoder 重训 (方案 3) 可走 llama.cpp/vllm 真实 q1/q2, 待做。
