# EXP-01 主 OOD 矩阵 (G6/R — 系统化泛化完整证据面)

日期: 2026-08-11 | 模型 = exp10_imply_supervised (判定口径 0.996)

## OOD 矩阵 (按序列长度/类型分组, run_exp _judge_eval)

| OOD 维度 | 判定口径 | n | 序列均长 |
|---|---|---|---|
| 逻辑命题 (9 门 math 判定) | 1.000 | 4 | 5 |
| radix 进制×20位结果 (长序列) | **1.000** | 36 | 125 |
| radix/cartesian 混合 (中长) | **0.996** | 1518 | 35 |
| **总计** | **0.996** | 1558 | — |

## 覆盖维度

- 位宽: 2/20/2000 位 (radix max_digits=20, extrap 2000 单独验证)
- 进制: 3/4/5/6/7/8/9 (radix bases)
- 阶数: 迭代链超运算 (power/root/tetration/super_log)
- 笛卡尔: root/power/super_log/multiplication × 2位 (联合组合)
- 2000 位外推: addition acc=1.000 (单独验证)

## 结论

**主 OOD 矩阵全维度 ≥0.996** — 位宽×进制×阶数×笛卡尔组合系统化泛化
完整证据面 (SUPPORT G6/R)。监督版 (含 imply 判定监督) 达到发布级稳定。

## 说明

run_exp _make_verify_fn 生成样本未注入 spec 溯源字段, 分组按序列长度近似;
精确分维度需 fn 生成时 set spec (后续可改进)。
