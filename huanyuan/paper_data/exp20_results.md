# EXP-20 平行概念移出 (G2) — 结果记录

日期: 2026-08-11 | 基线 = exp10_imply_supervised (整体判定 0.996, 8/9 门 ≥0.95)

## 方法

逻辑门家族为上位概念。变体 (从基线完整监督删除某门族全部监督 law+nested+arith+interdef):
- exp20a_rm_iffxor: 删 logical_iff + logical_xor 全部监督
- exp20b_rm_andor: 删 logical_and + logical_or 全部监督

## 结果 (run_exp _judge_eval 判定口径, 各门 math 命题 OOD)

| 门 | 基线 | exp20a (删 iff/xor) | exp20b (删 and/or) |
|---|---|---|---|
| logical_and | 1.000 | 0.833 | **0.000** (被删) |
| logical_or | 1.000 | 0.833 | **0.000** (被删) |
| logical_imply | 1.000 | 0.000 (无判定监督) | 0.000 |
| logical_iff | 1.000 | **0.000** (被删) | 0.833 |
| logical_xor | 1.000 | **0.000** (被删) | 0.833 |
| logical_nand/nor/xnor | 1.000 | 0.833 | 0.833 |
| **整体** | 0.996 | **0.798** | **0.814** |

## 结论

**平行概念移出验证 G2**:
1. 删某门族全部监督 → 该门判定崩 (0.000)
2. **其他门也从 1.0 降到 0.833** — 门族互训链断裂 (De Morgan 对偶/inverse 箭头
   依赖互译), 平行概念同训是学习信号
3. imply 两个变体都 0 — imply 需要自身判定监督 (见 exp10_syntax_report §7)

**SUPPORT** — 逻辑门家族平行训练是泛化必要条件, 删一族影响全体。
