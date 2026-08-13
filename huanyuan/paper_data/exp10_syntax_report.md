# EXP-10 语法置换实验报告 (维持发现) — 2026-08-11

## 维持的核心发现

### 1. 语法置换实验 (用户指出, 正面结果)

imply 的 ptoken grammar 是**中缀** `[arg:0, →, arg:1]` (imply 在中间符号位置),
但合成器曾用裸列表生成**前缀** `[is_true][logical_imply][T][F]` (op 位置, is_true 后)。

模型在 op 位置预测 `base` (D:143, 算术判定常见开头) 而非 logical_imply —
**模型坚持 op 位置的正确 token, 拒绝把 imply (中缀算子) 错放到 op 位置**。

**证明模型学会了 token 的语法角色 (op 位置 vs 对象位置), 而非机械记忆序列。**

### 2. 判定监督必要性 (修正后的正确结论)

加 imply law (4 行真值表) + nested 判定监督后:
- imply 裸真值表: **1.000** (含 F→F)
- imply math 命题 OOD: **1.000**
- 整体判定口径: **0.996**

**判定监督是 operator-specific 学会的必要条件** (仅定义等式 imply 严格口径 0.000)。

### 3. 语法统一原则 (实现固化)

`_assemble_logic(op, args, notation)` 统一逻辑组装:
- prefix: `[op][arg0][arg1]` (默认, 全体样本统一)
- infix: `[arg0][op][arg1]` (沿 ptoken)
全部逻辑合成器 (logic_samples/nested/arith/interdef) 走统一组装, config 可切换.

## 评估口径纪律 (教训)

- **判定口径 (run_exp._judge_eval 全序列重建) 是唯一可信口径** — 末尾真值会掩盖结构错误.
- 逻辑门评估必须批量 (单样本 collate 有边界问题).

## 关键模型

- exp10_imply_supervised_20260811_073639 (判定 0.996, imply 1.000) = 主基线
- exp02_supervised_s2_20260811_081412 (判定 1.000)

## 配置

- docs/paper_data/configs/exp10_imply_supervised.json (主配置)
