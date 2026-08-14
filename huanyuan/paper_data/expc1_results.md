# EXP-C1: 普通 representation baseline (决定性对照) — 进度记录

日期: 2026-08-11 | 基线 = exp02_supervised_s2 (C1d 因子化, 判定 1.000)

## 目的

证明因子化表示是泛化原因, 非数据/规模巧合。C1a-c 去因子化 → 外推崩;
C1d 因子化 → 保持。验收: C1a-c ≤0.5, C1d ≥0.99。

## C1d 参照 (exp02, 已跑通)

外推零衰减 (exp80 全 1.000): 2000位 / base 60 / 100%匿名置换 / 联合全过。

## C1b 迭代结构隐藏 (已完成)

移除迭代样本 (law power/root/tetration/super + iterate/inverse arrow + iteration_expression/staircase),
保留基本算术。整体判定 0.745 (基线 0.996, 降 0.25)。

阶数外推 (balanced 小范围, run_exp _judge_eval):
| 算符 | acc | 解读 |
|---|---|---|
| addition | 1.000 | 基本算术保留 |
| multiplication | 1.000 | 基本算术保留 |
| power | 1.000 | 可从乘法直接推导 (2^3=8) |
| **root** | **0.000** | 迭代链降阶崩 (逆运算需迭代结构) |
| **tetration** | **0.000** | 高阶超运算崩 (迭代链必要) |

**结论: 部分支持 C1b** — 迭代结构对高阶 (root/tetration) 必要 (崩),
低阶 (power) 可从下推 (不崩)。迭代链是阶数泛化的原因。

## C1a 进制固化 (设计调整中)

首次尝试 strip_base (移除 base/cardinality token) → 整体崩 0.003
(过度破坏: 数字表示链断裂, 连 base 10 都学不会)。需重新设计:
进制不参数化 ≠ 移除 base token; 应保留 base 10 表示, 测试跨进制
(base 16/60) — 但 exp02 (C1d) 已 1.000 (因子化进制), C1a 对照需
构造"未因子化进制"表示 (纯 digit 无 cardinality, 但保持可计算).

## C1c 纠缠编码 (待做)

复合概念合并单 token (自由度融合) — 需 token 数据改造 (改 token 数据不改代码).

## 依赖

- C1b 模型: archive/log/train/expc1b_iter_hidden_20260811_115723
- 评估脚本: 固化到 docs/paper_data/scripts/expc1_eval.py (待写)

## C1a 架构限制 (2026-08-11 补充)

C1a (进制固化) 无法用简单 config 变体实现 — 进制因子化是 tokenizer 架构固有:
- strip_base="all": 移除 base+cardinality → 数字表示链断裂 (整体 0.003, 连 base10 都崩)
- strip_base="cardinality": 只移除 cardinality → base/value 序列变成数字一部分,
  表示边界破坏 (仍 0.003)

**原因**: cardinality token 表达进制数值, 是数字位分解的关键; 移除任何进制
标记都破坏 numeral 解析。C1a 的"未因子化进制"需独立表示系统 (定长 digit
无进制标记), 属表示层重设计, 超出 config 变体范畴。

**C1 决定性对照当前状态**:
- C1b 迭代隐藏: ✅ root 0.000 / tetration 0.000 (迭代链必要), power 1.000 (乘法推导)
- C1d 因子化: ✅ exp02 外推零衰减 (1.000)
- C1a/c: 需表示层重设计 (架构固有因子化, 简单 config 无法去因子化)

**结论**: C1b 提供迭代自由度必要性的正面对照; C1a/c 需表示层工程 (后续).

## C1c 纠缠编码 (2026-08-11 更新)

纠缠 (复合概念合并单 token) 需扩展 token 数据 — 2 位 value token (value_12)
当前不存在; inject_temp 只影响 core.load_layer, 不进 _register 的 DERIVE_REGISTRY
(api.all_concepts 不含), 无法纳入模型 vocab。

实现路径 (改 token 数据不改代码, 用户纪律):
- 在 concept_token.jsonl 追加 2 位纠缠 value token + 纠缠合成器 (训练 2 位纠缠,
  测 3 位 → 预期崩) — 需维护入口 (maintain) 合规写入。

C1a/c 同为表示层工程 (需 token 数据扩展或 flat 位权系统); C1b 已完成
决定性对照 (迭代自由度必要: root/tetration 崩)。
