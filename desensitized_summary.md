# 可解释泛化 — 摘要 (脱敏)

> 用途: 外部交流/评审 | 语言: 中文 | 脱敏标准: 只保留理论主张、中性训练参数与可观测现象。

## 摘要

我提出并验证一个主张: **神经网络的系统化泛化不是涌现或先验的能力**。泛化不是黑箱属性, 而是可设计、可验证、可复现的工程目标。

## 训练参数

| 项       | 值                                                            |
| -------- | ------------------------------------------------------------- |
| 模型     | 2 层自回归 Transformer, 隐藏维度 64, 参数量**<1MB**i          |
| 样本     | 自动合成自动重采样                                            |
| 套件规模 | 基础算术 ≈2000 样本 / 3 轮; 完整逻辑+算术 ≈6000 样本 / 8 轮 |
| 评估     | 完整输出正确率                                                |
|          |                                                               |

## 结果

| 套件     | 样本数 | 训练轮数 | 结果            |
| -------- | ------ | -------- | --------------- |
| 基础算术 | ≈2000 | 3        | 完全泛化 (100%) |
| 完整     | ≈6000 | 8        | 全部正确 (100%) |

泛化同时发生:

- **长度外推**: 训练所见规模放大 10 倍以上仍 100% 正确 (含全程进位传播);
- **进制泛化**: 对训练中从未出现的进制, 仍高正确率;
- **组合外推**: 多个维度同时取未训练组合时仍正确;
- **置换不变性**: 训练时将全部数字随机改名, 测试换回, 泛化几乎不受影响 —— 学到的是结构关系。
- 样本、权重体积、epoch轮数仍有大幅降低优化的空间。

## 可解释泛化的训练曲线

### 整体形态

| 训练轮数   | 1    | 2   | 3   | 5   | 8     |
| ---------- | ---- | --- | --- | --- | ----- |
| 泛化正确率 | 0.1% | 21% | 24% | 78% | 99.6% |

训练接近收敛时急剧跃升 (第 5→8 轮 78% → 99.6%) —— 呈"编译式"而非渐进式形态。

### token曲线分布: 极端二值化

- **可解释泛化第 3 轮即达 100%**, 无渐进过渡;
- **未泛化: 恒为 0%**, 且不随训练轮数增加而修复;
- 整体曲线的"上升"实为**依次从不学到完全学会的叠加**, 几乎没有 30%/50% 的中间状态。

这一现象将泛化缺陷从"训练量"中彻底剥离: 未学会的不因多训练而好转。

训练曲线呈"要么完全学会、要么完全不学"的极端二值分布。

---

## Conclusion

Once, under a unified formal attention-syntactic structure, a neural network ultimately,
inevitably, and necessarily clearly, intuitively, and interpretably observes the boundary
between intuition and construction, we can with full confidence use the unified
attention-formal language to direct logical construction, use precise synthetic-CoT
logical construction to earn *flashing intuition*, and use distilled intuitive context
to exhaustively search attention forms. Form, construction, and intuition co-adapt
within the unified framework, forming a feedback iteration pipeline of teaching, training,
and reasoning in which they are no longer distinct — perhaps one of the faster shortcuts
among the many paths toward ASI.
