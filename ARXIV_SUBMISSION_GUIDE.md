# arXiv 投稿指南 (幻元 Huanyuan — 三篇论文定稿投稿准备)

> 调查日期: 2026-08-11 | 状态: 投稿准备中 (未提交)

## 1. 账号与背书 (Endorsement)

- **注册**: arXiv 账号需邮箱注册 (arxiv.org/user), 首次投稿需**背书 (endorsement)**
- **背书机制**: 需已在该领域发表过论文的研究者背书; 若曾在 arXiv 发文则自动获得
- **建议**: 如无背书人, 可选 arXiv 的自动背书流程 (作者提供 ORCID/出版记录证明);
  或先由合作者/导师背书 (cs.AI/cs.LG 类别)

## 2. 投稿要求 (核心)

| 项 | 要求 |
|---|---|
| **源文件格式** | **LaTeX (必选)** — arXiv 不接收 Word/PDF 直接投稿; 需提交 .tex + 完整源 |
| **编译** | arXiv 自动编译 (pdflatex 为主); 需保证 tex 无错误、无缺失宏包 |
| **元数据** | 标题 / 作者 / 摘要 / 类别 (≥1 主类别, 可加次类别) / 关键词 |
| **类别建议** | P0 → cs.LG (或 cs.AI); P1 → cs.AI/cs.LG; P2 → cs.LG/cs.PL |
| **许可** | 默认 arXiv 非独占授权; 可自选 CC-BY 等 (与我们的专有许可需协调) |
| **时间** | 提交后 1-2 个工作日内出现在 (announce) 队列; 每周一/二宣布 |
| **版本** | 可随时更新版本 (v1/v2...), 但 v1 不可撤回 (可 withdraw) |

## 3. 当前障碍 (需解决)

1. **论文是 Markdown 非 LaTeX** — 三篇英文定稿均为 .md, 需转 LaTeX:
   - 转换工具: pandoc (`pandoc -s paper.md -o paper.tex --pdf-engine=pdflatex`)
   - 或手写 LaTeX (论文不长, P1 ~240 行 md, 转换可控)
2. **数学公式** — md 中若有数学符号需转 LaTeX 数学环境; 检查 P1 的
   形式化符号 (∀T∈L, [[T]] 等) 需用 `\forall`/`\llbracket` 等
3. **图片** — 若有图需单独上传; 当前论文以表格为主, 需将结果表转 LaTeX 表格
4. **许可协调** — arXiv 默认非独占授权 (作者保留版权), 与"保留所有权利"可共存
   (arXiv 仅获得发布权, 不改变版权归属); 但需在论文页注明 "All Rights Reserved"

## 4. 投稿流程 (实施步骤)

```
1. 准备 LaTeX 源: 三篇 md → tex (pandoc + 手动校对公式)
2. 本地编译验证: pdflatex 无错误 (需 texlive)
3. 注册 arXiv 账号 + 处理背书 (如需)
4. 提交: 上传 tex 源 + 元数据 (标题/摘要/类别/关键词)
5. 检查编译预览: arXiv 生成 PDF 预览, 确认无误
6. 等待宣布 (announce): 周一/二 入队列
7. 发布后: 更新 RELEASE_MANIFEST (arXiv ID + DOI 关联)
```

## 5. 标题与摘要 (定稿)

### P0: Three-Channel Equivalence
- 标题: *Three-Channel Equivalence: Construction, Intuition, and Formal Rewriting in One Neural System*
- 类别: cs.LG / cs.AI

### P1: Generalization as Formalized Induction
- 标题: *Generalization as Formalized Induction: Token-Native Factorized Representation and Weight Compilation*
- 类别: cs.AI / cs.LG

### P2: Neural Macro Compilation
- 标题: *Neural Macro Compilation: Compiling Slow Construction Paths into Fast Reasoning Paths*
- 类别: cs.LG / cs.PL

## 6. 待办

- [ ] 检查三篇 md 中数学符号 (∀/⇒/[[·]]/θ 等) 的 LaTeX 表达
- [ ] pandoc 转换 + 本地 texlive 编译验证
- [ ] 决定背书策略 (找背书人 or 自动背书)
- [ ] 摘要 150-300 词精炼 (arXiv 摘要限制)
- [ ] 许可声明在 arXiv 页面注明 All Rights Reserved
