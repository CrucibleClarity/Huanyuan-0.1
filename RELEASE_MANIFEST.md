# 幻元 (Huanyuan) 发布完整性清单 (Release Integrity Manifest)

> 生成: 2026-08-11
> 用途: 公开脱敏摘要, 同时锁定**尚未公开**资产的 SHA256 — 证明内容在本人手中、时间点可验证, 公开时按此清单校验。
> 规则: 清单内的 hash 对应内容的精确字节; 公开某项前必须核对 SHA256 一致。

## 公开层 (本清单随脱敏摘要一起发布)

| 资产 | SHA256 |
|---|---|
| 脱敏摘要 `desensitized_summary.md` | `85df5c1411b96fea5fe5589723a7c85e55f0e290d84939db10e96d86a561a40f` |
| 论文 P1 初稿 PDF (v0.1, 待时间戳) `paper_P1_draft_v0.1.pdf` | `09929beed6d002919e7f18b52450651c27fa30967ee8e5bf453cb9f87e48a3a4` |
| 论文 P2 初稿 PDF (v0.1, 待时间戳) `paper_P2_draft_v0.1.pdf` | `2d2d0642ea543c37c98609f37898874cf8ae26e406a584456caf8075fc4f82d5` |
| 论文 P0 初稿 PDF (v0.1, 待时间戳) `paper_P0_draft_v0.1.pdf` | `81a160d15af22e46a3696ca734c300c720d0c03f30e429358779c009ee4fa279` |
| 完整性清单 `RELEASE_MANIFEST.md` | `b81a3c72391d7f3a092ad65211d4d761ffbf3ad5837fda74913bcc861a6cc63c` (排除自身行, 可复现校验) |
| 引用元数据 `CITATION.cff` | `b0bbe3519b1e28bc4969a3df0a735b06a663d3362cfbb533fd1854333c1a9461` |

## 锁定层 (内容未公开, 仅锁 hash)

### 论文 (独立资产, 不随代码包发布)

| 资产 | SHA256 |
|---|---|
| 论文 P1 初稿 (English) `docs/Generalization_as_Formalized_Induction_Token-Native_Factorized_Representation_Weight_Compilation.md` | `26821da968cf35ed3aba936d9efb0bed5cdc15c8a60dab64c759307186091741` |
| 论文 P2 初稿 (English) `docs/Neural_Macro_Compilation_Compiling_Slow_Construction_Paths_into_Fast_Reasoning_Paths.md` | `a60e4aa20b9842a8d2b5c76e107b7853a3f173f10855fbf48fbd3c1f457de72c` |
| 论文 P0 初稿 (English) `docs/Three-Channel_Equivalence_Construction_Intuition_Formal_Rewriting.md` | `17b07b2fcf2b006a5d35e08b2401dbbd2aa0fc4d5f8be2f3c0fbd24dd78f3b26` |

### 模型权重 (checkpoint, 核心证据)

| 模型 | 判定口径 | SHA256 |
|---|---|---|
| exp02_supervised_s2 (完整套件, 判定 1.000) | 1.000 | `2c0b811d6bdccdfa4f98d3d9358adbd30d5f9d7f365154467744a9a414b6bd23` |
| exp10_imply_supervised (imply 监督, 判定 0.996) | 0.996 | `9beb3b46265f74622a9ef429a21d7a6d62496c8c0149a25554a9de5ed8bdee52` |
| exp41_permute (符号置换, 0.987) | 0.987 | `f3853ebf2fd8bf5b47e81108117a5e5ffb789c4054d7d617c9b142c2bcf7d9f6` |

### 数据

| 资产 | SHA256 |
|---|---|
| 结果登记 `docs/paper_data/results.csv` | `489e98af1cd7de812c91801fbd761990415223d7b844f333a9fa911b73f0d1f7` |

### 代码

| 标识 | hash |
|---|---|
| git HEAD (私有 master, 2026-08-09) | `8ac7ca11b87d893006e11ccf3903b2d27bf90b24` |
| git HEAD tree | `c988ecd2d7f16c228b04297b80342b475edcf7bf` |
| **工作区树 (含 2026-08-11 全部实验/文档, stash create)** | `027228fe50753a674b06a88567bbf50d99c78f23` |

### Lean 形式化 (锁定层, 2026-08-12 补)

| 资产 | SHA256 |
|---|---|
| Lean 证明文件 (65 个 .lean, `src/relative-recursion/formal/`, 排除 .lake 依赖) | `85dcf907ad5f46cd48b9317fb53d391c739123f323d7457131c6e22e93181086` |
| 形式化完整目录 (含 lakefile/lean-toolchain, 排除 .lake) | `a223ed41c9373d22f515f47e688975a30f1f1c3d7dbdb3805b02150bbb97936c` |

## 本清单 hash (最终)

```bash
sha256sum release_v0.1/RELEASE_MANIFEST.md   # 修改后需重算回填
```

## 校验方法

```bash
# 任何资产公开前校验
sha256sum <file>   # 与上表比对
# 代码树校验
git -C <repo> stash create | grep ^027228fe50753a674b06a88567bbf50d99c78f23$
```

## 发布流程 (建议)

1. **GitHub**: 建公开仓库 (仅放 `desensitized_summary.md` + 本清单 + `CITATION.cff`)
2. **Zenodo**: 连接该 GitHub → 创建 `v0.1-preview` Release → Zenodo 自动归档 → 生成 DOI
3. **引用**: DOI 指向"脱敏摘要 + 完整性清单"这个不可变记录; 正文引用该 DOI
4. **将来公开**: 定稿/代码/权重公开时, 在清单中核对 SHA256 一致, 追加 `v1.0.0` Release (含完整资产 + 新 DOI)

## 可复现包 (2026-08-11 补充)

最小可复现发布包 `huanyuan/` (20MB):
- 全部代码 + token 数据 + 6 关键模型权重 + 配置/脚本/结果 + 论文草稿
- 复现指南: `huanyuan/README.md` (8 步复现论文结论)
- 验证: verify_judge / exp10_impl / exp20 / exp41 / exp01 / exp80 / expc1 / quant 全部可跑
- 排除: 调研内容 (_research_brief/语法调研) / 废弃内容 (0-IMPLY 系列) / Rust 编译产物 (calc_rust target)
