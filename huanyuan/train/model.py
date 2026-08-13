"""train/model.py —— token 体系 transformer 骨架

输入:
  input_mode='vector'  [B, L, dim] 整形向量 (counts/序列向量)
  input_mode='ids'     [B, L] token id 序列 → Embedding + 位置编码
    (用户方向: 用长 token 序列的排序和位置表达结构, 代替向量浮点数)
输出: [B, L, num_concepts] 位置级概念 logits。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TokenTransformer(nn.Module):
    def __init__(self, dim, num_concepts, num_layers=2, nhead=None, dim_feedforward=512,
                 input_mode="vector", causal=False):
        super().__init__()
        if nhead is None:
            nhead = next(n for n in (8, 4, 2, 1) if dim % n == 0)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(dim, num_concepts)
        self.valid_head = nn.Linear(dim, 2)
        self.input_mode = input_mode
        self.causal = causal
        if input_mode == "ids":
            self.embed = nn.Embedding(num_concepts, dim)
        else:
            self.embed = None

    def forward(self, x, mask=None, head_weight=None):
        """head_weight: [B,L] 每位置注意力先验 (谓词/比较权重), 调制嵌入。

        ids: [B,L] 概念 id → 嵌入 (无位置编码, 数字 token 自带位序位权);
        vector: [B,L,dim] 直接用。返回 (位置 logits, 合法性 logits)。
        causal: 下三角注意力掩码 (位置 i 只看 ≤i), 根除"copy 末尾"捷径。
        """
        if self.embed is not None:
            x = self.embed(x)
        if head_weight is not None:
            x = x * head_weight.unsqueeze(-1)
        src_mask = None
        if self.causal:
            L = x.size(1)
            src_mask = torch.triu(torch.full((L, L), float("-inf"), device=x.device), diagonal=1)
        h = self.encoder(x, src_key_padding_mask=mask, mask=src_mask)
        return self.head(h), self.valid_head(h.mean(dim=1))
