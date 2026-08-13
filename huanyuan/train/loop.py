"""train/loop.py —— 训练循环 (单批次最小骨架)

损失: 位置级概念分类 CE (正例, targets pad=-100 忽略) + 合法性 CE (对错区分, 奖励正例)。
head_weight: 每位置注意力先验 (谓词/比较权重) 注入模型输入调制。
"""
from __future__ import annotations

import torch.nn as nn


def train_epoch(model, batch, opt, loss_fn=None, valid_w=0.5):
    """一步训练: 位置 CE (正例) + 合法性 CE (对错区分, 奖励对)。"""
    model.train()
    opt.zero_grad()
    logits, vlogits = model(batch["inputs"], mask=batch["mask"],
                            head_weight=batch.get("head_weight"))
    loss_fn = loss_fn or nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fn(logits.permute(0, 2, 1), batch["targets"])
    valid_fn = nn.CrossEntropyLoss()
    loss = loss + valid_w * valid_fn(vlogits, batch["valid"])
    loss.backward()
    opt.step()
    return loss.item()


def evaluate(model, batch, loss_fn=None, valid_w=0.5):
    """评估: loss + 位置准确率 + 合法性准确率。"""
    import torch
    model.eval()
    with torch.no_grad():
        logits, vlogits = model(batch["inputs"], mask=batch["mask"],
                                head_weight=batch.get("head_weight"))
        loss_fn = loss_fn or nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fn(logits.permute(0, 2, 1), batch["targets"])
        valid_fn = nn.CrossEntropyLoss()
        loss = loss + valid_w * valid_fn(vlogits, batch["valid"])
        pred = logits.argmax(dim=-1)
        valid = batch["targets"] != -100
        correct = (pred == batch["targets"]) & valid
        pos_acc = correct.sum().item() / max(valid.sum().item(), 1)
        val_acc = (vlogits.argmax(-1) == batch["valid"]).float().mean().item()
    return loss.item(), pos_acc, val_acc
