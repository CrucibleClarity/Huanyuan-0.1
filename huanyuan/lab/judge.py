"""lab/judge.py —— 通用判定器 (lab 实验共享核心)

统一"真值判定"实验的样本构造/批次/评估, 消灭各实验重复逻辑与经典 bug:
  bug1 双 is_true (truth3 曾双加 _IS_TRUE)
  bug2 负例范围 (digits_of(负数) 死循环)
  bug3 分布偏置 (假样本淹没真 → 无脑判假, E22)
  bug4 逐层监督 vs 聚焦监督混用

用法:
  from lab.judge import Judge, judge_sequence
  j = Judge()
  seq = judge_sequence(prop, truth=True)       # 统一序列
  j.train(samples)                              # 统一训练
  j.evaluate(samples)                           # 整体/判真/判假/预测分布
"""
from __future__ import annotations


import torch

from tokenizer import api
from train.data import rev_vocab, vocab
from train.model import TokenTransformer

_IS_TRUE = api.role_token("is_true")
_TRUTH = api.role_token("truth")
_TRUE = _TRUTH[0]
_FALSE = _TRUTH[1]


def judge_sequence(prop, truth):
    """统一判定序列: 沿 judge 概念 (judgment gtoken/ptoken) 组装 (唯一构造点).

    judgment ptoken grammar = [⊤, arg:0, arg:1] (arg:0=命题, arg:1=真值),
    is_true/truth 符号与槽位来自 P 层, 零手写判定格式.
    """
    from tokenizer import api as _api
    tr = _TRUE if truth else _FALSE
    return _api.assemble_seq(_api.role_token("judge"), [list(prop), [tr]])


def truth_of(seq):
    return seq[-1] == _TRUE


def _truth_set():
    """真值 token 集合 {truth_true, truth_false} (tokenizer 统一接口检索)。"""
    return {_TRUE, _FALSE}


class Judge:
    """通用判定器: 统一批次/训练/评估。

    样本格式: {"seq": [is_true][命题][truth], "valid": 1, ...}
    默认聚焦监督 (末尾真值唯一 target), 可选 mode="layer" 逐层真值监督。
    """

    def __init__(self, dim=64, num_layers=2, epochs=15, lr=1e-3, causal=True,
                 mode="focused", seed=0):
        """epoch 默认 15 (固定, 原则上不允许调试 — 曲线逐 epoch 观测 token 学习)。"""
        self.dim = dim
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = lr
        self.causal = causal
        self.mode = mode  # "focused" (末尾) / "layer" (每层真值)
        self.seed = seed
        self.rv = rev_vocab()
        self.V = vocab()
        self.model = None

    # ---- 批次 ----
    def collate(self, samples):
        """统一批次。mode='focused': 只监督末尾真值; mode='layer': 每个真值位置都监督。

        含 valid 字段 (合法性标签, train.loop.train_epoch 需要)。
        """
        seqs = [list(s["seq"]) for s in samples]
        # 输入 = seq[:-1] (去掉末尾真值)
        in_seqs = [seq[:-1] for seq in seqs]
        L = max((len(x) for x in in_seqs), default=0)
        B = len(samples)
        inputs = torch.zeros(B, L, dtype=torch.long)
        targets = torch.full((B, L), -100, dtype=torch.long)
        mask = torch.zeros(B, L, dtype=torch.bool)
        valid = torch.ones(B, dtype=torch.long)
        for i, seq in enumerate(in_seqs):
            ln = len(seq)
            for j, e in enumerate(seq):
                if e in self.V:
                    inputs[i, j] = self.V[e]
            mask[i, ln:] = True
            if samples[i].get("valid") == 0:
                valid[i] = 0
            if self.mode == "focused":
                # 末尾真值: 输入最后一位置预测下一个 (真值)
                if seqs[i][-1] in self.V:
                    targets[i, ln - 1] = self.V[seqs[i][-1]]
            elif self.mode == "layer":
                # 每个 truth 位置: 前一位置预测它
                for j in range(ln):
                    nxt = seqs[i][j + 1] if j + 1 < len(seqs[i]) else None
                    if nxt in (_TRUE, _FALSE) and nxt in self.V:
                        targets[i, j] = self.V[nxt]
        return {"inputs": inputs, "targets": targets, "mask": mask, "valid": valid}

    # ---- 训练 ----
    def train(self, samples, archive_token=None, archive=True):
        """统一训练循环 (train.loop.train_epoch) + 自动归档 (archive.save_training)。

        默认 epoch=15 (固定, 原则上不允许调试), 逐 epoch 采集曲线 (train_curve):
        每 epoch 末尾真值位置预测准确率 (观测 token 学习动态, 非 loss — bool token loss 无意义)。
        archive_token: 归档名; archive=False 跳过归档。
        """
        from train.loop import train_epoch
        from archive import run_dir, save_training

        torch.manual_seed(self.seed)
        batch = self.collate(samples)
        self.model = TokenTransformer(
            dim=self.dim, num_concepts=len(self.V), num_layers=self.num_layers,
            input_mode="ids", causal=self.causal)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        losses = []
        train_curve = []
        for _ in range(self.epochs):
            losses.append(train_epoch(self.model, batch, opt))
            train_curve.append(self._batch_truth_acc(batch))
        self.losses = losses
        self.train_curve = train_curve
        if archive:
            token = archive_token or f"judge_{self.mode}_e{self.epochs}"
            self.run_dir = run_dir(token)
            ev = self.evaluate(samples)
            config = {
                "token": token,
                "judge": {"dim": self.dim, "num_layers": self.num_layers,
                          "epochs": self.epochs, "lr": self.lr, "causal": self.causal,
                          "mode": self.mode, "seed": self.seed},
                "model": {"num_concepts": len(self.V), "input_mode": "ids",
                          "causal": self.causal, "num_layers": self.num_layers,
                          "dim": self.dim},
            }
            save_training(self.run_dir, config, samples, self.model.state_dict(),
                          {"losses": losses, "train_curve": train_curve,
                           "acc": ev["acc"], "true_acc": ev["true_acc"],
                           "false_acc": ev["false_acc"], "consistency": ev["consistency"],
                           "type_acc": ev["type_acc"],
                           "baseline_fixed": ev["baseline_fixed"],
                           "delta_fixed": ev["delta_fixed"],
                           "baseline_random": ev["baseline_random"],
                           "delta_random": ev["delta_random"]})
        else:
            self.run_dir = None
        return self

    def _batch_truth_acc(self, batch):
        """批内末尾真值位置准确率 (向量化, 单次 forward)。"""
        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(batch["inputs"], mask=batch["mask"])
        B, L = logits.shape[0], logits.shape[1]
        real_lens = L - batch["mask"].sum(dim=1)
        pos = real_lens - 1
        preds = logits[torch.arange(B), pos].argmax(dim=1)
        tgt = batch["targets"][torch.arange(B), pos]
        valid_t = tgt != -100
        if valid_t.sum() == 0:
            return 0.0
        return (preds[valid_t] == tgt[valid_t]).float().mean().item()

    # ---- 评估 ----
    def pred(self, seq):
        """给完整判定序列 (含末尾真值), 预测末尾真值 token。

        输入 seq[:-1] (去掉真值, 保留 is_true), 在最后一个输入位置预测下一个 (真值)。
        """
        batch = self.collate([{"seq": seq, "valid": 1}])
        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(batch["inputs"], mask=batch["mask"])
        ln = len(seq) - 1  # 输入长度 (seq[:-1])
        return self.rv[logits[0][ln - 1].argmax().item()]

    def answer(self, prop):
        """给纯命题 (无 is_true, 无真值), 返回模型回答的真值 token。

        构造完整判定序列 [is_true][命题][占位真值], 用 pred 预测末尾真值。
        注意: 占位真值会被 collate 去掉 (输入 seq[:-1]), 不影响预测。
        """
        seq = [_IS_TRUE] + list(prop) + [_TRUE]
        return self.pred(seq)

    def evaluate(self, samples):
        """判定评估: acc + 判真/判假 + 类型正确率 + 基线残差 (delta)。

        批量 forward (单次), 向量化统计 — 20000 样本从 20000 次 forward → 1 次。
        扩展标签 (用户要求):
          type_acc      类型正确率: 预测是否落在真值类型 (truth_true/false) — 数字错但类型对也计
          baseline_fixed 未学习基线: 恒定输出单一真值 (全部 true 或全部 false) 的 acc
          delta_fixed   acc - baseline_fixed: 相对"无脑单值"的增益
          baseline_random 随机基线: 真/假 50% 随机预测期望 acc
          delta_random  acc - baseline_random
        """
        if not samples:
            return {"acc": 0.0, "true_acc": 0.0, "false_acc": 0.0, "consistency": 0.0,
                    "type_acc": 0.0, "baseline_fixed": 0.5, "delta_fixed": 0.0,
                    "baseline_random": 0.5, "delta_random": 0.0, "pred_dist": {}}
        batch = self.collate(samples)
        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(batch["inputs"], mask=batch["mask"])
        B, L = logits.shape[0], logits.shape[1]
        real_lens = L - batch["mask"].sum(dim=1)
        pos = real_lens - 1
        pred_idx = logits[torch.arange(B), pos].argmax(dim=1)
        pred_tokens = [self.rv[p] for p in pred_idx.tolist()]
        truths = [s["seq"][-1] for s in samples]
        truth_set = _truth_set()
        tot = len(samples)
        hit = tp = tn = tfn = tfp = type_hit = 0
        dist = {}
        for p, e in zip(pred_tokens, truths):
            hit += (p == e)
            dist[p] = dist.get(p, 0) + 1
            type_hit += (p in truth_set)
            if e == _TRUE:
                tp += (p == _TRUE)
                tfn += 1
            else:
                tn += (p == _FALSE)
                tfp += 1
        acc = hit / max(tot, 1)
        true_acc = tp / max(tfn, 1)
        false_acc = tn / max(tfp, 1)
        consistency = 1.0 - abs(true_acc - false_acc)
        pos_ratio = tfn / max(tot, 1)
        baseline_fixed = max(pos_ratio, 1 - pos_ratio)
        return {
            "acc": acc,
            "true_acc": true_acc,
            "false_acc": false_acc,
            "consistency": consistency,
            "type_acc": type_hit / max(tot, 1),
            "baseline_fixed": baseline_fixed,
            "delta_fixed": acc - baseline_fixed,
            "baseline_random": 0.5,
            "delta_random": acc - 0.5,
            "pred_dist": {api.name(k): v for k, v in dist.items()},
        }

    # ---- 统一验证 (判定专用视图, 归档 save_views) ----
    def verify(self, samples, depth_outs=None, save=True):
        """统一验证 (判定专用): 样本集 → 视图 dict, 归档到 run_dir。

        判定任务的视图 = evaluate (整体/判真/判假/分布) + 可选 depth_outs
        ({depth: 样本列表} → 每深度 acc)。用 Judge.evaluate 口径 (末尾真值),
        不用 verify.build_views (它是逐位置口径, 不适配判定任务)。

        返回视图 dict 并归档 (run_dir 存在时)。
        """
        from archive import save_views

        views = {"eval": self.evaluate(samples)}
        if depth_outs:
            views["depth"] = {d: self.evaluate(ss)["acc"] for d, ss in depth_outs.items()}
        if save and getattr(self, "run_dir", None):
            save_views(self.run_dir, views)
        self.views = views
        return views

    # ---- 编排器注入口 (lab.runner 被动调用) ----
    def as_train_fn(self, archive_token=None):
        """作为训练注入口: runner 调用时 self.train(samples)。"""
        from . import runner
        return runner.make_train_fn(self, archive_token)

    def as_verify_fn(self, depth_outs_fn=None):
        """作为验证注入口: runner 调用时 self.verify + depth 外推。"""
        from . import runner

        def _ev(model, samples):
            # 复用 self.evaluate (基于 self.model)
            return self.evaluate(samples)
        return runner.make_verify_fn(_ev, depth_outs_fn)
