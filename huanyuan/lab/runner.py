"""lab/runner.py —— 统一编排器 (lab 实验入口, 所有模块被动调用)

设计 (依赖倒置):
  编排器不关心谁提供样本/训练/验证, 只按阶段调用注入口。
  模块被动: synth 只提供生成器; verify 接收 samples 而非自己合成;
  train 不强制归档 (由编排器控制 archive=True/False)。
  lab 通过注入口劫持: 传入自定义样本/自定义判定器, 无需改模块内部。

流程:
  run(samples_provider, train_fn, verify_fn) → {train_out, views}

注入口:
  samples_provider: () -> (train_samples, ood_samples, gen_samples)  样本提供者
  train_fn: (train_samples) -> model 产物 (含 run_dir)              训练器
  verify_fn: (model, train/ood/gen) -> views                       验证器
"""
from __future__ import annotations


def run(samples_provider, train_fn, verify_fn, **kw):
    """统一编排: 取样本 → 训练 → 验证 → 返回 {train, views}。"""
    tr, ood, gen = samples_provider()
    train_out = train_fn(tr, **kw)
    views = verify_fn(train_out, tr, ood, gen, **kw)
    return {"train": train_out, "views": views, "samples": {"train": tr, "ood": ood, "gen": gen}}


def make_samples_provider(train_samples, ood_samples=None, gen_samples=None):
    """样本注入口: 固定样本集 (lab 自定义样本灌入点)。"""
    return lambda: (train_samples, ood_samples or [], gen_samples or [])


def make_train_fn(judge, archive_token=None):
    """训练注入口: 用统一判定器 Judge 训练。"""
    def _train(tr, **kw):
        j = judge
        j.train(tr, archive_token=archive_token, archive=kw.get("archive", True))
        return j
    return _train


def make_verify_fn(evaluator=None, depth_outs_fn=None):
    """验证注入口: evaluator(model, samples)->dict; depth_outs_fn(model)->{depth:samples}。"""
    def _verify(model, tr, ood, gen, **kw):
        ev = evaluator(model, tr) if evaluator else {}
        views = {"eval": ev}
        if depth_outs_fn:
            views["depth"] = {d: evaluator(model, ss) for d, ss in depth_outs_fn(model).items()}
        return views
    return _verify


__all__ = ["run", "make_samples_provider", "make_train_fn", "make_verify_fn"]
