"""verify/loader.py —— 模型加载器

从 archive 归档目录一次加载 (load_training):
  训练模型 (权重 + 架构配置)
  样本合成方法档案数据 (全合成参数 config)
  训练数据 + 训练指标
"""
from __future__ import annotations

from archive import load_training
from train import TokenTransformer


def load(run_dir: str) -> dict:
    """加载归档训练产物。

    返回 {model, config, train_samples, metrics, run_dir}。
    """
    data = load_training(run_dir)
    config = data["config"]
    model = TokenTransformer(
        dim=config["model"]["dim"],
        num_concepts=config["model"]["num_concepts"],
        num_layers=config["model"].get("num_layers", 2),
        input_mode=config["model"].get("input_mode", "vector"),
        causal=config["model"].get("causal", False),
    )
    model.load_state_dict(data["model_state"])
    return {
        "model": model,
        "config": config,
        "train_samples": data["samples"],
        "metrics": data["metrics"],
        "run_dir": run_dir,
    }
