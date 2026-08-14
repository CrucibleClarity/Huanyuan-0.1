"""archive/ —— 训练归档模块 (log/train/<run>)

归档内容:
  config.json     样本合成方法档案数据 (全合成参数, 可复现)
  model.pt        模型权重
  samples.jsonl   训练数据 (notation + seq)
  metrics.json    训练指标 (losses/acc)
  views.json      verify 产出视图 (整体正确率/泛化成功率/逐token曲线/正确率)
train 归档训练产物, verify 归档验证视图。路径统一在 log/train/。
"""
from __future__ import annotations

import json
import os
import time

import torch

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log", "train")


def run_dir(token: str, ts: str | None = None) -> str:
    ts = ts or time.strftime("%Y%m%d_%H%M%S")
    d = os.path.join(LOG_DIR, f"{token}_{ts}")
    os.makedirs(d, exist_ok=True)
    return d


def _p(run, name):
    return os.path.join(run, name)


def save_config(run: str, config: dict) -> None:
    with open(_p(run, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def save_samples(run: str, samples) -> None:
    """训练样本序列化 (完整 raw: seq + truth/depth/spec 溯源等全部 JSON 可序列化字段).

    归档时保存每个样本的原始构造数据 (含 spec 配置投影), 供后续诊断
    按样本类型/参数对比 (如找翻 0-acc 的样本差异), 不止 notation/seq.
    """
    with open(_p(run, "samples.jsonl"), "w", encoding="utf-8") as f:
        for s in samples:
            row = {}
            for k, v in s.items():
                if k == "seq":
                    row["seq"] = v
                elif isinstance(v, (str, int, float, bool, type(None))):
                    row[k] = v
                elif isinstance(v, (list, dict)):
                    row[k] = v  # spec 配置 dict / gap_pos answer 列表 (可序列化)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_model(run: str, state) -> None:
    torch.save(state, _p(run, "model.pt"))


def save_metrics(run: str, metrics: dict) -> None:
    with open(_p(run, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def save_views(run: str, views: dict) -> None:
    with open(_p(run, "views.json"), "w", encoding="utf-8") as f:
        json.dump(views, f, ensure_ascii=False, indent=2)


def save_training(run: str, config: dict, samples, model_state, metrics: dict) -> str:
    """一次归档训练产物 (train 调用): config + samples + 模型权重 + 指标。"""
    os.makedirs(run, exist_ok=True)
    save_config(run, config)
    save_samples(run, samples)
    save_model(run, model_state)
    save_metrics(run, metrics)
    return run


def load_training(run: str) -> dict:
    """一次加载训练产物 (verify 调用): {config, samples, model_state, metrics}。"""
    return {
        "config": load_config(run),
        "samples": load_samples(run),
        "model_state": load_model(run),
        "metrics": load_metrics(run),
    }


def load_config(run: str) -> dict:
    with open(_p(run, "config.json"), encoding="utf-8") as f:
        return json.load(f)


def load_samples(run: str) -> list[dict]:
    out = []
    with open(_p(run, "samples.jsonl"), encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                out.append(json.loads(s))
    return out


def load_model(run: str):
    return torch.load(_p(run, "model.pt"), weights_only=False)


def load_metrics(run: str) -> dict:
    with open(_p(run, "metrics.json"), encoding="utf-8") as f:
        return json.load(f)


def load_views(run: str) -> dict | None:
    p = _p(run, "views.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


__all__ = [
    "run_dir", "save_config", "save_samples", "save_model", "save_metrics", "save_views",
    "save_training", "load_training",
    "load_config", "load_samples", "load_model", "load_metrics", "load_views",
]
