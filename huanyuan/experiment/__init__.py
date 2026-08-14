"""experiment/ —— 实验组装器

输入待实验 ctoken → 自动跑完整管线:
  synth (沿定义链覆盖平行 ctoken) → train (归档) → verify (三类样本+视图归档)
返回 {token, run_dir, train, views}。

用法:
  from experiment import run
  run('D:250')  # 输入待训练 ctoken, 自动出训练+验证结果
"""
from __future__ import annotations

from train import train
from verify import verify


def run(token_eid, n_samples=50, epochs=3, depth=1, seed=None,
        shape_method="sequence_counts", tokens=None, **train_kw) -> dict:
    """待实验 ctoken → 完整管线 (合成 → 训练 → 验证 → 归档)。

    tokens: 多 token 训练集 (加减乘等于+逻辑等), 样本合并; 缺省=单 token。
    """
    res = train(token_eid, n_samples=n_samples, epochs=epochs, depth=depth,
                seed=seed, shape_method=shape_method, tokens=tokens, **train_kw)
    views = verify(res["run_dir"])
    return {"token": token_eid, "run_dir": res["run_dir"], "train": res, "views": views}


__all__ = ["run"]
