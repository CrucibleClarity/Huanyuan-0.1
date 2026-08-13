"""verify/ —— 验证模块 (4 子模块)

loader  模型加载器: 加载模型 + 样本合成方法档案数据
synth   校验样本合成器: 训练样本 / 样本外 / 可解释泛化三类
views   结果视图生成器: 整体正确率 / 泛化成功率 / 逐token曲线 / 逐token正确率
归档    调用 archive 将验证视图写入训练归档文件夹

接口:
  verify(run_dir, ...) → 视图 dict (已归档 views.json)
"""
from __future__ import annotations

from .loader import load
from .synth import verify_samples
from .views import build_views
from archive import save_views
from infer import infer


def verify(run_dir: str, seeds=None, gen_depths=(1, 2, 3), samples_fn=None,
           profile=False, batch_size=None) -> dict:
    """验证归档训练产物: 加载 → 三类样本 → 推理 (异步并行) → 视图 → 归档。

    samples_fn: 样本注入口 (seed, exclude, depth) → 样本列表. 提供则完全接管
      三类样本合成 (兼容 train_seq 归档等非 build_sample_set 管线), verify 不读
      config 的 synth/shaper 细节 (id 序列样本 input_mode=ids 无需 order/encode)。
    profile: 对 verify 整体跑 cProfile, 热点写入归档 views['profile'] (持久化性能观测).
    batch_size: 推理分批大小 (流式, 防长序列大样本 OOM; None=一次全批).
    """
    import os
    if profile or os.environ.get("VERIFY_PROFILE"):
        import cProfile
        import io as _io
        import pstats
        pr = cProfile.Profile()
        pr.enable()
        views = _verify_impl(run_dir, seeds, gen_depths, samples_fn, batch_size)
        pr.disable()
        s = _io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(25)
        _append_profile(run_dir, s.getvalue())
        return views
    return _verify_impl(run_dir, seeds, gen_depths, samples_fn, batch_size)


def _verify_impl(run_dir, seeds, gen_depths, samples_fn, batch_size=None) -> dict:
    from concurrent.futures import ThreadPoolExecutor
    ld = load(run_dir)
    cfg = ld["config"]
    sm = cfg.get("shaper", {}).get("output", "ids")
    order = cfg.get("shaper", {}).get("order", "preorder")
    encode = cfg.get("shaper", {}).get("encode", "counts")
    expand = cfg.get("shaper", {}).get("expand")

    sset = verify_samples(ld, seeds=seeds, gen_depths=gen_depths, samples_fn=samples_fn)
    # train_replay 采样评估 (全量推理冗余, 训练时已记录 train_acc; 均匀采样分析训练内学习)
    tr = sset["train_replay"]
    if len(tr) > 800:
        step = len(tr) // 800
        tr = tr[::step]
    # 三类推理 + gen_by_depth 异步并行 (independent, 共享只读 model)
    # batch_size: 长序列样本 (2000 位外推) 流式分批, 防 OOM
    jobs = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        jobs["train"] = ex.submit(infer, ld["model"], tr, shape_method=sm, order=order,
                                  encode=encode, expand=expand, batch_size=batch_size)
        jobs["ood"] = ex.submit(infer, ld["model"], sset["ood"], shape_method=sm, order=order,
                                encode=encode, expand=expand, batch_size=batch_size)
        jobs["gen"] = ex.submit(infer, ld["model"], sset["generalization"], shape_method=sm,
                                order=order, encode=encode, expand=expand, batch_size=batch_size)
        depth_jobs = {}
        for d, samples in sset["gen_by_depth"].items():
            depth_jobs[d] = ex.submit(infer, ld["model"], samples, shape_method=sm,
                                      order=order, encode=encode, expand=expand, batch_size=batch_size)
        outs_train = jobs["train"].result()
        outs_ood = jobs["ood"].result()
        outs_gen = jobs["gen"].result()
        depth_outs = {d: j.result() for d, j in depth_jobs.items()}

    metrics = ld["metrics"]
    views = build_views(outs_train, outs_ood, outs_gen, metrics.get("losses", []),
                        depth_outs=depth_outs)
    if metrics.get("acc") is not None:
        views["train_acc"] = metrics["acc"]
    # 带崩元凶分析 (verify 功能): 判定失败样本中, 错误位置的真值 token 分布
    # (哪个 token 的错导致最多整条判定失败; 聚合 ood + generalization 失败样本)
    views["crash_culprit"] = crash_culprit(outs_ood + outs_gen)
    save_views(run_dir, views)
    return views


def crash_culprit(outs, top=20) -> dict:
    """带崩元凶: 判定失败样本 (pred != true) 中, 每个错误位置的真值 token 计数.

    哪个 token 位置的错导致最多整条判定失败 (token 带崩训练最多).
    返回 {token 名: 带崩次数} (top, 降序).
    """
    from collections import Counter
    from tokenizer import api
    c = Counter()
    for o in outs:
        pred, true = o["pred"], o["true"]
        if pred == true:
            continue
        for p, t in zip(pred, true):
            if p != t:
                c[t] += 1
    return {api.name(k): v for k, v in c.most_common(top)}


def _append_profile(run_dir, profile_text):
    """profile 结果持久化: 追加到归档 views.json['profile'] (性能观测留存)."""
    import json
    import os
    p = os.path.join(run_dir, "views.json")
    views = {}
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as f:
            views = json.load(f)
    views["profile"] = profile_text
    with open(p, "w", encoding="utf-8") as f:
        json.dump(views, f, ensure_ascii=False, indent=2)


__all__ = ["verify", "load", "verify_samples", "build_views"]
