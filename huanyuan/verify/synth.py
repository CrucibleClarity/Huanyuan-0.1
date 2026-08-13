"""verify/synth.py —— 校验样本合成器

复用加载器加载的样本合成方法档案数据 (config), 按三类合成:
  train_replay     训练样本 (同 seed 重放) — 分析训练本身学习成功率
  ood              样本外 (同合成逻辑, 不同 seed) — 合成逻辑范围内的样本
  generalization   可解释泛化 (深度+1/不同 seed) — 检测未训练 token,
                   依赖 token 体系对未训练符号的泛化结果
"""
from __future__ import annotations

from synth import build_sample_set


def verify_samples(loader_out: dict, seeds=None, gen_depths=(1, 2, 3),
                   samples_fn=None) -> dict:
    """三类校验样本 + 多层嵌套泛化。

    train_replay/ood 沿用训练 exclude/深度 (同分布);
    generalization 含被排除 token (拼装重建), 标注 untrained;
    gen_by_depth: 训练深度+1/+2/+3 的多层嵌套泛化样本 (不 exclude token, 测结构深度泛化)。

    samples_fn: 样本注入口 (lab 灌入自定义样本, 依赖倒置)。若提供则完全接管
      train_replay/ood/generalization 合成 (形如 lambda seed, exclude, depth -> 样本列表),
      verify 不读 config 的 synth 参数 (兼容 train_seq 等归档); 否则默认用
      build_sample_set 按 config 合成。

    返回 {train_replay, ood, generalization, gen_by_depth}。
    """
    cfg = loader_out["config"].get("synth") or {}
    toks = loader_out["config"].get("tokens") or [loader_out["config"]["token"]]
    seeds = seeds or {}
    ood_seed = seeds.get("ood", 12345)
    gen_seed = seeds.get("gen", 54321)
    if samples_fn is not None:
        seed = cfg.get("seed", 0)
        exclude = cfg.get("exclude") or []
    else:
        n, depth, seed = cfg["n_synth"], cfg["depth"], cfg["seed"]
        exclude = cfg.get("exclude") or []

    def synth_for(seed_i, ex, d=None):
        if samples_fn is not None:
            return samples_fn(seed_i, ex, d)
        d = d if d is not None else depth
        out = []
        for t in toks:
            out.extend(build_sample_set(t, n_synth=n, depth=d, seed=seed_i, exclude=ex)["synth_samples"])
        return out

    trained = {e for s in loader_out["train_samples"] for e in s.get("seq", [])}
    # 三类样本生成异步并行 (independent)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=3) as ex:
        f_tr = ex.submit(synth_for, seed, exclude)
        f_ood = ex.submit(synth_for, ood_seed, exclude)
        f_gen = ex.submit(synth_for, gen_seed, [])
        train_replay = f_tr.result()
        ood = f_ood.result()
        gen = f_gen.result()
    for s in gen:
        s["untrained"] = [e for e in s["seq"] if e not in trained]

    gen_by_depth = {}
    if samples_fn is None:
        for k in gen_depths:
            g = synth_for(gen_seed + k, [], d=depth + k)
            for s in g:
                s["untrained"] = [e for e in s["seq"] if e not in trained]
            gen_by_depth[depth + k] = g

    return {"train_replay": train_replay, "ood": ood,
            "generalization": gen, "gen_by_depth": gen_by_depth}
