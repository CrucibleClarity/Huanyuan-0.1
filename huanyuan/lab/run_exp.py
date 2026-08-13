"""lab/run_exp.py —— 标准实验入口 (薄封装编排, 复用项目后端)

职责 (用户确立):
  lab 唯一职责 = 实验新 token 注入 (inject_temp 防破坏 tokenizer);
  其余全复用项目后端 (薄封装编排):
    synth   样本合成 (compose_samples/ood_samples, 样本设计消费方)
    train   train.train_seq (训练 + 自动归档)
    verify  verify.verify (三类样本 + infer + 视图归档)
  目标: 每次实验迭代累积 (自动归档 config/samples/metrics/views), 不反复犯同样错误.

用法:
  PYTHONPATH=src/llm_research_v5 python -m lab.run_exp --config configs/xxx.json
"""
from __future__ import annotations

import argparse
import json
import time

from tokenizer import api
from lab import synth_core
from lab.synth_core import compose_samples, ood_samples
from train import train_seq
from train.data import collate, rev_vocab
from verify import verify


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def _evaluable(op):
    """tokenizer 求值能力探测: 二元 eval_op 试算 1 op 2; 一元 (arrange=unary_connective)
    _unary_eval 试算 — 可求值返回 True (确保一元算子如 neg 纳入 OOD, 无 0 判定)."""
    from lab.synth_core import _unary_eval
    eid = api.eid_by_name(op)
    try:
        if api.arrange_of(eid) == "unary_connective":
            _unary_eval(eid, 1)
        else:
            synth_core.eval_op(eid, 1, 2)
        return True
    except ValueError:
        return False


def _archive_exp_config(run_dir, cfg):
    """归档完整实验配置 (含 synth.samples 组合列表) 到 config.json["exp"]."""
    import os
    p = os.path.join(run_dir, "config.json")
    if not os.path.isfile(p):
        return
    with open(p, encoding="utf-8") as f:
        arch = json.load(f)
    arch["exp"] = cfg
    with open(p, "w", encoding="utf-8") as f:
        json.dump(arch, f, ensure_ascii=False, indent=2)


def _archive_epoch_gen(run_dir, epoch_gen):
    """逐 epoch 泛化曲线入 views.json: epoch_gen(per-token) + epoch_gen_all/epoch_gen_no0
    (全 token 平均 / 剥离 0-acc 平均 — 真实反映做对/做错, 反馈监督基础设施)."""
    import os
    p = os.path.join(run_dir, "views.json")
    views = {}
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as f:
            views = json.load(f)
    views["epoch_gen"] = epoch_gen
    views["epoch_gen_all"] = [sum(v.values()) / max(len(v), 1) for v in epoch_gen if isinstance(v, dict)]
    views["epoch_gen_no0"] = []
    for v in epoch_gen:
        if not isinstance(v, dict):
            continue
        vals = [a for a in v.values() if a > 0.0]
        views["epoch_gen_no0"].append(sum(vals) / max(len(vals), 1))
    with open(p, "w", encoding="utf-8") as f:
        json.dump(views, f, ensure_ascii=False, indent=2)


def _archive_gen_diag(run_dir, train_samples, epoch_gen):
    """泛化诊断持久化到 views.json['gen_diag']: 0-acc 定位 + 0.9/1.0 最小样本量."""
    import os
    zero_rows, nine_min, one_min, one_count = _gen_diagnosis(train_samples, epoch_gen)
    diag = {"zero_acc": [{"token": k, "samples": ns, "rules": nr} for k, ns, nr in zero_rows],
            "nine_min": {"token": nine_min[0], "samples": nine_min[1]} if nine_min else None,
            "one_min": {"token": one_min[0], "samples": one_min[1]} if one_min else None,
            "one_count": one_count}
    p = os.path.join(run_dir, "views.json")
    views = {}
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as f:
            views = json.load(f)
    views["gen_diag"] = diag
    with open(p, "w", encoding="utf-8") as f:
        json.dump(views, f, ensure_ascii=False, indent=2)


def _gen_diagnosis(train_samples, epoch_gen):
    """泛化诊断 (0-acc 定位 + 最小样本量指标 + 完美泛化方法论证据).

    0-acc 定位: 每 0-acc token 的 训练样本量 vs 定义 rules 数 —
      样本量 0 = 配置未覆盖 (power/root/tetration); rules 0 = 定义无方程可归纳
      (multiplication/subtraction); 样本仅前缀 = 非算子用途 (neg).
    最小样本量指标 (两个):
      0.9_min   acc ≥ 0.9 的 token 中样本量最小者 (近满泛化下限)
      one_min   acc = 1.0 的 token 中样本量最小者 (完美泛化, 剥离 0-acc 后)
    返回 (zero_rows, nine_min, one_min, one_count).
    """
    from collections import Counter
    last = epoch_gen[-1] if epoch_gen else {}
    sample_cnt = Counter()
    for s in train_samples:
        for t in set(s["seq"]):
            sample_cnt[t] += 1

    def _nrules(eid):
        from tokenizer.maintain import core
        d = core.load_all().get(eid, {}).get("definition") or {}
        return len(d.get("rules") or [])

    zero_rows = [(k, sample_cnt.get(api.eid_by_name(k), 0), _nrules(api.eid_by_name(k)))
                 for k in sorted(last) if last[k] == 0.0]
    nine_min = one_min = None
    for k in sorted(last):
        a = last[k]
        n = sample_cnt.get(api.eid_by_name(k), 0)
        if a >= 0.9 and (nine_min is None or n < nine_min[1]):
            nine_min = (k, n)
        if a >= 1.0 and (one_min is None or n < one_min[1]):
            one_min = (k, n)
    return zero_rows, nine_min, one_min, sum(1 for a in last.values() if a >= 1.0)


def build_samples(cfg, seed):
    """合成训练样本 (compose_samples, 配置驱动) + 泛化配置."""
    s = cfg["synth"]
    if "samples" in s:
        train, npos, nneg = compose_samples(samples=s["samples"], seed=seed)
        # EXP-C1a 进制固化: 训练表示移除进制参数 (cardinality token)
        # strip="all" 全删 (过度破坏) / strip="cardinality" 只删进制参数 (正确粒度)
        strip = cfg.get("synth", {}).get("strip_base")
        if strip:
            from lab.synth_core_helpers_nobase import strip_base_tokens
            train = strip_base_tokens(train, mode=strip)
            log(f'  EXP-C1a 进制固化 (strip={strip}): 已移除进制参数 token')
        # EXP-41 符号置换: train 时随机置换 digit 概念 token (保持结构),
        # 测试用原 token — 若判定不变, 学的是关系非符号 (用户确立最强弹药).
        perm = cfg.get("synth", {}).get("permute_digits")
        if perm:
            from tokenizer import api as _api
            rng = __import__("random").Random(seed + 777)
            dnames = ["digit_zero","digit_one","digit_two","digit_three","digit_four",
                      "digit_five","digit_six","digit_seven","digit_eight","digit_nine"]
            digs = [_api.eid_by_name(n) for n in dnames]
            perm_map = list(digs)
            rng.shuffle(perm_map)
            cnt = 0
            for ts in train:
                seq = ts["seq"]
                for i, e in enumerate(seq):
                    if e in digs:
                        seq[i] = perm_map[digs.index(e)]
                        cnt += 1
            log(f'  EXP-41 符号置换: digit 映射 {cnt} 处')
        log(f'  训练样本: n={len(train)} 真={npos} 假={nneg} (组合 {len(s["samples"])} 类样本)')
        return train, cfg.get("verify", {}).get("ood", [])
    raise ValueError("配置必须提供 synth.samples (样本类型组合列表)")


def _make_verify_fn(cfg, train_samples, seed):
    """verify 样本注入口: train_replay=训练样本, ood/gen=全方向泛化合并.

    全方向算子沿 token 方向体系自动发现 (api.direction_ops), config.verify.ood
    可覆盖 (每个 spec: op/digits/n/mode/neg_mode); 无则默认全方向 × 5位 mixed.
    每个算子按 tokenizer 求值能力探测 (eval_op 试算), 不可求值的自动跳过
    (如 translation/inversion 求值未就绪) — 求值能力就绪即自动纳入, 零硬编码.
    """
    # 默认全方向 × 5位 mixed (每个算子均匀少量采样, token 泛化目标, 非海量).
    # 只测"有训练覆盖的算子" (沿 config.synth.samples 的 op, 未训练 token 的
    # OOD = 没教的泛化, 非 OOD 泛化 — 零硬编码: 从配置读, 不查算子名).
    # 迭代算符 (有迭代基础, 下层算符重复应用) 值域爆炸 → 小值域少量
    # (digits=2, n=20), 直接算符 5位×40
    trained_ops = {s.get("op") for s in (cfg.get("synth", {}).get("samples") or [])
                   if s.get("op")}
    ood_cfgs = []
    for d in api.direction_ops():
        if d["name"] not in trained_ops:
            continue
        eid = api.eid_by_name(d["name"])
        # 逻辑门 (op_domain=logical): 不走数字 OOD (bool 无位数), 由
        # logic_arith 逻辑命题 OOD 覆盖
        if api.op_domain(eid) == "logical":
            continue
        from tokenizer.eval.engine import _base_of_layer, _is_unary, _is_pred_based
        # 迭代算符 (有迭代基础): 操作数 2 位, 结果 20 位 (用户确立: 20 位是
        # 结果范围非输入, 防高阶溢出); 直接算符: 5 位操作数 × 20 位结果
        if _is_unary(eid) or (_base_of_layer(eid) is not None and not _is_pred_based(eid)):
            ood_cfgs.append({"op": d["name"], "digits": 2, "n": 20, "mode": "mixed",
                             "result_digits": 20})
        else:
            ood_cfgs.append({"op": d["name"], "digits": 5, "n": 40, "mode": "mixed",
                             "result_digits": 20})
    ood_cfgs = [s for s in ood_cfgs if _evaluable(s["op"])]
    # 未经训练的外推算子 (用户确立): 沿 A 层 iterate/inverse 箭头发现的
    # 迭代链高阶算符 (5 阶 super_root/super_log) + 混合迭代 (translation/
    # inversion 平移反演, differential/integral 升层降层) — 训练无样本,
    # 验证模型沿迭代链条解释外推. 少量 (n=10, digits=2, 值域受限防爆炸).
    untrained = cfg.get("verify", {}).get("untrained_ops") or []
    for op_name in untrained:
        if op_name in trained_ops:
            continue
        eid = api.eid_by_name(op_name)
        if not _evaluable(op_name):
            continue
        ood_cfgs.append({"op": op_name, "digits": 2, "n": 10, "mode": "mixed",
                         "kind": "untrained", "result_digits": 20})
    # config.verify.ood 附加 (extrap 外推等额外视角), 不替换默认全方向 —
    # 替换会导致判定口径 OOD 只剩外推 (2000 位退化全真, 一致=0)
    extra = cfg.get("verify", {}).get("ood") or []
    # extrap (2000 位外推) 不进 verify 主流程 (长序列推理极慢), 由 run_exp 单独串行验证
    extra_plain = [e for e in extra if e.get("kind") != "extrap"]
    ood_cfgs = [s for s in ood_cfgs if not any(
        s.get("op") == e.get("op") and s.get("kind") == e.get("kind") for e in extra_plain)]
    ood_cfgs.extend(extra_plain)
    extrap_cfgs = [e for e in extra if e.get("kind") == "extrap"]
    log(f'  泛化配置: ' + " | ".join(
        f'coercion提升hi={s.get("hi", 100)}' if s.get("kind") == "coercion"
        else f'逻辑{s["op"]}' if s.get("kind") == "logic"
        else f'填空{s["op"]}' if s.get("kind") == "fill"
        else f'选择{s["op"]}k={s.get("k", 2)}' if s.get("kind") == "choose"
        else f'深嵌套{s["op"]}depth={s.get("max_depth")}' if s.get("kind") == "balanced"
        else f'外推{s["op"]}depth{s.get("depths")}' if s.get("kind") == "deep_nest"
        else f'2000位外推{s["op"]}' if s.get("kind") == "extrap"
        else f'联合笛卡尔{s["ops"]}×{s.get("ndigits", 2)}位' if s.get("kind") == "cartesian"
        else f'进制{s.get("bases", (3,4,5,6,7,8,9))}×{s.get("max_digits", 20)}位结果{s["ops"]}' if s.get("kind") == "radix"
        else f'逻辑命题{s["ops"]}' if s.get("kind") == "logic_arith"
        else f'{s["mode"]}{s["digits"]}位{s["op"]}' for s in ood_cfgs))
    if extrap_cfgs:
        log(f'  2000位外推 (单独验证, 不进主 verify): ' + " | ".join(
            f'{s["op"]}' for s in extrap_cfgs))

    def fn(seed_i, exclude, depth):
        if seed_i == seed:
            return train_samples
        out = []
        for spec in ood_cfgs:
            kind = spec.get("kind")
            if kind == "coercion":
                ss, _, _ = synth_core.coercion_samples(hi=spec.get("hi", 100), seed=seed_i)
                out.extend(ss)
                continue
            if kind == "logic":
                ss, _, _ = synth_core.logic_samples(op=spec["op"], seed=seed_i)
                out.extend(ss)
                continue
            if kind == "logic_arith":
                # 逻辑门 OOD: 数学命题套用 (深嵌套多题型, 与训练同构)
                ss, _, _ = synth_core.logic_arith_samples(
                    ops=spec["ops"], hi=spec.get("hi", 9), seed=seed_i)
                out.extend(ss)
                continue
            if kind == "fill":
                ss, _, _ = synth_core.fill_samples(op=spec["op"], hi=spec.get("hi", 9), seed=seed_i)
                out.extend(ss)
                continue
            if kind == "choose":
                ss, _, _ = synth_core.choose_samples(op=spec["op"], hi=spec.get("hi", 9),
                                                     k=spec.get("k", 2), seed=seed_i)
                out.extend(ss)
                continue
            if kind == "balanced":
                ss, _, _ = synth_core.balanced_samples(
                    max_depth=spec["max_depth"], hi=spec.get("hi", 9),
                    op=spec["op"], neg_mode=spec.get("neg_mode", 0), seed=seed_i)
                out.extend(ss)
                continue
            if kind == "deep_nest":
                ss, _, _ = synth_core.deep_nest_samples(
                    op=spec["op"], hi=spec.get("hi", 5),
                    depths=spec.get("depths", [6, 11]),
                    n=spec.get("n", 100),
                    neg_mode=spec.get("neg_mode", 1), seed=seed_i)
                out.extend(ss)
                continue
            if kind == "extrap":
                ss, _, _ = synth_core.extrap_2000_samples(
                    op=spec["op"], ndigits=spec.get("ndigits", 2000), seed=seed_i)
                out.extend(ss)
                continue
            if kind == "cartesian":
                ss, _, _ = synth_core.cartesian_ood_samples(
                    ops=spec["ops"], ndigits=spec.get("ndigits", 2),
                    n=spec.get("n", 20), neg_mode=spec.get("neg_mode", 1), seed=seed_i)
                out.extend(ss)
                continue
            if kind == "radix":
                ss, _, _ = synth_core.radix_ood_samples(
                    ops=spec["ops"], bases=spec.get("bases", (3, 4, 5, 6, 7, 8, 9)),
                    max_digits=spec.get("max_digits", 20), n=spec.get("n", 20),
                    neg_mode=spec.get("neg_mode", 1), seed=seed_i)
                out.extend(ss)
                continue
            out.extend(ood_samples(op=spec["op"], digits=spec["digits"],
                                   n=spec.get("n", 300), mode=spec.get("mode", "random"),
                                   neg_mode=spec.get("neg_mode", 1), seed=seed_i,
                                   result_digits=spec.get("result_digits"))[0])
        return out
    return fn


def main():
    ap = argparse.ArgumentParser(description="标准实验入口 (薄封装编排)")
    ap.add_argument("--config", required=True, help="实验配置 JSON 路径")
    ap.add_argument("--compare-law", action="store_true",
                    help="matched-control: 对全部 law 样本用 correct/shuffle/"
                    "wrong-symmetry 三 mode 分别训练, 对比 OOD (证明 law 结构驱动泛化)")
    args = ap.parse_args()

    if args.compare_law:
        _run_law_mode_compare(args.config)
        return

    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    seed = cfg.get("seed", 0)
    name = cfg["name"]

    log(f'== 实验: {name} (seed={seed}) ==')
    t0 = time.time()

    # 1. 合成训练样本 (synth 消费方, 配置驱动)
    train_samples, ood_cfgs = build_samples(cfg, seed)
    t_synth = time.time()

    # 2. 训练 (项目后端 train_seq, 自动归档); 逐 epoch 泛化曲线 (反馈监督基础设施)
    tr = cfg.get("train", {})
    epoch_eval = None
    if tr.get("epoch_gen"):
        ood_fixed = _make_verify_fn(cfg, train_samples, seed)(99999, [], None)
        epoch_eval = _make_per_token_eval(ood_fixed)
    res = train_seq(train_samples, epochs=tr.get("epochs", 15), dim=tr.get("dim", 64),
                    num_layers=tr.get("layers", 2), seed=seed, token=name,
                    batch_size=tr.get("batch_size", 512), epoch_eval_fn=epoch_eval,
                    max_n=tr.get("max_n"), qat_bits=tr.get("qat_bits"))
    run_dir = res["run_dir"]
    # 归档完整实验配置 (含 synth.samples 组合列表, 可复现 + 按样本类型诊断)
    _archive_exp_config(run_dir, cfg)
    log(f'  训练归档: {run_dir}')
    log(f'  train acc={res["acc"]:.3f} valid_acc={res["valid_acc"]:.3f}')
    log(f'  loss 曲线: ' + " ".join(f"{x:.2f}" for x in res["losses"]))
    if res.get("epoch_gen"):
        eg = res["epoch_gen"]
        all_avg = [sum(v.values()) / max(len(v), 1) for v in eg]
        no0_avg = []
        z0_cnt = []
        for v in eg:
            vals = [a for a in v.values() if a > 0.0]
            no0_avg.append(sum(vals) / max(len(vals), 1))
            z0_cnt.append(sum(1 for a in v.values() if a == 0.0))
        log(f'  epoch_gen_all (全 token): ' + " ".join(f"{x:.2f}" for x in all_avg))
        log(f'  epoch_gen_no0 (剥离0acc): ' + " ".join(f"{x:.2f}" for x in no0_avg))
        log(f'  epoch_gen_0 (0acc数量):   ' + " ".join(f"{x:.0f}" for x in z0_cnt))
        ok = sum(1 for v in eg if all(a >= 0.98 for a in v.values()))
        log(f'  全 token 达 0.98 的 epoch: {ok}/{len(eg)}')
        log('  [可解释泛化调试] 学到的 token 泛化应 epoch 少即 1.0; 剥离 0-acc 曲线'
            ' 揭示做对/做错; 0-acc token = 定义/样本设计缺陷 (非 epoch/层数/堆量) — '
            '追求泛化快训练量少, 不靠堆量')
        # 泛化诊断: 0-acc 问题定位 (样本量|定义rules) + 0.9/1.0 最小样本量
        zero_rows, nine_min, one_min, one_count = _gen_diagnosis(train_samples, res.get("epoch_gen") or [])
        if zero_rows:
            log('  [泛化诊断] 0-acc token 定位 (样本量 | 定义rules):')
            for k, n_s, n_r in zero_rows:
                log(f'    {k}: 样本量 {n_s} | rules {n_r}'
                    f'{"" if n_s else "  ← 配置未覆盖 (无训练样本)"}'
                    f'{"  ← 定义无方程可归纳" if n_s and not n_r else ""}')
        if nine_min:
            log(f'  [≥0.9acc 最小样本量] {nine_min[0]}: 仅 {nine_min[1]} 样本即近满泛化')
        if one_min:
            log(f'  [1.0acc 最小样本量] {one_min[0]}: 仅 {one_min[1]} 样本即学会 '
                f'(共 {one_count} 个 1-acc) — 归纳不需过多样本, 定义清晰 + 各角度对比即可')
            log('  [完美泛化方法论] 学习是归纳: 定义方程 (rules) 清晰 + 少量样本 '
                '(1-2 定义样例) + 各角度对比样本 (正/反例、边界、换位) 即可学会; '
                '0-acc 根因 = 样本零覆盖 或 定义缺方程, 非堆量可解')
        # per-token 未达标列表 (默认诊断输出, 免手动拼装)
        last = eg[-1]
        below = sorted((k, a) for k, a in last.items() if a < 1.0)
        if below:
            log(f'  [per-token 未达标 {len(below)}/{len(last)}]: ' + " | ".join(
                f'{k}={a:.2f}' for k, a in below))
    t_train = time.time()

    # 3. 校验 (项目后端 verify, 注入泛化样本合成)
    # batch_size=32: 流式分批推理 (2000 位外推长序列防 OOM)
    views = verify(run_dir, samples_fn=_make_verify_fn(cfg, train_samples, seed), batch_size=32)
    log(f'  位置级: overall_acc={views.get("overall_acc", 0):.3f} '
        f'ood_acc={views.get("ood_acc", 0):.3f} '
        f'gen_acc={views.get("gen_acc", 0):.3f}')

    # 3b. 判定口径评估 (末尾真值, 对比位置级假象)
    ood = _make_verify_fn(cfg, train_samples, seed)(12345, [], None)
    jacc, jt, jf, jcons = _judge_eval(res["model"], ood)
    log(f'  判定口径 (全真值): acc={jacc:.3f} 判真={jt:.3f} 判假={jf:.3f} 一致={jcons:.3f}'
        f' (n={len(ood)})')

    # 3c. 2000 位外推 (默认执行, 用户确立): config.verify.ood kind=extrap 的
    # 算子逐一串行验证 — 长序列流式分批 (batch_size=1), 判定口径 acc
    extrap_cfgs = [e for e in (cfg.get("verify", {}).get("ood") or [])
                   if e.get("kind") == "extrap"]
    if extrap_cfgs:
        for spec in extrap_cfgs:
            _run_extrap(res["model"], spec["op"], spec.get("ndigits", 2000))
    t_verify = time.time()

    # 逐 epoch 泛化曲线入 views (verify 之后, 防被 save_views 覆盖)
    _archive_epoch_gen(run_dir, res.get("epoch_gen") or [])
    _archive_gen_diag(run_dir, train_samples, res.get("epoch_gen") or [])

    log(f'== 耗时拆解: 合成 {t_synth-t0:.1f}s | 训练 {t_train-t_synth:.1f}s | '
        f'verify {t_verify-t_train:.1f}s | 总 {t_verify-t0:.1f}s ==')


def _run_law_mode_compare(config_path):
    """matched-control 实验 (用户确立): law 三 mode 对比, 证明结构驱动 OOD.

    对配置中全部 law 样本, 分别用 correct/shuffle/wrong-symmetry 训练
    (样本量/token数/操作数分布/base10/single-digit 完全一致, 只变 law
    relation), 评估同 OOD 集, 输出对比表.
    预期: L_correct OOD=1 / L_shuffle≈0 / L_wrong≈0 → 非 extra data,
    是 specific relational structure ⟹ OOD.
    """
    import copy
    with open(config_path, encoding="utf-8") as f:
        base_cfg = json.load(f)
    seed = base_cfg.get("seed", 0)
    modes = ["correct", "shuffle", "wrong-symmetry"]
    results = {}
    for mode in modes:
        cfg = copy.deepcopy(base_cfg)
        cfg["name"] = f"{base_cfg['name']}_law_{mode}"
        for s in cfg["synth"]["samples"]:
            if s["kind"] == "law":
                s["mode"] = mode
        log(f'== matched-control mode={mode} ==')
        # 训练 + 判定口径 OOD
        train_samples, _ = build_samples(cfg, seed)
        res = train_seq(train_samples, epochs=cfg.get("train", {}).get("epochs", 8),
                        dim=cfg.get("train", {}).get("dim", 64),
                        num_layers=cfg.get("train", {}).get("layers", 2),
                        seed=seed, token=cfg["name"], archive_dir=None, batch_size=512,
                        qat_bits=cfg.get("train", {}).get("qat_bits"))
        ood = _make_verify_fn(cfg, train_samples, seed)(12345, [], None)
        jacc, _, _, _ = _judge_eval(res["model"], ood)
        # 仅-law 算符 (无 balanced) 的目标 OOD acc
        results[mode] = {"judge_acc": round(jacc, 3)}
        log(f'  mode={mode}: 判定口径 acc={jacc:.3f} (n={len(ood)})')
    log('== matched-control 对比 (law 三 mode, 样本完全一致) ==')
    for mode, r in results.items():
        log(f'  L_{mode}: OOD={r["judge_acc"]:.3f}')
    with open("/tmp/opencode/law_mode_compare.json", "w") as f:
        json.dump(results, f, indent=2)
    log('  已落盘 /tmp/opencode/law_mode_compare.json')


def _run_extrap(model, op, ndigits):
    """2000 位外推验证 (单算子): 生成外推样本, 流式判定, 报告 acc.

    长序列 (2000 位) 逐样本 batch_size=1 前向, 判定口径 = 完整序列重建.
    """
    import torch
    from train.data import collate, rev_vocab
    from tokenizer import api as _api
    try:
        samples, _, _ = synth_core.extrap_2000_samples(op=op, ndigits=ndigits)
    except ValueError as e:
        log(f'  2000位外推 {op}: 跳过 ({e})')
        return
    if not samples:
        log(f'  2000位外推 {op}: 无样本')
        return
    rv = rev_vocab()
    model.eval()
    hit = 0
    # 逐样本流式推理 (2000 位长序列单 batch 会 OOM)
    for s in samples:
        b = collate([s], input_mode="ids")
        dev = next(model.parameters()).device
        b = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
        with torch.no_grad():
            logits, _ = model(b["inputs"], mask=b["mask"])
        rl = b["lengths"][0]
        preds = [rv[p] for p in logits[0, :rl].argmax(dim=1).tolist()]
        if all(p == t for p, t in zip(preds, s["seq"])):
            hit += 1
    log(f'  2000位外推 {op}: acc={hit/max(len(samples),1):.3f} (n={len(samples)})')


def _make_per_token_eval(ood):
    """逐 epoch per-token 泛化评估器: model → {token 名: 泛化 acc}.

    OOD 样本逐 token 预测 acc (每 token 位置正确率), 逐 epoch 调用
    (精细化控制训练量 + 反馈监督基础设施: 每 token 达标 epoch).
    """
    import torch
    from tokenizer import api as _api
    from train.data import collate, rev_vocab
    from collections import defaultdict
    rv = rev_vocab()
    fixed = ood

    def ev(model):
        dev = next(model.parameters()).device
        batch = collate(fixed, input_mode="ids")
        batch = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        model.eval()
        with torch.no_grad():
            logits, _ = model(batch["inputs"], mask=batch["mask"])
        lens = batch["lengths"]
        cnt = defaultdict(lambda: [0, 0])
        for i, s in enumerate(fixed):
            rl = lens[i]
            preds = [rv[p] for p in logits[i, :rl].argmax(dim=1).tolist()]
            for j, t in enumerate(s["seq"]):
                cnt[t][1] += 1
                if j < len(preds) and preds[j] == t:
                    cnt[t][0] += 1
        return {_api.name(t): c[0] / max(c[1], 1) for t, c in cnt.items()}
    return ev


def _judge_eval(model, samples):
    """判定口径 (全真值): 完整序列逐 token 重建, 全对才计正确 (用户确立, 末尾真值淘汰).

    判定 = 每样本非 padding 全部位置预测 argmax 与真实 seq 一致 (truth 蕴含整序列).
    指标: acc (全对率) / 判真率 / 判假率 / 一致性.
    """
    import torch
    from tokenizer import api as _api
    _TRUTH = _api.role_token("truth")
    _TRUE = _TRUTH[0]
    _FALSE = _TRUTH[1]
    batch = collate(samples, input_mode="ids")
    dev = next(model.parameters()).device
    batch = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    model.eval()
    with torch.no_grad():
        logits, _ = model(batch["inputs"], mask=batch["mask"])
    B, L = logits.shape[0], logits.shape[1]
    real_lens = L - batch["mask"].sum(dim=1)
    pred_idx = logits.argmax(dim=2)  # 全位置
    rv = rev_vocab()
    hit = tp = tn = tfn = tfp = 0
    for i, s in enumerate(samples):
        rl = real_lens[i].item()
        preds = [rv[p] for p in pred_idx[i, :rl].tolist()]
        truth = all(p == t for p, t in zip(preds, s["seq"]))
        if truth:
            hit += 1
            if s["seq"][-1] == _TRUE:
                tp += 1; tfn += 1
            else:
                tn += 1; tfp += 1
        else:
            if s["seq"][-1] == _TRUE:
                tfn += 1
            else:
                tfp += 1
    n = max(len(samples), 1)
    acc = hit / n
    tt = tp / max(tfn, 1)
    ff = tn / max(tfp, 1)
    return acc, tt, ff, 1.0 - abs(tt - ff)
    pgen = views.get("per_token_gen", {})
    log(f'  逐 token 泛化 ({len(pgen)} token): ' + json.dumps(
        {k: f'{v["acc"]:.2f}({v["total"]})' for k, v in pgen.items() if v.get("total")}, ensure_ascii=False))

    log(f'== 完成, 总耗时 {time.time()-t0:.1f}s ==')


if __name__ == "__main__":
    main()
