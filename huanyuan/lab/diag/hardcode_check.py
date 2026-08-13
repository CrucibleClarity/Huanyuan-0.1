#!/usr/bin/env python3
"""hardcode_check.py —— 硬编码检测器 (mask 单 token, 观测输出是否仍可生成)

方法论 (用户确立):
  同一映射下的 ascbgp token (A/S/C/B/G/P) 逐个 mask (临时移除/破坏其定义),
  观测目标脚本的输出。若 mask 掉某 token 后输出仍可生成/一致 →
  说明该 token 的贡献被硬编码 (代码没读它, 靠假设/映射表推断), 报告硬编码。

mask 策略 (每个 token 两种破坏):
  remove  从注册表临时移除 (代码若硬编码它 → 仍能跑, 触发)
  corrupt 破坏其定义 (glyph→'', definition→{}, maps_to→{}) (代码若硬编码其
          字段值 → 仍能跑, 触发)

用法:
  PYTHONPATH=src python -m lab.diag.hardcode_check \
      --target "lab.synth_core:compose_samples" --config lab/configs/numeral_v4.json \
      --scope C --report-top 20
  观测: 目标函数跑完, 比较 mask 前后输出是否一致。
"""
import argparse
import importlib
import json
import sys
import traceback


def _load_target(target_spec):
    """'module:func' → callable (模块 import + 取属性)."""
    mod_name, _, attr = target_spec.partition(":")
    mod = importlib.import_module(mod_name)
    obj = mod
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


class HardcodeChecker:
    """mask 单 token → 跑目标 → 对比输出 → 报告硬编码嫌疑."""

    def __init__(self, scope_layers=("B", "C", "S", "G", "P", "A")):
        from tokenizer.maintain import core
        self.core = core
        self.scope_layers = scope_layers
        self.baseline = {}   # eid → (layer, 原字段dict)

    def snapshot(self):
        """备份全部 scope token 原字段 (供恢复)."""
        for layer in self.scope_layers:
            for eid, fields in self.core.load_layer(layer).items():
                self.baseline[eid] = (layer, dict(fields))

    def restore(self):
        """恢复全部 token 原字段 (清缓存重载)."""
        self.core.invalidate()
        from tokenizer import _register
        _register.load_baseloop(); _register.load_derive()
        _register.load_symbols(); _register.load_arrows()
        from tokenizer import cte
        cte.invalidate_all()
        from tokenizer import api
        api._ARRANGE_CACHE.clear()

    def mask(self, eid, mode):
        """mask 单 token: remove=从层中移除; corrupt=破坏定义字段.

        目标注册表 = 消费方真正读取的内存 registry (_register 的 B/C/S/A
        TOKEN_BY_NAME/DERIVE_BY_NAME/SYMBOL_BY_GLYPH/ARROW_BY_NAME) +
        core._LAYER_CACHE (G/P 及 maintain 路径读) — 只 mask core 缓存
        对经 _register 查名的合成器零效果 (曾 372/372 假阳性).
        """
        layer = self.baseline[eid][0]
        from tokenizer import _register
        if layer in ("B", "C", "S", "A"):
            reg, by_name = {
                "B": (_register.TOKEN_REGISTRY, _register.TOKEN_BY_NAME),
                "C": (_register.DERIVE_REGISTRY, _register.DERIVE_BY_NAME),
                "S": (_register.SYMBOL_REGISTRY, _register.SYMBOL_BY_NAME),
                "A": (_register.ARROW_REGISTRY, _register.ARROW_BY_NAME),
            }[layer]
            if eid in reg:
                td = reg.pop(eid)
                by_name.pop(td.name, None)
                if layer == "S" and hasattr(td, "glyph"):
                    sidx = _register.SYMBOL_BY_GLYPH.get(td.glyph, [])
                    _register.SYMBOL_BY_GLYPH[td.glyph] = [s for s in sidx if s != eid]
        rows = {r["eid"]: r for r in self.core._LAYER_CACHE.get(layer, [])}
        if eid in rows:
            if mode == "remove":
                del rows[eid]
            else:
                for k in ("glyph", "definition", "maps_to", "grammar", "reduction",
                          "source", "target", "concept", "precedence", "associativity"):
                    if k in rows[eid]:
                        rows[eid][k] = "" if k == "glyph" else ({} if k in ("definition", "maps_to") else [])
            self.core._LAYER_CACHE[layer] = list(rows.values())
        self.core._ALL_CACHE = None
        from tokenizer import cte
        cte.invalidate_all()
        from tokenizer import api
        api._ARRANGE_CACHE.clear()

    def run_masked(self, target, args, eid, mode, timeout_s=60):
        """mask eid 后跑 target, 返回 (输出, 异常). eid='__none__'=不 mask (baseline)."""
        if eid != "__none__":
            self.mask(eid, mode)
        try:
            return target(*args), None
        except Exception as e:
            return None, f"{type(e).__name__}: {e}"
        finally:
            self.restore()


def main():
    ap = argparse.ArgumentParser(description="硬编码检测器 (mask token 观测输出)")
    ap.add_argument("--target", required=True, help="目标 'module:func'")
    ap.add_argument("--config", default=None, help="实验配置 (传给 target 的额外参数)")
    ap.add_argument("--scope", default="BCSGPA", help="mask 层范围 (B/C/S/G/P/A 子集)")
    ap.add_argument("--mode", default="remove", choices=["remove", "corrupt"],
                    help="mask 方式: remove=移除 / corrupt=破坏定义")
    ap.add_argument("--top", type=int, default=20, help="报告前 N 个")
    args = ap.parse_args()

    target = _load_target(args.target)
    extra = []
    if args.config:
        extra = [json.load(open(args.config, encoding="utf-8"))]

    checker = HardcodeChecker(scope_layers=tuple(args.scope))
    checker.snapshot()

    # baseline (无 mask)
    base_out, base_err = checker.run_masked(target, extra, "__none__", "remove")
    print(f"[baseline] target={args.target} 输出: {str(base_out)[:120]} 异常: {base_err}")

    # 逐个 mask
    from tokenizer import api
    results = []
    for eid in sorted(checker.baseline):
        out, err = checker.run_masked(target, extra, eid, args.mode)
        # 硬编码嫌疑: 无异常 且 输出与 baseline 一致 (代码没读该 token 仍能生成)
        if err is None and base_err is None and out is not None and out == base_out:
            results.append((eid, "一致 (mask 后仍生成, 硬编码嫌疑)"))
        elif err is None and base_err is not None:
            results.append((eid, "mask 后无异常 (baseline 有异常, 疑似硬编码兜底)"))
    print(f"\n== 硬编码嫌疑 ({len(results)}/{len(checker.baseline)}) ==")
    for eid, note in results[: args.top]:
        name = api.name(eid)
        print(f"  {eid} {name}: {note}")

    checker.restore()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
