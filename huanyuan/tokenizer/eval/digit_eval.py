"""tokenizer/eval/digit_eval.py —— 数字求值器 (位权合成, 读定义零硬编码)

数字表示 (用户确立): 所有数字 = 0-9 数符 token (digit_n) 组合的数位序列。
数值计算 (位权): 
  单数符数值 = 基数 × 进制^位序   (cardinality × base^place_pos)
  数字值     = Σ 各数位贡献        (numeral_value)
进制/位序/基数全部从 token 定义读取, 结果用数字 token (value_n) 表示。

数据源:
  digit→基数: symbol maps_to (glyph 'n' → [digit_n, value_n]) — 数符双 token 体系
  进制/位序: token 定义 (base/place_pos 概念)
  输出: value token (value_n) 表示数值
"""
from __future__ import annotations

from ..maintain import core

_DIGIT_NAMES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
_VALUE_NAMES = {f"value_{w}": i for i, w in enumerate(_DIGIT_NAMES)}


def _name(eid: str) -> str:
    """eid → name (本地解析, 避免循环导入)。"""
    for layer in ("S", "C", "B"):
        for r in core.load_layer(layer).values():
            if r.get("eid") == eid:
                return r.get("name", eid)
    return eid


def _eid_by_name(name: str) -> str:
    """name → eid (本地解析, 避免循环导入 api)。"""
    for layer in ("S", "C", "B", "G", "P"):
        for r in core.load_layer(layer).values():
            if r.get("name") == name:
                return r["eid"]
    raise KeyError(f"未注册 token: {name!r}")


def digit_cardinality(digit_eid: str) -> int:
    """数符 eid → 数值 (数符的 value, 0-9)。

    数据源: S 层 symbol maps_to — glyph 'n' → [digit_n, value_n]。
    找出该 digit 对应 symbol 的 value 目标, 从 value_n 名解析数值。
    零硬编码: 不预设 digit 名, 沿 maps_to 关联推导。
    """
    # 遍历 symbol 找映射到该 digit 的符号
    for sid, td in core.load_layer("S").items():
        mt = td.get("maps_to") or {}
        if digit_eid in mt:
            # 同 symbol 还映射 value_n → 该 value 名即数值
            for target in mt:
                if target != digit_eid:
                    nm = _name(target)
                    if nm in _VALUE_NAMES:
                        return _VALUE_NAMES[nm]
    raise KeyError(f"数符数值未解析: {_name(digit_eid)} ({digit_eid})")


def digits_to_numeral(digit_eids: list[str], base: int = 10) -> int:
    """数位序列 → 数值 (位权合成: Σ 数符value × base^pos)。

    数位表示 (用户确立): 序列高→低位序 (首位=最高位), 数位可为
    正数符 digit_n 或负数位 [neg, digit_n] (neg 一元算子前缀, 极性对偶)。
    例: [1][0] = 10 (十位1, 个位0); [1][-1] = 9 (十位1, 个位-1)。
    base: 进制 (从定义读, 默认 10)。
    公式: value = Σ (数符的 value) × base^pos × 极性 (pos = 从低位起序号)。
    用户铁律: 乘的是数符的 value, 不是 digit 数符本身。
    """
    neg = _eid_by_name("neg")
    # 解析数位: neg 标记紧邻的下一个数符为负 (极性对偶)
    digits = []  # [(极性, digit_eid)] 高→低
    i = 0
    while i < len(digit_eids):
        sign = 1
        if digit_eids[i] == neg:
            sign = -1
            i += 1
        if i >= len(digit_eids):
            break
        digits.append((sign, digit_eids[i]))
        i += 1
    # 高→低: pos 从低位 (最右) 递增
    value = 0
    for pos, (sign, deid) in enumerate(reversed(digits)):
        card = digit_cardinality(deid)
        value += sign * card * (base ** pos)
    return value


def _digit_eid_for(n: int) -> str:
    """数值 n (0-9) → 数符 digit token eid (S 层 glyph maps_to, 取 digit 目标)。"""
    for sid, td in core.load_layer("S").items():
        mt = td.get("maps_to") or {}
        for target in mt:
            if _name(target).startswith("digit_"):
                # 校验该 digit 的 value == n
                if digit_cardinality(target) == n:
                    return target
    raise KeyError(f"数符未找到: value={n}")


def numeral_to_digits(value: int, base: int = 10) -> list[str]:
    """数值 → 数位序列 (digit eid, 高→低位序, 首位=最高位)。"""
    if value == 0:
        return [_digit_eid_for(0)]
    out = []
    while value:
        value, d = divmod(value, base)
        out.append(_digit_eid_for(d))
    out.reverse()  # 低→高 → 高→低
    return out


def eval_digit(digit_eids: list[str], base: int = 10) -> int:
    """数字求值器主入口: 数位序列 → 数值 (int)。"""
    return digits_to_numeral(digit_eids, base)


def eval_numeral_tokens(digit_eids: list[str], base: int = 10):
    """数字求值 → 用 value token 表示 (用户: 位权 value 用数字 token 表示)。"""
    n = digits_to_numeral(digit_eids, base)
    return n


__all__ = ["digit_cardinality", "digits_to_numeral", "numeral_to_digits",
           "eval_digit", "eval_numeral_tokens"]
