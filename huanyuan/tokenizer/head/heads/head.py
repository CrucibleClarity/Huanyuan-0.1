"""heads/head.py —— 头实例 (一个注意头的配置)

头 = (采样器, 计算配置, 参数) 的组合, 复用 select/compute registry。
一个头关注一个维度 (同类/异类、语法槽位、分层、深度等), 通过 run 独立执行。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..router import run


@dataclass
class Head:
    name: str
    selector: str = "all"
    algorithm: str | None = None
    weight: str | None = None
    params: dict = field(default_factory=dict)

    def run(self, sequence, ctx=None):
        """执行本头: select (采样) → compute (计算)。weight 指定走权重算法, 否则归类算法。"""
        return run(
            sequence,
            selector=self.selector,
            algorithm=self.algorithm,
            weight=self.weight,
            ctx=ctx,
            **self.params,
        )
