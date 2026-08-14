"""tokenizer/ —— 唯一定义源 (意思层)

定义"是什么"——token、元素、向量、序列语法。
零依赖: 不 import 本项目任何模块。

内部:
  _register.py — 三层 token 注册器 (B/C/S 加载 + 查询)
  _axes.py     — token 轴空间
  _vector.py   — 元素 → 向量
  maintain/    — 维护/诊断 (PROJECTIONS 六层: B/C/S/G/P/X, 单一入口)
  grammar/     — G 层 gtoken (排列方法) + P 层呈现 (concrete syntax) 实现
  construct/   — 构造器 (resolve/render/expand/nl: 符号 ↔ 四形态 + NL 定义序列)
  head/        — 注意算子 (同类/异类归类, 可插拔 select × compute)
"""
