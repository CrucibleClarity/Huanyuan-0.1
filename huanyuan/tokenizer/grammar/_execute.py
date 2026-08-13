"""tokenizer/grammar/_execute.py —— 语法执行器

批量执行查询/组装 (query/scope/assemble), 逐条报告, 不中断。
tasks = [{op, ...}]; op ∈ query|scope|assemble。
"""
from ._query import query, scope
from ._assemble import assemble


def execute(tasks):
    """批量执行查询/组装, 返回逐条结果 (错误不中止)。"""
    results = []
    for t in tasks:
        try:
            op = t.get('op')
            if op == 'query':
                results.append({'op': op, 'result': query(t['node'])})
            elif op == 'scope':
                results.append({'op': op, 'result': scope(t['node'])})
            elif op == 'assemble':
                results.append({'op': op, 'result': assemble(t['node'], t.get('children', []))})
            else:
                results.append({'op': op, 'error': f'未知操作: {op!r}'})
        except Exception as e:
            results.append({'op': t.get('op'), 'error': str(e)})
    return results
