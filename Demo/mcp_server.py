import ast
import operator

from mcp.server.fastmcp import FastMCP

# 创建MCP服务器
mcp = FastMCP("计算器演示")

# 安全的表达式求值器（不使用 eval/exec，避免任意代码执行）
_ALLOWED_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST) -> float:
    """递归求值，只允许数字字面量与四则运算，禁止任何函数调用/属性访问。"""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINARY_OPS:
        return _ALLOWED_BINARY_OPS[type(node.op)](
            _safe_eval(node.left), _safe_eval(node.right)
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY_OPS:
        return _ALLOWED_UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"不支持的表达式: {type(node).__name__}")


def safe_calculate(expression: str) -> float:
    """计算仅含四则运算与数字字面量的表达式。"""
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("表达式不能为空")
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree)
    except SyntaxError as e:
        raise ValueError(f"表达式语法错误: {e}") from e
    if isinstance(result, complex) or result != result:  # 拒绝复数/NaN
        raise ValueError("表达式结果无效")
    return result


# 添加工具
@mcp.tool()
def calculate(expression: str) -> float:
    """计算四则运算表达式（仅支持数字与 + - * / % // ** 及括号）"""
    return safe_calculate(expression)


# 启动服务器（确保没有任何多余的 print 或空行干扰）
if __name__ == "__main__":
    mcp.run(transport='stdio')
