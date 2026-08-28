from mcp.server.fastmcp import FastMCP

# 创建MCP服务器
mcp = FastMCP("计算器演示")

# 添加工具
@mcp.tool()
def calculate(expression: str) -> float:
    """计算四则运算表达式"""
    # 注意：生产环境慎用 eval，这里仅作 Demo 演示
    return eval(expression)

# 启动服务器（确保没有任何多余的 print 或空行干扰）
if __name__ == "__main__":
    mcp.run(transport='stdio')