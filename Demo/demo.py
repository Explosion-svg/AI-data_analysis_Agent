from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters
import asyncio

# 配置Server参数，指定启动Server的命令
server_params = StdioServerParameters(
    command="python",  # 启动命令
    args=["./mcp_server.py"],  # 启动的文件路径
    env=None
)


async def main():
    # 建立与Server的连接
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write, sampling_callback=None) as session:
            # 初始化 Session（固定步骤）
            await session.initialize()

            print('\n正在调用计算器工具...')
            # 调用Server中的 calculate 工具，传入表达式参数
            result = await session.call_tool("calculate", {"expression": "188*23-34"})

            # 打印计算结果
            print(f"计算结果：{result.content}")


# 运行Client
asyncio.run(main())