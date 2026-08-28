"""
tools/tool_registry.py — 工具注册中心（Service Locator Pattern）

职责：
  统一管理所有已注册的 Agent 工具，提供：
  1. 工具注册（register）：链式调用风格，支持覆盖（配合 WARNING 日志）
  2. 工具查找（get / __contains__）：按名称查找工具实例
  3. OpenAI schema 导出（to_openai_tools）：导出所有工具的函数描述
  4. 懒加载单例（get_registry）：全局共享一个注册表实例

设计模式：
  Service Locator Pattern（服务定位器）：
  - 组件通过 get_registry().get("sql_query") 而非依赖注入获取工具
  - 简化了工具的获取方式，但降低了可测试性（单元测试需要注意注册状态）
  - 在 Executor 和 AgentLoop 中使用此模式，避免在构造函数中传递所有工具

链式注册（Fluent Interface）：
  register() 返回 self，支持链式调用：
    registry.register(SQLTool()).register(PythonTool()).register(ChartTool())
  这使 assembler.py 中的注册代码更简洁。

工具覆盖策略：
  - 覆盖同名工具时记录 WARNING（不抛异常）
  - 允许覆盖的用例：插件系统、A/B 测试（注册自定义工具替换默认工具）
  - WARNING 确保开发者知道覆盖发生了（防止意外覆盖）

与 assembler.py 的关系：
  - assembler.py 调用 build_default_registry() 注册所有默认工具
  - AgentLoop/Executor 通过 get_registry() 获取注册表
  - 单元测试可以清空并重新注册测试工具

全局注册表的时序：
  1. assembler.py 调用 build_default_registry() → 注册 5 个默认工具
  2. AgentLoop.run() 调用 get_registry() → 获取已注册的注册表
  3. Executor.execute() 调用 get_registry() → 查找步骤对应的工具
"""
from __future__ import annotations

from typing import Any

from ai_data_agent.tools.base_tool import BaseTool
from ai_data_agent.observability.logger import get_logger

logger = get_logger(__name__)


class ToolRegistry:
    """
    工具注册中心，维护名称到工具实例的映射。

    设计为可重复注册（覆盖），而不是抛异常，
    原因是在插件场景和测试场景中，覆盖是合法且常见的操作。

    内部使用 dict[str, BaseTool]：
    - key：工具名称（tool.name，全小写下划线）
    - value：BaseTool 子类实例
    - 查找时间复杂度 O(1)（哈希表）
    """

    def __init__(self) -> None:
        """初始化空的工具映射字典。"""
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> "ToolRegistry":
        """
        注册工具到注册表，返回 self 以支持链式调用。

        覆盖行为：
        - 同名工具已存在时，记录 WARNING 日志并覆盖
        - 不抛异常，因为覆盖可能是有意为之（如测试工具替换生产工具）

        Args:
            tool: BaseTool 子类实例（必须已实现 name、description、_run）

        Returns:
            self（ToolRegistry 实例），支持链式调用

        Example::
            registry.register(SQLTool()).register(PythonTool())
        """
        if tool.name in self._tools:
            logger.warning("tool_registry.overwrite", name=tool.name)
        self._tools[tool.name] = tool
        logger.info("tool_registry.registered", name=tool.name)
        return self

    def get(self, name: str) -> BaseTool:
        """
        按名称查找工具实例，不存在则抛出 KeyError。

        Args:
            name: 工具名称（如 "sql_query"、"python_analysis"）

        Returns:
            对应的 BaseTool 子类实例

        Raises:
            KeyError: 工具名称未注册，消息中包含所有可用工具名（方便调试）
        """
        if name not in self._tools:
            available = list(self._tools.keys())
            raise KeyError(f"Tool '{name}' not found. Available: {available}")
        return self._tools[name]

    def list_tools(self) -> list[BaseTool]:
        """
        返回所有已注册工具实例的列表。

        Returns:
            BaseTool 实例列表（顺序与注册顺序一致）
        """
        return list(self._tools.values())

    def list_names(self) -> list[str]:
        """
        返回所有已注册工具的名称列表。

        用途：
        - Planner.plan() 传入 available_tools，让 LLM 知道有哪些工具
        - 日志记录（启动时打印所有已注册工具）

        Returns:
            工具名称列表（如 ["sql_query", "python_analysis", ...]）
        """
        return list(self._tools.keys())

    def to_openai_tools(self) -> list[dict[str, Any]]:
        """
        导出所有工具的 OpenAI function calling 格式描述。

        这是 AgentLoop 中传给 router.generate(tools=...) 的数据。
        每个工具调用 to_openai_function() 生成标准格式：
        [
            {
                "type": "function",
                "function": {
                    "name": "sql_query",
                    "description": "...",
                    "parameters": {...}
                }
            },
            ...
        ]

        Returns:
            所有工具的 OpenAI function calling 格式列表
        """
        return [t.to_openai_function() for t in self._tools.values()]

    def __contains__(self, name: str) -> bool:
        """
        支持 `name in registry` 语法（in 运算符）。

        用于 Executor._execute_step() 检查工具是否存在：
            if step.tool not in registry:
                step.error = f"Tool '{step.tool}' not found."

        Args:
            name: 工具名称

        Returns:
            True 表示工具已注册
        """
        return name in self._tools

    def __len__(self) -> int:
        """
        返回已注册工具数量，支持 len(registry) 语法。

        Returns:
            已注册工具的数量
        """
        return len(self._tools)


# ── 全局注册表单例 ─────────────────────────────────────────────────────────────

_registry: ToolRegistry | None = None


def get_registry() -> ToolRegistry:
    """
    获取全局 ToolRegistry 单例（懒加载）。

    为什么使用全局单例：
    - AgentLoop 和 Executor 需要共享同一个注册表
    - 工具注册在应用启动时只发生一次（assembler.py 调用 build_default_registry）
    - 之后所有查找都针对同一实例

    注意：第一次调用时返回空注册表（未注册任何工具），
    必须先调用 build_default_registry() 或手动注册工具，
    否则 Executor 查找工具时会得到 KeyError。

    Returns:
        全局 ToolRegistry 实例（不存在则创建空实例）
    """
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry


def build_default_registry() -> ToolRegistry:
    """
    初始化并注册所有默认工具，返回配置好的注册表。

    注册的 5 个默认工具：
    1. SQLTool（"sql_query"）：SQL 查询执行
    2. PythonTool（"python_analysis"）：Python 代码沙盒执行
    3. ChartTool（"generate_chart"）：Plotly 图表生成
    4. SchemaTool（"get_schema"）：数据库 schema 查询
    5. RAGTool（"search_documents"）：知识库语义检索

    延迟导入（import 在函数内部）的原因：
    - 避免循环导入（工具文件可能间接依赖本模块）
    - 减少模块加载时间（工具可能依赖重型库如 pandas、plotly）
    - 保证在需要时才实例化工具（节省内存）

    被 assembler.py 在应用启动时调用一次，之后通过 get_registry() 访问。

    Returns:
        已注册所有默认工具的 ToolRegistry 实例
    """
    from ai_data_agent.tools.sql_tool import SQLTool
    from ai_data_agent.tools.python_tool import PythonTool
    from ai_data_agent.tools.chart_tool import ChartTool
    from ai_data_agent.tools.schema_tool import SchemaTool
    from ai_data_agent.tools.rag_tool import RAGTool

    registry = get_registry()
    registry.register(SQLTool())
    registry.register(PythonTool())
    registry.register(ChartTool())
    registry.register(SchemaTool())
    registry.register(RAGTool())
    logger.info("tool_registry.built", tools=registry.list_names())
    return registry
