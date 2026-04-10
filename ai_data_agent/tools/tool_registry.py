"""
tools/tool_registry.py — 工具注册中心
统一管理所有工具，提供名称查找和 OpenAI schema 导出
"""
from __future__ import annotations

from typing import Any

from ai_data_agent.tools.base_tool import BaseTool
from ai_data_agent.observability.logger import get_logger

logger = get_logger(__name__)


class ToolRegistry:
    """工具注册中心（单例模式）。"""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> "ToolRegistry":
        """注册工具，支持链式调用。"""
        if tool.name in self._tools:
            logger.warning("tool_registry.overwrite", name=tool.name)
        self._tools[tool.name] = tool
        logger.info("tool_registry.registered", name=tool.name)
        return self

    def get(self, name: str) -> BaseTool:
        if name not in self._tools:
            available = list(self._tools.keys())
            raise KeyError(f"Tool '{name}' not found. Available: {available}")
        return self._tools[name]

    def list_tools(self) -> list[BaseTool]:
        return list(self._tools.values())

    def list_names(self) -> list[str]:
        return list(self._tools.keys())

    def to_openai_tools(self) -> list[dict[str, Any]]:
        """导出所有工具的 OpenAI function calling 格式。"""
        return [t.to_openai_function() for t in self._tools.values()]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)


# 全局注册表实例
_registry: ToolRegistry | None = None


def get_registry() -> ToolRegistry:
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry


def build_default_registry() -> ToolRegistry:
    """初始化并注册所有默认工具。"""
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
