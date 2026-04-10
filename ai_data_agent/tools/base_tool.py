"""
tools/base_tool.py — 工具基类
所有工具必须继承 BaseTool，统一接口
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ai_data_agent.observability.logger import get_logger
from ai_data_agent.observability.metrics import metrics

logger = get_logger(__name__)


@dataclass
class ToolInput:
    """工具输入基类，子类可扩展字段。"""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    success: bool
    data: Any = None                 # 返回给 Agent 的结构化数据
    text: str = ""                   # 文本摘要（供 LLM 消费）
    error: str = ""
    tool_name: str = ""
    latency_ms: float = 0.0

    def to_observation(self) -> str:
        """业界公认的将结果转换为 Agent 可读的 Observation 字符串。"""
        if not self.success:
            return f"[{self.tool_name}] ERROR: {self.error}"
        return f"[{self.tool_name}] {self.text}"


class BaseTool(ABC):
    """所有 Agent 工具的抽象基类。"""

    @property
    @abstractmethod
    def name(self) -> str:
        """工具唯一名称，用于注册和调用。"""

    @property
    @abstractmethod
    def description(self) -> str:
        """自然语言描述，供 LLM 选择工具时参考。"""

    @property
    def parameters_schema(self) -> dict[str, Any]:
        """OpenAI function calling 格式的参数 JSON Schema。"""
        return {"type": "object", "properties": {}, "required": []}

    @abstractmethod
    async def _run(self, **kwargs: Any) -> ToolResult:
        """实际执行逻辑，由子类实现。"""

    async def run(self, **kwargs: Any) -> ToolResult:
        """带监控的执行入口。"""
        # perf_counter比time.time()精准
        start = time.perf_counter()
        metrics.tool_calls_total.labels(tool_name=self.name).inc()
        logger.debug("tool.start", tool=self.name, kwargs=str(kwargs)[:200])
        try:
            result = await self._run(**kwargs)
            result.tool_name = self.name
            result.latency_ms = (time.perf_counter() - start) * 1000
            if not result.success:
                metrics.tool_errors_total.labels(tool_name=self.name).inc()
            metrics.tool_latency.labels(tool_name=self.name).observe(
                result.latency_ms / 1000
            )
            logger.debug(
                "tool.done",
                tool=self.name,
                success=result.success,
                latency_ms=round(result.latency_ms, 1),
            )
            return result
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            metrics.tool_errors_total.labels(tool_name=self.name).inc()
            logger.error("tool.exception", tool=self.name, error=str(exc))
            return ToolResult(
                success=False,
                error=str(exc),
                tool_name=self.name,
                latency_ms=elapsed,
            )

    def to_openai_function(self) -> dict[str, Any]:
        """生成 OpenAI tools 格式的函数描述。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }
