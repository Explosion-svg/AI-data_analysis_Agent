"""
tools/base_tool.py — 工具抽象基类（Template Method Pattern）

职责：
  定义所有 Agent 工具必须遵循的统一接口，并通过模板方法模式（Template Method）
  在 run() 中提供通用的监控、限流、异常处理能力，让子类只需实现核心 _run() 逻辑。

架构位置（工具层 - 第 5 层）：
  AgentLoop → ToolRegistry → BaseTool.run() → BaseTool._run()（子类实现）

模板方法模式（Template Method Pattern）：
  run()  是"模板方法"，定义了执行骨架：
    1. 记录开始时间和 metrics
    2. 获取并发槽（ConcurrencyLimiter）
    3. 调用 _run()（子类实现的核心逻辑）
    4. 设置 tool_name 和 latency_ms
    5. 记录错误 metrics（如果失败）
    6. 更新延迟 Histogram
    7. 处理未捕获的异常（返回 ToolResult(success=False)）

  _run() 是"抽象步骤"，子类只需关注业务逻辑：
    - 不需要处理监控、限流、异常包装
    - 只需要返回 ToolResult 或让异常自然传播（run() 会捕获）

ToolResult 数据契约：
  - success: 工具是否执行成功（业务逻辑层面）
  - data: 结构化数据（供后续工具使用，如 sql_query → python_analysis）
  - text: 文本摘要（供 LLM 消费，作为 Observation）
  - error: 错误信息（success=False 时填充）
  - tool_name: 由 run() 自动设置（子类不需要填）
  - latency_ms: 由 run() 自动计算（子类不需要填）

to_openai_function() 导出格式：
  遵循 OpenAI function calling 规范，包含：
  - type: "function"
  - function.name: 工具唯一名称
  - function.description: 自然语言描述（LLM 用来决策是否调用此工具）
  - function.parameters: JSON Schema（LLM 用来生成参数）

并发控制：
  run() 内部通过 ConcurrencyLimiter.limit(self.name) 获取并发槽，
  意味着每个工具的并发量独立受限（舱壁隔离）。
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ai_data_agent.observability.logger import get_logger
from ai_data_agent.observability.metrics import metrics
from ai_data_agent.reliability.concurrency import get_limiter

logger = get_logger(__name__)


@dataclass
class ToolInput:
    """
    工具输入基类（预留扩展点）。

    目前只包含 raw 字段（原始参数字典），
    子类可扩展为类型化的输入（如 SQLInput(sql=..., max_rows=...)）。

    当前阶段工具直接使用 **kwargs，此类作为未来强类型化的预留接口。
    """
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """
    工具执行结果的统一数据结构。

    字段设计分层：
    - success + error：执行状态（是否成功，失败原因）
    - data：机器可读数据（供后续工具链消费，如 DataFrame 的 records 格式）
    - text：人类可读摘要（供 LLM 消费，作为 ReAct 循环的 Observation）
    - tool_name + latency_ms：可观测性元数据（由 run() 自动填充）

    data vs text 的区别：
    - data：原始结构化数据（list[dict]、dict、number 等），供下游工具处理
    - text：格式化的摘要字符串，如 "Query returned 5 rows: | col1 | col2 |..."
    - LLM 只看 text（token 有限），后续工具只看 data

    to_observation() 方法：
    - 将 ToolResult 转化为 ReAct 循环的标准 Observation 格式
    - 格式：[工具名] 结果内容 或 [工具名] ERROR: 错误信息
    - AgentLoop._execute_tool_call() 调用此方法构建观察消息
    """
    success: bool
    data: Any = None                 # 返回给 Agent 的结构化数据
    text: str = ""                   # 文本摘要（供 LLM 消费）
    error: str = ""
    tool_name: str = ""
    latency_ms: float = 0.0

    def to_observation(self) -> str:
        """
        将工具结果转换为 ReAct 循环的 Observation 字符串。

        这是 Agent Observation 的标准格式：
        - 成功：[工具名] 文本摘要
        - 失败：[工具名] ERROR: 错误信息

        为什么返回 text 而不是 data？
        - data 可能是大型 DataFrame，转成字符串会消耗大量 token
        - text 是预先格式化的摘要，包含关键信息（行数、列名、部分数据）
        - LLM 根据 text 决定下一步行动，不需要完整原始数据

        Returns:
            格式化的 Observation 字符串
        """
        if not self.success:
            return f"[{self.tool_name}] ERROR: {self.error}"
        return f"[{self.tool_name}] {self.text}"


class BaseTool(ABC):
    """
    所有 Agent 工具的抽象基类，提供统一接口和通用监控能力。

    子类实现要求：
    1. 实现 name 属性（工具唯一名称，与 ToolRegistry 的 key 对应）
    2. 实现 description 属性（自然语言描述，供 LLM 选择工具时参考）
    3. 可选覆盖 parameters_schema 属性（返回 JSON Schema，默认空对象）
    4. 实现 _run(**kwargs) 方法（核心业务逻辑）

    子类不需要：
    - 手动记录 metrics（run() 已处理）
    - 手动获取并发槽（run() 通过 ConcurrencyLimiter 处理）
    - 手动捕获所有异常（run() 的 try/except 已兜底）
    - 手动记录执行日志（run() 已记录 tool.start 和 tool.done）

    典型子类（5 个工具）：
    - SQLTool：执行 SQL 查询
    - PythonTool：执行 Python 代码（沙盒）
    - ChartTool：生成 Plotly 图表
    - SchemaTool：查询数据库 schema
    - RAGTool：检索知识库文档
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        工具的唯一名称（用于 ToolRegistry 注册和 LLM 调用时的函数名）。

        规范：
        - 全小写，下划线连接（如 "sql_query"、"python_analysis"）
        - 与 ConcurrencyLimiter 的桶名一致（用于并发控制）
        - 与 Executor._PARAM_GEN_SYSTEM 中的工具名一致（用于参数生成）

        Returns:
            工具唯一名称字符串
        """

    @property
    @abstractmethod
    def description(self) -> str:
        """
        工具的自然语言描述（LLM 根据此描述决策是否调用此工具）。

        好的描述应包含：
        - 工具能做什么（能力）
        - 什么时候用（触发条件）
        - 输入/输出的关键信息

        避免：
        - 过于简短（LLM 无法判断何时使用）
        - 过于详细（消耗 token，LLM 忽略细节）

        Returns:
            描述字符串（通常 1-3 句话）
        """

    @property
    def parameters_schema(self) -> dict[str, Any]:
        """
        工具参数的 JSON Schema（用于 OpenAI function calling）。

        默认返回空对象 schema（表示工具不需要参数或参数可选）。
        子类应覆盖此属性，提供精确的参数约束：
        - properties：各参数名称、类型、描述
        - required：必填参数列表

        JSON Schema 的重要性：
        - LLM 根据 schema 生成参数（错误的 schema → 错误的参数格式）
        - Executor._generate_params() 将 schema 注入参数生成 prompt

        Returns:
            符合 JSON Schema 规范的 dict
        """
        return {"type": "object", "properties": {}, "required": []}

    @abstractmethod
    async def _run(self, **kwargs: Any) -> ToolResult:
        """
        工具的核心执行逻辑（抽象方法，由子类实现）。

        子类实现规范：
        - 参数通过 **kwargs 接收（与 parameters_schema 中的 properties 对应）
        - 成功时返回 ToolResult(success=True, data=..., text=...)
        - 业务层失败时返回 ToolResult(success=False, error=...)
        - 让异常自然传播（run() 的 try/except 会兜底，转换为 ToolResult(success=False)）

        注意：不需要在 _run() 内设置 tool_name 和 latency_ms，
        这两个字段由 run() 模板方法自动填充。

        Args:
            **kwargs: 工具参数（由 LLM 通过 Executor._generate_params() 生成）

        Returns:
            ToolResult 对象
        """

    async def run(self, **kwargs: Any) -> ToolResult:
        """
        工具的公共执行入口（模板方法，提供统一的监控和错误处理）。

        执行流程：
        1. 记录开始时间（time.perf_counter，比 time.time 精度更高）
        2. 递增 tool_calls_total 指标（Prometheus Counter）
        3. 记录 DEBUG 日志（tool.start，包含参数预览）
        4. 获取并发槽（get_limiter().limit(self.name)，舱壁隔离）
        5. 调用 _run(**kwargs)（子类核心逻辑）
        6. 设置 tool_name（自动注入工具名）
        7. 计算并设置 latency_ms（执行时间）
        8. 更新失败计数（tool_errors_total）和延迟直方图（tool_latency）
        9. 记录 DEBUG 日志（tool.done，包含成功状态和耗时）
        10. 异常兜底：捕获所有未处理异常，返回 ToolResult(success=False)

        注意：
        - tool_name 由 run() 自动设置，_run() 不需要填
        - latency_ms 包含了等待并发槽的时间（公平地反映真实耗时）
        - 异常兜底 catch 不会吞掉异常信息（转化为 error 字段保留）

        Args:
            **kwargs: 工具参数（透传给 _run()）

        Returns:
            ToolResult 对象（永远不会抛出异常，异常已转化为 success=False）
        """
        # perf_counter 比 time.time() 精准（单调时钟，不受系统时间调整影响）
        start = time.perf_counter()
        metrics.tool_calls_total.labels(tool_name=self.name).inc()
        logger.debug("tool.start", tool=self.name, kwargs=str(kwargs)[:200])
        try:
            # 获取并发槽（舱壁隔离，防止单工具拖垮系统）
            async with get_limiter().limit(self.name):
                result = await self._run(**kwargs)
            result.tool_name = self.name
            result.latency_ms = (time.perf_counter() - start) * 1000
            if not result.success:
                metrics.tool_errors_total.labels(tool_name=self.name).inc()
            # 延迟直方图：buckets 按秒单位，latency_ms 需除以 1000
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
            # 兜底：任何未捕获的异常都转为 success=False（防止 AgentLoop 崩溃）
            return ToolResult(
                success=False,
                error=str(exc),
                tool_name=self.name,
                latency_ms=elapsed,
            )

    def to_openai_function(self) -> dict[str, Any]:
        """
        生成 OpenAI function calling 格式的工具描述。

        输出格式遵循 OpenAI Chat Completions API 规范：
        {
            "type": "function",
            "function": {
                "name": "sql_query",
                "description": "Execute a SELECT SQL query...",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sql": {"type": "string", ...},
                        "max_rows": {"type": "integer", ...}
                    },
                    "required": ["sql"]
                }
            }
        }

        此格式被 ToolRegistry.to_openai_tools() 收集，
        传给 AgentLoop 中的 router.generate(tools=[...])，
        让 LLM 能够选择并调用工具。

        Returns:
            符合 OpenAI function calling 格式的字典
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }
