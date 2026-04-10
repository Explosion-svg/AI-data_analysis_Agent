"""
orchestration/agent_loop.py — Agent 核心循环
实现 ReAct (Reasoning + Acting) 模式
使用 OpenAI function calling 进行工具选择
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from ai_data_agent.config.config import settings
from ai_data_agent.context.prompt_builder import PromptBuilder
from ai_data_agent.context.query_rewriter import QueryRewriter
from ai_data_agent.context.schema_context import SchemaContextBuilder
from ai_data_agent.memory.conversation_memory import get_memory
from ai_data_agent.memory.cache_memory import get_cache
from ai_data_agent.model_gateway.base_model import Message
from ai_data_agent.model_gateway.router import get_router, TaskType
from ai_data_agent.observability.logger import get_logger
from ai_data_agent.observability.metrics import metrics
from ai_data_agent.observability.tracer import span
from ai_data_agent.tools.tool_registry import get_registry
from ai_data_agent.reliability.circuit_breaker import get_breaker

logger = get_logger(__name__)


@dataclass
class AgentResponse:
    answer: str
    conversation_id: str
    iterations: int = 0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    charts: list[dict[str, Any]] = field(default_factory=list)
    data: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0
    error: str = ""
    success: bool = True


class AgentLoop:
    """
    ReAct Agent 主循环。
    每轮：Think → Act (call tool) → Observe → 继续或结束
    """

    def __init__(self) -> None:
        self._prompt_builder = PromptBuilder()
        self._query_rewriter = QueryRewriter()
        self._schema_builder = SchemaContextBuilder()

    async def run(
        self,
        query: str,
        conversation_id: str,
        use_cache: bool = True,
    ) -> AgentResponse:
        start = time.perf_counter()
        metrics.agent_requests_total.inc()

        with span("agent_loop.run", {"conversation_id": conversation_id}):
            # 1. 缓存检查
            if use_cache:
                cache = get_cache()
                cache_key = cache.make_key("agent", query, conversation_id)
                cached = cache.get(cache_key)
                if cached:
                    logger.info("agent_loop.cache_hit", conversation_id=conversation_id)
                    return cached

            try:
                response = await self._react_loop(query, conversation_id)
            except Exception as e:
                logger.error("agent_loop.failed", error=str(e), conversation_id=conversation_id)
                metrics.agent_errors_total.labels(error_type=type(e).__name__).inc()
                response = AgentResponse(
                    answer=f"I encountered an error: {e}",
                    conversation_id=conversation_id,
                    success=False,
                    error=str(e),
                )

        elapsed_ms = (time.perf_counter() - start) * 1000
        response.latency_ms = elapsed_ms
        metrics.agent_latency.observe(elapsed_ms / 1000)
        metrics.agent_iterations.observe(response.iterations)

        # 存入缓存（只缓存成功结果）
        if use_cache and response.success:
            cache.set(cache_key, response)

        return response

    async def _react_loop(
        self,
        query: str,
        conversation_id: str,
    ) -> AgentResponse:
        """核心 ReAct 循环。"""
        memory = get_memory()
        registry = get_registry()
        router = get_router()
        breaker = get_breaker("llm")

        # ── 准备上下文 ────────────────────────────────────────────────────────
        # Query rewrite
        rewrite_result = await self._query_rewriter.rewrite(query)
        logger.debug("agent_loop.rewrite", result=rewrite_result)

        # Schema context
        schema_ctx = await self._schema_builder.build(query)

        # RAG（多路查询提升召回）
        rag_docs: list[dict[str, Any]] = []
        if registry.list_names():
            try:
                rag_tool = registry.get("search_documents")
                rag_result = await rag_tool.run(query=rewrite_result["rewritten"])
                if rag_result.success and rag_result.data:
                    rag_docs = rag_result.data
            except Exception as e:
                logger.debug("agent_loop.rag_skip", error=str(e))

        # 历史对话
        history = memory.get_messages(conversation_id)

        # 构建初始 messages
        messages = self._prompt_builder.build(
            query=query,
            rag_docs=rag_docs,
            schema_context=schema_ctx,
            history=history,
        )

        # ── ReAct 循环 ────────────────────────────────────────────────────────
        tool_calls_log: list[dict[str, Any]] = []
        charts: list[dict[str, Any]] = []
        latest_data: list[dict[str, Any]] = []
        iteration = 0
        tools_schema = registry.to_openai_tools()

        while iteration < settings.agent_max_iterations:
            iteration += 1
            logger.debug("agent_loop.iteration", n=iteration, conversation_id=conversation_id)

            # LLM 调用（带熔断器）
            resp = await breaker.call(
                router.generate,
                messages=messages,
                task_type=TaskType.COMPLEX,
                tools=tools_schema,
                tool_choice="auto",
            )

            # 无工具调用 → 最终答案
            if not resp.tool_calls:
                final_answer = resp.content
                logger.info(
                    "agent_loop.final_answer",
                    conversation_id=conversation_id,
                    iterations=iteration,
                )
                # 保存对话历史
                memory.add(conversation_id, "user", query)
                memory.add(conversation_id, "assistant", final_answer)

                return AgentResponse(
                    answer=final_answer,
                    conversation_id=conversation_id,
                    iterations=iteration,
                    tool_calls=tool_calls_log,
                    charts=charts,
                    data=latest_data,
                    success=True,
                )

            # 有工具调用 → 执行工具
            # 先把 assistant 消息（含 tool_calls）加入 messages
            messages.append(
                Message(
                    role="assistant",
                    content=resp.content or "",
                    tool_calls=resp.tool_calls,
                )
            )

            for tc in resp.tool_calls:
                tool_name = tc["function"]["name"]
                tool_args_str = tc["function"]["arguments"]
                tool_call_id = tc["id"]

                # 解析参数
                try:
                    tool_args = json.loads(tool_args_str)
                except json.JSONDecodeError:
                    tool_args = {}

                logger.info(
                    "agent_loop.tool_call",
                    tool=tool_name,
                    args=str(tool_args)[:200],
                    iteration=iteration,
                )

                # 执行工具
                try:
                    tool = registry.get(tool_name)
                    tool_result = await tool.run(**tool_args)
                except KeyError:
                    observation = f"Error: Tool '{tool_name}' not found."
                    tool_result = None
                except Exception as e:
                    observation = f"Error executing tool '{tool_name}': {e}"
                    tool_result = None
                else:
                    observation = tool_result.to_observation()
                    tool_calls_log.append(
                        {"tool": tool_name, "args": tool_args, "success": tool_result.success}
                    )
                    # 收集图表和数据
                    if tool_name == "generate_chart" and tool_result.success:
                        charts.append(tool_result.data)
                    if tool_name == "sql_query" and tool_result.success:
                        latest_data = tool_result.data or []

                # 把 tool 结果加入 messages
                messages.append(
                    Message(
                        role="tool",
                        content=observation,
                        tool_call_id=tool_call_id,
                        name=tool_name,
                    )
                )

        # 超过最大迭代次数
        logger.warning(
            "agent_loop.max_iterations",
            conversation_id=conversation_id,
            iterations=iteration,
        )
        # 最后一次 LLM 调用，要求给出最终答案
        messages.append(
            Message(
                role="user",
                content="Please summarize what you've found so far and provide your best answer.",
            )
        )
        final_resp = await breaker.call(
            router.generate,
            messages=messages,
            task_type=TaskType.COMPLEX,
        )
        final_answer = final_resp.content

        memory.add(conversation_id, "user", query)
        memory.add(conversation_id, "assistant", final_answer)

        return AgentResponse(
            answer=final_answer,
            conversation_id=conversation_id,
            iterations=iteration,
            tool_calls=tool_calls_log,
            charts=charts,
            data=latest_data,
            success=True,
        )
