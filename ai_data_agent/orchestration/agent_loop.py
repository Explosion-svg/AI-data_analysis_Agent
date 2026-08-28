"""
orchestration/agent_loop.py — Agent 核心编排循环（ReAct 模式）

职责：
  实现 ReAct（Reasoning + Acting）模式的 Agent 主循环，
  是整个系统最核心的编排组件，连接所有其他子系统。

ReAct 模式原理（每轮循环）：
  Think → 调用 LLM，让模型推理"下一步应该做什么"
  Act   → 如果模型请求工具调用，执行对应工具
  Observe → 把工具执行结果作为 observation 追加到消息历史
  重复上述循环，直到模型不再请求工具（给出最终答案）

两种执行路径：
  1. Plan-and-Execute（如果启用了 Planner 且问题足够复杂）：
     Planner 生成多步计划 → Executor 并行执行各步骤 → 合成最终答案
  2. ReAct 循环（默认路径，适合大多数场景）：
     直接进入 Think-Act-Observe 循环，模型自主决定工具调用顺序

职责边界（AgentLoop 只做编排，不做具体逻辑）：
  ✓ 控制循环的推进条件（何时继续、何时结束）
  ✓ 协调各组件之间的数据流（记忆、缓存、工具、LLM）
  ✗ 不解析 schema（交给 SchemaContextBuilder）
  ✗ 不压缩工具结果（交给 WorkMemorySummarizer）
  ✗ 不做 SQL 安全校验（交给 SQLGuard + SQLTool）
  ✗ 不管理 LLM 重试（交给 ModelRouter + async_retry）

缓存策略：
  相同的 (tenant_id, query, conversation_id) 组合命中缓存时直接返回，
  跳过完整的 ReAct 循环（节省 LLM 调用和数据库查询开销）。
  只缓存成功结果（success=True 的 AgentResponse）。

上下文传播：
  通过 RequestContext 传播 request_id/user_id/tenant_id，
  确保所有日志、审计和指标都可以按请求追踪。
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from ai_data_agent.config.config import settings
from ai_data_agent.context.request_context import RequestContext, get_request_context
from ai_data_agent.memory.work_memory_summarizer import WorkMemorySummarizer
from ai_data_agent.model_gateway.base_model import Message
from ai_data_agent.model_gateway.router import TaskType
from ai_data_agent.observability.logger import get_logger
from ai_data_agent.observability.metrics import metrics
from ai_data_agent.reliability.concurrency import ConcurrencyLimitExceeded, get_limiter
from ai_data_agent.observability.tracer import span

if TYPE_CHECKING:
    from ai_data_agent.context.prompt_builder import PromptBuilder
    from ai_data_agent.context.query_rewriter import QueryRewriter
    from ai_data_agent.context.schema_context import SchemaContextBuilder
    from ai_data_agent.memory.conversation_memory import ConversationMemory
    from ai_data_agent.memory.cache_memory import CacheMemory
    from ai_data_agent.memory.work_memory import WorkMemory
    from ai_data_agent.model_gateway.router import ModelRouter
    from ai_data_agent.orchestration.executor import Executor
    from ai_data_agent.orchestration.planner import Planner, PlanStep
    from ai_data_agent.tools.tool_registry import ToolRegistry
    from ai_data_agent.reliability.circuit_breaker import CircuitBreaker

logger = get_logger(__name__)


@dataclass
class AgentResponse:
    """
    Agent 执行一次请求的完整结果。

    字段说明：
    - answer: Agent 的最终自然语言回答（始终存在，即使失败也有错误描述）
    - conversation_id: 用户传入的会话 ID（未加作用域前缀的原始 ID）
    - iterations: ReAct 循环迭代次数（1 = 第一次就直接回答，无工具调用）
    - tool_calls: 本次请求中所有工具调用的摘要列表（用于前端展示和审计）
    - charts: 图表 JSON 列表（来自 generate_chart 工具）
    - data: 最近 SQL 查询的原始结果（list[dict]，供前端表格展示）
    - latency_ms: 端到端延迟（从 run() 开始到返回，包含所有子操作）
    - error: 错误信息（success=False 时有内容）
    - success: 是否成功（False 时 answer 包含错误描述）

    注意：latency_ms 在 run() 函数最后才赋值，
    AgentResponse 构造时不需要传入（默认 0.0）。
    """
    answer: str                                    # 最终回答文本
    conversation_id: str                           # 原始会话 ID（未加作用域）
    iterations: int = 0                            # ReAct 循环次数
    tool_calls: list[dict[str, Any]] = field(default_factory=list)  # 工具调用记录
    charts: list[dict[str, Any]] = field(default_factory=list)      # 图表 JSON 列表
    data: list[dict[str, Any]] = field(default_factory=list)        # SQL 结果数据
    latency_ms: float = 0.0                        # 端到端延迟（毫秒）
    error: str = ""                                # 错误信息（成功时为空）
    success: bool = True                           # 是否成功


class AgentLoop:
    """
    ReAct Agent 主循环（Plan-and-Execute 可选路径）。

    架构定位：
    - AgentLoop 是整个应用的编排核心
    - 它消费所有其他子系统（记忆、缓存、工具、LLM、可靠性组件）
    - 但它不创建这些依赖，全部通过构造函数注入（Composition Root 模式）
    - 实际组装在 assembler.py 的 _init_orchestration() 中完成

    依赖注入的好处：
    - 单元测试时可以注入 MockLLM、MockToolRegistry 等，不需要真实 API
    - 组件替换（如切换 LLM provider）不需要修改 AgentLoop
    - 依赖关系显式声明，代码更容易理解和维护

    线程安全：
    - AgentLoop 是无状态的（所有状态存储在注入的组件中）
    - 可以在多个并发请求之间共享同一个实例
    - 并发安全性由 ConcurrencyLimiter 和 asyncio.Lock 保证
    """

    def __init__(
        self,
        *,
        prompt_builder: "PromptBuilder",
        query_rewriter: "QueryRewriter",
        schema_builder: "SchemaContextBuilder",
        memory: "ConversationMemory",
        cache: "CacheMemory",
        work_memory: "WorkMemory",
        registry: "ToolRegistry",
        router: "ModelRouter",
        breaker: "CircuitBreaker",
        planner: "Planner | None" = None,
        executor: "Executor | None" = None,
    ) -> None:
        """
        通过依赖注入初始化 AgentLoop。

        所有依赖均为必填（除了可选的 planner 和 executor）。
        planner/executor 为 None 时，启用 Plan-and-Execute 路径的配置会被忽略。

        Args:
            prompt_builder: 构建 LLM 输入消息的组件
            query_rewriter: LLM 查询改写器
            schema_builder: 数据库 schema 语义检索器
            memory: 对话历史管理器（三层记忆）
            cache: 结果缓存（LRU + TTL）
            work_memory: 单次任务状态记录器
            registry: 工具注册中心（所有可用工具）
            router: 模型路由器（选择 LLM 和 Fallback）
            breaker: 熔断器（保护 LLM 调用）
            planner: 任务规划器（可选，复杂任务的高层规划）
            executor: 计划执行器（可选，与 planner 配合）
        """
        self._prompt_builder = prompt_builder
        self._query_rewriter = query_rewriter
        self._schema_builder = schema_builder
        self._memory = memory
        self._cache = cache
        self._work_memory = work_memory
        self._registry = registry
        self._router = router
        self._breaker = breaker
        self._planner = planner
        self._executor = executor

    async def run(
        self,
        query: str,
        conversation_id: str,
        request_context: RequestContext | None = None,
        use_cache: bool = True,
    ) -> AgentResponse:
        """
        执行一次完整的 Agent 请求（对外公开的主入口）。

        外层职责（与具体 ReAct 逻辑无关）：
        1. 解析请求上下文（RequestContext）
        2. 检查缓存，命中则直接返回（跳过 LLM 和工具调用）
        3. 通过并发限制器控制同时进行的 Agent 请求数
        4. 创建 OpenTelemetry span（分布式追踪）
        5. 调用 _react_loop 执行实际逻辑
        6. 捕获异常，返回 success=False 的 AgentResponse（不向调用方抛异常）
        7. 记录 Prometheus 指标（请求数、延迟、迭代次数）
        8. 将成功结果写入缓存

        RequestContext 优先级：
        - 显式传入（HTTP 层通过 Depends 注入）> contextvars 传播 > 默认系统上下文
        - 这样兼顾了 HTTP 请求场景和内部调用场景

        Args:
            query: 用户的自然语言查询
            conversation_id: 用户提供的会话 ID（未加租户作用域）
            request_context: 请求上下文（可选，包含 request_id/user_id/tenant_id）
            use_cache: 是否使用结果缓存（默认 True，eval 时设为 False）

        Returns:
            AgentResponse（无论成功还是失败都返回，不抛出异常）
        """
        start = time.perf_counter()
        metrics.agent_requests_total.inc()

        # 解析请求上下文
        req_ctx = request_context or get_request_context() or RequestContext(
            request_id="system",
            user_id=settings.default_user_id,
            tenant_id=settings.default_tenant_id,
        )
        # 加租户前缀，防止不同租户的 conversation_id 冲突
        scoped_conversation_id = req_ctx.scoped_conversation_id(conversation_id)

        async with get_limiter().limit("agent_request"):
            with span(
                "agent_loop.run",
                {
                    "request_id": req_ctx.request_id,
                    "user_id": req_ctx.user_id,
                    "tenant_id": req_ctx.tenant_id,
                    "conversation_id": conversation_id,
                },
            ):
                # 缓存检查（只在成功路径缓存，失败不缓存）
                cache_key = self._cache.make_key("agent", req_ctx.tenant_id, query, conversation_id)
                if use_cache:
                    # P2-15：Redis 缓存同步读包 to_thread，避免阻塞事件循环
                    cached = await asyncio.to_thread(self._cache.get, cache_key)
                    if cached:
                        logger.info(
                            "agent_loop.cache_hit",
                            request_id=req_ctx.request_id,
                            user_id=req_ctx.user_id,
                            tenant_id=req_ctx.tenant_id,
                            conversation_id=conversation_id,
                        )
                        return cached

                try:
                    response = await self._react_loop(
                        query=query,
                        conversation_id=conversation_id,
                        scoped_conversation_id=scoped_conversation_id,
                        request_context=req_ctx,
                    )
                except ConcurrencyLimitExceeded:
                    # P2-10：并发过载不降级为 success=False 响应，
                    # 原样上抛，交由 main.py 的 503 处理器返回 503 Service Unavailable。
                    raise
                except Exception as e:
                    # 失败时更新工作记忆状态（为了快照显示正确的失败状态）
                    await asyncio.to_thread(self._work_memory.fail_run, scoped_conversation_id, str(e))
                    logger.error(
                        "agent_loop.failed",
                        request_id=req_ctx.request_id,
                        user_id=req_ctx.user_id,
                        tenant_id=req_ctx.tenant_id,
                        error=str(e),
                        conversation_id=conversation_id,
                    )
                    metrics.agent_errors_total.labels(error_type=type(e).__name__).inc()
                    # 不向调用方抛异常，返回 success=False 的响应
                    response = AgentResponse(
                        answer=f"I encountered an error: {e}",
                        conversation_id=conversation_id,
                        success=False,
                        error=str(e),
                    )

        # 记录端到端延迟（在 limit 上下文外，包含等待并发槽位的时间）
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.latency_ms = elapsed_ms
        metrics.agent_latency.observe(elapsed_ms / 1000)
        metrics.agent_iterations.observe(response.iterations)

        # 只缓存成功结果
        if use_cache and response.success:
            await asyncio.to_thread(self._cache.set, cache_key, response)

        return response

    async def _react_loop(
        self,
        query: str,
        conversation_id: str,
        scoped_conversation_id: str,
        request_context: RequestContext,
    ) -> AgentResponse:
        """
        核心 ReAct 循环（内部方法，负责实际的 Think-Act-Observe 编排）。

        执行流程（两阶段）：
        阶段一：准备上下文
          1. 初始化工作记忆（start_run）
          2. 改写查询（query rewriting）
          3. 生成 schema 上下文
          4. 可选 RAG 检索
          5. 合并历史消息（会话记忆）
          6. 构建初始消息列表

        阶段二：尝试 Plan-and-Execute（如果启用且问题复杂）
          - Planner 生成计划，Executor 并行执行
          - 如果计划为空或问题简单，退回 ReAct 循环

        阶段三：ReAct 循环（如果未走 Plan-and-Execute）
          Think（LLM 推理）→ Act（工具调用）→ Observe（追加结果）→ 重复
          直到：
          - 模型不再请求工具（正常结束）
          - 达到 max_iterations 限制（强制结束）

        两阶段放在同一方法内的原因：
        - 两段逻辑共享大量运行态对象（messages、schema_ctx 等）
        - 拆分到两个方法会导致需要在方法间传递过多变量
        - 通过私有方法（_build_initial_messages、_try_plan_and_execute 等）
          提取子逻辑，让这里的流程骨架保持清晰

        Args:
            query: 原始用户查询
            conversation_id: 原始会话 ID（未加租户作用域）
            scoped_conversation_id: 加租户前缀的会话 ID（用于内部组件访问）
            request_context: 请求上下文（日志用）

        Returns:
            AgentResponse（成功时的完整响应）
        """
        # 为本次请求建立新的工作状态（覆盖同 conversation 的旧状态）
        await asyncio.to_thread(self._work_memory.start_run, scoped_conversation_id, query)

        # 阶段一：准备初始消息列表和 schema 上下文
        messages, schema_ctx = await self._build_initial_messages(
            query=query,
            scoped_conversation_id=scoped_conversation_id,
        )

        # 阶段二：尝试 Plan-and-Execute（问题复杂时提前返回）
        planned_response = await self._try_plan_and_execute(
            query=query,
            conversation_id=conversation_id,
            scoped_conversation_id=scoped_conversation_id,
            schema_context=schema_ctx,
        )
        if planned_response is not None:
            return planned_response

        # ── 阶段三：ReAct 循环 ─────────────────────────────────────────────────
        tool_calls_log: list[dict[str, Any]] = []    # 工具调用审计日志
        charts: list[dict[str, Any]] = []            # 图表产物收集
        latest_data: list[dict[str, Any]] = []       # 最近 SQL 结果数据
        iteration = 0
        tools_schema = self._registry.to_openai_tools()  # OpenAI function calling 格式

        while iteration < settings.agent_max_iterations:
            iteration += 1
            await asyncio.to_thread(self._work_memory.set_iterations, scoped_conversation_id, iteration)
            logger.debug(
                "agent_loop.iteration",
                request_id=request_context.request_id,
                user_id=request_context.user_id,
                tenant_id=request_context.tenant_id,
                n=iteration,
                conversation_id=conversation_id,
            )

            # Think：调用 LLM 推理下一步行动（通过熔断器保护）
            resp = await self._breaker.call(
                self._router.generate,
                messages=messages,
                task_type=TaskType.COMPLEX,
                tools=tools_schema,
                tool_choice="auto",  # "auto" = 让模型自己决定是否调用工具
            )

            # 模型没有请求工具 → 给出了最终答案，正常结束循环
            if not resp.tool_calls:
                return await self._build_final_response(
                    query=query,
                    conversation_id=conversation_id,
                    scoped_conversation_id=scoped_conversation_id,
                    final_answer=resp.content,
                    iteration=iteration,
                    tool_calls_log=tool_calls_log,
                    charts=charts,
                    latest_data=latest_data,
                )

            # 模型请求了工具调用 → Act 阶段
            # 先把包含 tool_calls 的 assistant 消息追加到历史（OpenAI API 要求）
            messages.append(
                Message(
                    role="assistant",
                    content=resp.content or "",
                    tool_calls=resp.tool_calls,
                )
            )

            # Observe 阶段：执行每个请求的工具调用
            for tc in resp.tool_calls:
                tool_result = await self._execute_tool_call(
                    tool_call=tc,
                    conversation_id=conversation_id,
                    scoped_conversation_id=scoped_conversation_id,
                    iteration=iteration,
                    messages=messages,
                    tool_calls_log=tool_calls_log,
                    charts=charts,
                )
                # 更新最近 SQL 结果数据（只保留最新的一份）
                if (
                    tool_result is not None
                    and tool_result.success
                    and tc["function"]["name"] == "sql_query"
                ):
                    latest_data = tool_result.data or []

        # 达到最大迭代次数限制 → 强制收尾（兜底路径）
        return await self._build_forced_final_response(
            query=query,
            conversation_id=conversation_id,
            scoped_conversation_id=scoped_conversation_id,
            iteration=iteration,
            messages=messages,
            tool_calls_log=tool_calls_log,
            charts=charts,
            latest_data=latest_data,
        )

    async def _build_initial_messages(
        self,
        *,
        query: str,
        scoped_conversation_id: str,
    ) -> tuple[list[Message], str]:
        """
        准备 ReAct 循环开始前所需的初始消息列表和 schema 上下文。

        这个方法执行所有前置准备工作（都不属于 ReAct 循环本身）：
        1. Query Rewriting：让 LLM 改写查询，提高 schema 检索和 RAG 的召回率
        2. Schema Context：语义检索相关表的结构信息，为 LLM 提供"数据地图"
        3. RAG 检索：可选的知识库检索，提供业务定义和分析方法
        4. 历史消息：从 ConversationMemory 获取对话历史（用于多轮对话）
        5. Prompt 构建：通过 PromptBuilder 把以上所有信息组装成 messages 列表

        抽取为独立方法的原因：
        - _react_loop() 的代码长度和逻辑复杂度下降
        - 准备阶段逻辑集中在一处，方便调试和修改
        - 返回 (messages, schema_ctx) 两个值，让 _react_loop() 可以分别使用

        Args:
            query: 原始用户查询
            scoped_conversation_id: 加租户前缀的会话 ID

        Returns:
            (初始消息列表, schema 上下文文本) 元组
        """
        # Step 1: Query Rewriting（LLM 改写，提高 schema/RAG 召回率）
        rewrite_result = await self._query_rewriter.rewrite(query)
        logger.debug("agent_loop.rewrite", result=rewrite_result)
        await asyncio.to_thread(
            self._work_memory.set_rewritten_query,
            scoped_conversation_id,
            rewrite_result.get("rewritten", ""),
        )
        if rewrite_result.get("reason"):
            await asyncio.to_thread(
                self._work_memory.add_finding,
                scoped_conversation_id,
                f"Query rewritten rationale: {rewrite_result['reason']}",
            )

        # Step 2: Schema Context（语义检索相关表结构）
        schema_ctx = await self._schema_builder.build(query)
        await asyncio.to_thread(
            self._work_memory.set_schema_context,
            scoped_conversation_id,
            schema_ctx,
            selected_tables=self._schema_builder.extract_table_names(schema_ctx),
        )

        # Step 3: RAG 检索（可选，失败不影响主流程）
        rag_docs = await self._retrieve_rag_docs(
            query=rewrite_result.get("rewritten", query),
            scoped_conversation_id=scoped_conversation_id,
        )

        # Step 4: 获取历史消息（三层记忆 → Messages）
        history = await asyncio.to_thread(self._memory.get_messages, scoped_conversation_id)

        # Step 5: 组装完整消息列表
        return self._prompt_builder.build(
            query=query,
            rag_docs=rag_docs,
            schema_context=schema_ctx,
            history=history,
            work_context=self._work_memory.build_prompt_context(scoped_conversation_id),
        ), schema_ctx

    async def _try_plan_and_execute(
        self,
        *,
        query: str,
        conversation_id: str,
        scoped_conversation_id: str,
        schema_context: str,
    ) -> "AgentResponse | None":
        """
        尝试 Plan-and-Execute 路径（如果启用且适用）。

        触发条件（三个条件必须同时满足）：
        1. settings.agent_enable_planning = True（配置开关）
        2. planner 和 executor 均已注入（组件完整）
        3. Planner 评估问题为 moderate/complex 级别（不是 simple）

        不走 Plan-and-Execute 的场景（返回 None）：
        - 功能未启用
        - 组件未注入
        - Planner 认为问题简单（is_simple=True）或计划为空（is_empty=True）
        - Planner 调用失败（降级为 Plan.complexity="simple"）

        Args:
            query: 原始用户查询
            conversation_id: 原始会话 ID
            scoped_conversation_id: 加租户前缀的会话 ID
            schema_context: 已构建的 schema 上下文文本

        Returns:
            AgentResponse（走了 Plan-and-Execute 时）或 None（退回 ReAct 时）
        """
        # 配置开关或组件未就绪时跳过
        if not settings.agent_enable_planning:
            return None
        if self._planner is None or self._executor is None:
            return None

        # 让 Planner 评估问题复杂度并生成计划
        plan = await self._planner.plan(
            query=query,
            available_tools=self._registry.list_names(),
            schema_context=schema_context,
        )

        # 计划为空或问题简单时，退回到 ReAct 循环
        if plan.is_empty or plan.is_simple:
            return None

        # 记录 Plan-and-Execute 决策（方便调试）
        await asyncio.to_thread(
            self._work_memory.add_finding,
            scoped_conversation_id,
            f"Planner selected {plan.complexity} plan with {len(plan.steps)} steps.",
        )
        await asyncio.to_thread(self._work_memory.set_iterations, scoped_conversation_id, len(plan.steps))

        # 让 Executor 按拓扑顺序执行所有步骤
        steps = await self._executor.execute(plan, schema_context=schema_context)
        return await self._build_planned_response(
            query=query,
            conversation_id=conversation_id,
            scoped_conversation_id=scoped_conversation_id,
            steps=steps,
        )

    async def _retrieve_rag_docs(
        self,
        *,
        query: str,
        scoped_conversation_id: str,
    ) -> list[dict[str, Any]]:
        """
        执行一次可选的 RAG 知识库检索。

        为什么独立成方法？
        - RAG 是"可选增强"，不是核心流程
        - 失败时不应中断主流程（内部吞掉异常）
        - 隔离 "非核心、可失败" 的逻辑是好的设计实践

        如果注册表中没有 search_documents 工具（未注册 RAGTool），
        会在 registry.get() 时抛出 KeyError，被 except Exception 捕获，
        记录 finding 后返回空列表。

        Args:
            query: 改写后的查询（改写版本通常比原始查询检索效果更好）
            scoped_conversation_id: 加租户前缀的会话 ID（用于 finding 写入）

        Returns:
            检索到的文档列表（list[dict]），失败时返回空列表
        """
        if not self._registry.list_names():
            return []

        try:
            rag_tool = self._registry.get("search_documents")
            rag_result = await rag_tool.run(query=query)
            if rag_result.success and rag_result.data:
                await asyncio.to_thread(
                    self._work_memory.add_finding,
                    scoped_conversation_id,
                    f"Retrieved {len(rag_result.data)} relevant knowledge document(s).",
                )
                return rag_result.data
        except Exception as e:
            logger.debug("agent_loop.rag_skip", error=str(e))
            await asyncio.to_thread(
                self._work_memory.add_finding,
                scoped_conversation_id,
                f"RAG retrieval skipped: {e}",
            )

        return []

    async def _execute_tool_call(
        self,
        *,
        tool_call: dict[str, Any],
        conversation_id: str,
        scoped_conversation_id: str,
        iteration: int,
        messages: list[Message],
        tool_calls_log: list[dict[str, Any]],
        charts: list[dict[str, Any]],
    ):
        """
        执行单个工具调用（Act 阶段 + Observe 阶段的组合）。

        完整 Act-Observe 流程：
        1. 解析 tool arguments（JSON 字符串 → dict，宽松解析）
        2. 记录 WorkStep 开始（start_tool_step）
        3. 提前写入 latest_sql（sql_query 工具在执行前即可知道 SQL）
        4. 执行工具（registry.get(tool_name).run(**tool_args)）
        5. 提取 observation（ToolResult.to_observation()）
        6. 应用副作用（图表收集、数据摘要更新、finding 记录）
        7. 更新 WorkStep 状态（finish_tool_step）
        8. 把工具结果追加到消息历史（role="tool"）

        返回值说明：
        - 返回 tool_result（可能是 None，如工具不存在或执行前出错）
        - 调用方 _react_loop 根据返回值决定是否更新 latest_data

        错误处理策略：
        - 工具不存在（KeyError）：返回 None，observation 包含错误信息
        - 工具执行异常（Exception）：返回 None，observation 包含错误信息
        - 工具本身失败（tool_result.success=False）：正常返回，observation 包含错误描述

        这样的设计保证：任何工具的失败都不会中断 ReAct 循环，
        模型会看到错误的 observation，然后决定如何继续（重试、换工具或给出答案）。

        Args:
            tool_call: OpenAI function calling 格式的工具调用请求
            conversation_id: 原始会话 ID
            scoped_conversation_id: 加租户前缀的会话 ID
            iteration: 当前迭代次数
            messages: 消息历史（此方法会追加 tool role 消息）
            tool_calls_log: 工具调用审计日志（此方法会追加记录）
            charts: 图表产物收集列表（此方法可能追加）

        Returns:
            ToolResult 对象（工具成功或失败时），或 None（执行前就出错时）
        """
        tool_name = tool_call["function"]["name"]
        tool_args_str = tool_call["function"]["arguments"]
        tool_call_id = tool_call["id"]
        tool_args = self._parse_tool_args(tool_args_str)

        # 记录工具步骤开始（在执行前，保证快照中有 running 状态记录）
        work_step = await asyncio.to_thread(
            self._work_memory.start_tool_step,
            conversation_id=scoped_conversation_id,
            iteration=iteration,
            tool=tool_name,
            args=tool_args,
        )

        logger.info(
            "agent_loop.tool_call",
            tool=tool_name,
            args=str(tool_args)[:200],
            iteration=iteration,
        )

        # 对于 sql_query，提前写入 SQL（即使工具还未执行）
        if tool_name == "sql_query":
            sql = tool_args.get("sql")
            if isinstance(sql, str):
                await asyncio.to_thread(self._work_memory.set_latest_sql, scoped_conversation_id, sql)

        # 执行工具（三种可能的错误路径）
        try:
            tool = self._registry.get(tool_name)
            tool_result = await tool.run(**tool_args)
        except KeyError:
            observation = f"Error: Tool '{tool_name}' not found."
            tool_result = None
        except ConcurrencyLimitExceeded:
            # P2-10：工具并发超限属于系统过载，不降级为观察消息，原样上抛。
            raise
        except Exception as e:
            observation = f"Error executing tool '{tool_name}': {e}"
            tool_result = None
        else:
            # 工具执行完成（成功或失败，tool_result 均不为 None）
            observation = tool_result.to_observation()
            tool_calls_log.append(
                {"tool": tool_name, "args": tool_args, "success": tool_result.success}
            )
            # 处理工具成功执行后的附带副作用
            await asyncio.to_thread(
                self._apply_tool_result_side_effects,
                tool_name=tool_name,
                tool_result=tool_result,
                scoped_conversation_id=scoped_conversation_id,
                charts=charts,
            )

        # 生成工作记忆步骤摘要（无论成功还是失败）
        result_summary = WorkMemorySummarizer.summarize_tool_result(
            tool_name,
            tool_args,
            tool_result,
            observation,
        )
        # 记录工具步骤完成
        await asyncio.to_thread(
            self._work_memory.finish_tool_step,
            scoped_conversation_id,
            work_step.step_id,
            success=bool(tool_result and tool_result.success),
            observation=observation,
            result_summary=result_summary,
            error="" if (tool_result and tool_result.success) else observation,
        )

        # 把工具结果追加到消息历史（OpenAI API 要求 tool role 消息与 assistant tool_calls 对应）
        messages.append(
            Message(
                role="tool",
                content=observation,
                tool_call_id=tool_call_id,  # 必须对应 assistant 消息中的 tool_call id
                name=tool_name,
            )
        )
        return tool_result

    def _apply_tool_result_side_effects(
        self,
        *,
        tool_name: str,
        tool_result: Any,
        scoped_conversation_id: str,
        charts: list[dict[str, Any]],
    ) -> None:
        """
        处理工具成功执行后的附带状态更新（副作用）。

        为什么把副作用集中到这里？
        - _execute_tool_call 已经足够复杂，继续添加副作用逻辑会让它难以阅读
        - 副作用本身是"基于工具结果的派生行为"，单独抽取逻辑更清晰
        - 未来添加新工具的副作用只需在这里扩展，不需要修改主执行流程

        各工具的副作用：
        - generate_chart：把图表 JSON 追加到 charts 列表，注册产物引用
        - sql_query：更新 latest_data_summary，注册产物引用
        - 所有成功工具：如果有 text 摘要，追加一条 finding

        注意：这个方法只在工具成功时调用（失败时的副作用在 finish_tool_step 中处理）。

        Args:
            tool_name: 工具名称
            tool_result: 工具执行结果（ToolResult 对象，已确认 success=True）
            scoped_conversation_id: 加租户前缀的会话 ID
            charts: 图表产物收集列表（可能被修改）
        """
        if tool_name == "generate_chart" and tool_result.success:
            # 把图表 JSON 加入本次响应的 charts 列表
            charts.append(tool_result.data)
            self._work_memory.add_artifact(
                scoped_conversation_id,
                artifact_type="chart",
                preview=tool_result.text,
                metadata={"tool": tool_name},
            )

        if tool_name == "sql_query" and tool_result.success:
            # 更新数据摘要（行数、列名、首行预览）
            self._work_memory.set_latest_data_summary(
                scoped_conversation_id,
                WorkMemorySummarizer.summarize_rows(tool_result.data or []),
            )
            self._work_memory.add_artifact(
                scoped_conversation_id,
                artifact_type="sql_result",
                preview=tool_result.text,
                metadata={"rows": len(tool_result.data or [])},
            )

        # 所有成功工具的文本摘要都追加为 finding
        if tool_result.success and tool_result.text:
            self._work_memory.add_finding(
                scoped_conversation_id,
                f"{tool_name}: {tool_result.text[:300]}",
            )

    async def _build_final_response(
        self,
        *,
        query: str,
        conversation_id: str,
        scoped_conversation_id: str,
        final_answer: str,
        iteration: int,
        tool_calls_log: list[dict[str, Any]],
        charts: list[dict[str, Any]],
        latest_data: list[dict[str, Any]],
    ) -> AgentResponse:
        """
        统一的正常结束路径处理（成功收敛）。

        当模型不再请求工具时调用此方法，完成：
        1. WorkMemory 收尾（complete_run）
        2. ConversationMemory 写回（user 问题 + assistant 回答）
        3. 提取 pinned_facts（业务口径、用户偏好等值得长期固定的事实）
        4. 构造并返回 AgentResponse

        设计决策：正常结束和强制结束（_build_forced_final_response）
        都通过此方法完成收尾，确保两条路径的处理一致（DRY 原则）。

        ConversationMemory 写回时序：
        - 先写 user（原始问题），再写 assistant（最终回答 + metadata）
        - metadata 包含 work_memory 的桥接摘要和提取的 pinned_facts
        - assistant 消息的 metadata 面向 AgentLoop/ConversationMemory，不面向模型

        Args:
            query: 原始用户查询
            conversation_id: 原始会话 ID（写入 AgentResponse）
            scoped_conversation_id: 加租户前缀的会话 ID（内部组件用）
            final_answer: 模型的最终回答文本
            iteration: 最终迭代次数
            tool_calls_log: 本次所有工具调用的审计日志
            charts: 本次生成的所有图表 JSON
            latest_data: 最近 SQL 查询的原始结果

        Returns:
            成功的 AgentResponse
        """
        logger.info(
            "agent_loop.final_answer",
            conversation_id=conversation_id,
            iterations=iteration,
        )
        # 标记工作状态完成
        await asyncio.to_thread(self._work_memory.complete_run, scoped_conversation_id, final_answer)

        # 构建 ConversationMemory 写回的 metadata
        bridge_meta = await self._build_conversation_metadata(
            scoped_conversation_id=scoped_conversation_id,
            query=query,
            final_answer=final_answer,
        )

        # 写回对话历史（用户问题 + 助手回答）
        await self._memory.add(scoped_conversation_id, "user", query)
        await self._memory.add(
            scoped_conversation_id,
            "assistant",
            final_answer,
            metadata=bridge_meta,
        )
        return AgentResponse(
            answer=final_answer,
            conversation_id=conversation_id,
            iterations=iteration,
            tool_calls=tool_calls_log,
            charts=charts,
            data=latest_data,
            success=True,
        )

    async def _build_forced_final_response(
        self,
        *,
        query: str,
        conversation_id: str,
        scoped_conversation_id: str,
        iteration: int,
        messages: list[Message],
        tool_calls_log: list[dict[str, Any]],
        charts: list[dict[str, Any]],
        latest_data: list[dict[str, Any]],
    ) -> AgentResponse:
        """
        达到最大迭代次数后的兜底收尾（强制结束路径）。

        ReAct 循环需要一个硬上限，防止无限循环（模型可能陷入循环调用工具的死循环）。
        当循环次数达到 settings.agent_max_iterations 时，通过这个方法强制收束：
        1. 追加一条 user 消息请求总结
        2. 再做最后一次 LLM 调用（不提供工具，强制给出文本答案）
        3. 用这个强制生成的答案调用 _build_final_response 完成收尾

        为什么不直接截断？
        - 截断会让用户得到不完整的回答（"已执行 5 步但没有结论"）
        - 通过最后一次 LLM 调用，可以基于已有 observations 合成一个有意义的答案
        - 比截断体验好得多，虽然可能不如"正常结束"完整

        Args:
            query: 原始用户查询
            conversation_id: 原始会话 ID
            scoped_conversation_id: 加租户前缀的会话 ID
            iteration: 最终迭代次数（= agent_max_iterations）
            messages: 当前完整消息历史（包含所有工具调用和 observations）
            tool_calls_log: 工具调用审计日志
            charts: 图表产物收集
            latest_data: 最近 SQL 结果

        Returns:
            基于已有 observations 合成的最终 AgentResponse
        """
        logger.warning(
            "agent_loop.max_iterations",
            conversation_id=conversation_id,
            iterations=iteration,
        )
        # 追加一条请求总结的用户消息
        messages.append(
            Message(
                role="user",
                content="Please summarize what you've found so far and provide your best answer.",
            )
        )
        # 最后一次 LLM 调用（不提供 tools，强制输出文本）
        final_resp = await self._breaker.call(
            self._router.generate,
            messages=messages,
            task_type=TaskType.COMPLEX,
        )
        # 复用正常结束路径完成收尾
        return await self._build_final_response(
            query=query,
            conversation_id=conversation_id,
            scoped_conversation_id=scoped_conversation_id,
            final_answer=final_resp.content,
            iteration=iteration,
            tool_calls_log=tool_calls_log,
            charts=charts,
            latest_data=latest_data,
        )

    async def _build_planned_response(
        self,
        *,
        query: str,
        conversation_id: str,
        scoped_conversation_id: str,
        steps: list["PlanStep"],
    ) -> AgentResponse:
        """
        Plan-and-Execute 路径的响应构建。

        收集所有 PlanStep 的执行结果，整理成 AgentResponse 格式，
        然后通过 LLM 把所有步骤的 evidence 合成为最终答案。

        与 ReAct 路径的区别：
        - ReAct：模型自主决定工具调用顺序，消息历史是累积的
        - Plan-and-Execute：预先规划好步骤，并行执行，最后合成答案
        - 合成使用 _generate_grounded_final_answer()，确保答案有据可查

        Args:
            query: 原始用户查询
            conversation_id: 原始会话 ID
            scoped_conversation_id: 加租户前缀的会话 ID
            steps: Executor 执行完的 PlanStep 列表（包含每步的结果）

        Returns:
            AgentResponse（基于所有步骤结果合成的响应）
        """
        tool_calls_log: list[dict[str, Any]] = []
        charts: list[dict[str, Any]] = []
        latest_data: list[dict[str, Any]] = []

        for step in steps:
            success = bool(step.result and step.result.success)
            tool_calls_log.append(
                {
                    "tool": step.tool,
                    "args": step.tool_params,
                    "success": success,
                }
            )
            if not step.done:
                continue
            if success and step.result:
                # 处理成功步骤的副作用
                await asyncio.to_thread(
                    self._apply_tool_result_side_effects,
                    tool_name=step.tool,
                    tool_result=step.result,
                    scoped_conversation_id=scoped_conversation_id,
                    charts=charts,
                )
                if step.tool == "sql_query":
                    latest_data = step.result.data or []
                await asyncio.to_thread(
                    self._work_memory.add_finding,
                    scoped_conversation_id,
                    f"Plan step {step.step} [{step.tool}] succeeded.",
                )
            elif step.error:
                await asyncio.to_thread(
                    self._work_memory.add_finding,
                    scoped_conversation_id,
                    f"Plan step {step.step} [{step.tool}] failed: {step.error[:200]}",
                )

        # 基于所有步骤的 evidence 合成最终答案
        final_answer = await self._generate_grounded_final_answer(
            query=query,
            steps=steps,
        )
        return await self._build_final_response(
            query=query,
            conversation_id=conversation_id,
            scoped_conversation_id=scoped_conversation_id,
            final_answer=final_answer,
            iteration=max((step.step for step in steps), default=0),
            tool_calls_log=tool_calls_log,
            charts=charts,
            latest_data=latest_data,
        )

    async def _generate_grounded_final_answer(
        self,
        *,
        query: str,
        steps: list["PlanStep"],
    ) -> str:
        """
        基于计划步骤的执行证据，生成"有据可查"的最终答案。

        "Grounded"的含义：
        - 答案中每条陈述都有对应的 evidence（来自具体的工具执行结果）
        - 如果证据不足，明确说明缺少什么，而不是编造
        - 引用步骤编号（如 [Step 2]）让用户知道答案来源

        两种实现路径：
        1. settings.agent_force_grounded_answer=False（默认）：
           直接把前 3 个步骤的 evidence 拼接返回（简单但效果差）
        2. settings.agent_force_grounded_answer=True：
           调用 LLM 做 evidence 综合，生成结构化、有引用的最终答案

        如果 LLM 合成失败（熔断器开启等），退化为通用错误消息。

        Args:
            query: 原始用户查询
            steps: 所有 PlanStep（包含成功和失败的）

        Returns:
            合成的最终答案字符串
        """
        # 构建证据列表（每个步骤的结果摘要）
        evidence_lines: list[str] = []
        for step in steps:
            if step.result and step.result.success:
                preview = (step.result.text or "")[:800]
                evidence_lines.append(
                    f"[Step {step.step} | {step.tool} | success]\n"
                    f"Goal: {step.goal}\nEvidence: {preview}"
                )
            elif step.error:
                evidence_lines.append(
                    f"[Step {step.step} | {step.tool} | failed]\n"
                    f"Goal: {step.goal}\nError: {step.error[:400]}"
                )

        if not evidence_lines:
            return "I could not gather enough evidence to answer reliably."

        # 简单模式：直接返回 evidence（不调用 LLM）
        if not settings.agent_force_grounded_answer:
            return "\n\n".join(evidence_lines[:3])

        # 完整模式：用 LLM 综合 evidence 生成有引用的答案
        prompt = (
            "You are a grounded answer synthesizer for a data analysis agent.\n\n"
            "Use only the evidence below. Do not invent facts, numbers, tables, or causes.\n"
            "If the evidence is insufficient, explicitly say what is missing.\n"
            "When evidence supports a statement, cite the step ids in square brackets such as [Step 2].\n"
            "Prefer concise business language.\n\n"
            f"User question:\n{query}\n\n"
            f"Evidence:\n{chr(10).join(evidence_lines)}"
        )
        try:
            resp = await self._breaker.call(
                self._router.generate,
                messages=[
                    Message(
                        role="system",
                        content="You only synthesize answers from evidence produced by tools.",
                    ),
                    Message(role="user", content=prompt),
                ],
                task_type=TaskType.COMPLEX,
                temperature=0.0,
                max_tokens=800,
            )
            content = resp.content.strip()
            return content or "I could not synthesize a grounded final answer."
        except Exception as e:
            logger.warning("agent_loop.grounded_answer_failed", error=str(e))
            return "I could not synthesize a grounded final answer from the available evidence."

    async def _build_conversation_metadata(
        self,
        *,
        scoped_conversation_id: str,
        query: str,
        final_answer: str,
    ) -> dict[str, Any]:
        """
        构建写回 ConversationMemory 的 assistant metadata。

        metadata 包含两部分：
        1. work_memory 桥接摘要（run_id、selected_tables、latest_sql 等）
        2. pinned_facts（由 LLM 从本轮对话中提取，用于长期记忆）

        metadata 面向系统内部（AgentLoop + ConversationMemory），
        不直接面向模型消费（ConversationMemory 会选择性地把 pinned_facts 注入 prompt）。

        为什么在 assistant 消息的 metadata 里提取 pinned_facts？
        - assistant 消息整合了本轮 query、工具结果和最终解释，
          比只看 user query 更容易判断哪些内容值得长期保留
        - ConversationMemory.add() 会检查 metadata["pinned_facts"] 并存入长期记忆

        Args:
            scoped_conversation_id: 加租户前缀的会话 ID
            query: 原始用户查询
            final_answer: Agent 生成的最终回答

        Returns:
            包含 work_memory 桥接摘要和 pinned_facts 的 metadata 字典
        """
        metadata = self._work_memory.build_conversation_bridge(scoped_conversation_id)
        pinned_facts = await self._extract_pinned_facts(
            query=query,
            final_answer=final_answer,
            bridge_meta=metadata,
        )
        if pinned_facts:
            metadata["pinned_facts"] = pinned_facts
        return metadata

    async def _extract_pinned_facts(
        self,
        *,
        query: str,
        final_answer: str,
        bridge_meta: dict[str, Any],
    ) -> list[str]:
        """
        从本轮问答中提取值得长期固定的会话事实。

        提取标准（只保留"后续轮次可能复用"的内容）：
        ✓ 用户明确偏好（展示格式、分析粒度、语言偏好）
        ✓ 业务口径（指标定义、过滤条件、时间口径、归因口径）
        ✓ 稳定映射（某业务词对应的表、字段、维度）
        ✓ 后续追问需要继承的明确约束

        不提取：
        ✗ 一次性的查询结果或具体数值（会随时间变化）
        ✗ 工具调用过程、SQL 细节（内部实现细节）
        ✗ 模糊、未确认或模型自己推测的信息

        模型选择：使用 TaskType.SIMPLE（fast model / gpt-4o-mini），
        原因：提取任务结构简单，不需要最强模型，节省成本。

        输出格式：JSON 数组（list[str]），由 _parse_pinned_facts() 宽松解析。

        Args:
            query: 原始用户查询
            final_answer: Agent 的最终回答（最多使用前 3000 字符）
            bridge_meta: work_memory 桥接摘要（提供上下文参考）

        Returns:
            提取的 pinned facts 列表（最多 5 条，每条最多 80 字）；
            提取失败时返回空列表（不中断主流程）
        """
        prompt = (
            "你是数据分析 Agent 的长期记忆提取器。请从本轮用户问题和助手答案中，"
            "提取后续对话值得长期保留的 pinned facts。\n\n"
            "只保留以下类型：\n"
            "- 用户长期偏好，例如展示格式、分析粒度、语言偏好。\n"
            "- 业务口径，例如指标定义、过滤条件、时间口径、归因口径。\n"
            "- 稳定映射，例如某业务词对应的表、字段、维度。\n"
            "- 后续追问需要继承的明确约束。\n\n"
            "不要保留：\n"
            "- 一次性的查询结果或具体数值。\n"
            "- 工具调用过程、SQL 细节、报错信息。\n"
            "- 模糊、未确认或模型自己推测的信息。\n\n"
            "输出要求：\n"
            "- 只输出 JSON 数组。\n"
            "- 每项是一条简短中文事实，最多 80 字。\n"
            "- 如果没有值得长期固定的信息，输出 []。\n"
            "- 最多输出 5 条。\n\n"
            f"用户问题：\n{query}\n\n"
            f"助手答案：\n{final_answer[:3000]}\n\n"
            f"本轮工作摘要 metadata：\n{json.dumps(bridge_meta, ensure_ascii=False)[:1200]}"
        )
        try:
            resp = await self._breaker.call(
                self._router.generate,
                messages=[
                    Message(
                        role="system",
                        content="你只负责抽取长期记忆事实，不回答用户问题。",
                    ),
                    Message(role="user", content=prompt),
                ],
                task_type=TaskType.SIMPLE,
                max_tokens=400,
                temperature=0.0,
            )
            facts = self._parse_pinned_facts(resp.content)
            metrics.pinned_facts_total.labels(
                outcome="extracted" if facts else "empty"
            ).inc()
            return facts
        except Exception as e:
            metrics.pinned_facts_total.labels(outcome="failed").inc()
            logger.debug("agent_loop.pinned_facts_skip", error=str(e))
            return []

    @staticmethod
    def _parse_pinned_facts(content: str) -> list[str]:
        """
        宽松解析 LLM 返回的 pinned facts JSON 数组。

        "宽松"的含义：
        - LLM 被要求只输出 JSON 数组，但实际可能包一层解释或 markdown
        - 先尝试直接解析，失败则尝试提取 JSON 片段（找 [ 和 ] 之间的内容）
        - 只接受 list[str]，拒绝其他类型
        - 对每个事实做去重、长度裁剪

        为什么不用严格解析？
        - LLM 输出不总是完全符合格式要求，严格解析会导致频繁失败
        - 宽松解析在大多数情况下能正确提取，极少数真正混乱的输出才会返回空列表
        - pinned_facts 是可选的增强，提取失败不影响主流程

        Args:
            content: LLM 返回的原始文本

        Returns:
            解析出的 pinned facts 列表（最多 5 条，每条最多 120 字符）；
            解析失败时返回空列表
        """
        raw = content.strip()
        # 去除 markdown code fence（```json ... ```）
        if raw.startswith("```"):
            raw = raw.strip("`").strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # 直接解析失败，尝试提取 JSON 数组片段
            start = raw.find("[")
            end = raw.rfind("]")
            if start == -1 or end == -1 or end <= start:
                return []
            try:
                parsed = json.loads(raw[start: end + 1])
            except json.JSONDecodeError:
                return []

        if not isinstance(parsed, list):
            return []

        facts: list[str] = []
        for item in parsed:
            if not isinstance(item, str):
                continue
            fact = " ".join(item.split()).strip()  # 压缩空白
            if not fact or fact in facts:
                continue  # 空或重复的跳过
            facts.append(fact[:120])
        return facts[:5]  # 最多返回 5 条

    @staticmethod
    def _parse_tool_args(tool_args_str: str) -> dict[str, Any]:
        """
        宽松解析 LLM 生成的工具参数 JSON 字符串。

        function calling 中，模型生成的 arguments 是 JSON 字符串，
        但模型偶尔会生成格式不完全正确的 JSON（如末尾多了逗号）。

        当前策略：直接 json.loads()，失败则返回空字典。
        空字典策略的后果：工具收到缺少参数的调用会在工具层报错，
        error observation 会追加到消息历史，模型可以根据错误重试。
        这比在这里做复杂的 JSON 修复要简单和可靠。

        Args:
            tool_args_str: LLM 生成的 JSON 参数字符串

        Returns:
            解析出的参数字典；解析失败时返回空字典
        """
        try:
            return json.loads(tool_args_str)
        except json.JSONDecodeError:
            return {}
