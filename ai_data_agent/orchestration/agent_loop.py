"""
orchestration/agent_loop.py — Agent 核心循环
实现 ReAct (Reasoning + Acting) 模式
使用 OpenAI function calling 进行工具选择
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from ai_data_agent.config.config import settings
from ai_data_agent.memory.work_memory_summarizer import WorkMemorySummarizer
from ai_data_agent.model_gateway.base_model import Message
from ai_data_agent.model_gateway.router import TaskType
from ai_data_agent.observability.logger import get_logger
from ai_data_agent.observability.metrics import metrics
from ai_data_agent.observability.tracer import span

if TYPE_CHECKING:
    from ai_data_agent.context.prompt_builder import PromptBuilder
    from ai_data_agent.context.query_rewriter import QueryRewriter
    from ai_data_agent.context.schema_context import SchemaContextBuilder
    from ai_data_agent.memory.conversation_memory import ConversationMemory
    from ai_data_agent.memory.cache_memory import CacheMemory
    from ai_data_agent.memory.work_memory import WorkMemory
    from ai_data_agent.model_gateway.router import ModelRouter
    from ai_data_agent.tools.tool_registry import ToolRegistry
    from ai_data_agent.reliability.circuit_breaker import CircuitBreaker

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

    职责边界：
    - 这里只负责编排流程和状态推进
    - schema 解析交给 SchemaContextBuilder
    - 工作记忆摘要交给 WorkMemorySummarizer

    换句话说，AgentLoop 负责“什么时候做什么”，
    而不负责“如何把 schema / tool result 格式化成摘要”。
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
    ) -> None:
        """
        AgentLoop 的依赖全部由外部注入。

        这样 orchestration 层只负责消费依赖，不负责创建依赖；
        组件的真实组装点收敛到 assembler，符合 composition root 设计。
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
                cache_key = self._cache.make_key("agent", query, conversation_id)
                cached = self._cache.get(cache_key)
                if cached:
                    logger.info("agent_loop.cache_hit", conversation_id=conversation_id)
                    return cached

            try:
                response = await self._react_loop(query, conversation_id)
            except Exception as e:
                self._work_memory.fail_run(conversation_id, str(e))
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
            self._cache.set(cache_key, response)

        return response

    async def _react_loop(
        self,
        query: str,
        conversation_id: str,
    ) -> AgentResponse:
        """
        核心 ReAct 循环。

        执行顺序固定分成两个阶段：
        1. 循环前准备上下文，并初始化本轮工作状态
        2. 进入 Think -> Act -> Observe 循环，直到模型直接给出答案

        这里刻意把“准备阶段”和“循环阶段”写在同一个方法内，
        因为这两段逻辑共享大量运行态对象。

        当前版本进一步把“准备上下文”“收尾写回”“单次工具执行”都拆成私有方法，
        让这里尽量接近一段真正的流程骨架代码：
        - 初始化运行态
        - 循环调用模型
        - 要么收尾返回，要么执行工具并继续
        """
        # 为本次请求建立一份全新的工作状态。
        # 当前版本按“每个请求一次运行”建模，这样可以清晰记录本轮执行轨迹。
        self._work_memory.start_run(conversation_id, query)
        messages = await self._build_initial_messages(
            query=query,
            conversation_id=conversation_id,
        )

        # ── ReAct 循环 ────────────────────────────────────────────────────────
        tool_calls_log: list[dict[str, Any]] = []
        charts: list[dict[str, Any]] = []
        latest_data: list[dict[str, Any]] = []
        iteration = 0
        tools_schema = self._registry.to_openai_tools()

        while iteration < settings.agent_max_iterations:
            iteration += 1
            self._work_memory.set_iterations(conversation_id, iteration)
            logger.debug("agent_loop.iteration", n=iteration, conversation_id=conversation_id)

            # LLM 调用（带熔断器）
            resp = await self._breaker.call(
                self._router.generate,
                messages=messages,
                task_type=TaskType.COMPLEX,
                tools=tools_schema,
                tool_choice="auto",
            )

            # 无工具调用 → 最终答案
            if not resp.tool_calls:
                return await self._build_final_response(
                    query=query,
                    conversation_id=conversation_id,
                    final_answer=resp.content,
                    iteration=iteration,
                    tool_calls_log=tool_calls_log,
                    charts=charts,
                    latest_data=latest_data,
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
                tool_result = await self._execute_tool_call(
                    tool_call=tc,
                    conversation_id=conversation_id,
                    iteration=iteration,
                    messages=messages,
                    tool_calls_log=tool_calls_log,
                    charts=charts,
                )
                if (
                    tool_result is not None
                    and tool_result.success
                    and tc["function"]["name"] == "sql_query"
                ):
                    latest_data = tool_result.data or []

        # 所有工具执行完毕 → 把所有结果整理成自然语言答案返回给用户
        return await self._build_forced_final_response(
            query=query,
            conversation_id=conversation_id,
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
        conversation_id: str,
    ) -> list[Message]:
        """
        准备进入 ReAct 循环前所需的初始消息列表。

        这个阶段本质上是“把原始用户问题加工成模型第一轮输入”：
        - 改写 query
        - 生成 schema context
        - 可选执行一次轻量 RAG 检索
        - 合并历史对话和工作记忆摘要

        这些步骤都发生在循环前，因此抽成单独方法后，_react_loop() 就能只关注
        “循环如何推进”，而不是被前置准备细节淹没。
        """
        rewrite_result = await self._query_rewriter.rewrite(query)
        logger.debug("agent_loop.rewrite", result=rewrite_result)
        self._work_memory.set_rewritten_query(
            conversation_id,
            rewrite_result.get("rewritten", ""),
        )
        if rewrite_result.get("reason"):
            self._work_memory.add_finding(
                conversation_id,
                f"Query rewritten rationale: {rewrite_result['reason']}",
            )

        schema_ctx = await self._schema_builder.build(query)
        self._work_memory.set_schema_context(
            conversation_id,
            schema_ctx,
            selected_tables=self._schema_builder.extract_table_names(schema_ctx),
        )

        rag_docs = await self._retrieve_rag_docs(
            query=rewrite_result.get("rewritten", query),
            conversation_id=conversation_id,
        )
        history = self._memory.get_messages(conversation_id)

        return self._prompt_builder.build(
            query=query,
            rag_docs=rag_docs,
            schema_context=schema_ctx,
            history=history,
            work_context=self._work_memory.build_prompt_context(conversation_id),
        )

    async def _retrieve_rag_docs(
        self,
        *,
        query: str,
        conversation_id: str,
    ) -> list[dict[str, Any]]:
        """
        执行一次可选的知识检索，并把检索结果摘要写入工作记忆。

        这里单独成方法的原因不是复用，而是隔离“非核心、可失败”的前置增强逻辑。
        RAG 检索失败不应影响主循环，因此方法内部自行吞掉异常并记录 finding。
        """
        if not self._registry.list_names():
            return []

        try:
            rag_tool = self._registry.get("search_documents")
            rag_result = await rag_tool.run(query=query)
            if rag_result.success and rag_result.data:
                self._work_memory.add_finding(
                    conversation_id,
                    f"Retrieved {len(rag_result.data)} relevant knowledge document(s).",
                )
                return rag_result.data
        except Exception as e:
            logger.debug("agent_loop.rag_skip", error=str(e))
            self._work_memory.add_finding(
                conversation_id,
                f"RAG retrieval skipped: {e}",
            )

        return []

    async def _execute_tool_call(
        self,
        *,
        tool_call: dict[str, Any],
        conversation_id: str,
        iteration: int,
        messages: list[Message],
        tool_calls_log: list[dict[str, Any]],
        charts: list[dict[str, Any]],
    ):
        """
        执行单个 tool call，并把结果同步回消息历史与工作记忆。

        这个方法封装的是一次完整的“Act -> Observe”子流程：
        - 解析参数
        - 记录 step 开始
        - 执行工具
        - 提取 observation
        - 更新产物、finding、摘要和 messages

        注意这里不再负责维护外层的 latest_data 状态。
        它只返回本次工具调用产生的 tool_result，由主循环决定是否据此更新
        “当前循环持有的最新数据”。这样状态归属会更清楚：工具执行负责产出结果，
        主循环负责维护跨轮状态。
        """
        tool_name = tool_call["function"]["name"]
        tool_args_str = tool_call["function"]["arguments"]
        tool_call_id = tool_call["id"]
        tool_args = self._parse_tool_args(tool_args_str)

        work_step = self._work_memory.start_tool_step(
            conversation_id=conversation_id,
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

        if tool_name == "sql_query":
            sql = tool_args.get("sql")
            if isinstance(sql, str):
                self._work_memory.set_latest_sql(conversation_id, sql)

        try:
            tool = self._registry.get(tool_name)
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
            self._apply_tool_result_side_effects(
                tool_name=tool_name,
                tool_result=tool_result,
                conversation_id=conversation_id,
                charts=charts,
            )

        result_summary = WorkMemorySummarizer.summarize_tool_result(
            tool_name,
            tool_args,
            tool_result,
            observation,
        )
        self._work_memory.finish_tool_step(
            conversation_id,
            work_step.step_id,
            success=bool(tool_result and tool_result.success),
            observation=observation,
            result_summary=result_summary,
            error="" if (tool_result and tool_result.success) else observation,
        )

        messages.append(
            Message(
                role="tool",
                content=observation,
                tool_call_id=tool_call_id,
                name=tool_name,
            )
        )
        return tool_result

    def _apply_tool_result_side_effects(
        self,
        *,
        tool_name: str,
        tool_result: Any,
        conversation_id: str,
        charts: list[dict[str, Any]],
    ) -> None:
        """
        处理工具成功执行后的附带状态更新。

        这些更新不是“执行工具”的核心动作，而是围绕工具结果产生的派生副作用：
        - chart 产物登记
        - SQL 结果摘要写入
        - finding 记录

        把副作用集中起来后，工具执行主逻辑会更清楚，也更方便以后继续拆分。
        """
        if tool_name == "generate_chart" and tool_result.success:
            charts.append(tool_result.data)
            self._work_memory.add_artifact(
                conversation_id,
                artifact_type="chart",
                preview=tool_result.text,
                metadata={"tool": tool_name},
            )

        if tool_name == "sql_query" and tool_result.success:
            self._work_memory.set_latest_data_summary(
                conversation_id,
                WorkMemorySummarizer.summarize_rows(tool_result.data or []),
            )
            self._work_memory.add_artifact(
                conversation_id,
                artifact_type="sql_result",
                preview=tool_result.text,
                metadata={"rows": len(tool_result.data or [])},
            )

        if tool_result.success and tool_result.text:
            self._work_memory.add_finding(
                conversation_id,
                f"{tool_name}: {tool_result.text[:300]}",
            )

    async def _build_final_response(
        self,
        *,
        query: str,
        conversation_id: str,
        final_answer: str,
        iteration: int,
        tool_calls_log: list[dict[str, Any]],
        charts: list[dict[str, Any]],
        latest_data: list[dict[str, Any]],
    ) -> AgentResponse:
        """
        处理正常结束路径。

        当模型不再请求工具时，说明当前 ReAct 回合已经收敛。
        这里统一完成：
        - work_memory 收尾
        - conversation_memory 桥接写回
        - AgentResponse 构造

        这样可以保证“正常结束”和“强制结束”都复用同一套收尾逻辑。
        """
        logger.info(
            "agent_loop.final_answer",
            conversation_id=conversation_id,
            iterations=iteration,
        )
        self._work_memory.complete_run(conversation_id, final_answer)
        bridge_meta = await self._build_conversation_metadata(
            conversation_id=conversation_id,
            query=query,
            final_answer=final_answer,
        )
        await self._memory.add(conversation_id, "user", query)
        await self._memory.add(
            conversation_id,
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
        iteration: int,
        messages: list[Message],
        tool_calls_log: list[dict[str, Any]],
        charts: list[dict[str, Any]],
        latest_data: list[dict[str, Any]],
    ) -> AgentResponse:
        """
        处理达到最大迭代次数后的兜底收尾。

        ReAct 循环本身是开放式的，因此必须存在一个硬上限。
        当达到上限时，这里会追加一个“请基于已有发现给出最佳答案”的用户消息，
        再做最后一次模型调用，把开放式循环收束成一个确定的最终响应。
        """
        logger.warning(
            "agent_loop.max_iterations",
            conversation_id=conversation_id,
            iterations=iteration,
        )
        messages.append(
            Message(
                role="user",
                content="Please summarize what you've found so far and provide your best answer.",
            )
        )
        final_resp = await self._breaker.call(
            self._router.generate,
            messages=messages,
            task_type=TaskType.COMPLEX,
        )
        return await self._build_final_response(
            query=query,
            conversation_id=conversation_id,
            final_answer=final_resp.content,
            iteration=iteration,
            tool_calls_log=tool_calls_log,
            charts=charts,
            latest_data=latest_data,
        )

    async def _build_conversation_metadata(
        self,
        *,
        conversation_id: str,
        query: str,
        final_answer: str,
    ) -> dict[str, Any]:
        """
        生成写回 conversation_memory 的 assistant metadata。
        metadata是给agent_loop看的

        这里做两件事：
        - 保留 work_memory 给 conversation_memory 的轻量桥接摘要
        - 额外提取 pinned_facts，用于长期记住业务口径、用户偏好、稳定约束

        pinned_facts 明确放在 assistant 回复的 metadata 上，是因为：
        - assistant 回复通常已经整合了本轮 query、工具结果和最终解释
        - 比只看 user query 更容易判断哪些内容已被确认、值得长期保留
        - conversation_memory 只消费 metadata["pinned_facts"]，不会把整段 metadata 注入 prompt
        """
        metadata = self._work_memory.build_conversation_bridge(conversation_id)
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

        只提取“后续轮次可能继续复用”的内容，例如：
        - 用户明确偏好
        - 业务指标口径
        - 稳定过滤条件
        - 表/字段/术语映射

        不提取一次性的分析结果、临时报错、工具执行过程或过细的数值结果。
        这能避免 pinned facts 变成另一个无限增长的结果缓存。
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
            return self._parse_pinned_facts(resp.content)
        except Exception as e:
            logger.debug("agent_loop.pinned_facts_skip", error=str(e))
            return []

    @staticmethod
    def _parse_pinned_facts(content: str) -> list[str]:
        """
        宽松解析 pinned facts 提取结果。

        LLM 被要求返回 JSON 数组，但实际输出可能包一层解释或 markdown。
        这里只接受 list[str]，并做去重、裁剪，避免脏数据进入长期记忆。
        """
        raw = content.strip()
        if raw.startswith("```"):
            raw = raw.strip("`").strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("[")
            end = raw.rfind("]")
            if start == -1 or end == -1 or end <= start:
                return []
            try:
                parsed = json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                return []

        if not isinstance(parsed, list):
            return []

        facts: list[str] = []
        for item in parsed:
            if not isinstance(item, str):
                continue
            fact = " ".join(item.split()).strip()
            if not fact or fact in facts:
                continue
            facts.append(fact[:120])
        return facts[:5]

    @staticmethod
    def _parse_tool_args(tool_args_str: str) -> dict[str, Any]:
        """
        解析模型返回的 tool arguments。

        function calling 返回的是 JSON 字符串，但模型输出并不总是完全可靠。
        这里统一做宽松解析，失败时退回空参数，避免解析异常直接打断整轮执行。
        """
        try:
            return json.loads(tool_args_str)
        except json.JSONDecodeError:
            return {}
