"""
orchestration/langgraph_agent_loop.py — 基于 LangGraph 的 Agent 编排循环

对比 agent_loop.py（手写 while 循环 ReAct），本文件用 LangGraph 框架将相同的逻辑
表达为有向状态图（StateGraph），把每个步骤抽象为节点（Node），
把每个分支决策抽象为条件边（Conditional Edge）。

LangGraph 核心概念：
  StateGraph：有向图，每个节点是一个函数（async 或 sync）。
    - 节点输入：当前全量状态（AgentState TypedDict）
    - 节点输出：要更新的状态字段（部分 dict，LangGraph 自动 merge）

  State Reducer（归约器）：
    - 默认（无 Annotated）：新值直接覆盖旧值（overwrite semantics）
    - Annotated[list, operator.add]：追加语义（accumulate semantics）
      → 节点只需返回"新增的部分"，LangGraph 自动 extend 到现有列表
      → 适合跨迭代收集工具调用记录（tool_calls_log）和图表（charts）

  条件边（add_conditional_edges）：
    - 接受一个路由函数（routing function）
    - 路由函数根据当前状态返回下一个节点名称（字符串）
    - 等价于原代码中的 if/elif/return 分支结构

节点清单（9 个）：
  prepare_context      — 等价于 AgentLoop._build_initial_messages()
  planner              — 调用 Planner 生成执行计划
  executor             — 执行 Planner 生成的计划
  build_planned_response — 从计划步骤结果构建最终答案
  think                — ReAct Think 步：调用 LLM 推理下一步行动
  act_and_observe      — ReAct Act+Observe 步：执行工具调用
  build_final_response — 正常结束：模型不再请求工具
  force_summarize      — 超限结束：达到最大迭代次数，强制 LLM 总结
  save_memory          — 统一收尾：写 ConversationMemory，构建 AgentResponse

图的边（原 while 循环和 if/return 分支 → 条件边）：
  START → prepare_context
  prepare_context →(routing)→ {planner | think}       ← 是否启用 Plan-and-Execute
  planner →(routing)→ {executor | think}              ← 计划是否值得执行
  executor → build_planned_response → save_memory → END
  think →(routing)→ {act_and_observe | build_final_response | force_summarize}
  act_and_observe → think                             ← ReAct 循环的核心回边
  build_final_response → save_memory → END
  force_summarize → save_memory → END

LangGraph vs 手写循环 的优势（本文件展示的能力）：
  - 可视化：graph.get_graph().draw_mermaid() 生成 Mermaid 流程图
  - 流式输出：graph.astream() 支持逐步返回中间状态（前端 SSE 流式展示）
  - 检查点（Checkpoint）：可接入 LangGraph Checkpoint，支持断点续跑和状态持久化
  - 测试友好：每个节点是独立的 async 函数，可单独单元测试
  - 状态隔离：每次 ainvoke() 创建独立状态，无需手动清理中间变量

依赖说明（需要安装）：
  langgraph 包尚未在 requirements.txt 中声明，使用前需手动添加：
    pip install "langgraph>=0.2.0"
  或在 requirements.txt 中添加：
    langgraph>=0.2.0
"""
from __future__ import annotations

import json
import operator
import time
from typing import Annotated, Any, TYPE_CHECKING
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from ai_data_agent.config.config import settings
from ai_data_agent.context.request_context import RequestContext, get_request_context
from ai_data_agent.memory.work_memory_summarizer import WorkMemorySummarizer
from ai_data_agent.model_gateway.base_model import Message
from ai_data_agent.model_gateway.router import TaskType
from ai_data_agent.observability.logger import get_logger
from ai_data_agent.observability.metrics import metrics
from ai_data_agent.reliability.concurrency import get_limiter
from ai_data_agent.observability.tracer import span

# 从原始模块复用 AgentResponse，保持调用方接口兼容性
from ai_data_agent.orchestration.agent_loop import AgentResponse

if TYPE_CHECKING:
    from ai_data_agent.context.prompt_builder import PromptBuilder
    from ai_data_agent.context.query_rewriter import QueryRewriter
    from ai_data_agent.context.schema_context import SchemaContextBuilder
    from ai_data_agent.memory.conversation_memory import ConversationMemory
    from ai_data_agent.memory.cache_memory import CacheMemory
    from ai_data_agent.memory.work_memory import WorkMemory
    from ai_data_agent.model_gateway.router import ModelRouter
    from ai_data_agent.orchestration.executor import Executor
    from ai_data_agent.orchestration.planner import Planner
    from ai_data_agent.tools.tool_registry import ToolRegistry
    from ai_data_agent.reliability.circuit_breaker import CircuitBreaker

logger = get_logger(__name__)


# ── LangGraph 状态定义 ──────────────────────────────────────────────────────────

class AgentState(TypedDict, total=False):
    """
    LangGraph 图的全局状态（State Schema）。

    total=False 表示所有字段都是可选的（Optional）：
    - 节点只需返回它"更新了哪些字段"的子集，LangGraph 自动 merge 到当前状态
    - 初始状态只需包含入参字段，其余字段在各节点执行过程中逐渐填充

    Reducer 语义说明：
    - 无 Annotated（默认）：覆盖语义（overwrite）
      → messages、iteration、llm_resp 等单值字段
      → 每次节点返回新值时，直接替换旧值
    - Annotated[list, operator.add]：追加语义（accumulate）
      → tool_calls_log：每次迭代新增的调用记录，LangGraph 自动 extend 到已有列表
      → charts：每次迭代新增的图表 JSON，同样追加

    字段分组（按生命周期）：
    ① 入参字段（run() 注入，图运行期间只读）
    ② 上下文阶段产物（prepare_context 节点写入，后续节点只读）
    ③ ReAct 迭代状态（think / act_and_observe 节点读写，跨迭代演进）
    ④ Plan-and-Execute 状态（planner / executor 节点读写）
    ⑤ 最终结果字段（save_memory 节点写入，run() 从这里取结果）
    """

    # ── ① 入参字段 ──────────────────────────────────────────────────────────
    query: str                          # 原始用户查询（自然语言）
    conversation_id: str                # 用户提供的会话 ID（未加租户前缀）
    scoped_conversation_id: str         # 加租户前缀的内部会话 ID（组件间通信用）
    use_cache: bool                     # 是否启用结果缓存
    request_context: Any                # RequestContext 对象（日志/审计上下文）

    # ── ② 上下文阶段产物 ─────────────────────────────────────────────────────
    messages: list                      # list[Message]，覆盖语义（每轮更新整个列表）
    schema_ctx: str                     # schema 上下文文本（LLM 的"数据地图"）
    rewrite_result: dict                # 查询改写结果（含 rewritten、reason 字段）
    rag_docs: list                      # RAG 检索结果列表（list[dict]）

    # ── ③ ReAct 迭代状态 ──────────────────────────────────────────────────────
    iteration: int                      # 当前 ReAct 迭代次数（每次 think 前 +1）
    llm_resp: Any                       # 最近一次 LLM 响应（LLMResponse 对象）
    latest_data: list                   # 最近 SQL 查询的原始结果（list[dict]，覆盖）
    # Annotated[list, operator.add]：追加 reducer，节点只返回"新增部分"
    tool_calls_log: Annotated[list, operator.add]   # 工具调用审计日志（跨迭代积累）
    charts: Annotated[list, operator.add]           # 图表 JSON 列表（跨迭代积累）

    # ── ④ Plan-and-Execute 状态 ───────────────────────────────────────────────
    plan: Any                           # Planner 生成的 Plan 对象
    plan_steps: list                    # Executor 执行完的 PlanStep 列表

    # ── ⑤ 最终结果字段 ────────────────────────────────────────────────────────
    final_answer: str                   # 最终答案文本（所有结束路径写入）
    response: Any                       # AgentResponse（save_memory 构建，run() 读取）


# ── LangGraph Agent 主类 ────────────────────────────────────────────────────────

class LangGraphAgentLoop:
    """
    基于 LangGraph 的 ReAct Agent（功能等价于 AgentLoop，用声明式图取代手写循环）。

    与 AgentLoop 的等价性：
    - 相同的依赖注入接口（__init__ 参数完全一致）
    - 相同的 run() 公共方法签名和返回类型（AgentResponse）
    - 相同的缓存、并发限制、熔断器、Prometheus、OpenTelemetry 集成
    - 相同的 ReAct 循环语义（Think → Act → Observe → 循环）
    - 相同的 Plan-and-Execute 可选路径

    架构差异（LangGraph 特有能力）：
    - 可视化：self._graph.get_graph().draw_mermaid() 输出 Mermaid 流程图
    - 流式输出：self._graph.astream(initial_state) 逐步返回各节点输出
    - 检查点：可在 compile() 时传入 checkpointer 实现断点续跑
    - 早期错误检测：_build_graph() 在 __init__ 阶段执行 compile()，
      图结构错误（孤立节点、缺少 END 路径等）在启动时就会报错

    线程安全：
    - LangGraphAgentLoop 本身是无状态的（所有状态存储在注入的组件中）
    - graph.ainvoke() 每次创建独立的状态副本，天然隔离并发请求
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
        通过依赖注入初始化 LangGraphAgentLoop（接口与 AgentLoop.__init__ 完全一致）。

        所有参数均为关键字参数（keyword-only），防止位置参数顺序错误。
        planner/executor 为可选参数：
        - 两者都为 None 时，_route_strategy 直接路由到 think（ReAct 路径）
        - 两者都不为 None 且 settings.agent_enable_planning=True 时，尝试 Plan-and-Execute

        图在 __init__ 阶段编译（self._graph = self._build_graph()）：
        - compile() 执行拓扑排序，检测孤立节点、缺失边等图结构错误
        - 确保 LangGraphAgentLoop 实例创建成功即代表图结构合法

        Args:
            prompt_builder: 构建 LLM 输入消息的组件
            query_rewriter: LLM 查询改写器（提升 schema/RAG 召回率）
            schema_builder: 数据库 schema 语义检索器
            memory: 对话历史管理器（三层记忆结构）
            cache: 结果缓存（LRU + TTL）
            work_memory: 单次任务状态记录器
            registry: 工具注册中心（所有可用工具）
            router: 模型路由器（选择 LLM 及 Fallback 策略）
            breaker: 熔断器（保护 LLM 调用，防止级联故障）
            planner: 任务规划器（可选，用于 Plan-and-Execute 路径）
            executor: 计划执行器（可选，与 planner 配合使用）
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

        # 在 __init__ 阶段编译图（拓扑排序 + 类型检查），早期发现图结构错误
        self._graph = self._build_graph()

    # ── 图构建 ──────────────────────────────────────────────────────────────────

    def _build_graph(self):
        """
        声明并编译 LangGraph 状态图。

        图结构总览（节点 + 边）：
          START
            ↓
          prepare_context ─→ planner ─→ executor ─→ build_planned_response ─→ save_memory → END
            ↓                   ↓
           think ←─────────────╯
            ↓ (routing)
          act_and_observe ──→ think（循环）
            ↓ (routing)
          build_final_response ─→ save_memory → END
            ↓ (routing)
          force_summarize ─────→ save_memory → END

        add_conditional_edges 格式：
            graph.add_conditional_edges(
                source_node,       # 从哪个节点出发
                routing_fn,        # 路由函数（AgentState → str）
                {"返回值": "目标节点名", ...}  # 可选的值→节点名映射表
            )
        如果不传映射表，路由函数的返回值直接作为目标节点名。

        Returns:
            编译好的 CompiledGraph，支持 ainvoke() / astream() 调用
        """
        graph = StateGraph(AgentState)

        # ── 注册所有节点（顺序不影响执行，但按逻辑流程顺序有助于阅读）────────
        graph.add_node("prepare_context", self._node_prepare_context)
        graph.add_node("planner", self._node_planner)
        graph.add_node("executor", self._node_executor)
        graph.add_node("build_planned_response", self._node_build_planned_response)
        graph.add_node("think", self._node_think)
        graph.add_node("act_and_observe", self._node_act_and_observe)
        graph.add_node("build_final_response", self._node_build_final_response)
        graph.add_node("force_summarize", self._node_force_summarize)
        graph.add_node("save_memory", self._node_save_memory)

        # ── 静态边（确定性跳转，无条件）────────────────────────────────────────
        graph.add_edge(START, "prepare_context")
        # Plan-and-Execute 路径：executor → build_planned_response → save_memory
        graph.add_edge("executor", "build_planned_response")
        graph.add_edge("build_planned_response", "save_memory")
        # ReAct 循环的核心回边：act_and_observe → think（循环体）
        graph.add_edge("act_and_observe", "think")
        # 两条正常/强制结束路径汇聚到 save_memory
        graph.add_edge("build_final_response", "save_memory")
        graph.add_edge("force_summarize", "save_memory")
        graph.add_edge("save_memory", END)

        # ── 条件边（等价于原代码的 if/elif/return 分支结构）────────────────────
        # 1. prepare_context 后：是否启用 Plan-and-Execute
        graph.add_conditional_edges(
            "prepare_context",
            self._route_strategy,
            {
                "planner": "planner",   # 启用 Plan-and-Execute → 先让 Planner 评估
                "think": "think",       # 直接走 ReAct → 进入 Think 节点
            },
        )
        # 2. planner 后：计划是否值得执行（非空且非 simple）
        graph.add_conditional_edges(
            "planner",
            self._route_after_plan,
            {
                "executor": "executor",  # 计划不为空且不是 simple → 执行计划
                "think": "think",        # 计划为空或 simple → 退回 ReAct
            },
        )
        # 3. think 后：ReAct 循环的三路分支
        graph.add_conditional_edges(
            "think",
            self._route_after_think,
            {
                "act_and_observe": "act_and_observe",       # 有工具调用且未达上限
                "build_final_response": "build_final_response",  # 无工具调用（正常结束）
                "force_summarize": "force_summarize",       # 达到最大迭代次数
            },
        )

        return graph.compile()

    # ── 公共入口 ─────────────────────────────────────────────────────────────────

    async def run(
        self,
        query: str,
        conversation_id: str,
        request_context: RequestContext | None = None,
        use_cache: bool = True,
    ) -> AgentResponse:
        """
        执行一次完整的 Agent 请求（与 AgentLoop.run() 接口完全一致）。

        外层逻辑与 AgentLoop.run() 完全相同，LangGraph 只影响内部编排方式：
        1. 解析 RequestContext（显式传入 > contextvars > 默认系统上下文）
        2. 缓存检查（命中则直接返回，跳过 LLM 和工具调用）
        3. 并发限制（get_limiter().limit("agent_request")）
        4. OpenTelemetry span（分布式追踪）
        5. graph.ainvoke(initial_state)——等价于原来的 _react_loop()
        6. 异常兜底（返回 success=False 的 AgentResponse，不向调用方抛异常）
        7. 延迟/迭代次数指标上报（Prometheus）
        8. 成功结果写入缓存

        graph.ainvoke() 与 graph.astream() 的区别：
        - ainvoke()：等待整个图执行完毕，返回最终 AgentState（适合同步 API）
        - astream()：逐步 yield 每个节点的输出，适合前端流式展示（SSE/WebSocket）

        Args:
            query: 用户的自然语言查询
            conversation_id: 用户提供的会话 ID（未加租户前缀）
            request_context: 请求上下文（可选）
            use_cache: 是否使用结果缓存（默认 True，eval 评估时设为 False）

        Returns:
            AgentResponse（无论成功还是失败都返回，不向调用方抛出异常）
        """
        start = time.perf_counter()
        metrics.agent_requests_total.inc()

        # 解析请求上下文（三级优先级：显式 > contextvars > 默认系统上下文）
        req_ctx = request_context or get_request_context() or RequestContext(
            request_id="system",
            user_id=settings.default_user_id,
            tenant_id=settings.default_tenant_id,
        )
        scoped_conversation_id = req_ctx.scoped_conversation_id(conversation_id)

        async with get_limiter().limit("agent_request"):
            with span(
                "langgraph_agent_loop.run",
                {
                    "request_id": req_ctx.request_id,
                    "user_id": req_ctx.user_id,
                    "tenant_id": req_ctx.tenant_id,
                    "conversation_id": conversation_id,
                },
            ):
                # 缓存检查（只缓存成功结果，失败不缓存）
                cache_key = self._cache.make_key(
                    "agent", req_ctx.tenant_id, query, conversation_id
                )
                if use_cache:
                    cached = self._cache.get(cache_key)
                    if cached:
                        logger.info(
                            "langgraph_agent_loop.cache_hit",
                            request_id=req_ctx.request_id,
                            conversation_id=conversation_id,
                        )
                        return cached

                try:
                    # 构建初始状态，作为图的输入
                    # Annotated[list, operator.add] 字段必须用空列表初始化，
                    # 否则第一个节点的 operator.add 找不到基础列表
                    initial_state: AgentState = {
                        "query": query,
                        "conversation_id": conversation_id,
                        "scoped_conversation_id": scoped_conversation_id,
                        "use_cache": use_cache,
                        "request_context": req_ctx,
                        "iteration": 0,
                        "tool_calls_log": [],   # operator.add reducer 需要初始化为 []
                        "charts": [],           # 同上
                        "latest_data": [],
                        "messages": [],
                        "rag_docs": [],
                    }
                    # 执行整个图，等待最终状态（等价于原来的 _react_loop()）
                    final_state: AgentState = await self._graph.ainvoke(initial_state)
                    response: AgentResponse = final_state["response"]

                except Exception as e:
                    # 任何未捕获的异常在这里兜底，确保不向调用方抛异常
                    self._work_memory.fail_run(scoped_conversation_id, str(e))
                    logger.error(
                        "langgraph_agent_loop.failed",
                        request_id=req_ctx.request_id,
                        error=str(e),
                        conversation_id=conversation_id,
                    )
                    metrics.agent_errors_total.labels(error_type=type(e).__name__).inc()
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
            self._cache.set(cache_key, response)

        return response

    # ── 节点实现（每个节点接受 AgentState，返回部分状态更新 dict）──────────────

    async def _node_prepare_context(self, state: AgentState) -> dict[str, Any]:
        """
        prepare_context 节点：等价于 AgentLoop._build_initial_messages()。

        执行 ReAct 循环前的所有准备工作（不属于 ReAct 循环本身）：
        1. 初始化 WorkMemory（start_run）
        2. Query Rewriting：LLM 改写查询，提升 schema 检索和 RAG 的召回率
        3. Schema Context：语义检索相关表结构，为 LLM 提供"数据地图"
        4. RAG 检索：可选的知识库检索（失败不中断主流程）
        5. 获取历史消息：ConversationMemory → list[Message]
        6. Prompt 构建：PromptBuilder 组装完整 messages 列表

        节点输出（状态更新字段）：
        - messages: 初始消息列表（系统提示 + RAG + schema + 历史 + 用户问题）
        - schema_ctx: schema 上下文文本（planner 节点也需要读取）
        - rewrite_result: 查询改写结果字典
        - rag_docs: RAG 检索文档列表
        """
        query = state["query"]
        scoped_cid = state["scoped_conversation_id"]

        # 初始化本次请求的工作记忆状态（start_run 标记为运行中）
        self._work_memory.start_run(scoped_cid, query)

        # Step 1: 查询改写（提升后续 schema 检索和 RAG 检索的召回率）
        rewrite_result = await self._query_rewriter.rewrite(query)
        logger.debug("langgraph_agent_loop.rewrite", result=rewrite_result)
        self._work_memory.set_rewritten_query(scoped_cid, rewrite_result.get("rewritten", ""))
        if rewrite_result.get("reason"):
            self._work_memory.add_finding(
                scoped_cid,
                f"Query rewritten rationale: {rewrite_result['reason']}",
            )

        # Step 2: Schema 上下文构建（语义检索最相关的表结构）
        schema_ctx = await self._schema_builder.build(query)
        self._work_memory.set_schema_context(
            scoped_cid,
            schema_ctx,
            selected_tables=self._schema_builder.extract_table_names(schema_ctx),
        )

        # Step 3: RAG 检索（可选，失败时返回空列表，不中断主流程）
        rag_docs = await self._retrieve_rag_docs(
            query=rewrite_result.get("rewritten", query),
            scoped_conversation_id=scoped_cid,
        )

        # Step 4: 从 ConversationMemory 获取对话历史（多轮对话上下文）
        history = self._memory.get_messages(scoped_cid)

        # Step 5: 组装完整初始消息列表
        messages = self._prompt_builder.build(
            query=query,
            rag_docs=rag_docs,
            schema_context=schema_ctx,
            history=history,
            work_context=self._work_memory.build_prompt_context(scoped_cid),
        )

        return {
            "messages": messages,
            "schema_ctx": schema_ctx,
            "rewrite_result": rewrite_result,
            "rag_docs": rag_docs,
        }

    async def _node_planner(self, state: AgentState) -> dict[str, Any]:
        """
        planner 节点：调用 Planner 评估问题复杂度并生成执行计划。

        只在 _route_strategy 返回 "planner" 时执行（即配置开关打开且组件已注入）。
        Planner 的评估结果（plan.is_simple / plan.is_empty）由后续路由函数判断，
        本节点只负责调用 Planner 并保存结果。

        Planner 失败时（抛出异常）：
        - 异常传播到 run() 外层 try/except，返回 success=False 响应
        - 生产级实现可在此节点内捕获异常并返回 {"plan": None}，
          让路由函数降级为 ReAct 路径（更健壮）

        节点输出（状态更新字段）：
        - plan: Planner 生成的 Plan 对象（后续由 _route_after_plan 判断是否执行）
        """
        plan = await self._planner.plan(
            query=state["query"],
            available_tools=self._registry.list_names(),
            schema_context=state.get("schema_ctx", ""),
        )
        return {"plan": plan}

    async def _node_executor(self, state: AgentState) -> dict[str, Any]:
        """
        executor 节点：执行 Planner 生成的计划（按拓扑顺序并行执行各步骤）。

        只在 _route_after_plan 返回 "executor" 时执行
        （即 plan 不为空且不是 simple）。

        节点输出（状态更新字段）：
        - plan_steps: 执行完的 PlanStep 列表（每步包含 result/error/done 字段）
        """
        plan = state["plan"]
        schema_ctx = state.get("schema_ctx", "")
        steps = await self._executor.execute(plan, schema_context=schema_ctx)
        return {"plan_steps": steps}

    async def _node_build_planned_response(self, state: AgentState) -> dict[str, Any]:
        """
        build_planned_response 节点：从计划执行结果构建最终答案。

        等价于 AgentLoop._build_planned_response()：
        1. 遍历 PlanStep，收集 tool_calls_log、charts、latest_data
        2. 处理各步骤的副作用（图表追加、SQL 数据摘要、finding 记录）
        3. 调用 LLM 综合所有步骤的 evidence 生成"有据可查"的最终答案

        关键设计：tool_calls_log 和 charts 使用 operator.add reducer，
        本节点返回"增量"（本批次新增的记录），不返回全量累积列表。

        节点输出（状态更新字段）：
        - tool_calls_log: 本批次工具调用日志（增量，operator.add 追加）
        - charts: 本批次图表（增量，operator.add 追加）
        - latest_data: 最近 SQL 结果（覆盖）
        - final_answer: LLM 综合后的最终答案文本
        """
        steps = state.get("plan_steps", [])
        scoped_cid = state["scoped_conversation_id"]
        query = state["query"]

        # 遍历所有步骤，收集产物（new_tool_calls 和 new_charts 是"增量"）
        new_tool_calls: list[dict[str, Any]] = []
        new_charts: list[dict[str, Any]] = []
        latest_data: list[dict[str, Any]] = []

        for step in steps:
            success = bool(step.result and step.result.success)
            new_tool_calls.append({
                "tool": step.tool,
                "args": step.tool_params,
                "success": success,
            })
            if not step.done:
                continue
            if success and step.result:
                self._apply_tool_result_side_effects(
                    tool_name=step.tool,
                    tool_result=step.result,
                    scoped_conversation_id=scoped_cid,
                    charts=new_charts,
                )
                if step.tool == "sql_query":
                    latest_data = step.result.data or []
                self._work_memory.add_finding(
                    scoped_cid,
                    f"Plan step {step.step} [{step.tool}] succeeded.",
                )
            elif step.error:
                self._work_memory.add_finding(
                    scoped_cid,
                    f"Plan step {step.step} [{step.tool}] failed: {step.error[:200]}",
                )

        self._work_memory.add_finding(
            scoped_cid,
            f"Planner selected plan with {len(steps)} steps.",
        )
        self._work_memory.set_iterations(scoped_cid, len(steps))

        # 调用 LLM 综合所有步骤的 evidence 生成最终答案
        final_answer = await self._generate_grounded_final_answer(
            query=query,
            steps=steps,
        )

        return {
            "tool_calls_log": new_tool_calls,   # operator.add：追加到已有列表
            "charts": new_charts,               # operator.add：追加到已有列表
            "latest_data": latest_data,         # 覆盖：只保留最近一份
            "final_answer": final_answer,
        }

    async def _node_think(self, state: AgentState) -> dict[str, Any]:
        """
        think 节点：ReAct Think 步骤 — 调用 LLM 推理下一步行动。

        等价于原 while 循环中的 LLM 调用段：
            resp = await self._breaker.call(self._router.generate, ...)

        职责：
        1. iteration 计数 +1（+ 更新 WorkMemory 迭代状态）
        2. 调用 LLM（通过熔断器保护）
        3. 如果有工具调用，把包含 tool_calls 的 assistant 消息追加到 messages
           （OpenAI API 要求：tool 消息必须跟在含 tool_calls 的 assistant 消息后面）
        4. 保存 llm_resp 到状态（路由函数通过 llm_resp.tool_calls 决定下一步）

        注意：messages 使用覆盖 reducer，本节点返回完整更新后的列表。
        无工具调用时不追加 assistant 消息（与原代码一致），
        最终答案通过 ConversationMemory.add() 在 save_memory 节点写入。

        节点输出（状态更新字段）：
        - iteration: 递增后的迭代次数
        - llm_resp: 本次 LLM 响应对象（LLMResponse）
        - messages: （可选）追加了 assistant 消息后的完整列表（有工具调用时才更新）
        """
        messages = list(state.get("messages", []))
        iteration = state.get("iteration", 0) + 1
        scoped_cid = state["scoped_conversation_id"]

        # 更新工作记忆中的迭代计数
        self._work_memory.set_iterations(scoped_cid, iteration)
        logger.debug(
            "langgraph_agent_loop.think",
            n=iteration,
            conversation_id=state["conversation_id"],
        )

        # Think：调用 LLM（通过熔断器保护，防止 LLM 服务不稳定级联故障）
        tools_schema = self._registry.to_openai_tools()
        resp = await self._breaker.call(
            self._router.generate,
            messages=messages,
            task_type=TaskType.COMPLEX,
            tools=tools_schema,
            tool_choice="auto",  # "auto" = 让模型自己决定是否调用工具
        )

        updates: dict[str, Any] = {
            "iteration": iteration,
            "llm_resp": resp,
        }

        # 有工具调用时，把 assistant 消息追加到历史（OpenAI API 协议要求）
        if resp.tool_calls:
            messages.append(
                Message(
                    role="assistant",
                    content=resp.content or "",
                    tool_calls=resp.tool_calls,
                )
            )
            # 返回完整更新后的消息列表（覆盖语义）
            updates["messages"] = messages

        return updates

    async def _node_act_and_observe(self, state: AgentState) -> dict[str, Any]:
        """
        act_and_observe 节点：ReAct Act + Observe 步骤 — 执行所有工具调用。

        等价于原 while 循环中的 for tc in resp.tool_calls 段：
        遍历 LLM 请求的所有工具调用，依次执行（Act），
        并把每个结果追加到 messages（Observe）。

        关键设计决策（状态 Reducer）：
        - tool_calls_log 和 charts 使用 Annotated[list, operator.add] reducer
          → 本节点只返回"本次迭代新增的"记录（增量），不返回累积全量
          → LangGraph 自动将增量 extend 到跨迭代积累的列表
        - messages 使用覆盖 reducer
          → 返回包含所有新追加 tool role 消息的完整列表
        - latest_data 使用覆盖语义
          → 只保留最近一次 SQL 查询结果（与原代码一致）

        节点输出（状态更新字段）：
        - messages: 追加了所有 tool role 消息后的完整列表（覆盖）
        - tool_calls_log: 本次迭代新增的工具调用记录（增量，operator.add）
        - charts: 本次迭代新增的图表（增量，operator.add）
        - latest_data: 最近 SQL 查询结果（覆盖）
        """
        messages = list(state.get("messages", []))
        llm_resp = state["llm_resp"]
        iteration = state.get("iteration", 1)
        scoped_cid = state["scoped_conversation_id"]
        conversation_id = state["conversation_id"]

        # 本次迭代的增量（operator.add reducer 的输入）
        new_tool_calls: list[dict[str, Any]] = []
        new_charts: list[dict[str, Any]] = []
        latest_data: list[dict[str, Any]] = state.get("latest_data", [])

        for tc in llm_resp.tool_calls:
            tool_result = await self._execute_tool_call(
                tool_call=tc,
                conversation_id=conversation_id,
                scoped_conversation_id=scoped_cid,
                iteration=iteration,
                messages=messages,              # _execute_tool_call 会 in-place append
                tool_calls_log=new_tool_calls,  # 收集本次迭代的增量
                charts=new_charts,              # 收集本次迭代的增量
            )
            # 更新最近 SQL 结果（只保留最新一份，覆盖语义）
            if (
                tool_result is not None
                and tool_result.success
                and tc["function"]["name"] == "sql_query"
            ):
                latest_data = tool_result.data or []

        return {
            "messages": messages,               # 覆盖：完整列表（含新追加的 tool 消息）
            "tool_calls_log": new_tool_calls,   # 增量：operator.add 追加
            "charts": new_charts,               # 增量：operator.add 追加
            "latest_data": latest_data,         # 覆盖：只保留最近一份
        }

    async def _node_build_final_response(self, state: AgentState) -> dict[str, Any]:
        """
        build_final_response 节点：正常结束 — 模型不再请求工具时的收尾。

        等价于原代码中 if not resp.tool_calls: return _build_final_response(...) 的正常结束分支。
        本节点只负责记录日志和提取 final_answer，
        实际的 ConversationMemory 写回和 AgentResponse 构建统一在 save_memory 完成（DRY 原则）。

        节点输出（状态更新字段）：
        - final_answer: 模型的最终回答文本（save_memory 节点使用）
        """
        llm_resp = state["llm_resp"]
        final_answer = llm_resp.content or ""

        logger.info(
            "langgraph_agent_loop.final_answer",
            conversation_id=state["conversation_id"],
            iterations=state.get("iteration", 0),
        )

        return {"final_answer": final_answer}

    async def _node_force_summarize(self, state: AgentState) -> dict[str, Any]:
        """
        force_summarize 节点：达到最大迭代次数时的强制总结。

        等价于原代码的 _build_forced_final_response()：
        1. 追加一条 user 消息，要求模型基于已有 observations 给出总结
        2. 不提供工具（强制 LLM 输出文本，而非继续工具调用）
        3. 最后一次 LLM 调用生成强制总结答案

        为什么不直接截断？
        - 截断导致用户得到不完整的回答（如"已执行 5 步但没有结论"）
        - 通过最后一次 LLM 调用，可以基于已有 observations 合成有意义的答案
        - 虽然质量可能不如正常结束，但体验远好于截断

        节点输出（状态更新字段）：
        - final_answer: 基于已有 observations 强制合成的最终答案
        - messages: 追加了总结请求消息的完整列表（可选，供调试）
        """
        messages = list(state.get("messages", []))
        conversation_id = state["conversation_id"]
        iteration = state.get("iteration", settings.agent_max_iterations)

        logger.warning(
            "langgraph_agent_loop.max_iterations",
            conversation_id=conversation_id,
            iterations=iteration,
        )

        # 追加总结请求消息（不提供 tools，强制 LLM 输出文本而非继续工具调用）
        messages.append(
            Message(
                role="user",
                content="Please summarize what you've found so far and provide your best answer.",
            )
        )
        # 最后一次 LLM 调用（不传 tools 参数，强制文本输出）
        final_resp = await self._breaker.call(
            self._router.generate,
            messages=messages,
            task_type=TaskType.COMPLEX,
        )

        return {
            "final_answer": final_resp.content,
            "messages": messages,
        }

    async def _node_save_memory(self, state: AgentState) -> dict[str, Any]:
        """
        save_memory 节点：统一收尾 — 写 ConversationMemory，构建最终 AgentResponse。

        所有执行路径（ReAct 正常结束、ReAct 强制结束、Plan-and-Execute）
        都收敛到这个节点，确保收尾逻辑只有一份（DRY 原则）。

        完整收尾流程（等价于 AgentLoop._build_final_response 的收尾部分）：
        1. work_memory.complete_run()：标记工作状态为"已完成"
        2. 构建 ConversationMemory bridge metadata（run_id、tables、latest_sql 等）
        3. _extract_pinned_facts()：用 LLM 从本轮问答中提取长期记忆事实
        4. 写回 ConversationMemory（user 问题 + assistant 回答 + metadata）
        5. 构建并返回 AgentResponse

        节点输出（状态更新字段）：
        - response: 完整的 AgentResponse 对象（run() 从 final_state["response"] 读取）
        """
        query = state["query"]
        conversation_id = state["conversation_id"]
        scoped_cid = state["scoped_conversation_id"]
        final_answer = state.get("final_answer", "")
        iteration = state.get("iteration", 0)
        tool_calls_log = state.get("tool_calls_log", [])
        charts = state.get("charts", [])
        latest_data = state.get("latest_data", [])

        # 标记工作记忆为已完成状态
        self._work_memory.complete_run(scoped_cid, final_answer)

        # 构建 ConversationMemory 写回的 metadata（bridge 摘要 + pinned_facts）
        metadata = self._work_memory.build_conversation_bridge(scoped_cid)
        pinned_facts = await self._extract_pinned_facts(
            query=query,
            final_answer=final_answer,
            bridge_meta=metadata,
        )
        if pinned_facts:
            metadata["pinned_facts"] = pinned_facts

        # 写回对话历史（顺序：先 user，再 assistant）
        await self._memory.add(scoped_cid, "user", query)
        await self._memory.add(
            scoped_cid,
            "assistant",
            final_answer,
            metadata=metadata,
        )

        response = AgentResponse(
            answer=final_answer,
            conversation_id=conversation_id,
            iterations=iteration,
            tool_calls=tool_calls_log,
            charts=charts,
            data=latest_data,
            success=True,
        )

        return {"response": response}

    # ── 路由函数（条件边的决策函数，输入 AgentState，返回下一节点名称）────────

    def _route_strategy(self, state: AgentState) -> str:
        """
        prepare_context 后的策略路由：是否尝试 Plan-and-Execute。

        等价于原 _react_loop() 中 _try_plan_and_execute() 的前置检查：
            if not settings.agent_enable_planning: return None  → "think"
            if self._planner is None: return None              → "think"
            else: ...                                          → "planner"

        三个条件必须同时满足才走 Plan-and-Execute：
        1. settings.agent_enable_planning = True（配置总开关）
        2. self._planner 不为 None（组件已注入）
        3. self._executor 不为 None（组件已注入）

        Returns:
            "planner" — 尝试 Plan-and-Execute（让 planner 节点评估是否真的需要）
            "think"   — 直接走 ReAct 路径
        """
        if (
            settings.agent_enable_planning
            and self._planner is not None
            and self._executor is not None
        ):
            return "planner"
        return "think"

    def _route_after_plan(self, state: AgentState) -> str:
        """
        planner 节点后的路由：计划是否值得执行。

        等价于原 _try_plan_and_execute() 中：
            if plan.is_empty or plan.is_simple: return None  → "think"（退回 ReAct）
            else: ...                                        → "executor"（执行计划）

        plan 为 None 的情况（理论上不应发生，因为 _node_planner 会写入 plan，
        但作为防御性编程处理）：视为 is_empty，退回 ReAct。

        Returns:
            "executor" — 计划不为空且不是 simple，继续执行
            "think"    — 计划为空/simple，退回 ReAct 循环
        """
        plan = state.get("plan")
        if plan is None or plan.is_empty or plan.is_simple:
            return "think"
        return "executor"

    def _route_after_think(self, state: AgentState) -> str:
        """
        think 节点后的路由：ReAct 循环的三路分支决策。

        等价于原 while 循环中的两个分支点：
            if not resp.tool_calls: return _build_final_response(...)  → "build_final_response"
            ...（继续执行工具）
            while iteration < max_iterations: ...
            → 超出时到 _build_forced_final_response()              → "force_summarize"

        三路路由逻辑（优先级从高到低）：
        ① 无工具调用 → 正常结束（模型已给出最终答案，不需要执行工具）
        ② 有工具调用 + 已达迭代上限 → 强制结束（防止无限循环）
        ③ 有工具调用 + 未达上限 → 继续 ReAct 循环（执行工具）

        注意：原始代码中，"达到上限"发生在 while 条件检查处（迭代开始前）。
        LangGraph 版本中，think 节点已经将 iteration +1，
        所以 iteration >= max_iterations 表示当前这次 Think 已是最后一轮。

        Returns:
            "act_and_observe"    — 继续 ReAct 循环（有工具调用且未达上限）
            "build_final_response" — 正常结束（无工具调用）
            "force_summarize"    — 达到最大迭代次数（强制总结）
        """
        llm_resp = state.get("llm_resp")
        iteration = state.get("iteration", 0)

        # ① 无工具调用 → 模型已给出最终答案，正常结束
        if not llm_resp or not llm_resp.tool_calls:
            return "build_final_response"

        # ② 有工具调用但已达迭代上限 → 强制结束（不再执行工具）
        if iteration >= settings.agent_max_iterations:
            return "force_summarize"

        # ③ 有工具调用且未达上限 → 继续 Act + Observe
        return "act_and_observe"

    # ── 共享辅助方法（逻辑与 AgentLoop 完全一致）────────────────────────────────

    async def _retrieve_rag_docs(
        self,
        *,
        query: str,
        scoped_conversation_id: str,
    ) -> list[dict[str, Any]]:
        """
        执行可选的 RAG 知识库检索（与 AgentLoop._retrieve_rag_docs 逻辑完全一致）。

        RAG 是"可选增强"，失败时不中断主流程（内部吞掉异常，返回空列表）。
        如果注册表中没有 search_documents 工具，会在 registry.get() 时抛出 KeyError，
        被 except Exception 捕获，记录 finding 后返回空列表。

        Args:
            query: 改写后的查询（改写版本通常比原始查询检索效果更好）
            scoped_conversation_id: 加租户前缀的会话 ID

        Returns:
            检索到的文档列表（list[dict]），失败时返回空列表
        """
        if not self._registry.list_names():
            return []

        try:
            rag_tool = self._registry.get("search_documents")
            rag_result = await rag_tool.run(query=query)
            if rag_result.success and rag_result.data:
                self._work_memory.add_finding(
                    scoped_conversation_id,
                    f"Retrieved {len(rag_result.data)} relevant knowledge document(s).",
                )
                return rag_result.data
        except Exception as e:
            logger.debug("langgraph_agent_loop.rag_skip", error=str(e))
            self._work_memory.add_finding(
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
        执行单个工具调用（与 AgentLoop._execute_tool_call 逻辑完全一致）。

        Act + Observe 组合：
        1. 解析 tool arguments（JSON 字符串 → dict）
        2. 记录 WorkStep 开始（start_tool_step）
        3. 对 sql_query 提前写入 SQL 到 WorkMemory
        4. 执行工具（三种错误路径：工具不存在 / 执行异常 / 工具本身失败）
        5. 提取 observation 文本
        6. 应用副作用（图表收集、数据摘要更新、finding 记录）
        7. 更新 WorkStep 状态（finish_tool_step）
        8. 把工具结果追加到 messages（role="tool"，Observe 步骤）

        错误处理策略：
        - 任何工具错误都不中断 ReAct 循环
        - 错误信息作为 observation 追加到 messages
        - 模型看到错误 observation 后可以重试、换工具或直接给出答案

        注意：此方法直接修改 messages 列表（in-place append），
        调用方（_node_act_and_observe）持有引用，无需返回 messages。

        Args:
            tool_call: OpenAI function calling 格式的工具调用请求
            conversation_id: 原始会话 ID
            scoped_conversation_id: 加租户前缀的会话 ID
            iteration: 当前迭代次数
            messages: 消息历史（此方法会 in-place 追加 tool role 消息）
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
        work_step = self._work_memory.start_tool_step(
            conversation_id=scoped_conversation_id,
            iteration=iteration,
            tool=tool_name,
            args=tool_args,
        )

        logger.info(
            "langgraph_agent_loop.tool_call",
            tool=tool_name,
            args=str(tool_args)[:200],
            iteration=iteration,
        )

        # 对于 sql_query，提前写入 SQL（即使工具还未执行，快照中即可看到）
        if tool_name == "sql_query":
            sql = tool_args.get("sql")
            if isinstance(sql, str):
                self._work_memory.set_latest_sql(scoped_conversation_id, sql)

        # 执行工具（三种错误路径，都不中断循环）
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
            # 工具执行完成（成功或 tool_result.success=False 都走这里）
            observation = tool_result.to_observation()
            tool_calls_log.append(
                {"tool": tool_name, "args": tool_args, "success": tool_result.success}
            )
            self._apply_tool_result_side_effects(
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
        self._work_memory.finish_tool_step(
            scoped_conversation_id,
            work_step.step_id,
            success=bool(tool_result and tool_result.success),
            observation=observation,
            result_summary=result_summary,
            error="" if (tool_result and tool_result.success) else observation,
        )

        # 把工具结果追加到消息历史（Observe 步骤）
        # OpenAI API 要求：tool 消息的 tool_call_id 必须对应 assistant 消息中的 id
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
        scoped_conversation_id: str,
        charts: list[dict[str, Any]],
    ) -> None:
        """
        处理工具成功执行后的附带状态更新（与 AgentLoop._apply_tool_result_side_effects 完全一致）。

        各工具的副作用：
        - generate_chart：追加图表 JSON 到 charts，注册产物引用
        - sql_query：更新 latest_data_summary（行数/列名/首行预览），注册产物引用
        - 所有成功工具：如果有 text 摘要，追加一条 finding

        Args:
            tool_name: 工具名称
            tool_result: 工具执行结果（已确认 success=True）
            scoped_conversation_id: 加租户前缀的会话 ID
            charts: 图表产物收集列表（本方法可能追加）
        """
        if tool_name == "generate_chart" and tool_result.success:
            charts.append(tool_result.data)
            self._work_memory.add_artifact(
                scoped_conversation_id,
                artifact_type="chart",
                preview=tool_result.text,
                metadata={"tool": tool_name},
            )

        if tool_name == "sql_query" and tool_result.success:
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

        if tool_result.success and tool_result.text:
            self._work_memory.add_finding(
                scoped_conversation_id,
                f"{tool_name}: {tool_result.text[:300]}",
            )

    async def _generate_grounded_final_answer(
        self,
        *,
        query: str,
        steps: list,
    ) -> str:
        """
        基于计划步骤的执行证据生成"有据可查"的最终答案（与 AgentLoop 完全一致）。

        "Grounded"含义：答案中每条陈述都有对应的 evidence（来自具体工具执行结果），
        引用步骤编号（如 [Step 2]）让用户知道答案来源。

        两种路径：
        - settings.agent_force_grounded_answer=False：直接拼接前 3 步 evidence（简单）
        - settings.agent_force_grounded_answer=True：LLM 综合 evidence 生成结构化答案

        Args:
            query: 原始用户查询
            steps: 所有 PlanStep（包含成功和失败的）

        Returns:
            合成的最终答案字符串
        """
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

        if not settings.agent_force_grounded_answer:
            return "\n\n".join(evidence_lines[:3])

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
            logger.warning("langgraph_agent_loop.grounded_answer_failed", error=str(e))
            return "I could not synthesize a grounded final answer from the available evidence."

    async def _extract_pinned_facts(
        self,
        *,
        query: str,
        final_answer: str,
        bridge_meta: dict[str, Any],
    ) -> list[str]:
        """
        从本轮问答中提取值得长期固定的会话事实（与 AgentLoop._extract_pinned_facts 完全一致）。

        使用 TaskType.SIMPLE（快速模型）执行提取，节省 API 成本。
        失败时返回空列表，不中断主流程（pinned_facts 是可选的长期记忆增强）。

        Args:
            query: 原始用户查询
            final_answer: Agent 的最终回答（最多前 3000 字符）
            bridge_meta: work_memory 桥接摘要（提供上下文参考）

        Returns:
            提取的 pinned facts 列表（最多 5 条）；失败时返回空列表
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
            logger.debug("langgraph_agent_loop.pinned_facts_skip", error=str(e))
            return []

    @staticmethod
    def _parse_pinned_facts(content: str) -> list[str]:
        """
        宽松解析 LLM 返回的 pinned facts JSON 数组（与 AgentLoop._parse_pinned_facts 完全一致）。

        "宽松"的含义：
        - LLM 被要求只输出 JSON 数组，但实际可能包一层解释或 markdown 代码块
        - 先尝试直接解析，失败则尝试提取 [ ... ] 之间的 JSON 片段
        - 只接受 list[str]，拒绝其他类型
        - 对每条事实做去重和长度裁剪

        Args:
            content: LLM 返回的原始文本

        Returns:
            解析出的事实列表（最多 5 条，每条最多 120 字符）；失败时返回空列表
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
        宽松解析 LLM 生成的工具参数 JSON 字符串（与 AgentLoop._parse_tool_args 完全一致）。

        解析失败时返回空字典（{}）：
        - 工具收到缺少参数的调用会在工具层报错
        - 错误 observation 追加到消息历史，模型可根据错误重试
        - 比在这里做复杂 JSON 修复更简单可靠

        Args:
            tool_args_str: LLM 生成的 JSON 参数字符串

        Returns:
            解析出的参数字典；解析失败时返回空字典
        """
        try:
            return json.loads(tool_args_str)
        except json.JSONDecodeError:
            return {}

    # ── 可视化辅助方法（LangGraph 特有能力）──────────────────────────────────────

    def get_graph_mermaid(self) -> str:
        """
        获取图的 Mermaid 流程图字符串（LangGraph 特有能力）。

        可以将返回的字符串粘贴到 https://mermaid.live 可视化查看图的拓扑结构，
        或在 Jupyter Notebook 中使用 IPython.display 渲染：
            from IPython.display import Image
            Image(agent.get_graph_image())

        Returns:
            Mermaid 格式的图描述字符串
        """
        return self._graph.get_graph().draw_mermaid()
