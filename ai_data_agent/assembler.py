"""
assembler.py — 应用装配器（Composition Root）

职责：
  系统唯一的"连线"入口，负责按正确顺序创建、配置、组装所有组件。
  其他模块只依赖接口（BaseTool、BaseLLM…），不关心具体实现如何被创建。

设计原则（Composition Root 模式）：
  ┌─────────────────────────────────────────────────────┐
  │  main.py / lifespan  →  AppContainer.startup()      │
  │                          ↓                           │
  │  组件只从 AppContainer 获取依赖                      │
  │  层与层之间不直接 import 对方的实现类               │
  └─────────────────────────────────────────────────────┘

为什么要有 Composition Root？
  - 如果组件 A 直接 import 并实例化组件 B，那么组件 A 就与 B 的具体实现耦合了
  - 测试时无法轻松替换 B 为 MockB
  - 用 Composition Root 集中装配，所有组件都"等待被注入"，便于替换和测试

8层初始化顺序：
  Config（已通过 Pydantic Settings 完成）
    → Observability（日志/追踪/指标：最先初始化，后续所有层都需要日志）
    → Infra（Warehouse / VectorStore：基础设施层）
    → Model Gateway（LLM 路由器：需要配置才能初始化）
    → Tools（SQL / Python / Chart / Schema / RAG：需要 infra + router）
    → Context（Prompt / Query Rewriter / Schema Context：需要 tools + router）
    → Memory（Conversation / Cache / Work：需要 router + breaker）
    → Reliability（熔断器等均为懒加载，无需显式初始化）
    → Orchestration（Planner / Executor / AgentLoop：需要所有上层）

使用方式（在 main.py lifespan 中）：
    container = AppContainer()
    await container.startup()
    ...
    await container.shutdown()

或者直接通过全局单例（在已启动的上下文中）：
    from ai_data_agent.assembler import get_container
    agent = get_container().get_agent_loop()
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ai_data_agent.config.config import Settings, settings as _global_settings
from ai_data_agent.observability.logger import configure_logging, get_logger

if TYPE_CHECKING:
    # TYPE_CHECKING 块内的 import 只在静态类型检查时执行，运行时不执行。
    # 这样可以避免循环导入，同时让 IDE 能正确推断类型。
    from sqlalchemy.ext.asyncio import AsyncEngine
    import chromadb
    from ai_data_agent.model_gateway.router import ModelRouter
    from ai_data_agent.tools.tool_registry import ToolRegistry
    from ai_data_agent.memory.conversation_memory import ConversationMemory
    from ai_data_agent.memory.cache_memory import CacheMemory
    from ai_data_agent.memory.work_memory import WorkMemory
    from ai_data_agent.context.prompt_builder import PromptBuilder
    from ai_data_agent.context.query_rewriter import QueryRewriter
    from ai_data_agent.context.schema_context import SchemaContextBuilder
    from ai_data_agent.orchestration.planner import Planner
    from ai_data_agent.orchestration.executor import Executor
    from ai_data_agent.orchestration.agent_loop import AgentLoop
    from ai_data_agent.reliability.circuit_breaker import CircuitBreaker

logger = get_logger(__name__)


@dataclass
class AppContainer:
    """
    应用容器（Composition Root），持有所有组件的唯一实例。

    通过 startup() / shutdown() 管理整个系统的生命周期。

    为什么用 dataclass？
    - 所有组件字段一眼可见
    - 避免 __init__ 手动赋值
    - repr() 自动包含所有字段（但我们重写了 __repr__ 来控制输出）

    字段分组：
    - cfg: 应用配置（Settings 单例）
    - warehouse_engine / chroma_client: 基础设施
    - router: 模型路由器
    - tool_registry: 工具注册中心
    - prompt_builder / query_rewriter / schema_builder: 上下文构建
    - conversation_memory / cache / work_memory: 记忆系统
    - planner / executor / agent_loop: 编排层
    - _started: 启动状态标志（防止重复初始化）

    层级装配顺序（从底到顶）：
        Config
          → Observability（日志/追踪/指标最先初始化，方便后续层记录日志）
          → Infra（Warehouse / VectorStore）
          → Model Gateway（LLM 路由器）
          → Tools（SQL / Python / Chart / Schema / RAG）
          → Context（Prompt / Query Rewriter / Schema Context）
          → Memory（Conversation / Cache）
          → Reliability（熔断器等均为懒加载，无需显式初始化）
          → Orchestration（Planner / Executor / AgentLoop）
    """

    cfg: Settings = field(default_factory=lambda: _global_settings)

    # ── 组件（startup() 后填充）─────────────────────────────────────────────
    # 所有组件初始值为 None，startup() 调用后才真正赋值。
    # 使用 field(default=None, init=False) 表示不通过构造函数传入，由内部方法赋值。

    # Infra 层
    warehouse_engine: "AsyncEngine | None" = field(default=None, init=False)
    chroma_client: "chromadb.ClientAPI | None" = field(default=None, init=False)

    # Model Gateway 层
    router: "ModelRouter | None" = field(default=None, init=False)

    # Tools 层
    tool_registry: "ToolRegistry | None" = field(default=None, init=False)

    # Context 层
    prompt_builder: "PromptBuilder | None" = field(default=None, init=False)
    query_rewriter: "QueryRewriter | None" = field(default=None, init=False)
    schema_builder: "SchemaContextBuilder | None" = field(default=None, init=False)

    # Memory 层
    conversation_memory: "ConversationMemory | None" = field(default=None, init=False)
    cache: "CacheMemory | None" = field(default=None, init=False)
    work_memory: "WorkMemory | None" = field(default=None, init=False)

    # Orchestration 层（最顶层，依赖所有其他层）
    planner: "Planner | None" = field(default=None, init=False)
    executor: "Executor | None" = field(default=None, init=False)
    agent_loop: "AgentLoop | None" = field(default=None, init=False)

    # 启动状态标志
    _started: bool = field(default=False, init=False)

    # ── 生命周期管理 ─────────────────────────────────────────────────────────

    async def startup(self) -> None:
        """
        按依赖顺序初始化所有组件。

        幂等设计：多次调用只初始化一次，第二次调用直接返回。
        这样可以防止热重载（uvicorn --reload）或测试 setup 中多次调用导致资源重复初始化。

        初始化步骤（严格按依赖顺序）：
          1. _init_observability  — 日志 / 追踪 / 指标
          2. _init_infra          — DB / Warehouse / VectorStore
          3. _init_model_gateway  — LLM 路由器
          4. _init_tools          — 工具注册
          5. _init_context        — Prompt / Rewriter / Schema
          6. _init_memory         — 对话记忆 / 结果缓存 / 工作记忆
          7. _init_orchestration  — Planner / Executor / AgentLoop
          8. _post_startup        — Schema 向量化（可选，失败不阻断）
        """
        if self._started:
            logger.debug("assembler.already_started")
            return

        logger.info("assembler.startup.begin", env=self.cfg.env.value)

        try:
            await self._init_observability()
            await self._init_infra()
            await self._init_model_gateway()
            await self._init_tools()
            await self._init_context()
            await self._init_memory()
            await self._init_orchestration()
            await self._post_startup()
        except Exception:
            # P2-21：启动中途失败时无条件清理已初始化的资源。
            # 若此处不清理，K8s CrashLoop 重试会持续累积引擎/连接句柄。
            # 每个 closer 都是幂等的（未初始化的资源关闭是 no-op）。
            logger.exception("assembler.startup_failed.cleaning_up")
            await self._cleanup_partial_startup()
            raise

        self._started = True
        logger.info("assembler.startup.done")

    async def shutdown(self) -> None:
        """
        释放所有资源（数据库连接池、向量数据库、Redis、LLM 客户端等）。

        在 FastAPI lifespan 的 shutdown 阶段调用，对应 K8s 的 SIGTERM 信号处理。
        只有在 _started=True 时才执行清理，避免未完成初始化时执行无效清理。

        P2-20：原来只关 DB 和 warehouse（2/5），
        现在补齐 Chroma、Redis 客户端、LLM httpx 客户端，并重置模块级单例。
        每个组件独立 try/except，单个关闭失败不影响其余组件的清理。
        """
        if not self._started:
            return
        logger.info("assembler.shutdown.begin")
        for name, closer in self._closers():
            try:
                await closer()
            except Exception as e:
                logger.warning(f"assembler.shutdown_failed.{name}", error=str(e))
        self._reset_singletons()
        self._started = False
        logger.info("assembler.shutdown.done")

    def _closers(self) -> list[tuple[str, Any]]:
        """
        返回按初始化逆序（LIFO）排列的资源关闭器列表（P2-20）。

        顺序：Orchestration 资源最后初始化，最先关闭；
        基础设施（warehouse/chroma）最先初始化，最后关闭。
        """
        return [
            ("llm_clients", self._close_llm_clients),
            ("redis", self._close_redis),
            ("chroma", self._close_chroma),
            ("warehouse", self._close_warehouse),
            # P3-11：tracer 最后关闭，flush 并导出前面各组件 close 阶段
            # 产生的 span（BatchSpanProcessor 有约 5 秒缓冲，直接退出会丢）。
            ("tracer", self._close_tracer),
        ]

    async def _cleanup_partial_startup(self) -> None:
        """
        启动中途失败时的资源清理（P2-21）。

        _started 仍为 False，不会走正常 shutdown() 路径，
        因此单独实现：按 _closers() 顺序关闭所有资源（幂等），
        再重置模块级单例，避免留下半初始化的全局状态。
        """
        for name, closer in self._closers():
            try:
                await closer()
            except Exception as e:
                logger.warning(f"assembler.startup_cleanup_failed.{name}", error=str(e))
        self._reset_singletons()

    # ── 组件关闭器（幂等，可安全重复调用）────────────────────────────────

    async def _close_warehouse(self) -> None:
        """关闭 OLAP 数据仓库连接池（P2-20）。未初始化时是 no-op。"""
        from ai_data_agent.infra import warehouse
        await warehouse.close_warehouse()
        self.warehouse_engine = None

    async def _close_chroma(self) -> None:
        """关闭 ChromaDB 客户端，释放持久化目录文件锁（P2-20）。"""
        from ai_data_agent.infra import vector_store
        vector_store.close_vector_store()
        self.chroma_client = None

    async def _close_redis(self) -> None:
        """关闭 Redis 记忆/缓存客户端连接池（P2-20）。未初始化或非 Redis 后端时是 no-op。"""
        for name, mem in [
            ("conversation", self.conversation_memory),
            ("cache", self.cache),
            ("work", self.work_memory),
        ]:
            if mem is not None and hasattr(mem, "close"):
                try:
                    mem.close()
                except Exception as e:
                    logger.warning(f"assembler.redis_close_failed.{name}", error=str(e))

    async def _close_llm_clients(self) -> None:
        """关闭所有 LLM 适配器背后的 httpx 连接池（P2-20）。"""
        if self.router is not None and hasattr(self.router, "close"):
            await self.router.close()

    async def _close_tracer(self) -> None:
        """关闭并冲刷 OTel tracer（P3-11）。未初始化时是 no-op。"""
        from ai_data_agent.observability import tracer as tracer_mod
        tracer_mod.shutdown_tracer()

    def _reset_singletons(self) -> None:
        """
        重置模块级全局单例，避免热重载/重启后旧资源被引用（P2-20）。

        与 tests/conftest.py 的 reset_singletons fixture 保持一致。
        warehouse/vector_store 的单例已由各自 close 函数重置。
        """
        from ai_data_agent.memory import cache_memory, conversation_memory, work_memory
        from ai_data_agent.model_gateway import router as router_mod
        from ai_data_agent.tools import tool_registry as tool_registry_mod

        router_mod._router = None
        conversation_memory._memory = None
        cache_memory._cache = None
        work_memory._work_memory = None
        tool_registry_mod._registry = None

        self.router = None
        self.tool_registry = None
        self.prompt_builder = None
        self.query_rewriter = None
        self.schema_builder = None
        self.conversation_memory = None
        self.cache = None
        self.work_memory = None
        self.planner = None
        self.executor = None
        self.agent_loop = None

    # ── 私有初始化步骤（严格按依赖顺序）────────────────────────────────────

    async def _init_observability(self) -> None:
        """
        Step 1：初始化可观测性基础设施（日志/追踪/指标）。

        为什么最先初始化？
        - 后续所有层在初始化期间也需要记录日志
        - 如果日志系统后初始化，早期启动日志会丢失或格式不一致

        初始化内容：
        - structlog 配置（JSON vs 彩色文本）
        - OpenTelemetry tracer（如果 enable_tracing=True）
        - Prometheus metrics HTTP server（如果 enable_metrics=True）
        """
        configure_logging(
            json_logs=self.cfg.log_json,
            log_level=self.cfg.log_level.value,     # 枚举成员 → 字符串
        )
        # 日志配置完成后重新获取 logger，确保使用已配置的正式 logger
        configured_logger = get_logger(__name__)
        from ai_data_agent.observability.tracer import init_tracer
        init_tracer()

        if self.cfg.enable_metrics:
            try:
                self._start_metrics_server()
                configured_logger.info("assembler.metrics_server", port=self.cfg.metrics_port)
            except OSError as e:
                # P2-22：端口冲突升级为 WARNING。
                # 多 worker 模式下只有一个进程能成功绑定 metrics 端口，
                # 其余 worker 的本地指标由抓取端通过 multiprocess 模式聚合，
                # 因此端口绑定失败不阻断主服务，但要明确告警便于排查。
                configured_logger.warning(
                    "assembler.metrics_port_busy",
                    port=self.cfg.metrics_port,
                    error=str(e),
                )
            except Exception as e:
                configured_logger.warning("assembler.metrics_failed", error=str(e))

        configured_logger.debug("assembler.observability_ready")

    def _start_metrics_server(self) -> None:
        """
        启动 Prometheus 指标 HTTP 服务（P2-22）。

        多 worker（已在 main.py 设置 PROMETHEUS_MULTIPROC_DIR）时，
        用 MultiProcessCollector 聚合所有 worker 写入的指标文件；
        单 worker 走默认 registry。

        start_wsgi_server 同步绑定端口，端口冲突会立即抛 OSError，
        由调用方记录 WARNING（多 worker 下只有一个进程能绑定成功）。
        """
        from prometheus_client import REGISTRY, CollectorRegistry
        from prometheus_client.exposition import start_wsgi_server
        from prometheus_client.multiprocess import MultiProcessCollector

        if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
            # 多 worker：各 worker 指标写入 mmap 文件，
            # 用 MultiProcessCollector 聚合（文件来自 main.py 创建的临时目录）。
            registry = CollectorRegistry()
            MultiProcessCollector(registry)
        else:
            # 单 worker：应用指标（AgentMetrics）已注册到默认 registry。
            registry = REGISTRY
        start_wsgi_server(
            port=self.cfg.metrics_port,
            addr="0.0.0.0",
            registry=registry,
        )

    async def _init_infra(self) -> None:
        """
        Step 2：初始化基础设施层（数据仓库 / 向量数据库）。

        两个基础设施是完全独立的，理论上可以并行初始化，
        但目前保持串行是为了让日志顺序可预测、问题好排查。

        初始化后将引擎句柄保存到容器字段，
        方便后续 health_report() 检查各组件状态。
        """
        from ai_data_agent.infra import warehouse, vector_store

        await warehouse.init_warehouse()
        self.warehouse_engine = warehouse.get_warehouse_engine()

        await vector_store.init_vector_store()
        # chroma_client 直接从模块级单例读取（chromadb 目前不支持异步接口）
        from ai_data_agent.infra.vector_store import _client
        self.chroma_client = _client

        logger.debug("assembler.infra_ready")

    async def _init_model_gateway(self) -> None:
        """
        Step 3：初始化 Model Gateway（LLM 路由器）。

        router 使用延迟初始化（在 get_router() 内）并以模块级单例维护。
        这里把它注册到容器，方便后续通过 get_router() 获取。

        注意：router 初始化会尝试连接各 LLM API（如果配置了的话），
        但不会发送真实请求，只是建立 AsyncOpenAI client 连接对象。
        """
        from ai_data_agent.model_gateway.router import get_router
        self.router = get_router()
        logger.debug(
            "assembler.model_gateway_ready",
            models=self.router.list_models(),
        )

    async def _init_tools(self) -> None:
        """
        Step 4：创建并注册所有 Agent 工具。

        工具实例化比较轻量（只是创建对象，不做 IO）。
        register() 返回 self 支持链式调用，这里保持显式调用更清晰。

        同步到全局单例（_tr_module._registry = registry）是一个临时兼容措施，
        允许通过 get_registry() 函数访问已装配的注册中心。
        后续重构时可以移除这个全局单例，改为纯依赖注入。
        """
        from ai_data_agent.tools.tool_registry import ToolRegistry
        from ai_data_agent.tools.sql_tool import SQLTool
        from ai_data_agent.tools.python_tool import PythonTool
        from ai_data_agent.tools.chart_tool import ChartTool
        from ai_data_agent.tools.schema_tool import SchemaTool
        from ai_data_agent.tools.rag_tool import RAGTool

        registry = ToolRegistry()
        registry.register(SQLTool())
        registry.register(PythonTool())
        registry.register(ChartTool())
        registry.register(SchemaTool())
        registry.register(RAGTool())

        # 同步到全局单例，兼容直接调用 get_registry() 的代码
        from ai_data_agent.tools import tool_registry as _tr_module
        _tr_module._registry = registry

        self.tool_registry = registry
        logger.debug(
            "assembler.tools_ready",
            tools=registry.list_names(),
        )

    async def _init_context(self) -> None:
        """
        Step 5：初始化上下文构建层。

        三个上下文组件都是无状态的纯对象（只依赖全局单例或配置），
        因此直接实例化即可，不需要异步操作。

        - PromptBuilder：把多种上下文来源组装成 LLM 消息列表
        - QueryRewriter：扩展用户问题，提升 RAG 召回率
        - SchemaContextBuilder：动态选择与问题相关的表 schema
        """
        from ai_data_agent.context.prompt_builder import PromptBuilder
        from ai_data_agent.context.query_rewriter import QueryRewriter
        from ai_data_agent.context.schema_context import SchemaContextBuilder

        self.prompt_builder = PromptBuilder()
        # P3-5：QueryRewriter 注入已装配的 router（带熔断保护），
        # 不再通过内部 get_router() 打全局单例。
        self.query_rewriter = QueryRewriter(router=self.router)
        self.schema_builder = SchemaContextBuilder()
        logger.debug("assembler.context_ready")

    async def _init_memory(self) -> None:
        """
        Step 6：初始化记忆系统（对话历史 / 结果缓存 / 工作记忆）。

        通过 factory.py 根据配置（memory_backend / cache_backend）决定使用内存版还是 Redis 版：
        - memory_backend="memory"  → ConversationMemory（内存）
        - memory_backend="redis"   → RedisConversationMemory（Redis + 内存降级）

        LLM router 和 circuit breaker 注入给 ConversationMemory，
        是因为 conversation_memory 在生成滚动摘要时需要调用 LLM，
        且需要熔断器保护避免 LLM 故障影响对话流程。

        同步到全局单例，兼容通过 get_memory() / get_cache() / get_work_memory() 访问。
        """
        from ai_data_agent.memory.factory import (
            build_cache_memory,
            build_conversation_memory,
            build_work_memory,
        )
        from ai_data_agent.reliability.circuit_breaker import get_breaker

        assert self.router is not None

        self.conversation_memory = build_conversation_memory(
            router=self.router,
            breaker=get_breaker("llm"),
        )
        self.cache = build_cache_memory()
        self.work_memory = build_work_memory()

        # 同步到模块级全局单例（兼容 get_memory() 等访问方式）
        from ai_data_agent.memory import conversation_memory as _cm_mod
        from ai_data_agent.memory import cache_memory as _cache_mod
        from ai_data_agent.memory import work_memory as _wm_mod
        _cm_mod._memory = self.conversation_memory
        _cache_mod._cache = self.cache
        _wm_mod._work_memory = self.work_memory

        logger.debug("assembler.memory_ready")

    async def _init_orchestration(self) -> None:
        """
        Step 7：初始化编排层（Planner / Executor / AgentLoop）。

        编排层是系统的顶层，依赖所有其他层的组件。
        AgentLoop 的构造函数接受所有依赖的显式注入，这正是 Composition Root 的价值所在：
        - 所有依赖清晰可见
        - 测试时可以轻松替换任意组件为 Mock
        - 不需要从容器内部再次查询依赖

        assert 断言确保在依赖不齐全时快速失败，避免后续运行时才出现 NullPointerError。
        """
        from ai_data_agent.orchestration.planner import Planner
        from ai_data_agent.orchestration.executor import Executor
        from ai_data_agent.orchestration.agent_loop import AgentLoop
        from ai_data_agent.reliability.circuit_breaker import get_breaker

        # P3-5：Planner/Executor 注入已装配的 router（带熔断保护），
        # 不再通过内部 get_router() 打全局单例。
        self.planner = Planner(router=self.router)
        self.executor = Executor(router=self.router)

        # 断言检查确保所有依赖已在前序步骤中初始化
        assert self.prompt_builder is not None
        assert self.query_rewriter is not None
        assert self.schema_builder is not None
        assert self.conversation_memory is not None
        assert self.cache is not None
        assert self.work_memory is not None
        assert self.tool_registry is not None
        assert self.router is not None

        self.agent_loop = AgentLoop(
            prompt_builder=self.prompt_builder,
            query_rewriter=self.query_rewriter,
            schema_builder=self.schema_builder,
            memory=self.conversation_memory,
            cache=self.cache,
            work_memory=self.work_memory,
            registry=self.tool_registry,
            router=self.router,
            breaker=get_breaker("llm"),   # LLM 专属熔断器
            planner=self.planner,
            executor=self.executor,
        )
        logger.debug("assembler.orchestration_ready")

    async def _post_startup(self) -> None:
        """
        Step 8：启动后可选任务 — 将所有表 schema 向量化并存入 ChromaDB。

        为什么是可选的（try/except 吞异常）？
        - 如果数据仓库暂时不可用，不应阻断整个应用启动
        - Schema 索引是一个增强特性，主流程有降级方案（全量 schema 或关键词匹配）
        - 后续也可以通过管理接口手动触发重新索引

        索引完成后，SchemaContextBuilder.build() 可以通过向量相似度高效选表，
        而不是每次都把全部表 schema 塞入 prompt（节省 token）。
        """
        try:
            assert self.schema_builder is not None
            await self.schema_builder.index_all_tables()
        except Exception as e:
            logger.warning("assembler.schema_index_failed", error=str(e))

    # ── 快捷访问（带断言，确保容器已启动）───────────────────────────────────

    def get_agent_loop(self) -> "AgentLoop":
        """
        获取 AgentLoop 实例。

        断言确保在容器未启动时快速失败，提供明确的错误信息。
        调用方应先确保 startup() 已完成（通常由 main.py lifespan 保证）。
        """
        assert self.agent_loop is not None, "Container not started. Call await startup() first."
        return self.agent_loop

    def get_logger(self, name: str) -> Any:
        """
        获取指定名称的 structlog logger。

        通过容器获取 logger 而不是直接调用 get_logger()，
        是为了保持容器作为唯一访问入口的一致性。
        """
        return get_logger(name)

    def span(self, name: str, attributes: dict[str, Any] | None = None) -> Any:
        """
        创建一个 OpenTelemetry tracing span（contextmanager）。

        如果 tracing 未启用，返回 NoOp contextmanager，不影响业务逻辑。
        使用方式：
            with container.span("my.operation", {"key": "value"}):
                ...
        """
        from ai_data_agent.observability.tracer import span as tracer_span
        return tracer_span(name, attributes)

    def build_request_context(
        self,
        *,
        request_id: str,
        user_id: str,
        tenant_id: str,
    ) -> Any:
        """
        构造请求上下文对象（RequestContext）。

        RequestContext 是一个 frozen dataclass，包含一次 HTTP 请求的身份信息。
        通过容器构造而不是直接 import，方便在测试中替换为 Mock 版本。

        Args:
            request_id: 请求唯一标识，用于全链路日志追踪
            user_id: 业务用户标识，用于审计
            tenant_id: 租户标识，用于数据隔离
        """
        from ai_data_agent.context.request_context import RequestContext
        return RequestContext(
            request_id=request_id,
            user_id=user_id,
            tenant_id=tenant_id,
        )

    def set_request_context(self, ctx: Any) -> Any:
        """
        将 RequestContext 绑定到当前 asyncio 协程（通过 contextvars）。

        返回 Token，调用方必须在请求结束时调用 clear_request_context(token)，
        否则同一 asyncio Task 复用时会读到上一个请求的上下文。

        Returns:
            contextvars.Token，用于后续清理
        """
        from ai_data_agent.context.request_context import set_request_context
        return set_request_context(ctx)

    def clear_request_context(self, token: Any | None = None) -> None:
        """
        清理当前协程的请求上下文。

        传入 token 时使用 reset() 恢复旧值（正确处理嵌套调用）；
        不传 token 时直接置 None。

        Args:
            token: set_request_context() 返回的 Token
        """
        from ai_data_agent.context.request_context import clear_request_context
        clear_request_context(token)

    def get_tool_registry(self) -> "ToolRegistry":
        """获取工具注册中心，断言确保容器已启动。"""
        assert self.tool_registry is not None, "Container not started."
        return self.tool_registry

    def get_router(self) -> "ModelRouter":
        """获取 LLM 路由器，断言确保容器已启动。"""
        assert self.router is not None, "Container not started."
        return self.router

    def get_memory(self) -> "ConversationMemory":
        """获取对话记忆实例，断言确保容器已启动。"""
        assert self.conversation_memory is not None, "Container not started."
        return self.conversation_memory

    def get_cache(self) -> "CacheMemory":
        """获取结果缓存实例，断言确保容器已启动。"""
        assert self.cache is not None, "Container not started."
        return self.cache

    def get_work_memory(self) -> "WorkMemory":
        """获取工作记忆实例，断言确保容器已启动。"""
        assert self.work_memory is not None, "Container not started."
        return self.work_memory

    # ── 诊断 / 监控 ──────────────────────────────────────────────────────────

    def health_report(self) -> dict:
        """
        返回各组件健康状态快照，供运维巡检和 K8s readinessProbe 使用。

        注意：
        - 只检查组件是否被成功初始化（not None），不做真实 ping 检查
        - 真实连通性检查应由各组件自己的 health_check() 方法负责
        - 这里的 report 适合记录到启动日志，方便快速确认所有层都正常就绪

        Returns:
            包含所有层组件状态的嵌套字典
        """
        return {
            "started": self._started,
            "infra": {
                "warehouse": self.warehouse_engine is not None,
                "vector_store": self.chroma_client is not None,
            },
            "model_gateway": {
                "ready": self.router is not None,
                "models": self.router.list_models() if self.router else [],
            },
            "tools": {
                "ready": self.tool_registry is not None,
                "registered": self.tool_registry.list_names() if self.tool_registry else [],
            },
            "memory": {
                "conversation": self.conversation_memory is not None,
                "cache": self.cache.stats() if self.cache else None,
                "work_memory": self.work_memory.stats() if self.work_memory else None,
            },
            "orchestration": {
                "planner": self.planner is not None,
                "executor": self.executor is not None,
                "agent_loop": self.agent_loop is not None,
            },
        }

    def __repr__(self) -> str:
        """
        自定义 repr，显示启动状态而不是内存地址。

        dataclass 默认的 repr 会把所有字段都打印出来（包括大量 None），
        这里重写只显示关键的状态信息，让调试日志更易读。
        """
        status = "started" if self._started else "not_started"
        return f"AppContainer(status={status}, env={self.cfg.env.value})"


# ── 全局单例 ──────────────────────────────────────────────────────────────────

_container: AppContainer | None = None


def get_container() -> AppContainer:
    """
    获取全局容器单例。

    这是应用中获取各组件的统一入口。
    注意：容器必须在使用前通过 startup() 初始化，否则各组件字段均为 None。
    在 main.py 的 lifespan 中调用 startup() 后，全局可用。

    线程/协程安全性：
    - 容器创建是同步的，Python GIL 保证了简单赋值的原子性
    - 组件初始化在单个 asyncio 事件循环内串行执行，无并发问题

    Returns:
        全局唯一的 AppContainer 实例
    """
    global _container
    if _container is None:
        _container = AppContainer()
    return _container


async def startup() -> AppContainer:
    """
    便捷函数：获取并启动全局容器。

    在 main.py 的 lifespan 函数中调用：
        container = await startup()

    Returns:
        已启动的 AppContainer 实例
    """
    container = get_container()
    await container.startup()
    return container


async def shutdown() -> None:
    """
    便捷函数：关闭全局容器，释放所有资源。

    在 main.py 的 lifespan 函数中调用：
        await shutdown()

    只有在容器已启动（_started=True）时才执行清理，防止未启动时误调用。
    """
    global _container
    if _container and _container._started:
        await _container.shutdown()
