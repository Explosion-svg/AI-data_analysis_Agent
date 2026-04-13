"""
assembler.py — 应用装配器（Composition Root）

职责：
  系统唯一的"连线"入口，负责按正确顺序创建、配置、组装所有组件。
  其他模块只依赖接口（BaseTool、BaseLLM…），不关心具体实现如何被创建。

设计原则：
  ┌─────────────────────────────────────────────────────┐
  │  main.py / lifespan  →  AppContainer.startup()      │
  │                          ↓                           │
  │  组件只从 AppContainer 获取依赖                      │
  │  层与层之间不直接 import 对方的实现类               │
  └─────────────────────────────────────────────────────┘

使用方式（在 main.py lifespan 中）：
    container = AppContainer()
    await container.startup()
    ...
    await container.shutdown()

或者直接通过全局单例：
    from ai_data_agent.assembler import get_container
    agent = get_container().agent_loop
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ai_data_agent.config.config import Settings, settings as _global_settings
from ai_data_agent.observability.logger import configure_logging, get_logger

if TYPE_CHECKING:
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
    应用容器，持有所有组件的唯一实例。
    通过 startup() / shutdown() 管理生命周期。

    层级装配顺序（从底到顶）：
        Config
          → Observability（日志/追踪/指标最先初始化，方便后续层记录日志）
          → Infra（DB / Warehouse / VectorStore）
          → Model Gateway（LLM 路由器）
          → Tools（SQL / Python / Chart / Schema / RAG）
          → Context（Prompt / Query Rewriter / Schema Context）
          → Memory（Conversation / Cache）
          → Reliability（熔断器等均为懒加载，无需显式初始化）
          → Orchestration（Planner / Executor / AgentLoop）
    """

    cfg: Settings = field(default_factory=lambda: _global_settings)

    # ── 组件（启动后填充）────────────────────────────────────────────────────
    # Infra
    db_engine: "AsyncEngine | None" = field(default=None, init=False)
    warehouse_engine: "AsyncEngine | None" = field(default=None, init=False)
    chroma_client: "chromadb.ClientAPI | None" = field(default=None, init=False)

    # Model Gateway
    router: "ModelRouter | None" = field(default=None, init=False)

    # Tools
    tool_registry: "ToolRegistry | None" = field(default=None, init=False)

    # Context
    prompt_builder: "PromptBuilder | None" = field(default=None, init=False)
    query_rewriter: "QueryRewriter | None" = field(default=None, init=False)
    schema_builder: "SchemaContextBuilder | None" = field(default=None, init=False)

    # Memory
    conversation_memory: "ConversationMemory | None" = field(default=None, init=False)
    cache: "CacheMemory | None" = field(default=None, init=False)
    work_memory: "WorkMemory | None" = field(default=None, init=False)

    # Orchestration
    planner: "Planner | None" = field(default=None, init=False)
    executor: "Executor | None" = field(default=None, init=False)
    agent_loop: "AgentLoop | None" = field(default=None, init=False)

    # State
    _started: bool = field(default=False, init=False)

    # ── 生命周期 ─────────────────────────────────────────────────────────────

    async def startup(self) -> None:
        """
        按依赖顺序初始化所有组件。
        幂等：多次调用只初始化一次。
        """
        if self._started:
            logger.debug("assembler.already_started")
            return

        logger.info("assembler.startup.begin", env=self.cfg.env.value)

        await self._init_observability()
        await self._init_infra()
        await self._init_model_gateway()
        await self._init_tools()
        await self._init_context()
        await self._init_memory()
        await self._init_orchestration()
        await self._post_startup()

        self._started = True
        logger.info("assembler.startup.done")

    async def shutdown(self) -> None:
        """释放所有资源。"""
        if not self._started:
            return
        logger.info("assembler.shutdown.begin")
        from ai_data_agent.infra import database, warehouse
        await database.close_db()
        await warehouse.close_warehouse()
        self._started = False
        logger.info("assembler.shutdown.done")

    # ── 私有初始化步骤 ────────────────────────────────────────────────────────

    async def _init_observability(self) -> None:
        """Step 1：日志 / Tracing / Metrics（最先，方便后续层打日志）"""
        configure_logging(
            json_logs=self.cfg.log_json,
            log_level=self.cfg.log_level.value,
        )
        from ai_data_agent.observability.tracer import init_tracer
        init_tracer()

        if self.cfg.enable_metrics:
            try:
                from prometheus_client import start_http_server
                start_http_server(self.cfg.metrics_port)
                logger.info("assembler.metrics_server", port=self.cfg.metrics_port)
            except OSError:
                # 端口已占用（多次 reload 时常见），忽略
                logger.debug("assembler.metrics_port_busy", port=self.cfg.metrics_port)
            except Exception as e:
                logger.warning("assembler.metrics_failed", error=str(e))

        logger.debug("assembler.observability_ready")

    async def _init_infra(self) -> None:
        """Step 2：基础设施 — DB / Warehouse / VectorStore"""
        from ai_data_agent.infra import database, warehouse, vector_store

        await database.init_db()
        self.db_engine = database.get_engine()

        await warehouse.init_warehouse()
        self.warehouse_engine = warehouse.get_warehouse_engine()

        await vector_store.init_vector_store()
        # chroma_client 直接从模块级单例读取（chromadb 不支持异步接口）
        from ai_data_agent.infra.vector_store import _client
        self.chroma_client = _client

        logger.debug("assembler.infra_ready")

    async def _init_model_gateway(self) -> None:
        """Step 3：Model Gateway — 注册 LLM 适配器"""
        from ai_data_agent.model_gateway.router import get_router
        self.router = get_router()
        logger.debug(
            "assembler.model_gateway_ready",
            models=self.router.list_models(),
        )

    async def _init_tools(self) -> None:
        """Step 4：Tool System — 创建并注册所有工具"""
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

        # 同步到全局单例（legacy 兼容）
        from ai_data_agent.tools import tool_registry as _tr_module
        _tr_module._registry = registry

        self.tool_registry = registry
        logger.debug(
            "assembler.tools_ready",
            tools=registry.list_names(),
        )

    async def _init_context(self) -> None:
        """Step 5：Context Management — Prompt / Rewriter / Schema"""
        from ai_data_agent.context.prompt_builder import PromptBuilder
        from ai_data_agent.context.query_rewriter import QueryRewriter
        from ai_data_agent.context.schema_context import SchemaContextBuilder

        self.prompt_builder = PromptBuilder()
        self.query_rewriter = QueryRewriter()
        self.schema_builder = SchemaContextBuilder()
        logger.debug("assembler.context_ready")

    async def _init_memory(self) -> None:
        """Step 6：Memory — 对话历史 + 缓存 + 工作记忆"""
        from ai_data_agent.memory.conversation_memory import ConversationMemory
        from ai_data_agent.memory.cache_memory import CacheMemory
        from ai_data_agent.memory.work_memory import WorkMemory
        from ai_data_agent.reliability.circuit_breaker import get_breaker

        assert self.router is not None

        self.conversation_memory = ConversationMemory(
            max_turns=self.cfg.conversation_max_turns,
            router=self.router,
            breaker=get_breaker("llm"),
        )
        self.cache = CacheMemory(
            max_size=self.cfg.cache_max_size,
            ttl_seconds=self.cfg.cache_ttl_seconds,
        )
        self.work_memory = WorkMemory()

        # 同步到全局单例
        from ai_data_agent.memory import conversation_memory as _cm_mod
        from ai_data_agent.memory import cache_memory as _cache_mod
        from ai_data_agent.memory import work_memory as _wm_mod
        _cm_mod._memory = self.conversation_memory
        _cache_mod._cache = self.cache
        _wm_mod._work_memory = self.work_memory

        logger.debug("assembler.memory_ready")

    async def _init_orchestration(self) -> None:
        """Step 7：Orchestration — Planner / Executor / AgentLoop"""
        from ai_data_agent.orchestration.planner import Planner
        from ai_data_agent.orchestration.executor import Executor
        from ai_data_agent.orchestration.agent_loop import AgentLoop
        from ai_data_agent.reliability.circuit_breaker import get_breaker

        self.planner = Planner()
        self.executor = Executor()
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
            breaker=get_breaker("llm"),
        )
        logger.debug("assembler.orchestration_ready")

    async def _post_startup(self) -> None:
        """Step 8：启动后任务 — Schema 向量化索引（可选，失败不阻断启动）"""
        try:
            assert self.schema_builder is not None
            await self.schema_builder.index_all_tables()
        except Exception as e:
            logger.warning("assembler.schema_index_failed", error=str(e))

    # ── 快捷访问（带断言，确保已启动）───────────────────────────────────────

    def get_agent_loop(self) -> "AgentLoop":
        assert self.agent_loop is not None, "Container not started. Call await startup() first."
        return self.agent_loop

    def get_tool_registry(self) -> "ToolRegistry":
        assert self.tool_registry is not None, "Container not started."
        return self.tool_registry

    def get_router(self) -> "ModelRouter":
        assert self.router is not None, "Container not started."
        return self.router

    def get_memory(self) -> "ConversationMemory":
        assert self.conversation_memory is not None, "Container not started."
        return self.conversation_memory

    def get_cache(self) -> "CacheMemory":
        assert self.cache is not None, "Container not started."
        return self.cache

    def get_work_memory(self) -> "WorkMemory":
        assert self.work_memory is not None, "Container not started."
        return self.work_memory

    # ── 诊断 ─────────────────────────────────────────────────────────────────

    def health_report(self) -> dict:
        """返回各组件健康状态，供运维巡检使用。"""

        return {
            "started": self._started,
            "infra": {
                "db": self.db_engine is not None,
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
        status = "started" if self._started else "not_started"
        return f"AppContainer(status={status}, env={self.cfg.env.value})"


# ── 全局单例 ──────────────────────────────────────────────────────────────────

_container: AppContainer | None = None


def get_container() -> AppContainer:
    """
    获取全局容器单例。
    在 main.py lifespan 中调用 startup() 后，全局可用。
    """
    global _container
    if _container is None:
        _container = AppContainer()
    return _container


async def startup() -> AppContainer:
    """便捷函数：获取并启动容器（供 main.py 使用）。"""
    container = get_container()
    await container.startup()
    return container


async def shutdown() -> None:
    """便捷函数：关闭容器（供 main.py 使用）。"""
    global _container
    if _container and _container._started:
        await _container.shutdown()
