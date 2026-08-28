"""
config/config.py — 全局配置管理

职责：
  集中管理整个应用的所有配置项，通过 Pydantic Settings 实现：
  1. 类型安全：所有配置项都有明确类型，错误值在启动时即报错
  2. 多来源：支持 .env 文件、环境变量，优先级：环境变量 > .env 文件 > 默认值
  3. 单例：通过 @lru_cache 保证全局只有一份配置实例

使用示例：
  from ai_data_agent.config.config import settings
  print(settings.openai_api_key)

环境变量映射（case_insensitive=True）：
  DATABASE_URL=... 对应 settings.database_url
  OPENAI_API_KEY=... 对应 settings.openai_api_key
"""
from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Env(str, Enum):
    """运行环境枚举。"""
    dev = "dev"
    staging = "staging"
    prod = "prod"


class LogLevel(str, Enum):
    """日志级别枚举，对应 Python logging 模块的级别名称。"""
    debug = "DEBUG"
    info = "INFO"
    warning = "WARNING"
    error = "ERROR"


class Settings(BaseSettings):
    """
    应用全局配置。

    所有字段均有默认值，可通过 .env 文件或环境变量覆盖。
    字段按功能分组，每组用注释标注。

    Pydantic Settings 工作原理：
      1. 先读取 .env 文件（env_file=".env"）
      2. 再读取操作系统环境变量（优先级更高）
      3. 最后使用字段默认值
      4. 对所有值做类型转换和校验
    """
    model_config = SettingsConfigDict(
        env_file=".env",                # .env 文件路径（相对于运行目录）
        env_file_encoding="utf-8",
        case_sensitive=False,           # 环境变量名大小写不敏感
        extra="ignore",                 # 忽略 .env 中多余的字段，避免启动失败
    )

    # ── 应用基础信息 ─────────────────────────────────────────────────────────────
    app_name: str = "AI Data Analysis Agent"
    app_version: str = "1.0.0"
    env: Env = Env.dev                  # 当前运行环境，影响日志格式、文档可见性等
    debug: bool = False                 # True 时开启 uvicorn 热重载

    # ── API 服务器 ───────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"              # 监听地址，生产环境建议改为具体 IP
    port: int = 8000
    workers: int = 1                   # uvicorn worker 数量，生产环境建议 CPU 核数 * 2 + 1
    api_key: Optional[str] = None      # 保护 /chat 端点的 Bearer Token，None 表示不鉴权
    default_user_id: str = "anonymous"
    default_tenant_id: str = "public"

    # ── OLTP 数据库（事务型，SQLAlchemy async）─────────────────────────────────
    # 支持 sqlite+aiosqlite / postgresql+asyncpg / mysql+aiomysql
    database_url: str = "postgresql+asyncpg://user:password@localhost:5432/agent"
    db_pool_size: int = 10             # 连接池基础连接数
    db_max_overflow: int = 20          # 超过 pool_size 后最多额外创建的连接数
    db_echo: bool = False              # True 时打印所有 SQL（仅开发调试）

    # ── OLAP 数据仓库（分析型）──────────────────────────────────────────────────
    # 与 OLTP 分开，避免长时间分析查询耗尽 OLTP 连接池
    # 支持 sqlite / postgresql / clickhouse 等
    warehouse_url: str = "postgresql+asyncpg://user:password@localhost:5432/warehouse"

    # ── 向量数据库（ChromaDB）──────────────────────────────────────────────────
    vector_store_type: str = "chroma"              # 当前只支持 chroma
    chroma_persist_dir: str = "./data/chroma"      # ChromaDB 持久化目录
    chroma_docs_collection: str = "docs"           # RAG 文档 collection 名称
    chroma_schema_collection: str = "schema"       # 表结构语义 collection 名称

    # ── LLM / Model Gateway ──────────────────────────────────────────────────────
    # 支持同时配置多个 LLM，路由器会按优先级选择并在失败时 fallback
    openai_api_key: Optional[str] = None
    openai_api_base: str = "https://api.openai.com/v1"
    openai_default_model: str = "gpt-4o"           # 复杂任务使用
    openai_fast_model: str = "gpt-4o-mini"         # 简单任务/摘要使用，节省成本

    deepseek_api_key: Optional[str] = None
    deepseek_api_base: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"

    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-sonnet-4-6"

    local_llm_api_base: Optional[str] = None       # 本地 LLM 地址，如 Ollama: http://localhost:11434/v1
    local_llm_model: Optional[str] = None

    llm_temperature: float = 0.0                   # 0.0 = 确定性输出（数据分析推荐）
    llm_max_tokens: int = 4096
    llm_timeout: float = 60.0                      # 单次 LLM 调用超时（秒）
    llm_max_retries: int = 3

    # ── Embedding 模型 ────────────────────────────────────────────────────────────
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536                # 向量维度，需与 ChromaDB collection 一致

    # ── 可靠性参数 ────────────────────────────────────────────────────────────────
    sql_query_timeout: float = 30.0                # SQL 执行超时（秒）
    python_exec_timeout: float = 20.0              # Python 代码执行超时（秒）
    agent_max_iterations: int = 10                 # ReAct 循环最大轮次，防止死循环

    # 熔断器配置
    circuit_breaker_failure_threshold: int = 5    # 连续失败 N 次后熔断
    circuit_breaker_recovery_timeout: float = 60.0  # 熔断后 N 秒尝试恢复

    # 重试配置（指数退避）
    retry_max_attempts: int = 3
    retry_base_delay: float = 1.0                  # 首次重试等待基础秒数
    retry_max_delay: float = 30.0                  # 最大等待秒数上限

    # 并发限流（信号量）- 各资源独立隔离，防止单一依赖拖垮全系统
    agent_request_concurrency: int = 64           # 最大同时处理的 Agent 请求数
    llm_concurrency: int = 16                     # 同时进行的 LLM 调用数
    embedding_concurrency: int = 32
    tool_concurrency: int = 32                    # 工具调用总并发
    sql_tool_concurrency: int = 16                # SQL 工具专属并发（保护 DB 连接池）
    python_tool_concurrency: int = 8              # Python 沙盒并发（CPU 密集，限制小）
    rag_tool_concurrency: int = 16
    chart_tool_concurrency: int = 8
    schema_tool_concurrency: int = 16
    concurrency_acquire_timeout_seconds: float = 1.0  # 获取信号量的超时（秒），超时返回 503

    # Planner / Executor 配置
    agent_enable_planning: bool = True             # 是否启用 Plan-and-Execute 模式
    executor_max_parallel_steps: int = 3          # Executor 最大并行步骤数
    agent_force_grounded_answer: bool = True      # 是否强制 LLM 仅基于工具证据回答

    # ── 记忆系统 ─────────────────────────────────────────────────────────────────
    conversation_max_turns: int = 20              # 对话记忆保留最近 N 轮原始对话
    cache_ttl_seconds: int = 300                  # 结果缓存生存时间（秒）
    cache_max_size: int = 256                     # 内存缓存最大条目数（LRU 淘汰）

    memory_backend: str = "memory"               # 对话/工作记忆后端：memory | redis
    cache_backend: str = "memory"                # 结果缓存后端：memory | redis

    # Redis 连接参数（memory_backend="redis" 时生效）
    redis_url: str = "redis://localhost:6379/0"
    redis_cache_prefix: str = "ai_data_agent:cache"
    redis_socket_timeout: float = 2.0
    redis_connect_timeout: float = 2.0
    redis_health_check_interval: int = 30
    redis_retry_on_timeout: bool = True
    redis_cache_fail_open: bool = True            # Redis 故障时降级到无缓存而非报错
    redis_cache_startup_check: bool = True        # 启动时检查 Redis 可用性
    redis_optimistic_lock_retries: int = 3        # 乐观锁并发写冲突最大重试次数
    redis_work_prefix: str = "ai_data_agent:work"
    redis_work_ttl_seconds: int = 86400           # 工作记忆 TTL：1天
    redis_conversation_prefix: str = "ai_data_agent:conversation"
    redis_conversation_ttl_seconds: int = 604800  # 对话记忆 TTL：7天

    # SQL 表白名单（空列表表示不限制）
    sql_allowed_tables: list[str] = Field(default_factory=list)

    # ── 可观测性 ─────────────────────────────────────────────────────────────────
    log_level: LogLevel = LogLevel.info
    log_json: bool = True                         # True = JSON 格式（生产）；False = 彩色文本（开发）
    enable_tracing: bool = False                  # 是否启用 OpenTelemetry 分布式追踪
    otlp_endpoint: Optional[str] = None          # OTLP 收集器地址，如 http://jaeger:4317
    enable_metrics: bool = True                  # 是否启动 Prometheus metrics HTTP server
    metrics_port: int = 9090                     # Prometheus metrics 端口

    # ── 安全设置 ─────────────────────────────────────────────────────────────────
    sql_readonly: bool = True                    # 只允许 SELECT，禁止 DDL/DML（生产必须为 True）
    python_sandbox: bool = True                  # 在沙盒中执行 Python 代码

    # ── 字段验证器 ────────────────────────────────────────────────────────────────

    @field_validator("llm_temperature")
    @classmethod
    def _check_temperature(cls, v: float) -> float:
        """
        验证 temperature 必须在合法范围 [0.0, 2.0] 内。

        OpenAI 的 temperature 范围是 0.0（确定性）到 2.0（极随机）。
        数据分析场景推荐 0.0，避免 LLM 产生幻觉。
        """
        if not 0.0 <= v <= 2.0:
            raise ValueError("temperature must be in [0.0, 2.0]")
        return v

    @field_validator("sql_allowed_tables", mode="before")
    @classmethod
    def _normalize_sql_allowed_tables(cls, v: object) -> object:
        """
        允许通过逗号分隔字符串配置 SQL 白名单表。

        环境变量只能传字符串，但配置字段类型是 list[str]。
        这里做归一化，让两种格式都能正常工作：
          SQL_ALLOWED_TABLES="orders,products,users"  →  ["orders", "products", "users"]
          SQL_ALLOWED_TABLES=["orders", "products"]   →  ["orders", "products"]（直接传列表）
        """
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    # ── 便捷属性 ─────────────────────────────────────────────────────────────────

    @property
    def is_prod(self) -> bool:
        """是否为生产环境。用于控制文档可见性、日志格式等。"""
        return self.env == Env.prod


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    获取全局配置单例。

    使用 @lru_cache 保证整个进程只实例化一次 Settings，
    避免多次读取 .env 文件和环境变量的开销。

    Returns:
        全局唯一的 Settings 实例
    """
    return Settings()


# 模块级便捷访问点，其他模块直接 from ai_data_agent.config.config import settings 使用
settings = get_settings()
