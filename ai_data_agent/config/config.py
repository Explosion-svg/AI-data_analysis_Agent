"""
config/config.py — 全局配置管理
使用 pydantic-settings，支持 .env 文件与环境变量
"""
from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Env(str, Enum):
    dev = "dev"
    staging = "staging"
    prod = "prod"


class LogLevel(str, Enum):
    debug = "DEBUG"
    info = "INFO"
    warning = "WARNING"
    error = "ERROR"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── 应用 ────────────────────────────────────────────
    app_name: str = "AI Data Analysis Agent"
    app_version: str = "1.0.0"
    env: Env = Env.dev
    debug: bool = False

    # ── API Server ───────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    api_key: Optional[str] = None          # 保护 /chat 端点的 Bearer token

    # ── 数据库（OLTP）────────────────────────────────────
    database_url: str = "sqlite+aiosqlite:///./data/agent.db"
    # database_url = "postgresql+asyncpg://user:password@localhost:5432/agent"
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_echo: bool = False

    # ── 数据仓库（OLAP）─────────────────────────────────
    warehouse_url: str = "sqlite+aiosqlite:///./data/warehouse.db"
    # warehouse_url = "postgresql+asyncpg://user:password@localhost:5432/warehouse"

    # ── 向量数据库 ────────────────────────────────────────
    vector_store_type: str = "chroma"          # chroma | milvus | weaviate
    chroma_persist_dir: str = "./data/chroma"
    chroma_docs_collection: str = "docs"
    chroma_schema_collection: str = "schema"

    # ── LLM / Model Gateway ──────────────────────────────
    openai_api_key: Optional[str] = None
    openai_api_base: str = "https://api.openai.com/v1"
    openai_default_model: str = "gpt-4o"
    openai_fast_model: str = "gpt-4o-mini"

    deepseek_api_key: Optional[str] = None
    deepseek_api_base: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"

    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-sonnet-4-6"

    local_llm_api_base: Optional[str] = None   # e.g. http://localhost:11434/v1
    local_llm_model: Optional[str] = None

    llm_temperature: float = 0.0
    llm_max_tokens: int = 4096
    llm_timeout: float = 60.0               # 单次 LLM 调用超时（秒）
    llm_max_retries: int = 3

    # ── Embedding ───────────────────────────────────────
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536

    # ── Reliability ─────────────────────────────────────
    sql_query_timeout: float = 30.0
    python_exec_timeout: float = 20.0
    agent_max_iterations: int = 10
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: float = 60.0
    retry_max_attempts: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 30.0

    # ── Memory ──────────────────────────────────────────
    conversation_max_turns: int = 20        # 保留最近 N 轮
    cache_ttl_seconds: int = 300
    cache_max_size: int = 256

    # ── Observability ────────────────────────────────────
    log_level: LogLevel = LogLevel.info
    log_json: bool = True                   # True = JSON 格式（生产）
    enable_tracing: bool = False
    otlp_endpoint: Optional[str] = None     # OpenTelemetry collector
    enable_metrics: bool = True
    metrics_port: int = 9090

    # ── 安全 ─────────────────────────────────────────────
    sql_readonly: bool = True               # 只允许 SELECT
    python_sandbox: bool = True             # 沙盒执行 Python

    @field_validator("llm_temperature")
    @classmethod
    def _check_temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError("temperature must be in [0.0, 2.0]")
        return v

    @property
    def is_prod(self) -> bool:
        return self.env == Env.prod


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """单例配置，全局共享。"""
    return Settings()


settings = get_settings()
