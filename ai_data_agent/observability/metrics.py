"""
observability/metrics.py — Prometheus 指标收集
对外暴露 /metrics 端点（由 prometheus_client 提供）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from prometheus_client import Counter, Histogram, Gauge, Summary


@dataclass
class AgentMetrics:
    # ── LLM ──────────────────────────────────────────────
    llm_tokens_total: Counter = field(
        default_factory=lambda: Counter(
            "llm_tokens_total",
            "Total LLM tokens consumed",
            ["model"],
        )
    )
    llm_latency: Histogram = field(
        default_factory=lambda: Histogram(
            "llm_latency_seconds",
            "LLM call latency",
            ["model"],
            buckets=(0.5, 1, 2, 5, 10, 30, 60),
        )
    )
    llm_errors_total: Counter = field(
        default_factory=lambda: Counter(
            "llm_errors_total",
            "LLM errors by type",
            ["model", "error_type"],
        )
    )

    # ── Tool ──────────────────────────────────────────────
    tool_calls_total: Counter = field(
        default_factory=lambda: Counter(
            "tool_calls_total",
            "Total tool invocations",
            ["tool_name"],
        )
    )
    tool_errors_total: Counter = field(
        default_factory=lambda: Counter(
            "tool_errors_total",
            "Tool errors",
            ["tool_name"],
        )
    )
    tool_latency: Histogram = field(
        default_factory=lambda: Histogram(
            "tool_latency_seconds",
            "Tool execution latency",
            ["tool_name"],
            buckets=(0.01, 0.1, 0.5, 1, 5, 10, 30),
        )
    )

    # ── SQL ───────────────────────────────────────────────
    sql_queries_total: Counter = field(
        default_factory=lambda: Counter(
            "sql_queries_total",
            "Total SQL queries executed",
        )
    )
    sql_latency: Summary = field(
        default_factory=lambda: Summary(
            "sql_query_latency_seconds",
            "SQL query latency",
        )
    )
    sql_blocked_total: Counter = field(
        default_factory=lambda: Counter(
            "sql_blocked_total",
            "SQL queries blocked by safety guard",
        )
    )

    # ── Agent ─────────────────────────────────────────────
    agent_requests_total: Counter = field(
        default_factory=lambda: Counter(
            "agent_requests_total",
            "Total agent requests",
        )
    )
    agent_errors_total: Counter = field(
        default_factory=lambda: Counter(
            "agent_errors_total",
            "Agent errors",
            ["error_type"],
        )
    )
    agent_iterations: Histogram = field(
        default_factory=lambda: Histogram(
            "agent_loop_iterations",
            "Agent ReAct loop iterations per request",
            buckets=(1, 2, 3, 5, 7, 10, 15),
        )
    )
    agent_latency: Histogram = field(
        default_factory=lambda: Histogram(
            "agent_request_latency_seconds",
            "End-to-end agent request latency",
            buckets=(1, 2, 5, 10, 30, 60, 120),
        )
    )

    # ── Circuit Breaker ───────────────────────────────────
    circuit_breaker_open: Gauge = field(
        default_factory=lambda: Gauge(
            "circuit_breaker_open",
            "1 if circuit breaker is open (service unavailable)",
            ["service"],
        )
    )

    # ── Cache ─────────────────────────────────────────────
    cache_hits_total: Counter = field(
        default_factory=lambda: Counter(
            "cache_hits_total",
            "Cache hits",
        )
    )
    cache_misses_total: Counter = field(
        default_factory=lambda: Counter(
            "cache_misses_total",
            "Cache misses",
        )
    )


# 全局单例
metrics = AgentMetrics()
