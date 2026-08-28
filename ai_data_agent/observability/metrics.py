"""
observability/metrics.py — Prometheus 指标收集

职责：
  定义并收集 Agent 运行时的关键业务和系统指标，
  通过 Prometheus + Grafana 实现实时监控和告警。

Prometheus 四种指标类型：
  - Counter（计数器）：只增不减的累计值（如请求总数、错误总数）
    → 通过 PromQL rate() 计算每秒速率
  - Histogram（直方图）：记录观测值的分布（如响应时间分桶）
    → 内置 sum/count/buckets，可计算 p50/p95/p99 延迟
  - Gauge（仪表盘）：可增可减的瞬时值（如熔断器状态）
    → 直接读取当前值
  - Summary（摘要）：类似 Histogram，但在客户端计算百分位数
    → SQL 延迟使用 Summary（数据库查询分布不均匀时）

指标命名规范（遵循 Prometheus 最佳实践）：
  - 全小写，单词用下划线分隔
  - 格式：{subsystem}_{metric_name}_{unit}（如 llm_latency_seconds）
  - 计数类以 _total 结尾（Prometheus 约定）
  - 延迟类以 _seconds 结尾（SI 单位）

Label（标签）设计：
  - labels 允许多维度切片（如按 model 分析不同 LLM 的延迟）
  - 不要使用高基数标签（如 user_id，会导致指标数量爆炸）
  - 典型低基数标签：model（2-3 个）、tool_name（5 个）、outcome（3-4 个）

@dataclass 实现全局指标单例：
  - 所有指标定义在 AgentMetrics dataclass 中
  - field(default_factory=...) 延迟实例化（避免模块导入时 Prometheus 注册冲突）
  - 全局 metrics 单例确保所有模块共享同一套指标对象

Prometheus Histogram buckets 设计原则：
  - LLM 延迟（0.5, 1, 2, 5, 10, 30, 60）：LLM 调用通常 1-10 秒
  - Agent 延迟（1, 2, 5, 10, 30, 60, 120）：端到端可能 2-30 秒
  - 工具延迟（0.01, 0.1, 0.5, 1, 5, 10, 30）：工具执行从毫秒到秒不等
  - 迭代次数（1, 2, 3, 5, 7, 10, 15）：ReAct 循环通常 1-7 次
"""
from __future__ import annotations

from dataclasses import dataclass, field
from prometheus_client import Counter, Histogram, Gauge, Summary


@dataclass
class AgentMetrics:
    """
    Agent 系统的全量指标定义（Prometheus 格式）。

    指标创建时机（P3-11 修正注释）：
    - 字段用 field(default_factory=...) 定义：指标对象在 AgentMetrics 实例化时创建，
      而不是在类定义（class body 执行）时创建。
    - Prometheus 指标创建时即注册到全局注册表（CollectorRegistry），同名重复注册
      会抛 ValueError。default_factory 让"创建即注册"的时点收敛到 `metrics =
      AgentMetrics()`（模块底部），避免类定义/实例化时机混淆。
    - 注意：模块底部 `metrics = AgentMetrics()` 仍在模块导入时执行，因此导入本模块
      就会注册全部指标；测试中如需重建单例，须使用独立 CollectorRegistry 或先
      清理已注册指标，否则会因重名注册报错。

    指标分组：
    - LLM 指标：token 消耗、延迟、错误
    - 工具指标：调用次数、错误次数、延迟
    - SQL 指标：查询总数、错误数、延迟、被拦截次数、审计事件
    - Agent 指标：请求总数、错误、迭代次数、端到端延迟
    - 熔断器指标：状态（开/关）
    - 缓存指标：命中/未命中次数
    - 内存指标：摘要和 pinned facts 提取事件
    """

    # ── LLM 指标 ──────────────────────────────────────────────────────────────
    # 用于追踪 LLM API 成本（tokens）、性能（latency）和可靠性（errors）

    llm_tokens_total: Counter = field(
        default_factory=lambda: Counter(
            "llm_tokens_total",
            "Total LLM tokens consumed",
            ["model"],  # 按模型分标签（openai-gpt4、deepseek-chat 等）
        )
    )
    """LLM 消耗的 token 总数（按模型分标签）。用于估算 API 成本。"""

    llm_latency: Histogram = field(
        default_factory=lambda: Histogram(
            "llm_latency_seconds",
            "LLM call latency",
            ["model"],
            buckets=(0.5, 1, 2, 5, 10, 30, 60),  # LLM 通常 1-10 秒，最慢 60 秒
        )
    )
    """LLM 调用延迟直方图（秒）。用于监控 p95/p99 延迟，检测 LLM 性能退化。"""

    llm_errors_total: Counter = field(
        default_factory=lambda: Counter(
            "llm_errors_total",
            "LLM errors by type",
            ["model", "error_type"],  # 按模型和错误类型（rate_limit、timeout 等）分标签
        )
    )
    """LLM 错误总数（按模型和错误类型分标签）。用于监控 LLM 可靠性。"""

    # ── 工具指标 ──────────────────────────────────────────────────────────────
    # 追踪每个工具的使用情况和健康状态

    tool_calls_total: Counter = field(
        default_factory=lambda: Counter(
            "tool_calls_total",
            "Total tool invocations",
            ["tool_name"],  # sql_query、python_analysis、generate_chart 等
        )
    )
    """工具调用总次数（按工具名分标签）。用于分析工具使用分布。"""

    tool_errors_total: Counter = field(
        default_factory=lambda: Counter(
            "tool_errors_total",
            "Tool errors",
            ["tool_name"],
        )
    )
    """工具错误总次数（按工具名分标签）。工具错误率 = errors/calls。"""

    tool_latency: Histogram = field(
        default_factory=lambda: Histogram(
            "tool_latency_seconds",
            "Tool execution latency",
            ["tool_name"],
            buckets=(0.01, 0.1, 0.5, 1, 5, 10, 30),  # 工具从毫秒级（schema）到秒级（SQL）
        )
    )
    """工具执行延迟直方图（秒）。用于识别慢工具和性能瓶颈。"""

    # ── SQL 指标 ──────────────────────────────────────────────────────────────
    # SQL 专项指标（除通用工具指标外的额外监控）

    sql_queries_total: Counter = field(
        default_factory=lambda: Counter(
            "sql_queries_total",
            "Total SQL queries executed",
        )
    )
    """SQL 查询执行总次数（不含被拦截的）。"""

    sql_latency: Summary = field(
        default_factory=lambda: Summary(
            "sql_query_latency_seconds",
            "SQL query latency",
        )
    )
    """SQL 查询延迟摘要。使用 Summary 而非 Histogram，因为 SQL 延迟分布差异大。"""

    sql_blocked_total: Counter = field(
        default_factory=lambda: Counter(
            "sql_blocked_total",
            "SQL queries blocked by safety guard",
        )
    )
    """被 sql_guard 拦截的 SQL 总次数。高拦截率可能表示 LLM 生成了危险 SQL。"""

    sql_errors_total: Counter = field(
        default_factory=lambda: Counter(
            "sql_errors_total",
            "SQL queries that failed during execution",
        )
    )
    """SQL 执行失败总次数（P3-11）。用于监控仓库层故障率，与 sql_queries_total 一起算错误率。"""

    sql_audit_total: Counter = field(
        default_factory=lambda: Counter(
            "sql_audit_total",
            "SQL audit events by outcome",
            ["outcome"],  # blocked、failed、success
        )
    )
    """SQL 审计事件总次数（按结果分标签）。用于合规报告。"""

    # ── Agent 指标 ────────────────────────────────────────────────────────────
    # 端到端 Agent 请求的质量和性能指标

    agent_requests_total: Counter = field(
        default_factory=lambda: Counter(
            "agent_requests_total",
            "Total agent requests",
        )
    )
    """Agent 请求总次数。与 agent_errors_total 一起计算错误率。"""

    agent_errors_total: Counter = field(
        default_factory=lambda: Counter(
            "agent_errors_total",
            "Agent errors",
            ["error_type"],  # circuit_open、timeout、concurrency_exceeded 等
        )
    )
    """Agent 错误总次数（按错误类型分标签）。用于监控 Agent 健壮性。"""

    agent_iterations: Histogram = field(
        default_factory=lambda: Histogram(
            "agent_loop_iterations",
            "Agent ReAct loop iterations per request",
            buckets=(1, 2, 3, 5, 7, 10, 15),  # 大多数请求 1-5 次迭代
        )
    )
    """每次 Agent 请求的 ReAct 循环迭代次数。迭代次数多可能表示任务复杂或 LLM 效率低。"""

    agent_latency: Histogram = field(
        default_factory=lambda: Histogram(
            "agent_request_latency_seconds",
            "End-to-end agent request latency",
            buckets=(1, 2, 5, 10, 30, 60, 120),  # 端到端通常 2-30 秒
        )
    )
    """Agent 端到端请求延迟（秒）。用于监控用户体验（p95 延迟目标）。"""

    # ── 熔断器指标 ────────────────────────────────────────────────────────────

    circuit_breaker_open: Gauge = field(
        default_factory=lambda: Gauge(
            "circuit_breaker_open",
            "1 if circuit breaker is open (service unavailable)",
            ["service"],  # llm、database 等服务名
        )
    )
    """熔断器状态（0=正常/1=熔断）。Gauge 允许反复在 0/1 间切换，记录状态历史。"""

    # ── 缓存指标 ──────────────────────────────────────────────────────────────

    cache_hits_total: Counter = field(
        default_factory=lambda: Counter(
            "cache_hits_total",
            "Cache hits",
        )
    )
    """缓存命中总次数。命中率 = hits / (hits + misses)，应该越高越好。"""

    cache_misses_total: Counter = field(
        default_factory=lambda: Counter(
            "cache_misses_total",
            "Cache misses",
        )
    )
    """缓存未命中总次数。"""

    memory_summary_total: Counter = field(
        default_factory=lambda: Counter(
            "memory_summary_total",
            "Conversation rolling summary events",
            ["outcome"],  # success、failed（LLM 滚动摘要结果）
        )
    )
    """对话滚动摘要事件次数。失败时会降级为规则摘要，可通过 outcome 区分。"""

    pinned_facts_total: Counter = field(
        default_factory=lambda: Counter(
            "pinned_facts_total",
            "Pinned facts extraction events",
            ["outcome"],  # extracted（成功提取）、empty（LLM 返回空）、failed（调用失败）
        )
    )
    """Pinned Facts 提取事件次数（按结果分标签）。用于监控长期记忆提取的可靠性。"""


# ── 全局单例 ──────────────────────────────────────────────────────────────────

# metrics 是全局单例，所有模块通过 `from ai_data_agent.observability.metrics import metrics` 访问
# 创建时（此处）所有 Counter/Histogram/Gauge 会被注册到 Prometheus 全局注册表
metrics = AgentMetrics()
