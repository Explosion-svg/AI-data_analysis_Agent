"""
observability/tracer.py — 分布式追踪（OpenTelemetry）

职责：
  基于 OpenTelemetry 标准提供分布式追踪能力，记录请求在系统中的
  完整执行路径（span）以及各步骤耗时，用于：
  - 性能分析：定位哪个步骤最慢（如 LLM 调用 vs SQL 查询）
  - 故障排查：查看请求失败前的完整调用链
  - 依赖分析：可视化 Agent → LLM → DB 的调用关系

OpenTelemetry 架构：
  Tracer Provider（全局追踪提供者）
    → TracerProvider（SDK 实现）
       → Resource（服务标识，如 service.name=ai_data_agent）
       → BatchSpanProcessor（批量导出 span）
          → OTLPSpanExporter（通过 gRPC 发送到 Jaeger/Zipkin/Tempo）
  Tracer（从 Provider 获取的追踪器，按服务名标识）
  Span（单个追踪单元，有 name、attributes、startTime、endTime）

NoOp 降级策略：
  如果 settings.enable_tracing=False 或 settings.otlp_endpoint 未配置，
  或 opentelemetry 包未安装，_tracer 保持 None。
  span() 上下文管理器检测到 _tracer=None 时直接 yield None，
  不做任何追踪操作，对业务逻辑零影响（代码不需要 if tracing_enabled 判断）。

Span 概念：
  - Span 表示一次有开始和结束时间的操作
  - 可嵌套形成调用树（parent span → child spans）
  - 携带 attributes（键值对，如 step=1, tool=sql_query）

@contextlib.contextmanager 的工作原理：
  span() 函数使用 @contextlib.contextmanager 装饰器，
  通过 yield 将函数转换为上下文管理器：
  - yield 之前的代码：进入 span（等价于 __enter__）
  - yield 之后的代码：退出 span（等价于 __exit__）
  - yield 的值：传递给 with ... as s 的变量

BatchSpanProcessor vs SimpleSpanProcessor：
  - BatchSpanProcessor（生产）：异步批量发送，不阻塞业务代码
  - SimpleSpanProcessor（测试）：同步发送，方便断言验证
"""
from __future__ import annotations

import contextlib
import functools
from typing import Any, Callable, Generator

from ai_data_agent.config.config import settings
from ai_data_agent.observability.logger import get_logger

logger = get_logger(__name__)

# 延迟导入（避免未安装 opentelemetry 时报错）
# _tracer 和 _trace_module 在 _init_tracer() 中初始化
_tracer = None
_trace_module = None


def _init_tracer():
    """
    内部初始化函数，根据配置决定是否启用 OpenTelemetry 追踪。

    不启用的条件（任一满足则跳过）：
    1. settings.enable_tracing=False（追踪总开关关闭）
    2. settings.otlp_endpoint 为空（没有配置 collector 地址）

    初始化步骤：
    1. 创建 Resource（服务标识，在 Jaeger UI 中显示服务名）
    2. 创建 TracerProvider（管理所有 tracer 实例）
    3. 创建 OTLPSpanExporter（gRPC 导出器，发送到 OTLP 兼容的 collector）
       - insecure=True：不需要 TLS（适合内网 Jaeger/Tempo 集群）
    4. 添加 BatchSpanProcessor（异步批量导出，不阻塞业务代码）
    5. 设置全局 TracerProvider（trace.set_tracer_provider）
    6. 获取 tracer 实例（按 service.name 命名）

    异常处理：
    - ImportError：opentelemetry 包未安装，记录 WARNING 并跳过（不影响业务）
    - 其他异常：初始化失败（如 collector 不可达），记录 WARNING 并跳过

    这种软依赖设计（try/except ImportError）允许在不安装 opentelemetry 包的情况下
    正常运行应用，适合开发环境和资源受限环境。
    """
    global _tracer, _trace_module
    if not settings.enable_tracing or not settings.otlp_endpoint:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.resources import Resource

        resource = Resource.create({"service.name": settings.app_name})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=settings.otlp_endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer(settings.app_name)
        _trace_module = trace
        logger.info("tracer.initialized", endpoint=settings.otlp_endpoint)
    except ImportError:
        logger.warning("tracer.opentelemetry_not_installed")
    except Exception as e:
        logger.warning("tracer.init_failed", error=str(e))


def init_tracer() -> None:
    """
    公共初始化入口，在应用启动时调用（assembler.py 中）。

    只是对 _init_tracer() 的包装，提供更清晰的公共接口。
    分离 init_tracer（公共）和 _init_tracer（内部）的原因：
    - 将来可能在 init_tracer 中添加参数（如 service_version）
    - _init_tracer 保持内部实现细节的封装
    """
    _init_tracer()


# @contextlib.contextmanager 把一个函数变成可以用 with ...: 调用的代码块
# 自动执行「前置操作」和「后置清理」
# 分布式追踪（Tracing）中 yield 交出控制权，让业务代码在 span 上下文中运行
@contextlib.contextmanager
def span(name: str, attributes: dict[str, Any] | None = None) -> Generator:
    """
    创建一个追踪 span（上下文管理器），未初始化时为 NoOp。

    使用方式::
        with span("executor.step", {"step": 1, "tool": "sql_query"}):
            result = await tool.run(sql=...)
        # span 自动记录开始和结束时间，携带 step 和 tool 属性

    NoOp 行为（_tracer 为 None 时）：
    - `yield None` 直接交出控制权
    - 业务代码正常执行，没有任何追踪开销
    - `with span("...") as s:` 中 s 为 None，代码中应避免在 s 上调用方法
      （除非先检查 `if s:...`）

    attribute 类型处理：
    - OTel span.set_attribute 只接受 str/int/float/bool（或它们的列表）
    - `str(v)` 统一转换为字符串，避免类型错误

    典型使用场景（Executor._execute_step）：
        with span("executor.step", {"step": step.step, "tool": step.tool}):
            ...

    Args:
        name: span 名称（Jaeger UI 中显示的操作名，如 "executor.step"）
        attributes: span 属性字典（键值对，支持 str/int/float/bool 值）

    Yields:
        OTel Span 对象（_tracer 为 None 时 yield None）
    """
    if _tracer is None or _trace_module is None:
        yield None    # NoOp：直接 yield，不创建任何 span
        return
    with _tracer.start_as_current_span(name) as s:
        if attributes:
            for k, v in attributes.items():
                s.set_attribute(k, str(v))  # 统一转为字符串
        yield s


def trace_async(name: str | None = None) -> Callable:
    """
    装饰器工厂：自动为异步函数创建 span，方法级追踪。

    比手动在函数内部用 `with span(...)` 更简洁，适合需要追踪整个函数的场景。

    使用方式::
        @trace_async("agent_loop.run")
        async def run(self, query: str) -> AgentResponse:
            ...
        # 等价于：
        async def run(self, query: str) -> AgentResponse:
            with span("agent_loop.run"):
                return await _original_run(...)

    span_name 优先级：
    - 显式传入 name → 使用 name
    - 未传入（None）→ 使用 fn.__qualname__（如 "AgentLoop.run"）

    @functools.wraps(fn)：
    - 保留原函数的 __name__、__doc__ 等属性
    - 使装饰后的函数在日志、调试中仍显示原始函数名

    Args:
        name: span 名称（None 时使用 fn.__qualname__）

    Returns:
        装饰器函数（接受 async 函数，返回带 span 包装的 async 函数）
    """
    def decorator(fn: Callable) -> Callable:
        span_name = name or fn.__qualname__

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            with span(span_name):
                return await fn(*args, **kwargs)

        return wrapper
    return decorator


def get_current_span() -> Any:
    """
    获取当前正在运行的 span（用于手动添加属性或记录事件）。

    典型使用场景：
    - 在执行过程中动态添加 span 属性（不是在创建时就知道所有属性）
    - 在异常处理中调用 record_exception()

    Returns:
        当前 OTel Span 对象，如果追踪未初始化则返回 None
    """
    if _trace_module is None:
        return None
    return _trace_module.get_current_span()


def record_exception(exc: Exception) -> None:
    """
    在当前 span 中记录异常信息（包含 traceback）。

    比单纯记录日志更强大：
    - span 异常会在 Jaeger/Tempo 中标记为"错误 span"（显示为红色）
    - 可以直接在 trace UI 中查看错误信息，无需搜索日志
    - 与 span 的时序信息关联（知道异常发生在请求的哪个阶段）

    如果当前没有活跃 span（get_current_span() 返回 None）则静默忽略。

    Args:
        exc: 要记录的异常对象
    """
    current = get_current_span()
    if current:
        current.record_exception(exc)
