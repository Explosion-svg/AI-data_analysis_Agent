"""
observability/tracer.py — 分布式追踪
基于 OpenTelemetry，可对接 Jaeger / Zipkin / OTLP
未配置时退化为 NoOp，不影响业务逻辑
"""
from __future__ import annotations

import contextlib
import functools
from typing import Any, Callable, Generator

from ai_data_agent.config.config import settings
from ai_data_agent.observability.logger import get_logger

logger = get_logger(__name__)

# 延迟导入，避免未安装 opentelemetry 时报错
_tracer = None
_trace_module = None


def _init_tracer():
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
    _init_tracer()

# @contextlib.contextmanager把一个函数变成可以用 with ...: 调用的代码块，自动执行「前置操作」和「后置清理」
# 分布式追踪（Tracing),yield交出控制权
@contextlib.contextmanager
def span(name: str, attributes: dict[str, Any] | None = None) -> Generator:
    """创建一个 tracing span，未初始化时为 NoOp。"""
    if _tracer is None or _trace_module is None:
        yield None
        return
    with _tracer.start_as_current_span(name) as s:
        if attributes:
            for k, v in attributes.items():
                s.set_attribute(k, str(v))
        yield s


def trace_async(name: str | None = None) -> Callable:
    """装饰器：自动为异步函数创建 span。"""
    def decorator(fn: Callable) -> Callable:
        span_name = name or fn.__qualname__

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            with span(span_name):
                return await fn(*args, **kwargs)

        return wrapper
    return decorator


def get_current_span() -> Any:
    if _trace_module is None:
        return None
    return _trace_module.get_current_span()


def record_exception(exc: Exception) -> None:
    current = get_current_span()
    if current:
        current.record_exception(exc)
