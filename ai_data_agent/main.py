"""
main.py — 应用启动入口

职责：
  1. 创建 FastAPI 应用实例（工厂模式）
  2. 通过 lifespan 管理整个应用的启动和关闭流程
  3. 注册全局中间件（CORS）和全局异常处理器
  4. 将路由注册到应用

设计说明：
  main.py 保持极简，不含任何业务逻辑。
  所有组件的创建和装配工作全部委托给 assembler.AppContainer。
  这样 main.py 只需关心"应用如何启动"，而不需要知道"各组件如何组装"。

运行方式：
  # 开发环境（热重载）
  python -m ai_data_agent.main
  # 生产环境
  uvicorn ai_data_agent.main:app --host 0.0.0.0 --port 8000 --workers 4
"""
from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ai_data_agent.config.config import settings

# P2-22：多 worker（uvicorn --workers N）下，每个 worker 是独立进程，
# Counter/Histogram 若只写各自进程内存，抓取端将看不到其他 worker 的指标
# （静默丢失 3/4）。必须在任何指标对象创建之前设置 PROMETHEUS_MULTIPROC_DIR，
# 使各 worker 把指标写入 mmap 文件，再由 metrics 端口所在进程用
# MultiProcessCollector 聚合（见 assembler._start_metrics_server）。
if settings.enable_metrics and settings.workers > 1:
    _prometheus_multiproc_dir = tempfile.mkdtemp(prefix="prometheus_multiproc_")
    os.environ["PROMETHEUS_MULTIPROC_DIR"] = _prometheus_multiproc_dir
    atexit.register(lambda: shutil.rmtree(_prometheus_multiproc_dir, ignore_errors=True))

from ai_data_agent.api.chat_api import router as chat_router
from ai_data_agent.assembler import startup as container_startup, shutdown as container_shutdown
from ai_data_agent.observability.logger import configure_logging, get_logger
from ai_data_agent.reliability.concurrency import ConcurrencyLimitExceeded

# 模块级 logger：在 configure_logging() 调用前使用 bootstrap 默认配置，安全可用
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    FastAPI 应用生命周期管理器（替代已废弃的 on_startup/on_shutdown）。

    执行顺序：
      1. 配置日志（必须在 assembler 之前，否则启动期间日志无法输出）
      2. 调用 container_startup() 按顺序初始化所有组件
      3. 打印就绪日志和健康报告
      4. yield —— 应用开始接受请求
      5. 收到关闭信号后调用 container_shutdown() 释放资源

    为什么日志配置要放在 lifespan 最开头而不是 assembler 里？
      因为 lifespan 是第一个被调用的地方，如果把日志初始化放到 assembler，
      那么 lifespan 自身的"app.starting"日志就无法通过正式配置输出。
      因此这里先配置日志，再把剩余工作委托给 assembler。

    Args:
        app: FastAPI 应用实例（由框架自动传入，通常不直接使用）
    """
    # ── 第一步：配置日志系统 ────────────────────────────────────────────────────
    # 注意：configure_logging() 之后才能使用正式配置的 logger，
    # 所以这里重新获取一次 logger 对象。
    configure_logging(
        json_logs=settings.log_json,
        log_level=settings.log_level.value,
    )
    runtime_logger = get_logger(__name__)

    # ── 第二步：启动所有组件 ────────────────────────────────────────────────────
    runtime_logger.info("app.starting", name=settings.app_name, env=settings.env.value)

    # container_startup() 内部按层级顺序初始化：
    # Observability → Infra → Model Gateway → Tools → Context → Memory → Orchestration
    container = await container_startup()

    runtime_logger.info("app.ready", host=settings.host, port=settings.port)
    # 打印各组件健康状态，便于部署时快速确认所有依赖是否就绪
    runtime_logger.info("app.health", **container.health_report())

    # ── 应用运行阶段 ─────────────────────────────────────────────────────────────
    yield   # 将控制权交还给 FastAPI，开始接受 HTTP 请求

    # ── 第三步：关闭清理 ────────────────────────────────────────────────────────
    # 收到 SIGTERM（K8s pod 缩容）或 SIGINT（Ctrl+C）后执行
    runtime_logger.info("app.shutting_down")
    await container_shutdown()  # 关闭数据库连接池、释放资源
    runtime_logger.info("app.stopped")


def create_app() -> FastAPI:
    """
    FastAPI 应用工厂函数。

    使用工厂函数而非直接在模块级创建实例，有两个好处：
    1. 方便测试：测试时可以调用此函数创建独立的应用实例
    2. 职责清晰：应用配置（中间件、路由、异常处理）集中在这里

    Returns:
        配置好的 FastAPI 应用实例
    """
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Enterprise AI Data Analysis Agent — 8-Layer Architecture",
        # 生产环境关闭 Swagger UI，避免暴露 API 文档
        docs_url="/docs" if not settings.is_prod else None,
        redoc_url="/redoc" if not settings.is_prod else None,
        lifespan=lifespan,
    )

    # ── CORS 中间件 ─────────────────────────────────────────────────────────────
    # P4-7：CORS 完全由 settings 配置（cors_allow_origins / cors_allow_credentials
    # / cors_allow_methods / cors_allow_headers），不再硬编码。
    # 注意：allow_credentials=True 与 allow_origins=["*"] 是 CORS 规范禁止的组合，
    # 配置层（config._validate_cors）会直接拒绝该组合，这里无需重复防御。
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
    )

    # ── 全局异常处理器 ──────────────────────────────────────────────────────────
    # 注意：异常处理器按注册顺序匹配，更具体的异常类型应先注册

    @app.exception_handler(ConcurrencyLimitExceeded)
    async def _concurrency_limit_handler(request: Request, exc: ConcurrencyLimitExceeded) -> JSONResponse:
        """
        处理并发超限异常。

        当系统达到最大并发数时（由 reliability/concurrency.py 控制），
        抛出 ConcurrencyLimitExceeded，这里统一转换为 503 响应。

        503 而非 429 的原因：
          - 429 表示客户端请求速率过快（客户端责任）
          - 503 表示服务当前无法处理请求（服务端责任）
          这里是系统保护，属于服务端主动限流，语义上更接近 503。
        """
        logger.warning(
            "app.concurrency_limited",
            path=str(request.url),
            bucket=exc.bucket,
            timeout_seconds=exc.timeout_seconds,
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error": "Service overloaded",
                "detail": str(exc),
                "bucket": exc.bucket,
            },
        )

    @app.exception_handler(Exception)
    async def _global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """
        兜底全局异常处理器。

        捕获所有未被路由层自行处理的异常，统一返回 500。
        注意：这里不应该出现业务异常（业务异常应在路由层转换为 4xx/5xx）。
        出现在这里的通常是系统级错误（数据库连接断开、OOM 等）。

        P2-23：不在响应体回显内部异常原文（str(exc) 可能包含数据库 URL/密码、
        内部文件路径等敏感信息）。服务端记录完整错误（exc_info=True），
        客户端只收到通用消息 + request_id，凭 request_id 在日志中定位根因。
        """
        request_id = request.headers.get("x-request-id") or "unknown"
        logger.error(
            "app.unhandled_exception",
            path=str(request.url),
            request_id=request_id,
            error=str(exc),
            exc_info=True,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal server error",
                "request_id": request_id,
                "detail": "An unexpected error occurred. Please contact support with the request_id.",
            },
        )

    # ── 路由注册 ─────────────────────────────────────────────────────────────────
    # chat_router 包含 /api/v1/chat、/api/v1/health 等所有业务端点
    app.include_router(chat_router)

    return app


# 模块级应用实例，供 uvicorn 直接引用：
# uvicorn ai_data_agent.main:app
app = create_app()


if __name__ == "__main__":
    # 直接运行时（python -m ai_data_agent.main）使用此入口
    # uvicorn 本身也会调用 create_app()，因此这里需要先配置日志
    configure_logging(
        json_logs=settings.log_json,
        log_level=settings.log_level.value,
    )
    # P4-7：uvicorn 的 reload（热重载）与多 worker 互斥——reload 开启时 workers 会被
    # 静默忽略（始终单进程运行）。这里显式降级为单 worker 并告警，
    # 避免"配置了 N 个 worker 却只跑单进程"的配置误导。
    workers = settings.workers
    if settings.debug and workers > 1:
        workers = 1
        logger.warning(
            "main.reload_forces_single_worker",
            configured_workers=settings.workers,
            reason="uvicorn reload mode is incompatible with multiple workers",
        )
    uvicorn.run(
        "ai_data_agent.main:app",
        host=settings.host,
        port=settings.port,
        workers=workers,
        reload=settings.debug,          # 开发环境开启热重载
        log_config=None,                # 禁用 uvicorn 默认日志，改用 structlog
    )
