"""
main.py — 应用启动入口
FastAPI + lifespan 管理所有资源初始化和清理
所有组件装配委托给 AppContainer（assembler.py）
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ai_data_agent.api.chat_api import router as chat_router
from ai_data_agent.config.config import settings
from ai_data_agent.assembler import startup as container_startup, shutdown as container_shutdown
from ai_data_agent.observability.logger import configure_logging, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    应用生命周期管理。
    启动/关闭逻辑全部委托给 AppContainer，main.py 保持极简。
    """
    logger.info("app.starting", name=settings.app_name, env=settings.env.value)

    container = await container_startup()
    logger.info("app.ready", host=settings.host, port=settings.port)
    logger.info("app.health", **container.health_report())

    yield

    logger.info("app.shutting_down")
    await container_shutdown()
    logger.info("app.stopped")


def create_app() -> FastAPI:
    """FastAPI 应用工厂。"""
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Enterprise AI Data Analysis Agent — 8-Layer Architecture",
        docs_url="/docs" if not settings.is_prod else None,
        redoc_url="/redoc" if not settings.is_prod else None,
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 全局异常处理
    @app.exception_handler(Exception)
    async def _global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error(
            "app.unhandled_exception",
            path=str(request.url),
            error=str(exc),
            exc_info=True,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "Internal server error", "detail": str(exc)},
        )

    # 路由
    app.include_router(chat_router)

    return app


app = create_app()


if __name__ == "__main__":
    configure_logging(
        json_logs=settings.log_json,
        log_level=settings.log_level.value,
    )
    uvicorn.run(
        "ai_data_agent.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        reload=settings.debug,
        log_config=None,   # 使用 structlog，禁用 uvicorn 默认 log 配置
    )
