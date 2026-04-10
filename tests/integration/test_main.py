"""
tests/integration/test_main.py

应用入口层集成测试。

主要验证：
- lifespan 是否正确调用 startup/shutdown
- create_app 在不同环境下的文档路由配置
- 全局异常处理器的返回结构
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from fastapi import Request

from ai_data_agent.config.config import Env, settings
from ai_data_agent.main import create_app, lifespan


@pytest.mark.asyncio
async def test_lifespan_calls_container_startup_and_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    # lifespan 是 main.py 的关键逻辑，负责把应用生命周期委托给 assembler。
    calls: list[str] = []

    class FakeContainer:
        def health_report(self) -> dict:
            return {"started": True}

    async def fake_startup():
        calls.append("startup")
        return FakeContainer()

    async def fake_shutdown():
        calls.append("shutdown")

    monkeypatch.setattr("ai_data_agent.main.container_startup", fake_startup)
    monkeypatch.setattr("ai_data_agent.main.container_shutdown", fake_shutdown)
    app = create_app()

    async with lifespan(app):
        calls.append("inside")

    assert calls == ["startup", "inside", "shutdown"]


def test_create_app_respects_prod_docs_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    # 生产环境应关闭 docs/redoc，非生产环境保持开启。
    monkeypatch.setattr(settings, "env", Env.prod)
    prod_app = create_app()
    assert prod_app.docs_url is None
    assert prod_app.redoc_url is None

    monkeypatch.setattr(settings, "env", Env.dev)
    dev_app = create_app()
    assert dev_app.docs_url == "/docs"
    assert dev_app.redoc_url == "/redoc"


@pytest.mark.asyncio
async def test_global_exception_handler_returns_500_json() -> None:
    # create_app 里注册了全局异常处理器，这里直接调用 handler 验证返回结构。
    app = create_app()
    handler = app.exception_handlers[Exception]
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/boom",
            "headers": [],
            "query_string": b"",
            "server": ("testserver", 80),
            "client": ("127.0.0.1", 12345),
            "scheme": "http",
        }
    )

    response = await handler(request, RuntimeError("boom"))

    assert response.status_code == 500
    assert b"Internal server error" in response.body
