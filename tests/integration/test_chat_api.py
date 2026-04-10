"""
tests/integration/test_chat_api.py

HTTP API 集成测试。

主要验证：
- /api/v1/chat 正常路径
- 参数校验
- API Key 校验
- /health 健康检查
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from ai_data_agent.api import chat_api
from ai_data_agent.config.config import settings
from ai_data_agent.main import create_app
from ai_data_agent.orchestration.agent_loop import AgentResponse


class FakeAgent:
    # 用固定返回值替代真实 AgentLoop，保证 API 测试只关注 HTTP 契约。
    def __init__(self, response: AgentResponse) -> None:
        self._response = response

    async def run(self, query: str, conversation_id: str, use_cache: bool = True) -> AgentResponse:
        return self._response


@pytest.mark.asyncio
async def test_chat_api_success() -> None:
    # 正常请求应返回 200 和约定的响应结构。
    app = create_app()
    app.dependency_overrides[chat_api._get_agent_loop] = lambda: FakeAgent(
        AgentResponse(
            answer="ok",
            conversation_id="c1",
            iterations=1,
            tool_calls=[],
            charts=[],
            data=[],
            latency_ms=12.3,
            success=True,
        )
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/v1/chat", json={"query": "hello", "conversation_id": "c1"})

    assert resp.status_code == 200
    assert resp.json()["answer"] == "ok"
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_chat_api_validation_error() -> None:
    # query 为空时应由 Pydantic/FastAPI 返回 422。
    app = create_app()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/v1/chat", json={"query": ""})

    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_chat_api_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # 配置 API key 后，未带 Bearer token 的请求应被拒绝。
    app = create_app()
    monkeypatch.setattr(settings, "api_key", "secret")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/v1/chat", json={"query": "hello"})

    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_health_endpoint() -> None:
    # 健康检查只验证最基本状态，不依赖 Agent 主逻辑。
    app = create_app()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/health")

    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
