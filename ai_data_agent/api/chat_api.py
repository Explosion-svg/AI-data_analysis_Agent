"""
api/chat_api.py — HTTP 入口层（FastAPI Router）
职责：接收请求 → 参数校验 → 调用 Agent → 返回结果
绝不包含业务逻辑
"""
from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from ai_data_agent.config.config import settings
from ai_data_agent.orchestration.agent_loop import AgentLoop, AgentResponse
from ai_data_agent.observability.logger import get_logger
from ai_data_agent.observability.tracer import span
import structlog.contextvars

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["Agent"])
_bearer = HTTPBearer(auto_error=False)


# ── 请求 / 响应模型 ───────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096, description="用户问题")
    conversation_id: str | None = Field(
        default=None,
        description="会话 ID，留空则自动生成新会话",
    )
    use_cache: bool = Field(default=True, description="是否使用结果缓存")


class ToolCallLog(BaseModel):
    tool: str
    args: dict[str, Any]
    success: bool


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    iterations: int
    tool_calls: list[ToolCallLog]
    charts: list[dict[str, Any]]
    data: list[dict[str, Any]]
    latency_ms: float
    success: bool


class HealthResponse(BaseModel):
    status: str
    version: str
    env: str


class ErrorResponse(BaseModel):
    error: str
    detail: str = ""


# ── 依赖注入 ──────────────────────────────────────────────────────────────────

def _get_agent_loop() -> AgentLoop:
    return AgentLoop()


def _verify_api_key(
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
) -> None:
    """若配置了 API_KEY，则校验 Bearer token；未配置则跳过。"""
    if not settings.api_key:
        return
    if credentials is None or credentials.credentials != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )


# ── 路由 ──────────────────────────────────────────────────────────────────────

@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="与数据分析 Agent 对话",
)
async def chat(
    req: ChatRequest,
    _: None = Depends(_verify_api_key),
    agent: AgentLoop = Depends(_get_agent_loop),
) -> ChatResponse:
    conversation_id = req.conversation_id or str(uuid.uuid4())

    # 注入 tracing 上下文，全局logger始终绑定以下两个字段
    structlog.contextvars.bind_contextvars(
        conversation_id=conversation_id,
        query_preview=req.query[:50],
    )

    logger.info(
        "api.chat.received",
        conversation_id=conversation_id,
        query_len=len(req.query),
    )

    with span("api.chat", {"conversation_id": conversation_id}):
        response: AgentResponse = await agent.run(
            query=req.query,
            conversation_id=conversation_id,
            use_cache=req.use_cache,
        )

    if not response.success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=response.error or "Agent failed.",
        )

    logger.info(
        "api.chat.done",
        conversation_id=conversation_id,
        iterations=response.iterations,
        latency_ms=round(response.latency_ms, 1),
    )
    structlog.contextvars.clear_contextvars()

    return ChatResponse(
        conversation_id=response.conversation_id,
        answer=response.answer,
        iterations=response.iterations,
        tool_calls=[
            ToolCallLog(**tc) for tc in response.tool_calls
        ],
        charts=response.charts,
        data=response.data,
        latency_ms=round(response.latency_ms, 1),
        success=response.success,
    )


@router.get("/health", response_model=HealthResponse, summary="健康检查")
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        env=settings.env.value,
    )


@router.delete(
    "/conversations/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="清除指定会话历史",
)
async def clear_conversation(conversation_id: str, _: None = Depends(_verify_api_key)) -> None:
    from ai_data_agent.memory.conversation_memory import get_memory
    get_memory().clear(conversation_id)
    logger.info("api.conversation.cleared", conversation_id=conversation_id)
