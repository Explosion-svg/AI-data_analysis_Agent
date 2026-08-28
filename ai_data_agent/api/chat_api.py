"""
api/chat_api.py — HTTP 入口层（FastAPI Router）

职责：
  1. 接收 HTTP 请求并做参数校验（Pydantic 自动完成）
  2. 从请求头提取身份信息，构造 RequestContext
  3. 调用 AgentLoop 执行核心业务逻辑
  4. 将 AgentResponse 转换为 HTTP 响应格式返回

设计原则：
  - 这一层绝不包含任何业务逻辑，只做"转换"工作：
    HTTP 请求 → Python 数据结构 → 调用 Agent → Python 数据结构 → HTTP 响应
  - 所有依赖通过 FastAPI Depends 注入，方便测试时替换

端点：
  POST   /api/v1/chat                     — 与 Agent 对话
  GET    /api/v1/health                   — 健康检查
  DELETE /api/v1/conversations/{id}       — 清除指定会话历史
"""
from __future__ import annotations

import hmac
import uuid
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from ai_data_agent.assembler import get_container
from ai_data_agent.config.config import settings
import structlog.contextvars

# 路由器：所有端点共享 /api/v1 前缀和 "Agent" 标签（Swagger 分组用）
router = APIRouter(prefix="/api/v1", tags=["Agent"])

# HTTP Bearer Token 提取器，auto_error=False 表示 token 缺失时不自动 401，
# 由 _verify_api_key 自行判断（未配置 api_key 时允许匿名访问）
_bearer = HTTPBearer(auto_error=False)


# ── 请求/响应 Pydantic 模型 ───────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """POST /chat 的请求体。"""
    query: str = Field(..., min_length=1, max_length=4096, description="用户问题")
    conversation_id: str | None = Field(
        default=None,
        description="会话 ID，留空则自动生成新会话（UUID）",
    )
    use_cache: bool = Field(default=True, description="是否使用结果缓存（相同问题直接返回缓存）")


class ToolCallLog(BaseModel):
    """单次工具调用记录，用于前端展示执行过程。"""
    tool: str           # 工具名称，如 "sql_query"、"generate_chart"
    args: dict[str, Any]  # 工具入参（已序列化）
    success: bool       # 本次调用是否成功


class ChatResponse(BaseModel):
    """POST /chat 的响应体。"""
    conversation_id: str                    # 本次对话的会话 ID（前端用于连续对话）
    answer: str                             # Agent 最终回答
    iterations: int                         # ReAct 循环轮次（用于调试/监控）
    tool_calls: list[ToolCallLog]           # 调用过的工具列表
    charts: list[dict[str, Any]]            # 生成的图表 JSON（供前端 Plotly 渲染）
    data: list[dict[str, Any]]              # 最后一次 SQL 查询结果（行列表）
    latency_ms: float                       # 端到端响应延迟（毫秒）
    success: bool                           # 请求是否成功


class HealthResponse(BaseModel):
    """GET /health 的响应体。"""
    status: str     # "ok" 表示正常
    version: str    # 应用版本
    env: str        # 运行环境（dev/staging/prod）


class ErrorResponse(BaseModel):
    """错误响应格式，与 FastAPI 默认格式保持一致。"""
    error: str
    detail: str = ""


# ── FastAPI 依赖函数（Depends）────────────────────────────────────────────────

def _get_logger() -> Any:
    """
    获取当前模块的 logger（从全局容器获取，保证与应用配置一致）。

    为什么不直接用模块级 logger？
    因为容器启动后 structlog 配置可能被更新，从容器获取可以拿到最新配置的 logger。
    """
    return get_container().get_logger(__name__)


def _get_agent_loop() -> Any:
    """
    从全局容器获取 AgentLoop 实例（已在 lifespan 中初始化）。

    使用 Depends 注入而非直接调用，方便在单元测试中替换为 mock 对象。
    """
    return get_container().get_agent_loop()


def _verify_api_key(
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
) -> None:
    """
    校验 Bearer Token（若已配置 API_KEY）。

    逻辑：
      - 未配置 api_key（settings.api_key is None）→ 跳过校验，允许匿名访问
      - 已配置但请求未携带 token 或 token 错误 → 返回 401

    安全说明：
      使用 hmac.compare_digest() 做常数时间比较（P0-4），
      避免时序攻击（timing attack）泄露密钥信息。

    Args:
        credentials: FastAPI 自动从 Authorization: Bearer <token> 头中提取的凭证对象
    """
    if not settings.api_key:
        # 未配置 API Key，允许所有请求通过（适合内网部署）
        return
    if credentials is None or not hmac.compare_digest(
        credentials.credentials, settings.api_key
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )


def _build_request_context(
    x_request_id: str | None = Header(default=None, alias="X-Request-Id"),
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
) -> Any:
    """
    从请求头构造 RequestContext。

    字段来源：
      - X-Request-Id：由 API Gateway 或客户端传入，用于全链路追踪
        未传时自动生成 UUID（保证每个请求都有唯一 ID）
      - X-User-Id：业务用户标识，用于审计和限流
        未传时使用 settings.default_user_id（"anonymous"）
      - X-Tenant-Id：租户标识，用于隔离对话历史和缓存
        未传时使用 settings.default_tenant_id（"public"）

    注意：
      这里不做权限校验，只做身份信息提取。
      真正的权限校验由 _verify_api_key 负责。

    Args:
        x_request_id: 请求追踪 ID，从请求头 X-Request-Id 读取
        x_user_id: 用户 ID，从请求头 X-User-Id 读取
        x_tenant_id: 租户 ID，从请求头 X-Tenant-Id 读取

    Returns:
        RequestContext 数据对象（frozen dataclass）
    """
    return get_container().build_request_context(
        request_id=x_request_id or str(uuid.uuid4()),
        user_id=x_user_id or settings.default_user_id,
        tenant_id=x_tenant_id or settings.default_tenant_id,
    )


# ── HTTP 端点 ─────────────────────────────────────────────────────────────────

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
    _: None = Depends(_verify_api_key),                     # 校验 API Key（忽略返回值）
    request_context: Any = Depends(_build_request_context), # 构造请求上下文
    agent: Any = Depends(_get_agent_loop),                  # 注入 AgentLoop
) -> ChatResponse:
    """
    主对话端点：接收用户问题，调用 Agent，返回分析结果。

    完整流程：
      1. 生成/使用 conversation_id（支持多轮对话）
      2. 设置 RequestContext 到 contextvars（全链路日志绑定）
      3. 绑定 structlog 上下文变量（request_id/user_id 自动附加到所有日志）
      4. 开启 OTel Span（分布式追踪）
      5. 调用 agent.run() 执行 ReAct 循环
      6. 清理上下文，构造响应返回

    错误处理：
      - agent.run() 返回 success=False 时，抛出 500
      - 未捕获异常由 main.py 的全局异常处理器转换为 500

    Args:
        req: 请求体（query + conversation_id + use_cache）
        _: API Key 校验结果（None 表示通过，此处仅利用副作用）
        request_context: 从请求头构造的身份上下文
        agent: AgentLoop 实例

    Returns:
        ChatResponse 包含 Agent 答案、工具调用记录、图表数据等
    """
    # 生成或使用请求中的 conversation_id
    # 前端通过保存并回传 conversation_id 实现多轮对话
    conversation_id = req.conversation_id or str(uuid.uuid4())
    container = get_container()
    logger = _get_logger()

    # 将 RequestContext 写入 contextvars，供下游模块读取（无需显式传参）
    ctx_token = container.set_request_context(request_context)

    # 将关键字段绑定到 structlog 上下文，后续所有 logger.xxx() 调用会自动携带这些字段
    # 这样 Kibana/Loki 可以直接通过 request_id 过滤出一次完整请求的所有日志
    structlog.contextvars.bind_contextvars(
        request_id=request_context.request_id,
        user_id=request_context.user_id,
        tenant_id=request_context.tenant_id,
        conversation_id=conversation_id,
        query_preview=req.query[:50],   # 记录问题前 50 字，方便日志检索
    )

    logger.info(
        "api.chat.received",
        request_id=request_context.request_id,
        user_id=request_context.user_id,
        tenant_id=request_context.tenant_id,
        conversation_id=conversation_id,
        query_len=len(req.query),
    )

    try:
        # 开启 OpenTelemetry Span 追踪本次 API 调用
        # 如果未启用 tracing，span() 是 NoOp，不影响逻辑
        with container.span(
            "api.chat",
            {
                "request_id": request_context.request_id,
                "user_id": request_context.user_id,
                "tenant_id": request_context.tenant_id,
                "conversation_id": conversation_id,
            },
        ):
            response = await agent.run(
                query=req.query,
                conversation_id=conversation_id,
                request_context=request_context,
                use_cache=req.use_cache,
            )
    finally:
        # 无论成功失败，都必须清理上下文
        # 否则同一 asyncio Task 被复用时，下一个请求会读到上一个请求的 context
        container.clear_request_context(ctx_token)
        structlog.contextvars.clear_contextvars()

    # Agent 内部错误（非异常，但 success=False）转换为 HTTP 500
    if not response.success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=response.error or "Agent failed.",
        )

    logger.info(
        "api.chat.done",
        request_id=request_context.request_id,
        user_id=request_context.user_id,
        tenant_id=request_context.tenant_id,
        conversation_id=conversation_id,
        iterations=response.iterations,
        latency_ms=round(response.latency_ms, 1),
    )

    # 将 AgentResponse 转换为 API 响应格式
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
    """
    轻量健康检查端点。

    用途：
      - K8s livenessProbe / readinessProbe
      - 负载均衡器健康检测
      - 运维快速确认服务是否存活

    注意：此端点只检查应用进程本身是否存活，不检查数据库/Redis 连接状态。
    如需深层健康检查（检查依赖组件），可扩展此端点调用 container.health_report()。

    Returns:
        HealthResponse 包含 status="ok"、版本号、运行环境
    """
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        env=settings.env.value,
    )


@router.delete(
    "/conversations/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
    summary="清除指定会话历史",
)
async def clear_conversation(
    conversation_id: str,
    _: None = Depends(_verify_api_key),
    request_context: Any = Depends(_build_request_context),
) -> None:
    """
    清除指定会话的对话历史和工作记忆。

    使用场景：
      - 用户主动开始新的分析任务（避免历史对话干扰）
      - 调试时重置状态
      - GDPR 数据删除请求

    注意：
      - conversation_id 会通过 request_context.scoped_conversation_id()
        自动加上租户前缀（{tenant_id}:{conversation_id}），
        保证不同租户之间无法互相清除对方的会话
      - 返回 204 No Content（无响应体）

    Args:
        conversation_id: URL 路径中的会话 ID
        _: API Key 校验（与 /chat 一致，要求相同权限）
        request_context: 用于确定租户归属
    """
    container = get_container()
    logger = _get_logger()

    # 生成带租户前缀的 scoped key，确保租户隔离
    scoped_conversation_id = request_context.scoped_conversation_id(conversation_id)

    # 同时清除对话记忆和工作记忆
    container.get_memory().clear(scoped_conversation_id)
    container.get_work_memory().clear(scoped_conversation_id)

    logger.info(
        "api.conversation.cleared",
        request_id=request_context.request_id,
        user_id=request_context.user_id,
        tenant_id=request_context.tenant_id,
        conversation_id=conversation_id,
    )
    # 返回 None，FastAPI 自动映射为 204 No Content
