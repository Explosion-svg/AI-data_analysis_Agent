"""
context/request_context.py — 请求级上下文

这个模块负责维护一次 API 请求在系统内部传播时需要共享的身份与追踪信息。

设计目标：
- 让 API 层只解析一次身份信息（从 HTTP Header 中提取一次）
- 让 orchestration / tools / memory / observability 共享同一份上下文，无需在函数签名中逐层传递
- 避免 user_id / tenant_id / request_id 在函数参数中无序扩散（参数污染）

为什么用 contextvars 而不是线程本地存储（threading.local）？
- Python 的 asyncio 是单线程并发模型，一个线程上同时运行多个协程（coroutine）
- threading.local 按线程隔离，无法区分同一线程上不同协程的上下文
- contextvars.ContextVar 按协程（asyncio Task）隔离，天然适配 async/await 并发
- 每个 asyncio Task 拥有独立的 Context 副本，请求之间不会串数据

使用流程：
  1. API 层：set_request_context(ctx) → 返回 token
  2. 业务层：get_request_context() → 获取当前请求的 ctx
  3. API 层收尾：clear_request_context(token) → 恢复上下文

注意：
- 这里保存的是"请求级上下文"，不是长期用户资料
- 上下文只在当前 asyncio Task 生命周期内有效
"""
from __future__ import annotations

# ContextVar：异步安全的"协程本地存储"
from contextvars import ContextVar, Token
from dataclasses import dataclass
import re

# 租户/用户 ID 字符集白名单（P3-7）：
# 仅允许字母数字、下划线、连字符。禁止冒号/斜杠等分隔符与特殊字符，
# 防止通过 `X-Tenant-Id` 声明任意租户来冒充/越权（可冒充面）。
_IDENT_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# 会话 ID 长度上限（P4-7）：防止多 MB 输入进 dict key 和日志
_MAX_CONVERSATION_ID_LEN = 256


@dataclass(frozen=True)
class RequestContext:
    """
    单次请求的身份与追踪上下文（不可变值对象）。

    frozen=True 的含义：
    - dataclass 字段不可修改（防止意外覆盖上下文数据）
    - 可以作为字典键（hashable）
    - 对象创建后状态稳定，线程/协程安全

    字段设计遵循"最小必要原则"：
    - request_id: 单次 HTTP 请求的唯一标识，用于跨组件日志串联（如 Kibana 过滤）
    - user_id: 业务用户标识，用于审计日志和限流策略
    - tenant_id: 租户标识，决定 memory/cache/data 的隔离作用域

    有意不包含的字段：
    - IP 地址（路由层关注，业务层不需要）
    - HTTP Method/Path（同上）
    - 时间戳（每个组件自己记录）
    - 权限信息（权限校验在 API 层已完成）
    """

    request_id: str    # 全链路追踪 ID，从 X-Request-Id Header 读取，未提供时自动生成 UUID
    user_id: str       # 业务用户 ID，从 X-User-Id Header 读取，未提供时使用 "anonymous"
    tenant_id: str     # 租户 ID，从 X-Tenant-Id Header 读取，未提供时使用 "public"

    def __post_init__(self) -> None:
        """
        构造后校验身份字段（P3-7）。

        只允许 `[A-Za-z0-9_-]` 字符集（长度 1~64）：
        - 拒绝冒号/斜杠等分隔符，防止 `X-Tenant-Id` 声明任意租户冒充越权
          （若 tenant_id 可含冒号，`"a:b"+"c"` 与 `"a"+"b:c"` 会碰撞出同一 scoped key）
        - 拒绝换行/控制字符，防止日志注入与 dict key 污染

        校验失败抛 ValueError，由 API 层转换为 422（Pydantic 校验语义一致）。
        """
        for field_name, value in (("tenant_id", self.tenant_id), ("user_id", self.user_id)):
            if not isinstance(value, str) or not _IDENT_RE.match(value):
                raise ValueError(
                    f"Invalid {field_name}: {value!r}. "
                    f"Allowed: letters/digits/underscore/hyphen, 1-64 chars."
                )

    def scoped_conversation_id(self, conversation_id: str) -> str:
        """
        生成带租户作用域的 conversation key。

        为什么需要这个方法？
        - 不同租户的对话历史、工作记忆、结果缓存都应相互隔离
        - 直接用 conversation_id 作为键可能导致租户 A 读到租户 B 的数据
        - 通过在键前加上 tenant_id 前缀，保证不同租户的键空间完全分离

        格式："{tenant_id}:{conversation_id}"
        例如："tenant_a:conv_12345"

        在 memory 和 cache 中统一使用 scoped key，而不是裸 conversation_id，
        这样不需要在每个存储层单独实现租户隔离逻辑。

        防碰撞保证（P3-7）：
        - tenant_id 已限制为不含冒号的字符集，故
          `"{t1}:{c1}" == "{t2}:{c2}"` 当且仅当 `t1==t2 and c1==c2`
        - conversation_id 长度上限（P4-7）：拒绝多 MB 输入进 dict key 和日志

        Args:
            conversation_id: 原始会话 ID（通常是 UUID 字符串）

        Returns:
            带租户前缀的作用域键，如 "public:abc-123"
        """
        if len(conversation_id) > _MAX_CONVERSATION_ID_LEN:
            raise ValueError(
                f"conversation_id exceeds {_MAX_CONVERSATION_ID_LEN} characters"
            )
        return f"{self.tenant_id}:{conversation_id}"


# ── 协程本地存储（asyncio-safe）────────────────────────────────────────────

# ContextVar 是 Python 3.7+ 引入的协程本地变量机制
# default=None 表示在没有显式设置时返回 None
_current_request_context: ContextVar[RequestContext | None] = ContextVar(
    "current_request_context",
    default=None,
)


def set_request_context(ctx: RequestContext) -> Token[RequestContext | None]:
    """
    将 RequestContext 绑定到当前协程的上下文。

    这个函数只影响当前 asyncio Task 的上下文，不影响其他并发 Task。
    同一个 Task 内的后续代码（包括被 await 调用的子协程）都能通过
    get_request_context() 访问到这里设置的值。

    Args:
        ctx: 要绑定的请求上下文

    Returns:
        Token 对象，用于后续调用 clear_request_context(token) 恢复旧值。
        必须保存这个 token 并在请求结束时使用，否则嵌套上下文会出错。
    """
    return _current_request_context.set(ctx)


def get_request_context() -> RequestContext | None:
    """
    获取当前协程绑定的请求上下文。

    在没有 HTTP 请求上下文的环境（如后台任务、单元测试）中返回 None，
    调用方应处理 None 情况。

    Returns:
        当前请求的 RequestContext，或 None（未绑定时）
    """
    return _current_request_context.get()


def clear_request_context(token: Token[RequestContext | None] | None = None) -> None:
    """
    清理当前协程的请求上下文，防止上下文泄漏。

    必须在每个请求处理结束时调用（通常在 finally 块中）。
    如果不清理，同一 asyncio Task 被复用时（如连接池中的 Task）
    会读到上一个请求的上下文数据，导致信息泄漏。

    两种清理方式：
    - 传入 token（推荐）：使用 reset() 将变量恢复到 set() 之前的值
      适用于嵌套上下文，不会误清空外层上下文
    - 不传 token：直接将变量置为 None
      适用于简单场景，无嵌套上下文时

    Args:
        token: set_request_context() 返回的 Token（可选）
    """
    if token is not None:
        # 使用 reset() 恢复旧值，正确处理嵌套 set/clear 的情况
        _current_request_context.reset(token)
    else:
        # 没有 token 时直接置 None
        _current_request_context.set(None)
