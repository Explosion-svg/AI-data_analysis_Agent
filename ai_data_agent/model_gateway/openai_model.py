"""
model_gateway/openai_model.py — OpenAI 兼容适配器

职责：
  实现 BaseLLM 接口，对接 OpenAI Chat Completions API。
  支持所有兼容 OpenAI API 格式的服务端（openai/deepseek/本地Ollama/Azure等），
  通过修改 api_base 和 api_key 即可切换不同服务商，无需修改上层代码。

技术选型：
  使用 openai 官方 Python SDK 的 AsyncOpenAI 客户端（异步版本），
  原因是整个项目基于 asyncio 事件循环，使用异步客户端可以避免阻塞事件循环，
  支持高并发场景下多个请求并行调用 LLM。

重试策略：
  max_retries=0 表示 SDK 内部不做重试，
  重试逻辑统一由外层 reliability/retry.py 的 @async_retry 装饰器处理。
  原因：reliability 层的重试策略是全局统一的（指数退避 + 抖动），
  如果 SDK 内部也重试会导致重试次数叠加，影响超时控制的准确性。

function calling 流程：
  1. generate() 的 config.tools 包含 OpenAI function calling 格式的工具定义
  2. 模型决定调用哪个工具时，响应的 finish_reason="tool_calls"
  3. choice.message.tool_calls 包含模型请求执行的工具列表
  4. 这里把 SDK 的 ChatCompletionMessageToolCall 对象转换为简单的 dict，
     方便上层代码（AgentLoop）统一处理，不需要依赖 openai SDK 类型

工厂函数：
  build_openai_model()  — 从 settings 读取 OpenAI 配置
  build_deepseek_model() — 从 settings 读取 DeepSeek 配置（同 OpenAI 格式）
  build_local_model()   — 从 settings 读取本地 LLM 配置（如 Ollama）
"""
from __future__ import annotations

import time
from typing import AsyncIterator, Any

from openai import AsyncOpenAI, APIError, RateLimitError, APITimeoutError

from ai_data_agent.config.config import settings
from ai_data_agent.model_gateway.base_model import (
    BaseLLM,
    LLMConfig,
    LLMResponse,
    Message,
)
from ai_data_agent.observability.logger import get_logger
from ai_data_agent.observability.metrics import metrics

logger = get_logger(__name__)


def _to_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """
    将内部 Message 列表转换为 OpenAI Chat API 格式的 messages 列表。

    OpenAI API 要求 messages 是 list[dict]，每个 dict 必须有 role 和 content。
    可选字段（name、tool_call_id、tool_calls）只有非 None 时才包含，
    因为 OpenAI API 不接受值为 null 的多余字段（会返回 400 错误）。

    role 格式说明：
    - "system"    → 系统提示词
    - "user"      → 用户消息
    - "assistant" → 助手回复，可能包含 tool_calls 字段（请求执行工具）
    - "tool"      → 工具执行结果，必须包含 tool_call_id（对应工具调用请求）和 name

    Args:
        messages: 内部 Message 对象列表

    Returns:
        OpenAI API 格式的 messages 列表（list[dict]）
    """
    result = []
    for m in messages:
        # role 和 content 是必须字段
        msg: dict[str, Any] = {"role": m.role, "content": m.content}
        # 以下字段按需添加（None 的不加）
        if m.name:
            msg["name"] = m.name
        if m.tool_call_id:
            msg["tool_call_id"] = m.tool_call_id
        if m.tool_calls:
            msg["tool_calls"] = m.tool_calls
        result.append(msg)
    return result


class OpenAIModel(BaseLLM):
    """
    OpenAI / OpenAI 兼容 API 的 LLM 适配器。

    支持 function calling（工具调用）：
    - 通过 LLMConfig.tools 传入 OpenAI function calling 格式的工具列表
    - 模型选择调用工具时，LLMResponse.tool_calls 包含请求的工具信息
    - LLMResponse.finish_reason = "tool_calls" 表示需要执行工具

    错误处理策略（分类处理）：
    - RateLimitError (429)：速率限制，上层 retry.py 会重试
    - APITimeoutError：超时，上层 retry.py 会重试
    - APIError：其他 API 错误，记录日志后重新抛出
    - 不在适配器层做 catch-all，让错误传播到 reliability 层统一处理

    指标：
    - llm_tokens_total：记录每次调用的 token 消耗（按模型分组）
    - llm_latency：记录端到端延迟分布（按模型分组）
    """

    def __init__(
        self,
        api_key: str,
        api_base: str,
        model: str,
        adapter_name: str = "openai",
    ) -> None:
        """
        初始化 OpenAI 适配器。

        Args:
            api_key: OpenAI API Key（或兼容服务商的 Key）
            api_base: API base URL（如 "https://api.openai.com/v1"）
            model: 默认使用的模型 ID（如 "gpt-4o"）
            adapter_name: 适配器标识符（"openai"、"deepseek"、"local"），
                         用于日志记录和 ModelRouter 注册表键名
        """
        self._name = adapter_name
        self._model = model
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=settings.llm_timeout,
            max_retries=0,   # 重试由 reliability/retry.py 统一控制，SDK 内部不重试
        )

    @property
    def name(self) -> str:
        """
        适配器唯一标识符（用于 ModelRouter 注册表和日志标签）。

        Returns:
            适配器名称字符串（"openai"、"deepseek" 或 "local"）
        """
        return self._name

    async def generate(
        self,
        messages: list[Message],
        config: LLMConfig,
    ) -> LLMResponse:
        """
        调用 OpenAI Chat Completions API 生成完整响应（非流式）。

        完整流程：
        1. 构建请求参数（model, messages, temperature, max_tokens, tools 等）
        2. 调用 self._client.chat.completions.create(**kwargs)
        3. 提取响应内容（content + tool_calls）
        4. 将 SDK 的 ChatCompletionMessageToolCall 对象转换为 dict
        5. 提取 token 使用量，记录指标和日志
        6. 构造 LLMResponse 返回

        function calling 响应处理：
        - choice.message.tool_calls 是 SDK 的对象列表，需要手动转换为 dict
        - 转换格式：{"id": str, "type": "function", "function": {"name": str, "arguments": str}}
        - arguments 是 JSON 字符串（由模型生成），需要在 AgentLoop 中解析

        Args:
            messages: 对话消息列表（包含 system/user/assistant/tool 各类消息）
            config: 调用配置（model、temperature、max_tokens、tools 等）

        Returns:
            LLMResponse 对象，包含 content、tool_calls、usage、finish_reason、latency_ms

        Raises:
            RateLimitError: API 调用达到速率限制（429）
            APITimeoutError: 调用超时
            APIError: 其他 OpenAI API 错误
        """
        start = time.perf_counter()
        try:
            # 构建基础请求参数
            kwargs: dict[str, Any] = {
                "model": config.model or self._model,
                "messages": _to_openai_messages(messages),
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
            }
            # 可选参数（None 时不传，避免 API 报 400 invalid parameter）
            if config.stop:
                kwargs["stop"] = config.stop
            if config.tools:
                kwargs["tools"] = config.tools
            if config.tool_choice:
                kwargs["tool_choice"] = config.tool_choice

            resp = await self._client.chat.completions.create(**kwargs)
            choice = resp.choices[0]
            content = choice.message.content or ""  # 模型回答（工具调用时可能为空）

            # 把 SDK 对象转换为简单 dict，避免上层依赖 openai SDK 类型
            tool_calls = None
            if choice.message.tool_calls:
                # List[ChatCompletionMessageToolCall] → List[dict]
                tool_calls = [
                    {
                        "id": tc.id,                        # 工具调用 ID（用于匹配响应）
                        "type": tc.type,                    # 固定为 "function"
                        "function": {
                            "name": tc.function.name,       # 工具函数名
                            "arguments": tc.function.arguments,  # JSON 字符串参数
                        },
                    }
                    for tc in choice.message.tool_calls
                ]

            # 提取 token 使用量（resp.usage 可能为 None）
            usage = {}
            if resp.usage:
                usage = {
                    "prompt_tokens": resp.usage.prompt_tokens,
                    "completion_tokens": resp.usage.completion_tokens,
                    "total_tokens": resp.usage.total_tokens,
                }

            elapsed = self._elapsed_ms(start)  # 端到端延迟（毫秒）

            # 记录 Prometheus 指标
            metrics.llm_tokens_total.labels(model=kwargs["model"]).inc(
                usage.get("total_tokens", 0)
            )
            metrics.llm_latency.labels(model=kwargs["model"]).observe(elapsed / 1000)

            logger.debug(
                "llm.generate",
                model=kwargs["model"],
                tokens=usage.get("total_tokens"),
                latency_ms=round(elapsed, 1),
            )
            return LLMResponse(
                content=content,
                model=resp.model,
                usage=usage,
                finish_reason=choice.finish_reason or "stop",
                tool_calls=tool_calls,
                latency_ms=elapsed,
            )
        except RateLimitError as e:
            # 429 速率限制，由外层 retry.py 重试
            logger.warning("llm.rate_limit", adapter=self._name, error=str(e))
            raise
        except APITimeoutError as e:
            # 超时，由外层 retry.py 重试
            logger.warning("llm.timeout", adapter=self._name, error=str(e))
            raise
        except APIError as e:
            # 其他 API 错误，记录并上抛
            logger.error("llm.api_error", adapter=self._name, error=str(e))
            raise

    async def stream(
        self,
        messages: list[Message],
        config: LLMConfig,
    ) -> AsyncIterator[str]:
        """
        流式生成响应，逐 token 返回字符串片段。

        适合需要实时展示部分回复的场景（如 WebSocket 流式推送给前端）。
        当前主流程主要使用 generate()（非流式），stream() 为未来扩展预留。

        使用方式：
            async for chunk in model.stream(messages, config):
                print(chunk, end="", flush=True)

        注意：
        - 流式模式不支持 function calling（OpenAI 的技术限制，需要特殊处理）
        - 流式模式不返回 token 使用量（use_completion_tokens 需要单独统计）

        Args:
            messages: 对话消息列表
            config: 调用配置

        Yields:
            逐个 token 的字符串片段（delta.content）
        """
        kwargs: dict[str, Any] = {
            "model": config.model or self._model,
            "messages": _to_openai_messages(messages),
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "stream": True,
        }
        async with await self._client.chat.completions.create(**kwargs) as stream:
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        使用 OpenAI embedding 模型生成文本向量。

        使用场景：
        - RAG 检索：把 query 向量化，然后与文档库向量做余弦相似度搜索
        - schema 语义索引：把表/列描述向量化，支持自然语言 schema 检索

        模型选择：
        - 使用 settings.embedding_model（默认 text-embedding-3-small）
        - text-embedding-3-small：1536 维，性价比高，适合大多数场景
        - text-embedding-3-large：3072 维，精度更高但更贵

        批量处理：
        - texts 可以是多个文本（批量 embedding 比逐个调用效率高）
        - OpenAI API 对单次请求有文本数量和 token 限制，超出时需要分批

        Args:
            texts: 要向量化的文本列表

        Returns:
            每个文本对应的 float 列表（embedding 向量），维度由模型决定
        """
        resp = await self._client.embeddings.create(
            model=settings.embedding_model,
            input=texts,
        )
        return [item.embedding for item in resp.data]

    async def health_check(self) -> bool:
        """
        通过列出可用模型来检查 OpenAI API 是否可达。

        list() 是一个轻量操作（不产生 token 消耗），
        只需要 API Key 有效且网络连通就能成功。

        Returns:
            True 表示 API 可用，False 表示不可用（任何异常都视为不可用）
        """
        try:
            await self._client.models.list()
            return True
        except Exception:
            return False

    async def aclose(self) -> None:
        """
        关闭 AsyncOpenAI 客户端及其背后的 httpx 连接池（P2-20）。

        应用优雅关闭时由 ModelRouter.close() 逐适配器调用，
        避免 httpx 会话在 SIGTERM 后残留导致端口占用或连接泄漏。
        """
        await self._client.close()


# ── 工厂函数 ─────────────────────────────────────────────────────────────────

def build_openai_model() -> OpenAIModel | None:
    """
    从配置构建 OpenAI 适配器实例。

    当 OPENAI_API_KEY 未配置时返回 None（ModelRouter 会忽略 None 适配器）。
    这样可以做到"有 key 就用，没有 key 就跳过"，不强制要求所有服务商都配置。

    Returns:
        配置好的 OpenAIModel 实例，或 None（未配置 OPENAI_API_KEY 时）
    """
    if not settings.openai_api_key:
        return None
    return OpenAIModel(
        api_key=settings.openai_api_key,
        api_base=settings.openai_api_base,
        model=settings.openai_default_model,
        adapter_name="openai",
    )


def build_deepseek_model() -> OpenAIModel | None:
    """
    从配置构建 DeepSeek 适配器实例。

    DeepSeek 的 API 完全兼容 OpenAI 格式（支持 function calling），
    只需修改 api_key 和 api_base，使用同一个 OpenAIModel 类即可。
    adapter_name="deepseek" 用于区分日志和 ModelRouter 注册表。

    DeepSeek 特别适合代码生成（ModelRouter 中 CODE 任务优先路由 DeepSeek）。

    Returns:
        配置好的 DeepSeek 适配器实例，或 None（未配置 DEEPSEEK_API_KEY 时）
    """
    if not settings.deepseek_api_key:
        return None
    return OpenAIModel(
        api_key=settings.deepseek_api_key,
        api_base=settings.deepseek_api_base,
        model=settings.deepseek_model,
        adapter_name="deepseek",
    )


def build_local_model() -> OpenAIModel | None:
    """
    从配置构建本地 LLM 适配器实例（如 Ollama）。

    Ollama 等本地推理框架提供 OpenAI 兼容的 HTTP API，
    通过设置 LOCAL_LLM_API_BASE（如 http://localhost:11434/v1）
    和 LOCAL_LLM_MODEL（如 qwen2.5:7b）即可使用本地模型。

    适用场景：
    - 离线环境（无互联网访问）
    - 数据隐私要求（不希望数据发送到外部 API）
    - 降低成本（本地 GPU 推理）

    Returns:
        配置好的本地 LLM 适配器实例，或 None（未配置 LOCAL_LLM_API_BASE 时）
    """
    if not (settings.local_llm_api_base and settings.local_llm_model):
        return None
    return OpenAIModel(
        api_key="local",           # 本地服务不需要真实 API Key，用占位符
        api_base=settings.local_llm_api_base,
        model=settings.local_llm_model,
        adapter_name="local",
    )
