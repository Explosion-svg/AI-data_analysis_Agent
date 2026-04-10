"""
model_gateway/openai_model.py — OpenAI 兼容适配器
支持 OpenAI / DeepSeek / Azure / 本地 LLM（OpenAI-compatible API）
使用异步库AsyncOpenAI，处理多请求速度较快
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
    result = []
    for m in messages:
        msg: dict[str, Any] = {"role": m.role, "content": m.content}
        if m.name:
            msg["name"] = m.name
        if m.tool_call_id:
            msg["tool_call_id"] = m.tool_call_id
        if m.tool_calls:
            msg["tool_calls"] = m.tool_calls
        result.append(msg)
    return result


class OpenAIModel(BaseLLM):
    """OpenAI / OpenAI-compatible 适配器（支持 function calling）。"""

    def __init__(
        self,
        api_key: str,
        api_base: str,
        model: str,
        adapter_name: str = "openai",
    ) -> None:
        self._name = adapter_name
        self._model = model
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=settings.llm_timeout,
            max_retries=0,   # 重试由 reliability 层控制
        )

    @property
    def name(self) -> str:
        return self._name

    async def generate(
        self,
        messages: list[Message],
        config: LLMConfig,
    ) -> LLMResponse:
        start = time.perf_counter()
        try:
            kwargs: dict[str, Any] = {
                "model": config.model or self._model,
                "messages": _to_openai_messages(messages),
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
            }
            if config.stop:
                kwargs["stop"] = config.stop
            if config.tools:
                kwargs["tools"] = config.tools
            if config.tool_choice:
                kwargs["tool_choice"] = config.tool_choice

            resp = await self._client.chat.completions.create(**kwargs)
            choice = resp.choices[0]
            content = choice.message.content or ""
            tool_calls = None
            if choice.message.tool_calls:
                # 每个tool_call是List[ChatCompletionMessageToolCall]对象Object，把返回的SDK对象转换为json dict
                tool_calls = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in choice.message.tool_calls
                ]

            usage = {}
            if resp.usage:
                usage = {
                    "prompt_tokens": resp.usage.prompt_tokens,
                    "completion_tokens": resp.usage.completion_tokens,
                    "total_tokens": resp.usage.total_tokens,
                }
            elapsed = self._elapsed_ms(start)
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
            logger.warning("llm.rate_limit", adapter=self._name, error=str(e))
            raise
        except APITimeoutError as e:
            logger.warning("llm.timeout", adapter=self._name, error=str(e))
            raise
        except APIError as e:
            logger.error("llm.api_error", adapter=self._name, error=str(e))
            raise

    async def stream(
        self,
        messages: list[Message],
        config: LLMConfig,
    ) -> AsyncIterator[str]:
        """
        流式生成
        :param messages:
        :param config:
        :return:
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
        embedding模型
        :param texts:
        :return:
        """
        resp = await self._client.embeddings.create(
            model=settings.embedding_model,
            input=texts,
        )
        return [item.embedding for item in resp.data]

    async def health_check(self) -> bool:
        try:
            await self._client.models.list()
            return True
        except Exception:
            return False


# ── 工厂函数 ─────────────────────────────────────────────────────────────────

def build_openai_model() -> OpenAIModel | None:
    if not settings.openai_api_key:
        return None
    return OpenAIModel(
        api_key=settings.openai_api_key,
        api_base=settings.openai_api_base,
        model=settings.openai_default_model,
        adapter_name="openai",
    )


def build_deepseek_model() -> OpenAIModel | None:
    if not settings.deepseek_api_key:
        return None
    return OpenAIModel(
        api_key=settings.deepseek_api_key,
        api_base=settings.deepseek_api_base,
        model=settings.deepseek_model,
        adapter_name="deepseek",
    )


def build_local_model() -> OpenAIModel | None:
    if not (settings.local_llm_api_base and settings.local_llm_model):
        return None
    return OpenAIModel(
        api_key="local",
        api_base=settings.local_llm_api_base,
        model=settings.local_llm_model,
        adapter_name="local",
    )
