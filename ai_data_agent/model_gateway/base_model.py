"""
model_gateway/base_model.py — 抽象 LLM 接口
所有具体模型适配器必须继承 BaseLLM
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator


@dataclass
class Message:
    role: str   # system | user | assistant | tool
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


@dataclass
class LLMResponse:
    content: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str = "stop"
    tool_calls: list[dict[str, Any]] | None = None
    latency_ms: float = 0.0

    @property
    def prompt_tokens(self) -> int:
        return self.usage.get("prompt_tokens", 0)

    @property
    def completion_tokens(self) -> int:
        return self.usage.get("completion_tokens", 0)

    @property
    def total_tokens(self) -> int:
        return self.usage.get("total_tokens", 0)


@dataclass
class LLMConfig:
    model: str
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout: float = 60.0
    top_p: float = 1.0
    stop: list[str] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict | None = None


class BaseLLM(ABC):
    """所有 LLM 适配器的基类。"""

    @property
    @abstractmethod
    def name(self) -> str:
        """模型标识符，如 'openai-gpt4o'。"""

    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        config: LLMConfig,
    ) -> LLMResponse:
        """发送消息列表，返回完整响应（非流式）。"""

    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        config: LLMConfig,
    ) -> AsyncIterator[str]:
        """流式返回 token 字符串。"""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """生成文本 embedding。"""

    @abstractmethod
    async def health_check(self) -> bool:
        """健康检查，可用返回 True。"""

    def _make_config(self, **overrides: Any) -> LLMConfig:
        """构建带 override 的默认配置。"""
        from ai_data_agent.config.config import settings
        cfg = LLMConfig(
            model=overrides.pop("model", settings.openai_default_model),
            temperature=overrides.pop("temperature", settings.llm_temperature),
            max_tokens=overrides.pop("max_tokens", settings.llm_max_tokens),
            timeout=overrides.pop("timeout", settings.llm_timeout),
        )
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg

    @staticmethod
    def _elapsed_ms(start: float) -> float:
        return (time.perf_counter() - start) * 1000
