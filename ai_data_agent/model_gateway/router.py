"""
model_gateway/router.py — 智能模型路由
根据任务类型、复杂度自动选择最合适的模型
支持主备切换（Fallback）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ai_data_agent.config.config import settings
from ai_data_agent.model_gateway.base_model import BaseLLM, LLMConfig, LLMResponse, Message
from ai_data_agent.model_gateway.openai_model import (
    build_deepseek_model,
    build_local_model,
    build_openai_model,
)
from ai_data_agent.observability.logger import get_logger

logger = get_logger(__name__)


class TaskType(str, Enum):
    SIMPLE = "simple"        # 简单问答 -> fast model
    COMPLEX = "complex"      # 复杂规划 -> strong model
    CODE = "code"            # 代码生成 -> deepseek / strong model
    EMBEDDING = "embedding"  # embedding -> openai


@dataclass
class ModelRouter:
    """
    模型路由器：
      - 主模型池 primary
      - 备用模型池 fallback
    按优先级顺序尝试，circuit_breaker 由外层 reliability 管控。
    """
    _registry: dict[str, BaseLLM] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._build_registry()

    def _build_registry(self) -> None:
        # 注册模型
        for factory, key in [
            (build_openai_model, "openai"),
            (build_deepseek_model, "deepseek"),
            (build_local_model, "local"),
        ]:
            model = factory()
            if model:
                self._registry[key] = model
                logger.info("model_router.registered", adapter=key)

        if not self._registry:
            raise RuntimeError(
                "No LLM adapter configured. "
                "Set at least one of: OPENAI_API_KEY, DEEPSEEK_API_KEY, LOCAL_LLM_API_BASE."
            )

    # ── 路由策略 ──────────────────────────────────────────────────────────────

    def _select_model(self, task_type: TaskType) -> BaseLLM:
        """
        选择模型
        根据任务类型返回最优模型（按优先级）。
        """
        priority: list[str]
        if task_type == TaskType.SIMPLE:
            priority = ["openai", "deepseek", "local"]
        elif task_type == TaskType.COMPLEX:
            priority = ["openai", "deepseek", "local"]
        elif task_type == TaskType.CODE:
            priority = ["deepseek", "openai", "local"]
        else:
            priority = ["openai", "deepseek", "local"]

        for key in priority:
            if key in self._registry:
                return self._registry[key]
        # 兜底：取任意一个
        return next(iter(self._registry.values()))

    def _make_config(self, task_type: TaskType, **overrides: Any) -> LLMConfig:
        """
        生成模型配置
        为任务类型生成 LLMConfig。
        """
        model_obj = self._select_model(task_type)
        # 根据任务选择具体 model 名称
        if task_type == TaskType.SIMPLE and "openai" in self._registry:
            model_name = settings.openai_fast_model
        elif task_type == TaskType.CODE and "deepseek" in self._registry:
            model_name = settings.deepseek_model
        else:
            model_name = self._get_default_model(model_obj.name)

        cfg = LLMConfig(
            model=overrides.pop("model", model_name),
            temperature=overrides.pop("temperature", settings.llm_temperature),
            max_tokens=overrides.pop("max_tokens", settings.llm_max_tokens),
            timeout=overrides.pop("timeout", settings.llm_timeout),
        )
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg

    @staticmethod
    def _get_default_model(adapter_name: str) -> str:
        mapping = {
            "openai": settings.openai_default_model,
            "deepseek": settings.deepseek_model,
            "local": settings.local_llm_model or "local",
        }
        return mapping.get(adapter_name, settings.openai_default_model)

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    async def generate(
        self,
        messages: list[Message],
        task_type: TaskType = TaskType.COMPLEX,
        **config_kwargs: Any,
    ) -> LLMResponse:
        """
        核心函数，调用模型
        带 fallback 链的 generate。
        """
        model = self._select_model(task_type)
        config = self._make_config(task_type, **config_kwargs)
        try:
            resp = await model.generate(messages, config)
            return resp
        except Exception as e:
            # fallback机制触发
            logger.warning(
                "model_router.primary_failed",
                adapter=model.name,
                error=str(e),
            )
            # 尝试其他模型
            for key, fallback in self._registry.items():
                if fallback is model:
                    continue
                try:
                    fallback_cfg = LLMConfig(
                        model=self._get_default_model(key),
                        temperature=config.temperature,
                        max_tokens=config.max_tokens,
                        timeout=config.timeout,
                        tools=config.tools,
                        tool_choice=config.tool_choice,
                    )
                    logger.info("model_router.fallback", to=key)
                    return await fallback.generate(messages, fallback_cfg)
                except Exception as fe:
                    logger.warning("model_router.fallback_failed", adapter=key, error=str(fe))
            raise RuntimeError("All LLM adapters failed.") from e

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        embedding模型
        优先使用 OpenAI embedding。
        """
        model = self._registry.get("openai") or next(iter(self._registry.values()))
        return await model.embed(texts)

    def list_models(self) -> list[str]:
        return list(self._registry.keys())


# ── 单例 ──────────────────────────────────────────────────────────────────────
_router: ModelRouter | None = None


def get_router() -> ModelRouter:
    global _router
    if _router is None:
        _router = ModelRouter()
    return _router
