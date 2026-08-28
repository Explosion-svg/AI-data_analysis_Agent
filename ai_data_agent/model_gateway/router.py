"""
model_gateway/router.py — 智能模型路由器

职责：
  根据任务类型（TaskType）从已注册的 LLM 适配器中选择最优模型，
  支持多模型 Fallback 链（主模型失败自动切换到备用模型），
  并对所有 LLM 调用施加统一的并发限流和重试策略。

路由策略（按优先级）：
  SIMPLE   → openai → deepseek → local （简单问答，fast model 节省成本）
  COMPLEX  → openai → deepseek → local （复杂分析，最强模型保证质量）
  CODE     → deepseek → openai → local （代码生成，DeepSeek 代码能力更强）
  EMBEDDING→ openai → deepseek → local （embedding 用 OpenAI text-embedding 系列）

Fallback 链工作方式：
  1. 按 task_type 路由选出主模型（primary）
  2. 调用主模型，如果失败（任何异常）：
  3. 遍历注册表中的其他所有模型（按注册顺序）
  4. 逐个尝试，第一个成功的即返回
  5. 全部失败则抛出 RuntimeError("All LLM adapters failed.")

并发控制：
  通过 get_limiter().limit("llm") 施加全局 LLM 并发上限（settings.llm_concurrency）
  防止同时发起过多 LLM 请求导致 rate limit 或下游服务过载

重试：
  通过 @async_retry() 装饰器对 _generate_with_retry 施加指数退避重试
  重试粒度是"单个模型的单次调用"，不跨模型重试
  （跨模型切换由 generate() 的 fallback 逻辑处理）

模型注册表：
  _registry: dict[str, BaseLLM]，key 是适配器名称（"openai"/"deepseek"/"local"）
  启动时通过工厂函数自动注册所有已配置的适配器（未配置的跳过）
  至少需要一个适配器，否则抛出 RuntimeError
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

from openai import APIConnectionError, APITimeoutError, RateLimitError

from ai_data_agent.config.config import settings
from ai_data_agent.model_gateway.base_model import BaseLLM, LLMConfig, LLMResponse, Message
from ai_data_agent.model_gateway.openai_model import (
    build_deepseek_model,
    build_local_model,
    build_openai_model,
)
from ai_data_agent.observability.logger import get_logger
from ai_data_agent.reliability.concurrency import ConcurrencyLimitExceeded, get_limiter
from ai_data_agent.reliability.retry import async_retry

logger = get_logger(__name__)

# 可重试的 LLM 异常：仅供应商/传输类错误（P2-9）
# - 本地过载（ConcurrencyLimitExceeded）不是模型故障，不重试、不 fallback
# - APIError（如 400 Bad Request）重试无意义，直接进入 fallback 判定
_RETRYABLE_LLM_ERRORS = (RateLimitError, APITimeoutError, APIConnectionError)


class TaskType(str, Enum):
    """
    LLM 任务类型枚举，用于路由决策。

    继承 str 的原因：可以直接用字符串值比较（TaskType.SIMPLE == "simple"），
    也方便序列化（JSON 序列化时直接输出字符串值而不是枚举对象）。

    各类型的路由策略：
    - SIMPLE：简单问答（query rewriting、pinned facts 提取）→ fast model 节省成本
    - COMPLEX：复杂分析和规划（ReAct 主循环、grounded answer 生成）→ 最强模型
    - CODE：代码和 SQL 生成（Executor 参数生成）→ DeepSeek 优先（代码能力更强）
    - EMBEDDING：文本向量化（RAG 检索、schema 索引）→ OpenAI embedding 模型
    """
    SIMPLE = "simple"        # 简单问答 → fast model（gpt-4o-mini）
    COMPLEX = "complex"      # 复杂规划和分析 → 强模型（gpt-4o）
    CODE = "code"            # 代码 / SQL 生成 → DeepSeek（代码更强）
    EMBEDDING = "embedding"  # 文本向量化 → OpenAI text-embedding-3-small


@dataclass
class ModelRouter:
    """
    模型路由器：统一管理多个 LLM 适配器的选择和 Fallback。

    使用 @dataclass 的原因：
    - 字段声明更清晰（_registry 明确标注类型）
    - 支持 __post_init__ 在实例化后自动调用 _build_registry()
    - 与 field(default_factory=dict) 配合，避免可变默认值的 Python 陷阱

    注意：_registry 是私有字段（以 _ 开头），
    外部代码只能通过 list_models()、generate()、embed() 等公开接口访问。
    """
    _registry: dict[str, BaseLLM] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        dataclass 实例化后的初始化钩子。

        __post_init__ 在 __init__ 自动生成的参数赋值后调用，
        相当于手动 __init__ 中的后处理逻辑。
        这里用来触发 _build_registry()，注册所有已配置的 LLM 适配器。
        """
        self._build_registry()

    def _build_registry(self) -> None:
        """
        扫描并注册所有已配置的 LLM 适配器。

        注册顺序（openai → deepseek → local）决定了 Fallback 链的默认顺序：
        当主模型失败时，按注册顺序遍历其他适配器，第一个成功的即返回。
        这也是为什么 openai 排在最前面（通常最稳定）。

        工厂函数返回 None 表示对应服务商未配置（缺少 API Key），
        返回 None 的适配器会被跳过（不加入注册表）。

        Raises:
            RuntimeError: 没有任何适配器被成功注册
                         （所有服务商的 API Key 都未配置）
        """
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
        根据任务类型选择最优的 LLM 适配器。

        选择逻辑：
        - 按任务类型定义优先级列表（priority）
        - 遍历优先级列表，返回第一个已注册的适配器
        - 如果所有优先级列表中的适配器都未注册，取注册表中任意一个（兜底）

        当前 CODE 任务优先 DeepSeek 的原因：
        - DeepSeek-Coder 系列在代码生成基准测试中表现优异
        - SQL 生成也是代码能力的一部分
        - 但如果 DeepSeek 未配置，自动 fallback 到 OpenAI

        Args:
            task_type: 任务类型

        Returns:
            选择的 BaseLLM 适配器实例
        """
        priority: list[str]
        if task_type == TaskType.CODE:
            # 代码生成优先 DeepSeek（代码能力更强）
            priority = ["deepseek", "openai", "local"]
        else:
            # 其他任务都优先 OpenAI（通用能力更强，更稳定）
            priority = ["openai", "deepseek", "local"]

        for key in priority:
            if key in self._registry:
                return self._registry[key]
        # 所有优先候选都不在注册表时的兜底（理论上不会走到这里）
        return next(iter(self._registry.values()))

    def _make_config(self, task_type: TaskType, **overrides: Any) -> LLMConfig:
        """
        为任务类型构建 LLMConfig，允许通过 overrides 覆盖默认值。

        模型名称选择规则：
        - SIMPLE 且有 OpenAI → 使用 settings.openai_fast_model（gpt-4o-mini，便宜）
        - CODE 且有 DeepSeek → 使用 settings.deepseek_model（代码更强）
        - 其他 → 使用所选适配器的默认模型

        这里通过 overrides.pop() 从 kwargs 中提取已知字段，
        再通过 setattr 处理其余字段（如 tools、tool_choice、stop），
        避免 LLMConfig 构造函数需要知道所有可能的 override 参数。

        Args:
            task_type: 任务类型（用于决定默认模型名称）
            **overrides: 覆盖默认值的配置参数

        Returns:
            完整的 LLMConfig 对象
        """
        model_obj = self._select_model(task_type)
        # 按任务类型决定具体使用哪个模型名称
        if task_type == TaskType.SIMPLE and "openai" in self._registry:
            model_name = settings.openai_fast_model  # 便宜的 fast model
        elif task_type == TaskType.CODE and "deepseek" in self._registry:
            model_name = settings.deepseek_model     # DeepSeek 代码模型
        else:
            model_name = self._get_default_model(model_obj.name)

        cfg = LLMConfig(
            model=overrides.pop("model", model_name),
            temperature=overrides.pop("temperature", settings.llm_temperature),
            max_tokens=overrides.pop("max_tokens", settings.llm_max_tokens),
            timeout=overrides.pop("timeout", settings.llm_timeout),
        )
        # 处理剩余 override（如 tools、tool_choice、stop、top_p）
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg

    @staticmethod
    def _get_default_model(adapter_name: str) -> str:
        """
        根据适配器名称返回其对应的默认模型 ID。

        这个映射允许 ModelRouter 在不知道具体适配器实现细节的情况下
        获取默认模型名称（如 "openai" → settings.openai_default_model）。

        Args:
            adapter_name: 适配器名称（"openai"、"deepseek" 或 "local"）

        Returns:
            模型 ID 字符串（如 "gpt-4o"、"deepseek-chat"）
        """
        mapping = {
            "openai": settings.openai_default_model,
            "deepseek": settings.deepseek_model,
            "local": settings.local_llm_model or "local",
        }
        return mapping.get(adapter_name, settings.openai_default_model)

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    @async_retry(exceptions=_RETRYABLE_LLM_ERRORS)
    async def _call_with_limiter(
        self,
        model: BaseLLM,
        messages: list[Message],
        config: LLMConfig,
    ) -> LLMResponse:
        """
        带并发槽的 LLM 调用（重试装饰器施加于外层）。

        P2-9 修复点：
        1. 重试仅针对可恢复的供应商/传输错误
           （RateLimitError / APITimeoutError / APIConnectionError），
           不再对任意 Exception 重试（否则本地过载也会被当成模型故障）。
        2. 每次重试尝试都独立获取 llm 并发槽（limit 在方法内部），
           因此指数退避的 sleep 发生在 limit("llm") 作用域之外——
           退避等待不再白白占用并发槽，避免加剧过载。
        3. ConcurrencyLimitExceeded 不在重试列表内，会立即向上传播，
           由 generate() 捕获并原样抛出（不触发 fallback 扫射）。

        Args:
            model: 要调用的具体 LLM 适配器实例
            messages: 消息列表
            config: 调用配置

        Returns:
            LLM 响应对象

        Raises:
            ConcurrencyLimitExceeded: 本地并发超限（不重试、不 fallback）
            RateLimitError / APITimeoutError / APIConnectionError: 重试耗尽后上抛
            其他异常: 直接上抛（交由 generate() 的 fallback 逻辑处理）
        """
        async with get_limiter().limit("llm"):
            return await model.generate(messages, config)

    async def generate(
        self,
        messages: list[Message],
        task_type: TaskType = TaskType.COMPLEX,
        **config_kwargs: Any,
    ) -> LLMResponse:
        """
        路由并执行 LLM 调用，带 Fallback 链保护。

        核心调用流程：
        1. 按 task_type 选出主模型
        2. 调用 _call_with_limiter()（带并发槽 + 限定异常重试）
        3. 成功则返回
        4. ConcurrencyLimitExceeded（本地过载）→ 原样上抛，不 fallback
        5. 其他异常（供应商/传输错误）→ 进入 Fallback 链
        6. Fallback：遍历注册表中的其他所有适配器，逐个尝试
        7. 第一个成功的返回；全部失败则抛出 RuntimeError

        P2-9 修复点：
        - 本地过载不再触发全适配器 fallback 扫射（LLM 本身可能健康，
          过载时扫射只会加剧下游压力）。
        - fallback 仅在供应商/传输错误时触发。
        - fallback 配置保留 top_p/stop 等全部参数（P4-7 附带修复）。

        config_kwargs 使用场景（透传参数）：
            router.generate(messages, TaskType.COMPLEX, tools=my_tools, tool_choice="auto")
            router.generate(messages, TaskType.SIMPLE, max_tokens=400, temperature=0.3)

        Args:
            messages: 对话消息列表
            task_type: 任务类型（决定路由策略和默认模型）
            **config_kwargs: 覆盖默认 LLMConfig 的参数（如 tools、max_tokens）

        Returns:
            LLM 响应对象

        Raises:
            ConcurrencyLimitExceeded: 本地并发超限（原样上抛，供 503 处理器）
            RuntimeError: 所有适配器都失败（"All LLM adapters failed."）
        """
        model = self._select_model(task_type)
        config = self._make_config(task_type, **config_kwargs)
        try:
            return await self._call_with_limiter(model, messages, config)
        except ConcurrencyLimitExceeded:
            # 本地过载：LLM 本身可能健康，直接上抛（不重试、不 fallback）
            raise
        except Exception as e:
            # 主模型失败（供应商/传输错误），启动 Fallback 链
            logger.warning(
                "model_router.primary_failed",
                adapter=model.name,
                error=str(e),
            )
            # 遍历注册表中的所有其他适配器（跳过已经失败的主模型）
            for key, fallback in self._registry.items():
                if fallback is model:
                    continue  # 跳过已失败的主模型
                try:
                    # Fallback 保留 temperature/max_tokens/tools/stop/top_p 全部配置，
                    # 仅替换为该适配器的默认模型名
                    fallback_cfg = replace(
                        config, model=self._get_default_model(key)
                    )
                    logger.info("model_router.fallback", to=key)
                    return await self._call_with_limiter(fallback, messages, fallback_cfg)
                except ConcurrencyLimitExceeded:
                    # 本地过载：立即上抛，不继续扫射其他适配器
                    raise
                except Exception as fe:
                    logger.warning("model_router.fallback_failed", adapter=key, error=str(fe))
            # 所有 Fallback 都失败
            raise RuntimeError("All LLM adapters failed.") from e

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        生成文本向量（embedding），优先使用 OpenAI embedding 模型。

        为什么优先 OpenAI？
        - OpenAI text-embedding-3 系列是业界标杆，向量质量高
        - DeepSeek 和本地模型的 embedding 质量通常不如 OpenAI
        - 一旦向量库使用某个模型建立索引，查询时必须用同一个模型，
          切换模型需要重建整个向量库（成本高）

        并发控制：
        - 通过 limit("embedding") 独立限流（与 LLM 生成分开）
        - 原因：embedding 请求通常更小、更快，可以配置更高并发上限

        Args:
            texts: 要向量化的文本列表（支持批量）

        Returns:
            每个文本对应的 float 向量列表
        """
        model = self._registry.get("openai") or next(iter(self._registry.values()))
        async with get_limiter().limit("embedding"):
            return await model.embed(texts)

    def list_models(self) -> list[str]:
        """
        返回当前已注册的所有适配器名称列表。

        用于：
        - 日志记录（启动时打印已注册的模型）
        - API 端点（告知客户端有哪些模型可用）
        - 单元测试（验证注册是否成功）

        Returns:
            适配器名称列表（如 ["openai", "deepseek"]）
        """
        return list(self._registry.keys())

    async def close(self) -> None:
        """
        关闭所有已注册适配器的底层资源（httpx 连接池等，P2-20）。

        由 AppContainer.shutdown() 在优雅关闭时调用，
        避免 AsyncOpenAI 客户端背后的 httpx 会话泄漏。
        逐个适配器独立 try/except，单点失败不影响其余适配器关闭。
        """
        for key, model in list(self._registry.items()):
            try:
                await model.aclose()
            except Exception as e:  # pragma: no cover - 防御性
                logger.warning("model_router.close_failed", adapter=key, error=str(e))
        self._registry.clear()


# ── 全局单例 ──────────────────────────────────────────────────────────────────

_router: ModelRouter | None = None


def get_router() -> ModelRouter:
    """
    获取全局模型路由器单例（懒加载）。

    注意：这个函数创建的 ModelRouter 使用默认配置。
    如果需要在特定上下文中使用特殊配置的 router（如测试中注入 MockLLM），
    应直接创建 ModelRouter 实例，不通过此函数。

    Returns:
        全局唯一的 ModelRouter 实例

    Raises:
        RuntimeError: 没有任何 LLM 适配器配置（创建时抛出）
    """
    global _router
    if _router is None:
        _router = ModelRouter()
    return _router
