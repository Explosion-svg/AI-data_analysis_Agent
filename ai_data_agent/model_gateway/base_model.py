"""
model_gateway/base_model.py — LLM 抽象接口与核心数据类

职责：
  定义 LLM 交互的核心数据结构和抽象基类，
  让上层代码不依赖具体 LLM 实现（OpenAI/DeepSeek/Anthropic/本地）。

核心数据类：
  - Message：LLM 消息（role + content + 可选字段）
  - LLMResponse：LLM 响应（content + 使用量 + 延迟等）
  - LLMConfig：LLM 调用配置（模型名 + 温度 + max_tokens 等）

抽象基类：
  - BaseLLM：所有 LLM 适配器必须实现的接口
    统一了 generate / stream / embed / health_check 四个核心方法

为什么要有这层抽象？
  - 可替换性：上层代码只依赖 BaseLLM，可以轻松切换底层 LLM
  - 可测试性：单元测试时可以注入 MockLLM，不需要真实 API Key
  - 可扩展性：新增 Anthropic 适配器只需实现 BaseLLM，无需修改上层

OpenAI function calling 的实现位置：
  - Message.tool_calls 字段存储工具调用请求
  - Message.tool_call_id 字段存储工具调用 ID（用于工具响应）
  - 具体的 function calling 参数序列化在 openai_model.py 中处理
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator


@dataclass
class Message:
    """
    LLM 消息对象，对应 OpenAI Chat API 的 message 格式。

    role 枚举值及含义：
    - "system"：系统提示词，定义模型行为和上下文
    - "user"：用户消息
    - "assistant"：模型回复
    - "tool"：工具调用结果（function calling 响应）

    可选字段（function calling 场景）：
    - name：工具名称（当 role="tool" 时使用，标识哪个工具返回了结果）
    - tool_call_id：工具调用 ID（将工具结果与请求对应，OpenAI 要求必填）
    - tool_calls：工具调用请求列表（当 role="assistant" 且模型请求工具时填充）

    序列化说明：
    - 发送给 OpenAI API 前，Message 需要通过 _to_openai_messages() 转换为字典
    - None 字段不应包含在发送的字典中（OpenAI API 不接受多余字段）
    """
    role: str                              # system | user | assistant | tool
    content: str                           # 消息正文
    name: str | None = None               # 工具名称（role=tool 时使用）
    tool_call_id: str | None = None       # 工具调用 ID（与请求匹配）
    tool_calls: list[dict[str, Any]] | None = None  # 工具调用请求列表


@dataclass
class LLMResponse:
    """
    LLM 调用响应对象，统一了不同 LLM 厂商的返回格式。

    核心字段：
    - content：模型的文本回复（如果有工具调用则可能为空字符串）
    - model：实际使用的模型 ID（可能与请求时不同，如别名映射）
    - usage：token 使用量统计（用于成本核算和配额控制）
    - finish_reason：停止原因（"stop"=正常结束，"tool_calls"=需要工具，"length"=超 max_tokens）
    - tool_calls：模型请求执行的工具调用列表
    - latency_ms：端到端延迟（毫秒），由适配器层计时

    usage 字典格式：
        {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}

    便捷属性：
    - prompt_tokens / completion_tokens / total_tokens：直接访问 usage 中的各项
      避免每次都写 response.usage.get("prompt_tokens", 0)
    """
    content: str                           # 模型回复文本
    model: str                             # 实际使用的模型 ID
    usage: dict[str, int] = field(default_factory=dict)  # token 使用量
    finish_reason: str = "stop"           # 停止原因
    tool_calls: list[dict[str, Any]] | None = None  # 工具调用请求
    latency_ms: float = 0.0              # 端到端延迟（毫秒）

    @property
    def prompt_tokens(self) -> int:
        """输入 prompt 消耗的 token 数（用于成本核算）。"""
        return self.usage.get("prompt_tokens", 0)

    @property
    def completion_tokens(self) -> int:
        """模型生成的 token 数（决定了主要成本）。"""
        return self.usage.get("completion_tokens", 0)

    @property
    def total_tokens(self) -> int:
        """总 token 数（prompt + completion）。"""
        return self.usage.get("total_tokens", 0)


@dataclass
class LLMConfig:
    """
    LLM 调用配置，控制模型的行为参数。

    各字段含义：
    - model：使用的具体模型 ID（如 "gpt-4o"、"deepseek-chat"）
    - temperature：随机性（0.0=确定性最强，2.0=最随机），数据分析推荐 0.0
    - max_tokens：最大生成 token 数（控制回复长度和成本）
    - timeout：单次调用超时秒数（超时抛出 APITimeoutError）
    - top_p：核采样参数（通常与 temperature 二选一调整，不同时修改两个）
    - stop：停止词列表（遇到这些词时停止生成）
    - tools：OpenAI function calling 格式的工具列表（JSON Schema）
    - tool_choice：工具选择策略（"auto"=自动, "none"=不调用, 具体工具名=强制调用）
    """
    model: str
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout: float = 60.0
    top_p: float = 1.0
    stop: list[str] | None = None
    tools: list[dict[str, Any]] | None = None         # function calling 工具定义
    tool_choice: str | dict | None = None             # 工具选择策略


class BaseLLM(ABC):
    """
    所有 LLM 适配器的抽象基类。

    定义了 LLM 交互的最小接口：
    - generate：非流式完整响应（主要接口）
    - stream：流式 token 生成（用于实时展示）
    - embed：文本向量化（用于 RAG 检索）
    - health_check：健康探测

    子类实现规范：
    - 必须实现所有 @abstractmethod 方法
    - 可以调用 _make_config() 构建带 override 的配置
    - 可以调用 _elapsed_ms() 计算耗时

    当前已有实现：
    - OpenAIModel：支持 OpenAI / DeepSeek / Azure / 本地 Ollama
    （未来可添加 AnthropicModel、GeminiModel 等）
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        模型适配器的唯一标识符（如 "openai-gpt4o"、"deepseek"）。

        用于日志记录、指标标签和路由器的模型注册表键名。
        """

    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        config: LLMConfig,
    ) -> LLMResponse:
        """
        发送消息列表，返回完整响应（非流式）。

        这是主要的 LLM 调用接口，支持 function calling。
        适合绝大多数场景：Planner、Executor、Agent ReAct 循环。

        Args:
            messages: 对话消息列表（包含 system/user/assistant/tool 消息）
            config: 调用配置（模型、温度、工具列表等）

        Returns:
            包含模型回复、工具调用请求、token 使用量的 LLMResponse
        """

    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        config: LLMConfig,
    ) -> AsyncIterator[str]:
        """
        流式生成响应，逐 token 返回字符串。

        适合需要实时展示部分回复的场景（如 WebSocket 流式推送）。
        当前主流程主要使用 generate()（非流式），stream() 为未来扩展预留。

        Args:
            messages: 对话消息列表
            config: 调用配置

        Yields:
            逐个 token 的字符串片段
        """

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        生成文本的向量 embedding。

        用于 RAG 检索（文档和 query 都需要向量化）和 schema 语义索引。
        返回的向量维度由模型决定（如 text-embedding-3-small 返回 1536 维）。

        Args:
            texts: 要向量化的文本列表（批量处理更高效）

        Returns:
            向量列表，每个向量是 float 列表，长度等于 embedding_dimension
        """

    @abstractmethod
    async def health_check(self) -> bool:
        """
        检查模型 API 是否可用。

        通常通过发送一个轻量请求（如列出模型列表）来测试连通性。
        用于启动时检查和熔断器的恢复探测。

        Returns:
            True 表示 API 可用，False 表示不可用
        """

    def _make_config(self, **overrides: Any) -> LLMConfig:
        """
        构建带 override 的默认 LLMConfig。

        允许调用方只覆盖部分参数，其余使用全局默认值（settings）。
        通过 pop() 处理已知字段，然后用 setattr 处理其余字段。

        Args:
            **overrides: 要覆盖的配置参数（如 model="gpt-4o-mini"、temperature=0.3）

        Returns:
            完整的 LLMConfig 对象
        """
        from ai_data_agent.config.config import settings
        cfg = LLMConfig(
            model=overrides.pop("model", settings.openai_default_model),
            temperature=overrides.pop("temperature", settings.llm_temperature),
            max_tokens=overrides.pop("max_tokens", settings.llm_max_tokens),
            timeout=overrides.pop("timeout", settings.llm_timeout),
        )
        # 处理剩余的 override 参数（如 tools、tool_choice、stop 等）
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg

    @staticmethod
    def _elapsed_ms(start: float) -> float:
        """
        计算从 start 到现在的耗时（毫秒）。

        使用 time.perf_counter() 而非 time.time()，
        perf_counter 提供更高精度（纳秒级），适合性能测量。

        Args:
            start: 开始时间（time.perf_counter() 的返回值）

        Returns:
            耗时（毫秒，浮点数）
        """
        return (time.perf_counter() - start) * 1000
