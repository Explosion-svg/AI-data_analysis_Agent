"""
memory/interfaces.py — Memory/Cache 抽象接口定义

职责：
  定义对话记忆（ConversationMemory）和结果缓存（CacheMemory）的抽象基类，
  让编排层只依赖接口而不依赖具体实现，从而支持后端替换。

为什么要抽象接口？
  - 开发环境：使用内存版（InMemory），零依赖，启动快
  - 测试环境：可以注入 Mock 实现，精确控制测试行为
  - 生产环境：可以逐步切换到 Redis 版，无需修改调用方代码

接口稳定性：
  接口一旦定义，不应随意修改——修改接口意味着要同步修改所有实现。
  当前 BaseConversationMemory 和 BaseCacheMemory 已有内存版和 Redis 版两套实现，
  接口变更成本较高。

设计说明：
  - BaseConversationMemory：async add 方法，因为生成摘要需要调用 LLM（IO 操作）
    但 get_messages / get_turns / clear / summary 是同步的（只读内存状态）
  - BaseCacheMemory：全部同步，因为内存 LRU 操作不需要异步
    Redis 版需要把同步接口的实现改为同步 Redis 操作（不依赖 asyncio）
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ai_data_agent.model_gateway.base_model import Message


class BaseConversationMemory(ABC):
    """
    对话记忆的抽象接口。

    定义了管理多轮对话历史的最小必要接口：
    - add：追加一条对话消息（可能触发摘要生成，因此是 async）
    - get_messages：获取发给 LLM 的消息列表（同步，只读内存）
    - get_turns：获取原始 Turn 对象列表（同步，供调试和 API 展示）
    - clear：清除指定会话的所有历史（同步，只操作内存/Redis）
    - summary：返回会话统计摘要（同步，只读状态）

    实现约束：
    - 同一个 conversation_id 的数据必须严格隔离
    - get_messages 的输出要能直接传给 LLM router.generate()
    - clear 必须同时清除所有内部状态（recent_turns、rolling_summary、pinned_facts）
    """

    @abstractmethod
    async def add(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        追加一条对话消息。

        Args:
            conversation_id: 会话唯一标识（建议使用租户作用域键）
            role: 消息角色，"user" 或 "assistant"
            content: 消息内容（自然语言文本）
            metadata: 可选元数据（如 pinned_facts、run_id 等控制信息）
        """
        raise NotImplementedError

    @abstractmethod
    def get_messages(self, conversation_id: str) -> list[Message]:
        """
        获取发给 LLM 的完整消息列表。

        返回值包含：
        - 长期记忆 system block（rolling_summary + pinned_facts）
        - 近期原始对话（user/assistant messages）

        Args:
            conversation_id: 会话唯一标识

        Returns:
            Message 列表，可直接传给 LLM generate()
        """
        raise NotImplementedError

    @abstractmethod
    def get_turns(self, conversation_id: str) -> list[Any]:
        """
        获取原始 Turn 对象列表（近期对话）。

        与 get_messages 的区别：
        - get_messages 面向模型输入，包含长期记忆 system block
        - get_turns 面向内部检查/调试，只返回近期原始 Turn

        Args:
            conversation_id: 会话唯一标识

        Returns:
            Turn 对象列表
        """
        raise NotImplementedError

    @abstractmethod
    def clear(self, conversation_id: str) -> None:
        """
        清除指定会话的所有对话历史。

        必须清除所有层次的数据：recent_turns、rolling_summary、pinned_facts。
        用于：用户主动开始新任务、GDPR 数据删除请求、调试重置等。

        Args:
            conversation_id: 要清除的会话标识
        """
        raise NotImplementedError

    @abstractmethod
    def summary(self, conversation_id: str) -> dict[str, Any]:
        """
        返回指定会话的统计摘要。

        用于监控和调试，了解各会话的记忆使用情况。

        Args:
            conversation_id: 会话标识

        Returns:
            包含 turns、messages、rolling_summary_chars 等统计信息的字典
        """
        raise NotImplementedError


class BaseCacheMemory(ABC):
    """
    结果缓存的抽象接口。

    缓存相同问题的 Agent 响应，避免重复调用 LLM 和数据库。
    实现要求：TTL 过期 + LRU 淘汰，两者缺一不可。

    接口设计选择（全同步 vs 全异步）：
    - 内存版：所有操作都在内存中完成，同步即可
    - Redis 版：为保持接口一致，Redis 操作也使用同步 Redis 客户端（redis-py），
      而不是 async redis，这样不需要改接口

    Key 设计：
    - make_key() 是静态方法，接受任意参数并生成稳定的缓存键
    - 通常使用 SHA-256 哈希，确保相同参数总是生成相同的 key
    """

    @staticmethod
    @abstractmethod
    def make_key(*parts: Any) -> str:
        """
        将任意参数组合生成稳定、唯一的缓存键。

        实现要求：
        - 确定性：相同参数总是生成相同的 key
        - 碰撞概率极低：不同参数生成相同 key 的概率可忽略
        - 安全性：key 不能反推原始参数（推荐使用哈希）

        Args:
            *parts: 构成缓存键的各个部分（任意类型，内部序列化为 JSON）

        Returns:
            稳定的缓存键字符串（通常是 32-64 位十六进制哈希）
        """
        raise NotImplementedError

    @abstractmethod
    def get(self, key: str) -> Any:
        """
        获取缓存值。

        Args:
            key: 缓存键

        Returns:
            缓存值，未命中或已过期时返回 None
        """
        raise NotImplementedError

    @abstractmethod
    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """
        写入缓存。

        Args:
            key: 缓存键
            value: 要缓存的值
            ttl: 生存时间（秒），None 时使用默认 TTL
        """
        raise NotImplementedError

    @abstractmethod
    def delete(self, key: str) -> None:
        """
        删除指定缓存项。

        Args:
            key: 要删除的缓存键
        """
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """清除所有缓存（通常用于测试或调试）。"""
        raise NotImplementedError

    @abstractmethod
    def stats(self) -> dict[str, Any]:
        """
        返回缓存统计信息（size、max_size、ttl 等）。

        用于 health_report() 和监控。

        Returns:
            包含缓存统计信息的字典
        """
        raise NotImplementedError
