"""
memory/conversation_memory.py — 对话历史管理
滑动窗口策略：保留最近 N 轮对话
支持多 conversation_id 隔离
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from ai_data_agent.model_gateway.base_model import Message
from ai_data_agent.config.config import settings
from ai_data_agent.observability.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Turn:
    # _store键值对中的值
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)

    def to_message(self) -> Message:
        return Message(role=self.role, content=self.content)


class ConversationMemory:
    """
    线程安全的对话记忆（asyncio 单线程下安全）。
    每个 conversation_id 对应独立的历史记录。
    """

    def __init__(self, max_turns: int | None = None) -> None:
        self._max_turns = max_turns or settings.conversation_max_turns
        # conversation_id → list[Turn]
        self._store: dict[str, list[Turn]] = defaultdict(list)

    def add(self, conversation_id: str, role: str, content: str, metadata: dict | None = None) -> None:
        """追加一条消息。"""
        turn = Turn(role=role, content=content, metadata=metadata or {})
        history = self._store[conversation_id]
        history.append(turn)
        # 滑动窗口：保留最近 max_turns 轮（每轮 = user + assistant = 2条）
        max_messages = self._max_turns * 2
        if len(history) > max_messages:
            self._store[conversation_id] = history[-max_messages:]
        logger.debug(
            "memory.add",
            conversation_id=conversation_id,
            role=role,
            total=len(self._store[conversation_id]),
        )

    def get_messages(self, conversation_id: str) -> list[Message]:
        """返回 Message 列表，供 prompt_builder 使用。"""
        return [t.to_message() for t in self._store.get(conversation_id, [])]

    def get_turns(self, conversation_id: str) -> list[Turn]:
        # 返回整个值
        return list(self._store.get(conversation_id, []))

    def clear(self, conversation_id: str) -> None:
        self._store.pop(conversation_id, None)
        logger.info("memory.cleared", conversation_id=conversation_id)

    def list_conversations(self) -> list[str]:
        return list(self._store.keys())

    def summary(self, conversation_id: str) -> dict:
        history = self._store.get(conversation_id, [])
        return {
            "conversation_id": conversation_id,
            "turns": len(history) // 2,
            "messages": len(history),
        }


# 全局单例
_memory: ConversationMemory | None = None


def get_memory() -> ConversationMemory:
    global _memory
    if _memory is None:
        _memory = ConversationMemory()
    return _memory
