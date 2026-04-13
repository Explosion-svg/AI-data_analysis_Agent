"""
memory/conversation_memory.py — 对话历史管理

这层记忆只负责“对话语义”，不负责“执行状态”。

允许存储的内容：
- user / assistant 的自然语言消息
- 会话级别的轻量 metadata，例如 run_id、工作摘要、来源标签、pinned facts

不应该存储的内容：
- 每一步工具调用明细
- 完整 SQL、原始结果集、图表配置
- 当前任务的状态机、失败重试轨迹

这些执行期信息应该进入 work_memory，而不是 conversation_memory。

当前实现采用分层记忆策略：
- recent turns：保留最近 N 轮原始对话
- rolling summary：用 LLM 把被淘汰旧对话合并进长期摘要
- pinned facts：长期保留的业务口径、用户偏好、关键约束

和纯滑动窗口相比，这里的旧上下文不会被硬删除。
当 recent turns 超出窗口时，旧消息会和已有长期摘要一起被 LLM 重新压缩，
形成一段稳定的“滚动摘要”；如果 LLM 不可用，再降级为规则摘要，保证主流程不断。
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, TYPE_CHECKING

from ai_data_agent.config.config import settings
from ai_data_agent.model_gateway.base_model import Message
from ai_data_agent.model_gateway.router import TaskType
from ai_data_agent.observability.logger import get_logger

if TYPE_CHECKING:
    from ai_data_agent.model_gateway.router import ModelRouter
    from ai_data_agent.reliability.circuit_breaker import CircuitBreaker

logger = get_logger(__name__)

_SUMMARY_MAX_CHARS = 1800
_SUMMARY_INPUT_MAX_CHARS = 4000
_PINNED_FACTS_MAX_ITEMS = 12
_PINNED_FACT_LENGTH = 240


@dataclass
class Turn:
    """
    一条对话消息。

    metadata 的存在是为了给会话层留少量诊断/桥接信息。
    metadata 不会作为原始对话直接进入 prompt；只有明确声明为 pinned fact 的
    内容会进入长期记忆层，避免 conversation_memory 演变成无边界状态仓库。
    """

    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_message(self) -> Message:
        """
        转换成发给模型的 Message。

        这里只保留 role 和 content，故意丢弃 metadata。
        原因是 metadata 可能包含 run_id、工作摘要等控制信息，
        它们不属于“对话原文”，不应无控制地重新注入模型。
        """
        return Message(role=self.role, content=self.content)


@dataclass
class ConversationState:
    """
    单个会话的分层记忆状态。

    recent_turns 保留最近原文，负责短期指代和细节。
    rolling_summary 保留更早历史，负责长期语义。
    pinned_facts 保留事实锚点，负责跨轮稳定约束。
    """

    recent_turns: list[Turn] = field(default_factory=list)
    rolling_summary: str = ""
    pinned_facts: list[str] = field(default_factory=list)


class ConversationMemory:
    """
    对话记忆。

    职责边界：
    - 回答“用户和助手聊过什么”
    - 为下轮生成 prompt 提供近期原文、长期摘要、长期事实锚点

    非职责：
    - 不记录工具执行状态
    - 不承担任务规划状态
    - 不保存大体量执行产物

    add() 是 async，因为滚动摘要需要调用 LLM。
    get_messages() 仍然保持同步，因为它只读取已有状态并组装 prompt 消息。
    """

    def __init__(
        self,
        max_turns: int | None = None,
        *,
        router: "ModelRouter | None" = None,
        breaker: "CircuitBreaker | None" = None,
    ) -> None:
        self._max_turns = max_turns or settings.conversation_max_turns
        self._router = router
        self._breaker = breaker
        # conversation_id -> ConversationState
        self._store: dict[str, ConversationState] = defaultdict(ConversationState)

    async def add(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        追加一条对话消息。

        写入流程：
        1. 保存最近原始消息
        2. 从 metadata 中吸收 pinned facts
        3. 如果 recent window 超限，就把溢出的旧消息合并进 rolling summary

        注意：这里不会从任意 metadata 中自动取内容进 prompt。
        只有明确命名为 pinned_fact / pinned_facts 的内容才会长期保留。
        """
        turn = Turn(role=role, content=content, metadata=metadata or {})
        state = self._store[conversation_id]
        state.recent_turns.append(turn)
        self._update_pinned_facts(state, turn)
        await self._roll_recent_turns_into_summary(conversation_id, state)
        logger.debug(
            "memory.add",
            conversation_id=conversation_id,
            role=role,
            recent_messages=len(state.recent_turns),
            has_summary=bool(state.rolling_summary),
            pinned_facts=len(state.pinned_facts),
        )

    def get_messages(self, conversation_id: str) -> list[Message]:
        """
        返回发给模型的对话消息列表。

        返回顺序：
        1. system: 长期会话记忆，包括 rolling summary 和 pinned facts
        2. user/assistant: 最近原始对话

        长期记忆用 system message 注入，因为它不是原始用户/助手消息，
        而是系统压缩后的记忆视图。
        """
        state = self._store.get(conversation_id)
        if state is None:
            return []

        messages: list[Message] = []
        memory_block = self._build_memory_block(state)
        if memory_block:
            messages.append(Message(role="system", content=memory_block))
        messages.extend(t.to_message() for t in state.recent_turns)
        return messages

    def get_turns(self, conversation_id: str) -> list[Turn]:
        """
        返回最近原始 Turn 对象，供调试或 API 层查看。

        与 get_messages() 的区别：
        - get_messages() 面向模型输入，会包含长期记忆 system block
        - get_turns() 面向系统内部检查，只返回 recent_turns 原文
        """
        state = self._store.get(conversation_id)
        if state is None:
            return []
        return list(state.recent_turns)

    def clear(self, conversation_id: str) -> None:
        self._store.pop(conversation_id, None)
        logger.info("memory.cleared", conversation_id=conversation_id)

    def list_conversations(self) -> list[str]:
        return list(self._store.keys())

    def summary(self, conversation_id: str) -> dict:
        """
        返回会话层摘要。

        这里只统计 conversation_memory 自身的三个层次，不尝试汇总执行状态；
        执行状态应由 work_memory 自己汇总。
        """
        state = self._store.get(conversation_id)
        if state is None:
            return {
                "conversation_id": conversation_id,
                "turns": 0,
                "messages": 0,
                "has_rolling_summary": False,
                "rolling_summary_chars": 0,
                "pinned_facts": 0,
            }

        return {
            "conversation_id": conversation_id,
            "turns": len(state.recent_turns) // 2,
            "messages": len(state.recent_turns),
            "has_rolling_summary": bool(state.rolling_summary),
            "rolling_summary_chars": len(state.rolling_summary),
            "pinned_facts": len(state.pinned_facts),
        }

    async def _roll_recent_turns_into_summary(
        self,
        conversation_id: str,
        state: ConversationState,
    ) -> None:
        """
        将窗口外的旧消息合并进滚动摘要。

        这是 conversation_memory 的核心策略：
        - recent_turns 保留最近细节
        - overflow_turns 不硬删，而是合并进 rolling_summary
        - rolling_summary 每次都是“旧摘要 + 新溢出消息”的再压缩结果

        这样可以避免摘要无限增长，也避免旧上下文突然消失。
        """
        max_messages = self._max_turns * 2
        if len(state.recent_turns) <= max_messages:
            return

        overflow_count = len(state.recent_turns) - max_messages
        if overflow_count < 2:
            return

        # 对话通常以 user/assistant 成对出现。
        # 如果刚写入 user 就超窗，先暂时保留，等 assistant 回复写入后再一起摘要，
        # 避免长期摘要里出现脱离回答的孤立问题，削弱多轮语义连续性。
        overflow_count -= overflow_count % 2
        if overflow_count <= 0:
            return

        overflow_turns = state.recent_turns[:overflow_count]
        state.recent_turns = state.recent_turns[overflow_count:]

        try:
            state.rolling_summary = await self._summarize_with_llm(
                existing_summary=state.rolling_summary,
                overflow_turns=overflow_turns,
                pinned_facts=state.pinned_facts,
            )
        except Exception as e:
            logger.warning(
                "conversation_memory.summary_failed",
                conversation_id=conversation_id,
                error=str(e),
            )
            state.rolling_summary = self._fallback_merge_summary(
                existing_summary=state.rolling_summary,
                overflow_turns=overflow_turns,
            )

    async def _summarize_with_llm(
        self,
        *,
        existing_summary: str,
        overflow_turns: list[Turn],
        pinned_facts: list[str],
    ) -> str:
        """
        使用 LLM 生成滚动摘要。

        输入包含三部分：
        - 已有长期摘要
        - 本次被窗口淘汰的旧消息
        - 已固定的 pinned facts

        输出要求是紧凑的中文 bullet summary，重点保留：
        - 用户目标和偏好
        - 业务口径 / 指标定义 / 过滤条件
        - 已经确认的分析结论
        - 仍然未解决或需要延续的上下文
        """
        if self._router is None or self._breaker is None:
            raise RuntimeError("ConversationMemory LLM summarizer is not configured.")

        transcript = self._format_turns_for_summary(overflow_turns)
        prompt = (
            "你是数据分析 Agent 的会话记忆摘要器。请把旧的长期摘要和这次溢出窗口的"
            "对话合并为新的长期摘要。\n\n"
            "要求：\n"
            "- 只保留后续对话可能还会用到的信息。\n"
            "- 重点保留业务口径、指标定义、过滤条件、用户偏好、已确认结论和未完成问题。\n"
            "- 不要保留寒暄、重复解释、工具调用细节或临时错误信息。\n"
            "- 不要编造对话中没有出现的信息。\n"
            f"- 总长度控制在 {_SUMMARY_MAX_CHARS} 字以内。\n"
            "- 输出中文，使用简洁 bullet list。\n\n"
            f"已固定事实（这些已单独长期保留，不需要重复展开）：\n{self._format_pinned_facts(pinned_facts)}\n\n"
            f"旧的长期摘要：\n{existing_summary or '(无)'}\n\n"
            f"本次需要合并的旧对话：\n{transcript}\n"
        )
        resp = await self._breaker.call(
            self._router.generate,
            messages=[
                Message(
                    role="system",
                    content="你只负责压缩会话记忆，不回答用户问题。",
                ),
                Message(role="user", content=prompt),
            ],
            task_type=TaskType.SIMPLE,
            max_tokens=700,
            temperature=0.0,
        )
        return self._clean_summary(resp.content)

    def _fallback_merge_summary(
        self,
        *,
        existing_summary: str,
        overflow_turns: list[Turn],
    ) -> str:
        """
        LLM 摘要失败时的降级策略。

        这里仍然不是理想摘要，但它保证两个性质：
        - 主对话流程不会因为记忆摘要失败而中断
        - 旧上下文不会直接硬删除
        """
        fragments = [existing_summary.strip()] if existing_summary.strip() else []
        for turn in overflow_turns:
            text = self._compact_text(turn.content, max_len=220)
            if not text:
                continue
            prefix = "用户" if turn.role == "user" else "助手"
            fragments.append(f"- {prefix}: {text}")
        merged = "\n".join(fragments)
        return merged[-_SUMMARY_MAX_CHARS:]

    def _build_memory_block(self, state: ConversationState) -> str:
        """
        组装给模型看的长期会话记忆块。

        pinned facts 放在 rolling summary 前面。
        原因是 pinned facts 往往是业务口径或用户偏好，优先级比普通摘要更高。
        """
        if not state.rolling_summary and not state.pinned_facts:
            return ""

        lines = ["## Conversation Memory"]
        if state.pinned_facts:
            lines.append("")
            lines.append("Pinned facts:")
            for fact in state.pinned_facts[-_PINNED_FACTS_MAX_ITEMS:]:
                lines.append(f"- {fact}")
        if state.rolling_summary:
            lines.append("")
            lines.append("Rolling summary of earlier conversation:")
            lines.append(state.rolling_summary)
        return "\n".join(lines)

    def _update_pinned_facts(self, state: ConversationState, turn: Turn) -> None:
        """
        matedata是对话记录中给系统(Agent_loop看的)
        content是给LLM看的
        从 metadata 中提取应长期保留的事实锚点。

        支持两类输入：单条和多条
        - metadata["pinned_fact"] = str
        - metadata["pinned_facts"] = list[str]

        注意这里不自动分析 content。pinned facts 的提取由上游 AgentLoop 显式完成，
        这样可以把“哪些内容值得长期固定”的决策放在编排层，而不是让 memory
        在每条消息上偷偷做额外推理。
        """
        raw_facts: list[str] = []
        if isinstance(turn.metadata.get("pinned_fact"), str):
            raw_facts.append(turn.metadata["pinned_fact"])

        meta_facts = turn.metadata.get("pinned_facts")
        if isinstance(meta_facts, list):
            raw_facts.extend(item for item in meta_facts if isinstance(item, str))

        for fact in raw_facts:
            normalized = self._compact_text(fact, max_len=_PINNED_FACT_LENGTH)
            if not normalized or normalized in state.pinned_facts:
                continue
            state.pinned_facts.append(normalized)
        state.pinned_facts = state.pinned_facts[-_PINNED_FACTS_MAX_ITEMS:]

    @staticmethod
    def _format_turns_for_summary(turns: list[Turn]) -> str:
        lines: list[str] = []
        total_chars = 0
        for turn in turns:
            content = " ".join(turn.content.split())
            if not content:
                continue
            line = f"{turn.role}: {content}"
            total_chars += len(line)
            if total_chars > _SUMMARY_INPUT_MAX_CHARS:
                lines.append("[truncated]")
                break
            lines.append(line)
        return "\n".join(lines) or "(无)"

    @staticmethod
    def _format_pinned_facts(pinned_facts: list[str]) -> str:
        if not pinned_facts:
            return "(无)"
        return "\n".join(f"- {fact}" for fact in pinned_facts[-_PINNED_FACTS_MAX_ITEMS:])

    @staticmethod
    def _clean_summary(summary: str) -> str:
        """
        清理 LLM 摘要输出。

        摘要是长期上下文的一部分，必须控制体积；这里不做复杂解析，只做最小清洗。
        """
        cleaned = summary.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").strip()
        return cleaned[:_SUMMARY_MAX_CHARS]

    @staticmethod
    def _compact_text(text: str, *, max_len: int) -> str:
        cleaned = " ".join(text.split()).strip()
        if not cleaned:
            return ""
        return cleaned[:max_len]


# 全局单例
_memory: ConversationMemory | None = None


def get_memory() -> ConversationMemory:
    global _memory
    if _memory is None:
        _memory = ConversationMemory()
    return _memory
