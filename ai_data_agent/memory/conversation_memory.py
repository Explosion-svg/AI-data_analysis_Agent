"""
memory/conversation_memory.py — 对话历史管理（三层分层记忆）

职责：
  管理用户与 Agent 之间的多轮对话历史，
  让每次请求的 LLM 输入都包含足够的历史上下文，
  同时控制 prompt token 总量不无限增长。

分层记忆策略（三层）：
  1. recent_turns（近期原文）：保留最近 N 轮原始 user/assistant 消息
     - 优点：包含精确细节，支持短期指代（"这个"、"上面那个"）
     - 缺点：会随轮次增加而占满 token 窗口
  2. rolling_summary（滚动摘要）：当 recent_turns 超出窗口时，
     溢出的旧消息由 LLM 重新压缩进入长期摘要
     - 滚动：每次窗口溢出都把"旧摘要 + 新溢出"一起重压缩，不是简单追加
     - 降级：LLM 不可用时用规则截断保证主流程不断
  3. pinned_facts（锚定事实）：业务口径、指标定义、用户偏好等
     - 由 AgentLoop 通过 metadata["pinned_facts"] 显式注入
     - 永远不会被滚动摘要覆盖，持久保留

与 WorkMemory 的边界：
  - ConversationMemory 回答"用户和助手聊过什么"
  - WorkMemory 回答"当前任务执行到了什么状态"
  - 两者之间通过 build_conversation_bridge() 做单向轻量桥接

模块常量：
  _SUMMARY_MAX_CHARS     = 1800  — 滚动摘要最大字符数（防止无限增长）
  _SUMMARY_INPUT_MAX_CHARS = 4000 — 输入给 LLM 摘要的原文最大字符数
  _PINNED_FACTS_MAX_ITEMS = 12   — 最多保留 12 条锚定事实
  _PINNED_FACT_LENGTH    = 240   — 单条锚定事实最大字符数
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, TYPE_CHECKING

from ai_data_agent.config.config import settings
from ai_data_agent.memory.interfaces import BaseConversationMemory
from ai_data_agent.model_gateway.base_model import Message
from ai_data_agent.model_gateway.router import TaskType
from ai_data_agent.observability.logger import get_logger
from ai_data_agent.observability.metrics import metrics

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
    单条对话消息的完整记录（内部数据结构）。

    与 Message 的区别：
    - Message 是"发给模型的消息格式"（只含 role + content）
    - Turn 是"内部记录的消息格式"（包含时间戳和元数据）

    metadata 字段的使用约束：
    - metadata 只面向系统内部（AgentLoop 会读取）
    - metadata 不直接注入 LLM prompt（防止内部控制信息污染对话原文）
    - 只有明确声明为 pinned_fact/pinned_facts 的 metadata 内容会进入长期记忆
    - 其他 metadata（如 run_id、work_summary）只用于系统内部追踪

    to_message() 转换时故意丢弃 metadata，原因：
    - metadata 包含 run_id、工作摘要等控制信息
    - 这些信息不属于"对话原文"，不应无控制地重新注入模型
    - 模型只需要看 role + content，不需要知道系统内部如何组织数据
    """

    role: str                              # "user" | "assistant"
    content: str                           # 消息正文（自然语言）
    timestamp: datetime = field(default_factory=datetime.utcnow)  # UTC 时间戳
    metadata: dict[str, Any] = field(default_factory=dict)        # 系统内部元数据

    def to_message(self) -> Message:
        """
        将 Turn 转换为发给模型的 Message 格式。

        只保留 role 和 content，故意丢弃 timestamp 和 metadata。
        这样模型的上下文输入是干净的"对话原文"，
        不会被内部控制信息（run_id、工作摘要等）污染。

        Returns:
            只含 role 和 content 的 Message 对象
        """
        return Message(role=self.role, content=self.content)


@dataclass
class ConversationState:
    """
    单个会话（conversation_id）的完整分层记忆状态。

    三个字段对应三层记忆策略：
    - recent_turns：近期原文，用于短期语义（细节、指代）
    - rolling_summary：长期压缩摘要，用于跨窗口历史（目标、口径、结论）
    - pinned_facts：长期锚定事实，用于跨轮稳定约束（业务定义、用户偏好）

    内存占用估算（以默认配置为例）：
    - 10 轮对话 × 平均 500 字/条 × 2 = 10,000 字 recent_turns
    - rolling_summary 最多 1,800 字
    - pinned_facts 最多 12 × 240 = 2,880 字
    - 合计 < 15,000 字，对 token 预算影响可控
    """

    recent_turns: list[Turn] = field(default_factory=list)
    rolling_summary: str = ""
    pinned_facts: list[str] = field(default_factory=list)


class ConversationMemory(BaseConversationMemory):
    """
    基于三层分层记忆策略的对话历史管理器（内存版实现）。

    职责边界（做什么 / 不做什么）：
    ✓ 回答"用户和助手聊过什么"
    ✓ 为 prompt 提供近期原文、长期摘要、锚定事实
    ✓ 在窗口溢出时调用 LLM 生成滚动摘要
    ✗ 不记录工具调用详情（这是 WorkMemory 的职责）
    ✗ 不承担任务规划状态（这是 AgentLoop + Planner 的职责）
    ✗ 不保存原始 SQL、图表配置等大体量产物

    异步接口设计：
    - add() 是 async：因为滚动摘要生成需要调用 LLM（IO 密集操作）
    - get_messages()、get_turns()、clear() 均为同步：只读/操作内存，无 IO

    多租户隔离：
    - 通过 conversation_id 隔离不同会话的数据
    - 推荐使用 scoped_conversation_id（如 "tenant_id:conversation_id"）做跨租户隔离
    - 如果不隔离，A 租户可能看到 B 租户的对话历史

    Redis 版：
    - 如果需要多进程共享或持久化，请使用 RedisConversationMemory（接口完全相同）
    - 通过 memory/factory.py 中的 build_conversation_memory() 按配置自动选择后端
    """

    def __init__(
        self,
        max_turns: int | None = None,
        *,
        router: "ModelRouter | None" = None,
        breaker: "CircuitBreaker | None" = None,
    ) -> None:
        """
        初始化对话记忆。

        Args:
            max_turns: 保留的最大对话轮数（一轮 = 一个 user + 一个 assistant）
                      None 时从 settings.conversation_max_turns 读取（默认 10 轮）
            router: ModelRouter 实例，用于生成滚动摘要。
                   None 表示不使用 LLM 摘要（退化为规则摘要）
            breaker: CircuitBreaker 实例，保护 LLM 摘要调用。
                    当 LLM 不可用时（熔断器 OPEN），退化为规则摘要
        """
        self._max_turns = max_turns or settings.conversation_max_turns
        self._router = router
        self._breaker = breaker
        # defaultdict(ConversationState) 确保首次访问任意 conversation_id 都有初始状态
        # 不需要手动 _store.setdefault(cid, ConversationState())
        self._store: dict[str, ConversationState] = defaultdict(ConversationState)

    async def add(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        追加一条对话消息到指定会话。

        完整写入流程（顺序执行）：
        1. 创建 Turn 对象并追加到 recent_turns
        2. 从 metadata 中提取 pinned_fact/pinned_facts 更新锚定事实
        3. 检查 recent_turns 是否超出窗口，超出则触发滚动摘要压缩

        只有 metadata["pinned_fact"] 和 metadata["pinned_facts"] 会进入长期记忆，
        其他 metadata 字段（如 run_id、work_summary）不会自动提升为长期记忆。
        这是刻意设计：pinned_facts 的提取决策由 AgentLoop 的 _extract_pinned_facts()
        负责，不让 ConversationMemory 自己猜哪些内容值得长期固定。

        Args:
            conversation_id: 会话唯一标识（建议使用租户作用域键防止跨租户泄露）
            role: 消息角色，"user" 或 "assistant"
            content: 消息内容（自然语言文本）
            metadata: 可选元数据，仅 pinned_fact/pinned_facts 键会进入长期记忆
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
        构建发给模型的对话消息列表（三层记忆合并）。

        返回顺序（顺序很重要，会影响模型的注意力分配）：
        1. system message：长期会话记忆块（pinned_facts + rolling_summary）
           - 用 system role 注入，因为它是"系统视角的历史摘要"，不是原始对话原文
           - 放在最前面，让模型先了解长期约束再看近期对话
        2. user/assistant messages：近期原始对话（recent_turns）
           - 包含精确细节和短期指代，放在后面让模型聚焦最近上下文

        如果没有长期记忆（首次对话），只返回 recent_turns 转换的消息列表。

        Args:
            conversation_id: 会话唯一标识

        Returns:
            Message 列表，可直接传给 LLM router.generate()；
            会话不存在时返回空列表
        """
        state = self._store.get(conversation_id)
        if state is None:
            return []

        messages: list[Message] = []
        memory_block = self._build_memory_block(state)
        if memory_block:
            # 长期记忆用 system role 注入，表示这是系统压缩的历史视图
            messages.append(Message(role="system", content=memory_block))
        messages.extend(t.to_message() for t in state.recent_turns)
        return messages

    def get_turns(self, conversation_id: str) -> list[Turn]:
        """
        获取近期原始 Turn 对象列表（供内部调试和 API 层展示）。

        与 get_messages() 的区别：
        - get_messages() 面向"模型输入"，会合并长期记忆 system block
        - get_turns() 面向"系统内部检查"，只返回 recent_turns 原文

        典型使用场景：
        - API 端点展示对话历史给用户看（不希望暴露 system block）
        - 单元测试验证写入内容是否正确
        - 调试某轮回复的输入是否符合预期

        Args:
            conversation_id: 会话唯一标识

        Returns:
            Turn 对象列表（副本，修改不影响内部状态）；
            会话不存在时返回空列表
        """
        state = self._store.get(conversation_id)
        if state is None:
            return []
        return list(state.recent_turns)

    def clear(self, conversation_id: str) -> None:
        """
        清除指定会话的所有记忆（三层全部清除）。

        使用场景：
        - 用户点击"开始新对话"（不希望带入旧历史）
        - GDPR 数据删除请求（必须清除所有个人数据）
        - 测试环境重置（确保每个测试用例从干净状态开始）

        注意：这个操作是不可逆的，清除后无法恢复历史。

        Args:
            conversation_id: 要清除的会话标识
        """
        self._store.pop(conversation_id, None)
        logger.info("memory.cleared", conversation_id=conversation_id)

    def list_conversations(self) -> list[str]:
        """
        列出当前内存中所有活跃会话的 ID。

        用于监控和管理：查看有哪些会话占用了内存，
        可配合 clear() 做过期会话的主动清理。

        Returns:
            会话 ID 列表
        """
        return list(self._store.keys())

    def summary(self, conversation_id: str) -> dict:
        """
        返回指定会话的三层记忆统计摘要。

        用于监控和调试：了解各会话的记忆使用情况。
        只统计 ConversationMemory 自身的状态，
        不汇总 WorkMemory 的执行状态（各司其职）。

        Args:
            conversation_id: 会话标识

        Returns:
            包含以下统计信息的字典：
            - conversation_id: 会话标识
            - turns: 近期轮数（user+assistant 对数）
            - messages: 近期消息总条数
            - has_rolling_summary: 是否有滚动摘要
            - rolling_summary_chars: 滚动摘要字符数
            - pinned_facts: 锚定事实条数
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
        当 recent_turns 超出窗口时，将溢出的旧消息压缩进滚动摘要。

        核心设计决策：溢出消息不是被"硬删除"，而是被"LLM 重新压缩"。
        这样可以同时满足两个目标：
        1. 控制 token 预算（recent_turns 不超过 max_turns × 2）
        2. 保留历史语义（重要上下文通过滚动摘要长期保留）

        成对策略：
        对话通常以 user/assistant 成对出现。如果只写入了 user 消息（assistant 还未回复），
        等 assistant 回复写入后再一起摘要，避免长期摘要里出现"孤立问题"（无对应答案），
        削弱多轮语义的连续性。因此 overflow_count -= overflow_count % 2。

        错误降级：
        如果 LLM 摘要失败（网络问题、熔断器开启、API 异常），
        退化为规则截断（_fallback_merge_summary），保证主流程不因为摘要失败而中断。

        Args:
            conversation_id: 会话标识（用于日志记录）
            state: 当前会话的记忆状态（原地修改）
        """
        max_messages = self._max_turns * 2  # N 轮 = N×2 条消息
        if len(state.recent_turns) <= max_messages:
            return

        overflow_count = len(state.recent_turns) - max_messages
        if overflow_count < 2:
            return

        # 确保按 user/assistant 成对溢出，避免摘要里出现孤立问题
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
            metrics.memory_summary_total.labels(outcome="success").inc()
        except Exception as e:
            metrics.memory_summary_total.labels(outcome="fallback").inc()
            logger.warning(
                "conversation_memory.summary_failed",
                conversation_id=conversation_id,
                error=str(e),
            )
            # LLM 不可用时降级为规则拼接，不中断主流程
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
        调用 LLM 生成滚动摘要（三元输入 → 新摘要）。

        输入三元组：
        1. existing_summary：已有的长期摘要（可能来自上一轮压缩）
        2. overflow_turns：本次被窗口淘汰的旧消息（已成对）
        3. pinned_facts：当前所有锚定事实（告诉 LLM 这些已单独保留，不需要重复展开）

        输出要求（通过 prompt 约束）：
        - 只保留后续对话可能还会用到的信息
        - 优先保留：业务口径、指标定义、过滤条件、用户偏好、已确认结论、未完成问题
        - 不保留：寒暄、重复解释、工具调用细节、临时错误信息
        - 不编造对话中没有出现的信息
        - 总长度控制在 _SUMMARY_MAX_CHARS 字以内
        - 输出中文 bullet list

        模型选择：
        - 使用 TaskType.SIMPLE 路由（通常是 fast model / gpt-4o-mini）
        - 原因：摘要任务结构简单，不需要最强模型，用 fast model 节省成本
        - max_tokens=700：摘要应该短，防止 LLM 输出冗长内容

        Args:
            existing_summary: 已有的长期摘要（空字符串表示首次压缩）
            overflow_turns: 本次溢出的旧消息列表
            pinned_facts: 当前锚定事实列表

        Returns:
            新的滚动摘要字符串（已截断到 _SUMMARY_MAX_CHARS）

        Raises:
            RuntimeError: router 或 breaker 未配置时
            任意 LLM 异常：由调用方 _roll_recent_turns_into_summary 捕获并降级
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
        # 通过熔断器调用，防止 LLM 不可用时阻塞主流程
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
        LLM 摘要失败时的规则降级策略。

        这不是高质量摘要，但保证两个关键属性：
        1. 主对话流程不会因为摘要失败而中断（即使 LLM 完全不可用）
        2. 旧上下文不会直接硬删除（保留片段，不完美但好过什么都没有）

        实现：把已有摘要 + 溢出消息的截断文本拼接起来，
        最终取尾部 _SUMMARY_MAX_CHARS 字符（保留最近的内容）。

        Args:
            existing_summary: 已有的长期摘要
            overflow_turns: 溢出的旧消息列表

        Returns:
            规则拼接后截断的摘要字符串
        """
        fragments = [existing_summary.strip()] if existing_summary.strip() else []
        for turn in overflow_turns:
            text = self._compact_text(turn.content, max_len=220)
            if not text:
                continue
            prefix = "用户" if turn.role == "user" else "助手"
            fragments.append(f"- {prefix}: {text}")
        merged = "\n".join(fragments)
        # 取尾部：保留最近的内容比保留最早的内容更有价值
        return merged[-_SUMMARY_MAX_CHARS:]

    def _build_memory_block(self, state: ConversationState) -> str:
        """
        将三层记忆（pinned_facts + rolling_summary）组装成 system message 文本。

        注入顺序设计（pinned_facts 在前）：
        - pinned_facts 通常是业务口径或用户偏好，是更高优先级的约束
        - rolling_summary 是对历史对话的压缩，优先级低于锚定事实
        - 先说约束，后说历史，模型更容易把握重点

        空检查：
        - 两者都为空时返回空字符串，表示不需要注入 system block
        - 首次对话、刚清除历史后都会走这个路径

        Args:
            state: 会话的完整记忆状态

        Returns:
            组装好的 Markdown 格式 system block，或空字符串
        """
        if not state.rolling_summary and not state.pinned_facts:
            return ""

        lines = ["## Conversation Memory"]
        if state.pinned_facts:
            lines.append("")
            lines.append("Pinned facts:")
            # 只取最近的 _PINNED_FACTS_MAX_ITEMS 条，避免内容过多
            for fact in state.pinned_facts[-_PINNED_FACTS_MAX_ITEMS:]:
                lines.append(f"- {fact}")
        if state.rolling_summary:
            lines.append("")
            lines.append("Rolling summary of earlier conversation:")
            lines.append(state.rolling_summary)
        return "\n".join(lines)

    def _update_pinned_facts(self, state: ConversationState, turn: Turn) -> None:
        """
        从 Turn 的 metadata 中提取并更新锚定事实。

        设计说明：
        - metadata["pinned_fact"]  = str：单条锚定事实
        - metadata["pinned_facts"] = list[str]：多条锚定事实
        - 两种格式都支持，方便调用方选择

        为什么不自动分析 content？
        - "哪些内容值得长期固定"是业务判断，不应由 ConversationMemory 自己做
        - pinned_facts 的提取决策集中在 AgentLoop._extract_pinned_facts()
        - 这样可以把"提取逻辑"和"存储逻辑"分开，各自独立演化

        去重逻辑：
        - 完全相同的事实不重复写入（normalized 字符串级别的去重）
        - 超出 _PINNED_FACTS_MAX_ITEMS 限制时，保留最近的条目（截尾保留）

        Args:
            state: 当前会话状态（原地修改 pinned_facts）
            turn: 当前写入的 Turn 对象（从其 metadata 中提取）
        """
        raw_facts: list[str] = []
        # 支持单条格式：metadata["pinned_fact"] = "..."
        if isinstance(turn.metadata.get("pinned_fact"), str):
            raw_facts.append(turn.metadata["pinned_fact"])

        # 支持多条格式：metadata["pinned_facts"] = ["...", "..."]
        meta_facts = turn.metadata.get("pinned_facts")
        if isinstance(meta_facts, list):
            raw_facts.extend(item for item in meta_facts if isinstance(item, str))

        for fact in raw_facts:
            normalized = self._compact_text(fact, max_len=_PINNED_FACT_LENGTH)
            if not normalized or normalized in state.pinned_facts:
                continue  # 空或重复的事实跳过
            state.pinned_facts.append(normalized)
            metrics.pinned_facts_total.labels(outcome="stored").inc()
        # 超出上限时截断尾部，保留最近写入的事实
        state.pinned_facts = state.pinned_facts[-_PINNED_FACTS_MAX_ITEMS:]

    @staticmethod
    def _format_turns_for_summary(turns: list[Turn]) -> str:
        """
        将 Turn 列表格式化为 LLM 摘要 prompt 的输入文本。

        限制：
        - 总字符数不超过 _SUMMARY_INPUT_MAX_CHARS（防止 prompt 过长）
        - 超过限制时截断并追加 "[truncated]" 标记
        - 连续空白压缩为单空格（减少 token 消耗）

        Args:
            turns: 需要格式化的 Turn 列表

        Returns:
            格式化的文本，如 "user: ...\nassistant: ...\n..."，
            或 "(无)"（空列表时）
        """
        lines: list[str] = []
        total_chars = 0
        for turn in turns:
            content = " ".join(turn.content.split())  # 压缩连续空白
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
        """
        将锚定事实列表格式化为 bullet list 文本。

        只取最近的 _PINNED_FACTS_MAX_ITEMS 条（防止输出过长）。

        Args:
            pinned_facts: 锚定事实列表

        Returns:
            bullet list 格式的文本，或 "(无)"（空列表时）
        """
        if not pinned_facts:
            return "(无)"
        return "\n".join(f"- {fact}" for fact in pinned_facts[-_PINNED_FACTS_MAX_ITEMS:])

    @staticmethod
    def _clean_summary(summary: str) -> str:
        """
        清理 LLM 摘要输出，去除 markdown code fence 并截断。

        LLM 有时会把输出包在 ``` 代码块里，这里统一去除。
        摘要是长期上下文的一部分，必须控制体积，截断到 _SUMMARY_MAX_CHARS。

        Args:
            summary: LLM 生成的原始摘要文本

        Returns:
            清理并截断后的摘要字符串
        """
        cleaned = summary.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").strip()
        return cleaned[:_SUMMARY_MAX_CHARS]

    @staticmethod
    def _compact_text(text: str, *, max_len: int) -> str:
        """
        压缩文本：去除首尾空白、合并内部连续空白、截断到 max_len。

        这是一个通用的文本清洗工具，用于：
        - 存入 pinned_facts 前的文本标准化
        - LLM 输出清洗

        Args:
            text: 输入文本
            max_len: 最大字符数

        Returns:
            压缩后的文本，空字符串表示输入为空
        """
        cleaned = " ".join(text.split()).strip()
        if not cleaned:
            return ""
        return cleaned[:max_len]


# ── 全局单例 ──────────────────────────────────────────────────────────────────

_memory: ConversationMemory | None = None


def get_memory() -> ConversationMemory:
    """
    获取全局对话记忆单例（懒加载，不含 LLM 摘要功能）。

    注意：这个函数创建的实例没有 router/breaker，因此无法生成 LLM 滚动摘要。
    如果需要 LLM 摘要，应通过 assembler 或 memory/factory.py 来创建完整配置的实例。

    使用场景：
    - 开发/测试环境，不需要 LLM 摘要
    - 快速启动场景，不关心长期记忆质量

    Returns:
        全局唯一的 ConversationMemory 实例（无 LLM 摘要配置）
    """
    global _memory
    if _memory is None:
        _memory = ConversationMemory()
    return _memory
