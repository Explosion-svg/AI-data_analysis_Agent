"""
context/prompt_builder.py — Prompt 构建器

职责：
  将多个上下文来源（RAG 文档、Schema、对话历史、工作记忆）按固定顺序
  组装成发给 LLM 的完整 messages 列表。

设计原则：
  - 纯粹的"组装"工作，不含任何 IO 或模型调用
  - 所有上下文来源通过参数注入，而不是在内部获取（便于测试）
  - 严格控制各部分进入 prompt 的方式，防止内容越界

重要边界（避免污染）：
  - history 只接受"对话消息"（user/assistant 的自然语言）
  - work_context 只接受"工作状态摘要"（已压缩的执行状态文本）
  - 不要把工具原始结果集、完整运行轨迹直接塞进 prompt
  - 大块原始数据会撑爆 token 预算，并可能让 LLM 产生幻觉

Prompt 消息顺序（对 LLM 有语义影响）：
  [system] → [work_context] → [history] → [rag_docs] → [schema] → [user_query]

  顺序设计依据：
  - system 最先：定义角色和全局约束
  - work_context 紧跟 system：让模型把当前任务状态当作系统背景
  - history 在中间：保持对话连贯性
  - rag_docs 和 schema 在 user_query 前：模型能在回答时直接参考
  - user_query 最后：模型基于所有上下文来理解和回答
"""
from __future__ import annotations

from typing import Any

from ai_data_agent.model_gateway.base_model import Message
from ai_data_agent.observability.logger import get_logger

logger = get_logger(__name__)

# ── 系统 Prompt 模板 ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert AI data analyst assistant. You have access to a data warehouse and analytical tools.

## Your Capabilities
- Execute SQL queries to retrieve data from the warehouse
- Run Python (pandas/numpy) code to analyze and transform data
- Generate interactive charts and visualizations
- Search internal documents for context and definitions
- Inspect database schema to understand data structure

## Guidelines
1. **Always check schema first** if you don't know the table structure.
2. **Write safe SQL** — only SELECT statements, no data modification.
3. **Be precise** — validate your SQL before execution.
4. **Explain results** — after retrieving data, provide clear business insights.
5. **Use the right tool** — SQL for retrieval, Python for complex analysis, Chart for visualization.
6. **Be iterative** — if one approach fails, try an alternative.
7. **Stay grounded** — only state conclusions supported by tool outputs or retrieved documents.
8. **Admit uncertainty** — if evidence is missing or conflicting, say so explicitly instead of guessing.

## Response Format
- Lead with the key insight
- Show data in tables when relevant
- Offer chart generation when visual comparison would help
- Be concise and business-focused

Current date: {current_date}
"""


class PromptBuilder:
    """
    构建发送给 LLM 的完整 messages 列表。

    这是一个纯组装器（assembler），不包含任何模型调用或 IO 操作。
    所有输入通过 build() 参数注入，输出是一个标准的 Message 列表。

    组装完成的消息列表直接传给 LLM router.generate() 使用。
    """

    def build(
        self,
        query: str,
        *,
        rag_docs: list[dict[str, Any]] | None = None,
        schema_context: str | None = None,
        history: list[Message] | None = None,
        work_context: str | None = None,
    ) -> list[Message]:
        """
        按固定顺序组装 messages 列表。

        消息顺序：
          [system] → [work_context] → [history] → [rag_context] → [schema_context] → [user_query]

        各部分详细说明：

        1. system（必须）：
           - 包含角色定义、能力说明、操作指南
           - 注入当前日期（帮助 LLM 理解"今年"、"本月"等相对时间表达）

        2. work_context（可选）：
           - 以 system 消息注入，让模型把它当作系统信息而非对话内容
           - 内容是已压缩的任务执行状态摘要，不包含原始数据
           - 帮助模型知道"当前做到哪一步"，避免重复已完成的工作

        3. history（可选）：
           - 对话历史消息，保持 user/assistant 交替结构
           - 来自 conversation_memory，包含近期对话原文和长期摘要

        4. rag_docs（可选）：
           - 从知识库检索到的相关文档
           - 注入为 system 消息，让模型把它当作背景知识
           - 最多注入 5 篇（避免 token 溢出）

        5. schema_context（可选）：
           - 数据库表结构信息（已按相关性筛选）
           - 注入为 system 消息，提供数据访问的结构指导

        6. user_query（必须）：
           - 用户的原始问题（不做修改）
           - 始终放在最后，让模型基于所有上下文来理解和回答

        Args:
            query: 用户问题（原始文本）
            rag_docs: RAG 检索结果列表，每项包含 content、metadata、score
            schema_context: 格式化的数据库 schema 字符串
            history: 对话历史消息列表（Message 对象）
            work_context: 工作状态摘要文本（来自 WorkMemory.build_prompt_context()）

        Returns:
            按顺序组装好的 Message 列表，可直接传给 LLM router.generate()
        """
        from datetime import date
        messages: list[Message] = []

        # 1. System prompt（包含当前日期注入）
        system_content = SYSTEM_PROMPT.format(current_date=date.today().isoformat())
        messages.append(Message(role="system", content=system_content))

        # 2. 工作状态摘要（以 system 身份注入）
        # 作为 system 消息而不是 user 消息，原因：
        # - 这不是用户说的话，是系统维护的执行状态
        # - system role 在 LLM 中通常有更高的指令遵循优先级
        # - 防止模型把工作状态当成需要响应的用户请求
        if work_context:
            messages.append(
                Message(
                    role="system",
                    content=f"## Current Work State\n\n{work_context}",
                )
            )

        # 3. 历史对话（最近 N 轮，按时间顺序，保持 user/assistant 交替）
        # history 只包含自然语言对话，不应混入工具调用日志或执行结果
        if history:
            messages.extend(history)

        # 4. RAG 文档（以 system 身份注入知识背景）
        # 最多取 5 篇，避免长文档撑爆 token 预算
        # 每篇附带来源（source）和相关度分数（score），方便 LLM 判断可信度
        if rag_docs:
            doc_texts = []
            for i, doc in enumerate(rag_docs[:5], 1):
                src = doc.get("metadata", {}).get("source", "")
                score = doc.get("score", 0)
                content = doc.get("content", "")
                doc_texts.append(
                    f"[Document {i}]{f' ({src})' if src else ''} "
                    f"relevance={score:.2f}\n{content}"
                )
            rag_block = "## Relevant Knowledge Base Documents\n\n" + "\n\n---\n".join(doc_texts)
            messages.append(Message(role="system", content=rag_block))

        # 5. Schema 上下文（以 system 身份注入数据库结构）
        # 放在 user_query 之前，让模型在理解问题时就能参考可用的表和字段
        if schema_context:
            messages.append(
                Message(
                    role="system",
                    content=f"## Database Schema\n\n{schema_context}",
                )
            )

        # 6. 用户问题（始终是最后一条消息）
        messages.append(Message(role="user", content=query))

        logger.debug(
            "prompt_builder.built",
            total_messages=len(messages),
            has_rag=bool(rag_docs),
            has_schema=bool(schema_context),
            has_history=bool(history),
            has_work_context=bool(work_context),
        )
        return messages
