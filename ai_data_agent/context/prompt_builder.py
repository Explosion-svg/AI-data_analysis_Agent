"""
context/prompt_builder.py — Prompt 构建器
将 system prompt、query、RAG 文档、schema、历史对话组装成 messages[]
"""
from __future__ import annotations

from typing import Any

from ai_data_agent.model_gateway.base_model import Message
from ai_data_agent.observability.logger import get_logger

logger = get_logger(__name__)

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

## Response Format
- Lead with the key insight
- Show data in tables when relevant
- Offer chart generation when visual comparison would help
- Be concise and business-focused

Current date: {current_date}
"""


class PromptBuilder:
    """构建发送给 LLM 的完整 messages 列表。"""

    def build(
        self,
        query: str,
        *,
        rag_docs: list[dict[str, Any]] | None = None,
        schema_context: str | None = None,
        history: list[Message] | None = None,
        extra_context: str | None = None,
    ) -> list[Message]:
        """
        按如下顺序组装 messages：
          [system] → [history] → [rag_context] → [schema_context] → [user_query]
        """
        from datetime import date
        messages: list[Message] = []

        # 1. System prompt
        system_content = SYSTEM_PROMPT.format(current_date=date.today().isoformat())
        if extra_context:
            system_content += f"\n\n## Additional Context\n{extra_context}"
        messages.append(Message(role="system", content=system_content))

        # 2. 历史对话（最近 N 轮，按时间顺序）
        if history:
            messages.extend(history)

        # 3. RAG 文档
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

        # 4. Schema 上下文
        if schema_context:
            messages.append(
                Message(
                    role="system",
                    content=f"## Database Schema\n\n{schema_context}",
                )
            )

        # 5. 用户问题
        messages.append(Message(role="user", content=query))

        logger.debug(
            "prompt_builder.built",
            total_messages=len(messages),
            has_rag=bool(rag_docs),
            has_schema=bool(schema_context),
        )
        return messages
