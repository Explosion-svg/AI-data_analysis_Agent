"""
tools/rag_tool.py — RAG 检索工具（RAGTool，Retrieval-Augmented Generation）

职责：
  对用户查询执行语义检索，从内部知识库（ChromaDB 向量数据库）中
  找到与查询语义最相近的文档片段，返回给 LLM 作为上下文。

RAG 流程（三步）：
  1. 查询向量化（Embedding）：
     调用 router.embed([query]) 生成查询的向量表示（float 数组）。
     使用 OpenAI text-embedding-ada-002（或其他 embedding 模型）。

  2. 向量搜索（Vector Search）：
     调用 vector_store.search_docs(query_embedding, top_k) 在 ChromaDB 中
     使用余弦相似度（cosine similarity）找到最相近的 top_k 个文档。

  3. 相关性过滤（Score Filtering）：
     保留相似度分数 >= score_threshold 的文档（默认 0.5）。
     过滤掉低质量匹配，避免把无关文档注入 LLM 上下文。

为什么用向量检索而不是关键词检索：
  - 关键词检索：精确匹配，"GMV" 搜不到 "成交总额"
  - 向量检索：语义相似，能理解同义词和语义关系
  - 数据分析场景下，用户可能用自然语言问业务定义类问题

score_threshold 设计：
  - 太低（0.1）：返回太多不相关文档（LLM 上下文噪音增大）
  - 太高（0.9）：过于严格，可能什么也找不到
  - 0.5 是工程上的经验值，在 cosine similarity 空间中表示"中等相关"

输出格式：
  每个文档片段格式化为：
    [序号] (score=0.823, source=business_glossary.pdf)
    GMV（Gross Merchandise Volume）是指平台上所有交易的总成交额...

  多个文档片段用 "---" 分隔，让 LLM 能清楚识别文档边界。

与 AgentLoop 的关系：
  Plan.needs_rag=True 时，AgentLoop 可以在主循环前预先触发 RAG，
  将检索结果注入 system prompt，无需占用 ReAct 循环的一次迭代。
"""
from __future__ import annotations

from typing import Any

from ai_data_agent.tools.base_tool import BaseTool, ToolResult
from ai_data_agent.infra import vector_store
from ai_data_agent.model_gateway.router import get_router
from ai_data_agent.observability.logger import get_logger

logger = get_logger(__name__)


class RAGTool(BaseTool):
    """
    知识库语义检索工具，实现 RAG 流程中的 Retrieval 部分。

    工具名：search_documents
    并发槽：ConcurrencyLimiter 的 "search_documents" 桶
    """

    @property
    def name(self) -> str:
        """返回工具名称 "search_documents"。"""
        return "search_documents"

    @property
    def description(self) -> str:
        """
        工具描述，强调适用于"概念定义"和"策略查询"场景。

        "definitions, or policies" 的设计意图：
        - 帮助 LLM 识别何时应该用 RAG（业务定义类问题）
        - 而不是用 SQL（量化数据类问题）
        """
        return (
            "Search internal knowledge base and documents using semantic search. "
            "Use this to retrieve relevant context, definitions, or policies "
            "that might help answer the user's question."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        """
        RAG 工具的参数 JSON Schema。

        参数：
        - query（必填）：搜索查询字符串
          → LLM 应将用户问题提炼为简洁的搜索关键短语
        - top_k（可选，默认 5）：检索的文档数量
          → 更多文档提供更多上下文，但也增加 token 消耗
        - score_threshold（可选，默认 0.5）：最低相似度分数
          → 过滤掉低质量匹配
        """
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query to find relevant documents.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of documents to retrieve (default: 5).",
                    "default": 5,
                },
                "score_threshold": {
                    "type": "number",
                    "description": "Minimum relevance score [0, 1] (default: 0.5).",
                    "default": 0.5,
                },
            },
            "required": ["query"],
        }

    async def _run(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.5,
        **_: Any,
    ) -> ToolResult:
        """
        执行 RAG 检索流程（三步）。

        每一步的错误处理策略：
        - Embedding 失败：立即返回 ToolResult(success=False)
          → embedding 服务不可用时，无法进行向量搜索，快速失败
        - 向量搜索失败：立即返回 ToolResult(success=False)
          → ChromaDB 不可用，快速失败
        - 无匹配文档（过滤后为空）：返回 ToolResult(success=True, data=[], text="No relevant documents found.")
          → 这是正常情况（不是错误），告知 LLM 知识库中没有相关内容

        "没有找到文档" 为什么也返回 success=True：
        - success 表示工具本身是否正常执行（搜索过程成功完成）
        - 找不到文档是业务层面的结果，不是工具执行失败
        - AgentLoop 可以据此决策：继续用其他方式回答，而不是把"空结果"当错误处理

        输出格式（text）：
        每个文档格式化为：
          [序号] (score=0.823, source=文档来源)
          文档内容...
        多文档之间用 "\n\n---\n\n" 分隔

        Args:
            query: 搜索查询字符串
            top_k: 检索的文档数量（默认 5）
            score_threshold: 最低相似度分数过滤阈值（默认 0.5）
            **_: 忽略的额外参数

        Returns:
            ToolResult：
            - 成功且有结果：success=True, data=list[dict], text=格式化文档
            - 成功但无结果：success=True, data=[], text="No relevant documents found."
            - 失败（embedding/搜索错误）：success=False, error=失败原因
        """
        if not query.strip():
            return ToolResult(success=False, error="Empty query.")

        # Step 1: 生成查询向量（将文本查询转换为数值向量）
        router = get_router()
        try:
            embeddings = await router.embed([query])
            query_embedding = embeddings[0]  # embed 返回 list，取第一个（对应 query）
        except Exception as e:
            return ToolResult(success=False, error=f"Embedding failed: {e}")

        # Step 2: 向量数据库搜索（余弦相似度）
        try:
            docs = vector_store.search_docs(
                query_embedding=query_embedding,
                top_k=top_k,
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Vector search failed: {e}")

        # Step 3: 相关性过滤（去掉分数低于阈值的文档）
        docs = [d for d in docs if d["score"] >= score_threshold]

        if not docs:
            return ToolResult(
                success=True,
                data=[],
                text="No relevant documents found.",
            )

        # Step 4: 格式化输出
        # 每个文档块包含：序号、分数、来源、内容
        parts = []
        for i, doc in enumerate(docs, 1):
            src = doc.get("metadata", {}).get("source", "unknown")
            parts.append(f"[{i}] (score={doc['score']:.3f}, source={src})\n{doc['content']}")

        text = f"Found {len(docs)} relevant document(s):\n\n" + "\n\n---\n\n".join(parts)
        logger.debug("rag_tool.retrieved", query=query[:80], docs=len(docs))

        return ToolResult(
            success=True,
            data=docs,    # 原始文档列表（含 score、content、metadata）
            text=text,    # 格式化的文本（供 LLM 消费）
        )
