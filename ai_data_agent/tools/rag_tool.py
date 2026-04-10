"""
tools/rag_tool.py — RAG 检索工具
流程：query → embedding → vector search → rerank → 返回文档
"""
from __future__ import annotations

from typing import Any

from ai_data_agent.tools.base_tool import BaseTool, ToolResult
from ai_data_agent.infra import vector_store
from ai_data_agent.model_gateway.router import get_router
from ai_data_agent.observability.logger import get_logger

logger = get_logger(__name__)


class RAGTool(BaseTool):
    @property
    def name(self) -> str:
        return "search_documents"

    @property
    def description(self) -> str:
        return (
            "Search internal knowledge base and documents using semantic search. "
            "Use this to retrieve relevant context, definitions, or policies "
            "that might help answer the user's question."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
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
        if not query.strip():
            return ToolResult(success=False, error="Empty query.")

        # 1. 生成 query embedding
        router = get_router()
        try:
            embeddings = await router.embed([query])
            query_embedding = embeddings[0]
        except Exception as e:
            return ToolResult(success=False, error=f"Embedding failed: {e}")

        # 2. 向量搜索
        try:
            docs = vector_store.search_docs(
                query_embedding=query_embedding,
                top_k=top_k,
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Vector search failed: {e}")

        # 3. 过滤低分文档
        docs = [d for d in docs if d["score"] >= score_threshold]

        if not docs:
            return ToolResult(
                success=True,
                data=[],
                text="No relevant documents found.",
            )

        # 4. 格式化输出
        parts = []
        for i, doc in enumerate(docs, 1):
            src = doc.get("metadata", {}).get("source", "unknown")
            parts.append(f"[{i}] (score={doc['score']:.3f}, source={src})\n{doc['content']}")

        text = f"Found {len(docs)} relevant document(s):\n\n" + "\n\n---\n\n".join(parts)
        logger.debug("rag_tool.retrieved", query=query[:80], docs=len(docs))

        return ToolResult(
            success=True,
            data=docs,
            text=text,
        )
