"""
context/query_rewriter.py — 查询改写器
对原始用户问题进行扩展，提升 RAG 召回率
策略：Query Rewrite + Multi-Query + Keyword Extraction
"""
from __future__ import annotations

from ai_data_agent.model_gateway.router import get_router, TaskType
from ai_data_agent.model_gateway.base_model import Message, LLMConfig
from ai_data_agent.config.config import settings
from ai_data_agent.observability.logger import get_logger

logger = get_logger(__name__)

_REWRITE_PROMPT = """You are a query expansion expert. Given a user's natural language question, generate:
1. A rewritten, more precise version of the question
2. 2-3 alternative phrasings that capture different aspects

Return ONLY a JSON object in this format:
{
  "rewritten": "precise rewritten query",
  "alternatives": ["alt query 1", "alt query 2"],
  "keywords": ["keyword1", "keyword2", "keyword3"]
}

User question: {query}"""


class QueryRewriter:
    """将模糊的用户问题改写成更精确的搜索查询。"""

    async def rewrite(self, query: str) -> dict[str, str | list[str]]:
        """
        返回:
          {
            "rewritten": str,
            "alternatives": list[str],
            "keywords": list[str],
            "all_queries": list[str]   # 合并列表，供 RAG 多路召回
          }
        """
        import json

        router = get_router()
        prompt = _REWRITE_PROMPT.format(query=query)
        config = LLMConfig(
            model=settings.openai_fast_model,
            temperature=0.3,
            max_tokens=512,
        )
        try:
            resp = await router.generate(
                messages=[Message(role="user", content=prompt)],
                task_type=TaskType.SIMPLE,
                model=settings.openai_fast_model,
                temperature=0.3,
                max_tokens=512,
            )
            parsed = json.loads(resp.content)
        except Exception as e:
            logger.warning("query_rewriter.failed", error=str(e))
            # 降级：直接使用原始 query
            return {
                "rewritten": query,
                "alternatives": [],
                "keywords": query.split()[:5],
                "all_queries": [query],
            }

        rewritten = parsed.get("rewritten", query)
        alternatives = parsed.get("alternatives", [])
        keywords = parsed.get("keywords", [])
        all_queries = [query, rewritten] + alternatives
        # 去重保序
        seen: set[str] = set()
        unique: list[str] = []
        for q in all_queries:
            if q and q not in seen:
                seen.add(q)
                unique.append(q)

        logger.debug(
            "query_rewriter.done",
            original=query[:80],
            rewritten=rewritten[:80],
            n_alternatives=len(alternatives),
        )
        return {
            "rewritten": rewritten,
            "alternatives": alternatives,
            "keywords": keywords,
            "all_queries": unique,
        }
