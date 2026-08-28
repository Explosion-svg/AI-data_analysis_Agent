"""
context/query_rewriter.py — 查询改写器

职责：
  对原始用户问题进行语义扩展，提升后续 RAG（向量检索）的召回率。

为什么需要查询改写？
  用户问题往往简短、口语化（如"今年卖了多少"），
  而知识库文档使用的是正式的业务词汇（如"本财年销售总额"）。
  直接用原始问题向量化后搜索，可能因语义偏差导致漏召回。
  改写后同时使用原始问题 + 改写问题 + 多个同义表达进行多路并行检索，
  显著提升召回率（Multi-Query 策略）。

策略组合：
  - Query Rewrite：把口语化表达改写成更规范的分析语言
  - Multi-Query：生成 2-3 个表达同一意图的不同问法
  - Keyword Extraction：提取关键词，可用于关键词召回的补充

降级策略：
  LLM 调用失败时不报错，直接退化到使用原始 query。
  这样即使 query_rewriter 不可用，主流程也能继续工作。
"""
from __future__ import annotations

from ai_data_agent.model_gateway.router import get_router, TaskType
from ai_data_agent.model_gateway.base_model import Message, LLMConfig
from ai_data_agent.config.config import settings
from ai_data_agent.observability.logger import get_logger

logger = get_logger(__name__)

# ── 提示词模板 ────────────────────────────────────────────────────────────────

_REWRITE_PROMPT = """You are a query expansion expert. Given a user's natural language question, generate:
1. A rewritten, more precise version of the question
2. 2-3 alternative phrasings that capture different aspects

Return ONLY a JSON object in this format:
{{
  "rewritten": "precise rewritten query",
  "alternatives": ["alt query 1", "alt query 2"],
  "keywords": ["keyword1", "keyword2", "keyword3"]
}}

User question: {query}"""


class QueryRewriter:
    """
    将模糊的用户问题改写成更精确的搜索查询。

    使用 fast model（如 gpt-4o-mini）而非 strong model，
    原因：
    - 查询改写是简单的语义变换任务，不需要复杂推理能力
    - 使用 fast model 节省成本，降低延迟
    - temperature=0.3（略高于 0.0），允许一定的表达多样性

    输出格式：
      {
        "rewritten": str,          # 改写后的精确问题
        "alternatives": list[str], # 2-3 个同义问法
        "keywords": list[str],     # 关键词列表
        "all_queries": list[str]   # 合并后的去重列表（供 RAG 多路召回）
      }
    """

    async def rewrite(self, query: str) -> dict[str, str | list[str]]:
        """
        对用户问题进行语义扩展，返回改写结果字典。

        执行流程：
          1. 调用 fast model 生成改写结果（JSON 格式）
          2. 解析 JSON，提取 rewritten / alternatives / keywords
          3. 合并所有查询变体，去重保序，形成 all_queries 列表
          4. 任何步骤失败都优雅降级，不影响主流程

        降级行为（LLM 失败时）：
          - rewritten = 原始 query
          - alternatives = []
          - keywords = query 的前 5 个词（简单分词）
          - all_queries = [query]

        Args:
            query: 用户原始问题

        Returns:
            字典，包含 rewritten、alternatives、keywords、all_queries 四个字段

        Example:
            >>> result = await rewriter.rewrite("今年GMV多少")
            >>> result["all_queries"]
            ["今年GMV多少", "本财年总成交额是多少", "2026年GMV统计", "年度交易总额"]
        """
        import json

        router = get_router()
        prompt = _REWRITE_PROMPT.format(query=query)

        try:
            # 使用 fast model 降低成本和延迟，temperature=0.3 允许适当的表达多样性
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
            # 降级：直接使用原始 query，不影响主流程
            return {
                "rewritten": query,
                "alternatives": [],
                "keywords": query.split()[:5],
                "all_queries": [query],
            }

        rewritten = parsed.get("rewritten", query)
        alternatives = parsed.get("alternatives", [])
        keywords = parsed.get("keywords", [])

        # 合并所有查询变体：原始问题放第一位，权重最高
        all_queries = [query, rewritten] + alternatives

        # 去重保序：避免相同的 query 被多次向量化和搜索
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
            n_unique=len(unique),
        )
        return {
            "rewritten": rewritten,
            "alternatives": alternatives,
            "keywords": keywords,
            "all_queries": unique,
        }
