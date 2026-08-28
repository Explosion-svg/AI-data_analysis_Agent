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

import re

from ai_data_agent.model_gateway.router import get_router, TaskType
from ai_data_agent.model_gateway.base_model import Message
from ai_data_agent.observability.logger import get_logger

logger = get_logger(__name__)

# CJK 连续段（P4-7：用于关键词切分，避免整句中文被当成一个"关键词"）
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
# 拉丁字母/数字/下划线单词（英文、模型名、表名等）
_WORD_RE = re.compile(r"[A-Za-z0-9_]+")

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

    def __init__(self, router=None) -> None:
        """
        P3-5：模型路由器通过构造函数注入，而不是在内部调用全局 get_router()。

        注入 router 的好处：
        - 测试时可以注入 MockLLM，避免打真实全局 router
        - 生产环境中由 assembler 注入已装配（且带熔断保护）的 router
        - 不传时回退到全局单例，保持向后兼容

        Args:
            router: ModelRouter 实例（可选，默认使用全局单例）

        注意：router 为空时**惰性**解析全局单例（调用时再 get_router()），
        而不是在构造时急切解析。这样组件可以在全局 router 尚未装配时安全构造
        （测试、独立脚本、优雅启动阶段），也保留了 monkeypatch get_router 的测试方式。
        """
        # P3-5：经构造注入的 router 优先；None 时惰性回退全局单例
        self._router = router

    def _resolve_router(self):
        """返回注入的 router，或惰性解析全局单例（P3-5）。"""
        return self._router if self._router is not None else get_router()

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
            字典，包含 rewritten、alternatives、keywords、all_queries、reason 五个字段

        Example:
            >>> result = await rewriter.rewrite("今年GMV多少")
            >>> result["all_queries"]
            ["今年GMV多少", "本财年总成交额是多少", "2026年GMV统计", "年度交易总额"]
        """
        import json

        prompt = _REWRITE_PROMPT.format(query=query)

        try:
            # 使用 fast model 降低成本和延迟（P3-2：不强制指定 OpenAI 模型名，
            # 由 router 的 SIMPLE 路由自动选择快速模型，兼容 DeepSeek/Ollama 部署）
            resp = await self._resolve_router().generate(
                messages=[Message(role="user", content=prompt)],
                task_type=TaskType.SIMPLE,
                temperature=0.3,
                max_tokens=512,
            )
            raw = _strip_code_fence(resp.content)
            parsed = json.loads(raw)
        except Exception as e:
            logger.warning("query_rewriter.failed", error=str(e))
            # 降级：直接使用原始 query，不影响主流程
            return {
                "rewritten": query,
                "alternatives": [],
                # P4-7：中文查询用 CJK 2-gram 切分，不再把整句当"一个关键词"
                "keywords": _extract_keywords(query),
                "all_queries": [query],
                "reason": f"query rewriting failed, using original query: {e}",
            }

        rewritten = parsed.get("rewritten", query)
        alternatives = parsed.get("alternatives", [])
        keywords = parsed.get("keywords", [])
        if not isinstance(alternatives, list):
            alternatives = []
        if not isinstance(keywords, list):
            keywords = []
        if not isinstance(rewritten, str) or not rewritten:
            rewritten = query

        # 合并所有查询变体：原始问题放第一位，权重最高
        all_queries = [query, rewritten] + [a for a in alternatives if isinstance(a, str)]

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
            "alternatives": [a for a in alternatives if isinstance(a, str)][:3],
            "keywords": [k for k in keywords if isinstance(k, str)][:8],
            "all_queries": unique,
            "reason": parsed.get("reasoning", ""),
        }


def _strip_code_fence(text: str) -> str:
    """
    去除 LLM 可能输出的 markdown code fence（P3-1）。

    LLM 即使被要求"只返回 JSON"，也经常把 JSON 包在代码块里：
        ```json
        {"rewritten": "..."}
        ```
    直接 json.loads 会因为前导 ``` 而失败，导致每次静默降级。

    实现与 planner._strip_code_fence 一致（三重反引号），
    单独维护避免模块间循环导入。

    Args:
        text: 可能包含 markdown 代码块的文本

    Returns:
        去除代码块包装后的纯文本
    """
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # 去掉首行（```json 或 ```）
        lines = lines[1:]
        # 去掉末行（```）
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def _extract_keywords(query: str) -> list[str]:
    """
    从查询中提取检索关键词（P4-7 降级路径用）。

    之前的实现是 `query.split()[:5]`——对中文查询（无空格）会把整句
    当成一个"关键词"，检索时几乎无效。这里改为：
    - 拉丁字母/数字/下划线 → 按单词提取（如 "GMV"、"orders"）
    - 连续中文字段 → 按 2-gram（bigram）切分，这是中文检索的标准 token 化
    - 去掉纯数字等无检索意义的项，去重保序，最多返回 8 个

    Args:
        query: 用户原始查询

    Returns:
        关键词列表（可为空）
    """
    keywords: list[str] = list(_WORD_RE.findall(query))
    for cjk in _CJK_RE.findall(query):
        if len(cjk) <= 2:
            keywords.append(cjk)
        else:
            for i in range(len(cjk) - 1):
                keywords.append(cjk[i : i + 2])

    seen: set[str] = set()
    out: list[str] = []
    for kw in keywords:
        kw = kw.strip()
        if not kw or kw in seen or kw.isdigit():
            continue  # 去重 + 去掉纯数字
        seen.add(kw)
        out.append(kw)
    return out[:8]
