"""
orchestration/planner.py — 任务规划器

职责：评估复杂度 + 生成高层计划（每步"用什么工具、要达到什么目标"）
不负责生成具体参数（SQL、Python代码等），那是 Executor 的工作。

输出的 Plan.complexity 驱动 agent_loop 的路由决策：
  simple   → 直接走 ReAct，不需要计划
  moderate → Plan-and-Execute
  complex  → Plan-and-Execute
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ai_data_agent.model_gateway.router import get_router, TaskType
from ai_data_agent.model_gateway.base_model import Message
from ai_data_agent.config.config import settings
from ai_data_agent.observability.logger import get_logger

logger = get_logger(__name__)

# ── Prompt ────────────────────────────────────────────────────────────────────

_PLANNER_SYSTEM = """You are a data analysis task planner.
Your job: analyze the user's question and decide the MINIMUM steps needed to answer it.

Available tools:
{tools_description}

Return ONLY a valid JSON object (no markdown, no explanation):
{{
  "complexity": "simple|moderate|complex",
  "reasoning": "one sentence why",
  "needs_rag": true/false,
  "plan": [
    {{
      "step": 1,
      "tool": "<tool_name>",
      "goal": "<what specific result this step should produce>",
      "depends_on": []
    }}
  ]
}}

Complexity rules:
- simple   = direct question, single tool, ≤2 steps  (e.g. "list all tables", "what is GMV?")
- moderate = SQL + maybe chart, 2-4 steps
- complex  = multi-table join + analysis + chart, 4+ steps

Step ordering rules:
- Always do get_schema BEFORE sql_query if table structure is unknown
- python_analysis and generate_chart always depend on sql_query
- search_documents can run in parallel with other steps (depends_on: [])

IMPORTANT: The "goal" field must be precise and actionable.
Good:  "Query monthly sales amount from sales table grouped by month for year 2026"
Bad:   "get sales data"
"""


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class PlanStep:
    step: int
    tool: str
    goal: str                              # 精确目标，Executor 用来生成参数
    depends_on: list[int] = field(default_factory=list)
    # 以下字段由 Executor 填充
    tool_params: dict[str, Any] = field(default_factory=dict)   # Executor 生成的参数
    result: Any = None                     # 工具执行结果（ToolResult）
    done: bool = False
    error: str = ""

    @property
    def succeeded(self) -> bool:
        return self.done and self.result is not None and self.result.success


@dataclass
class Plan:
    complexity: str = "moderate"           # simple | moderate | complex
    reasoning: str = ""
    needs_rag: bool = False
    steps: list[PlanStep] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return len(self.steps) == 0

    @property
    def is_simple(self) -> bool:
        return self.complexity == "simple"

    def summary(self) -> str:
        lines = [f"Complexity: {self.complexity} — {self.reasoning}"]
        for s in self.steps:
            deps = f" (after {s.depends_on})" if s.depends_on else ""
            lines.append(f"  Step {s.step}: [{s.tool}]{deps} → {s.goal}")
        return "\n".join(lines)


# ── Planner ───────────────────────────────────────────────────────────────────

class Planner:
    """
    将用户问题规划为有序的工具调用步骤。
    只生成"做什么"，不生成"怎么做"（具体参数由 Executor 负责）。
    """

    async def plan(
        self,
        query: str,
        available_tools: list[str],
        schema_context: str = "",
    ) -> Plan:
        tools_desc = "\n".join(f"- {t}" for t in available_tools)
        router = get_router()

        messages = [
            Message(
                role="system",
                content=_PLANNER_SYSTEM.format(tools_description=tools_desc),
            ),
            Message(
                role="user",
                content=(
                    f"User question: {query}\n\n"
                    f"Known schema:\n{schema_context or '(unknown, may need get_schema first)'}"
                ),
            ),
        ]

        try:
            # 规划任务用 fast model 节省成本（结构简单）
            resp = await router.generate(
                messages=messages,
                task_type=TaskType.SIMPLE,
                temperature=0.0,
                max_tokens=1024,
            )
            raw = _strip_code_fence(resp.content)
            parsed = json.loads(raw)
        except Exception as e:
            logger.warning("planner.failed", error=str(e))
            # 降级：空计划，agent_loop 会走 ReAct
            return Plan(complexity="simple", reasoning="planning failed, fallback to ReAct")

        steps = [
            PlanStep(
                step=s["step"],
                tool=s.get("tool", ""),
                goal=s.get("goal", s.get("description", "")),
                depends_on=s.get("depends_on", []),
            )
            for s in parsed.get("plan", [])
            if s.get("tool")   # 过滤掉没有 tool 的无效步骤
        ]

        plan = Plan(
            complexity=parsed.get("complexity", "moderate"),
            reasoning=parsed.get("reasoning", ""),
            needs_rag=parsed.get("needs_rag", False),
            steps=steps,
        )

        logger.info(
            "planner.done",
            complexity=plan.complexity,
            steps=len(steps),
            reasoning=plan.reasoning,
        )
        logger.debug("planner.plan_summary", summary=plan.summary())
        return plan


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _strip_code_fence(text: str) -> str:
    """去除 LLM 可能输出的 markdown code fence。"""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # 去掉首行（```json）和末行（```）
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()
