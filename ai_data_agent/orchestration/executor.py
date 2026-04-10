"""
orchestration/executor.py — 计划执行器

职责：
  对 Plan 中的每一步：
    1. 调用 LLM 生成该步骤工具的具体参数（SQL、Python代码、图表配置…）
    2. 执行工具
    3. 将结果传递给后续依赖步骤

这就是 Planner 和 Executor 分工的关键：
  Planner  → 知道"做什么"（高层目标）
  Executor → 知道"怎么做"（具体参数，LLM动态生成）
"""
from __future__ import annotations

import json
from typing import Any

from ai_data_agent.orchestration.planner import Plan, PlanStep
from ai_data_agent.tools.tool_registry import get_registry
from ai_data_agent.tools.base_tool import ToolResult
from ai_data_agent.model_gateway.router import get_router, TaskType
from ai_data_agent.model_gateway.base_model import Message
from ai_data_agent.observability.logger import get_logger
from ai_data_agent.observability.tracer import span

logger = get_logger(__name__)

# ── Prompt：让 LLM 为单个步骤生成工具参数 ────────────────────────────────────

_PARAM_GEN_SYSTEM = """You are a precise tool parameter generator for a data analysis system.
Your job: generate ONLY the JSON parameters for a specific tool call.

Tool: {tool_name}
Tool parameter schema:
{tool_schema}

Current step goal: {goal}

Context from previous steps:
{previous_context}

Database schema:
{schema_context}

Rules:
- Return ONLY a valid JSON object matching the tool's parameter schema
- No explanation, no markdown, no extra text
- For sql_query: write a valid SELECT SQL. Use exact table/column names from the schema.
- For python_analysis: write complete runnable Python code. Use `df` for the data variable. Assign final result to `result`.
- For generate_chart: choose appropriate chart_type, x, y from the available data columns.
- For get_schema: use action "describe_table" if you know the table name, else "list_tables".
- For search_documents: extract a precise search query from the goal.
- Do NOT include "data" field for python_analysis or generate_chart — it will be injected automatically.
"""


class Executor:
    """
    Plan-and-Execute 执行引擎。

    每步执行流程：
      goal + 上下文  →  [LLM] 生成参数  →  执行工具  →  收集结果
    """

    async def execute(
        self,
        plan: Plan,
        schema_context: str = "",
    ) -> list[PlanStep]:
        """
        按拓扑顺序执行所有步骤。
        返回填充了结果的 steps 列表。
        """
        registry = get_registry()
        router = get_router()

        # step_num → result data（供后续依赖步骤使用）
        completed: dict[int, Any] = {}

        for step in plan.steps:
            with span("executor.step", {"step": step.step, "tool": step.tool}):
                # ── 检查依赖是否完成 ──────────────────────────────────────────
                missing_deps = [
                    d for d in step.depends_on if d not in completed
                ]
                if missing_deps:
                    # 依赖步骤失败，跳过当前步骤
                    step.error = f"Skipped: dependent steps {missing_deps} did not complete."
                    step.done = True
                    logger.warning(
                        "executor.step_skipped",
                        step=step.step,
                        missing_deps=missing_deps,
                    )
                    continue

                if step.tool not in registry:
                    step.error = f"Tool '{step.tool}' not registered."
                    step.done = True
                    logger.warning("executor.tool_not_found", tool=step.tool)
                    continue

                tool = registry.get(step.tool)

                # ── Step 1: LLM 生成工具参数 ──────────────────────────────────
                tool_params = await self._generate_params(
                    step=step,
                    tool_schema=tool.parameters_schema,
                    completed=completed,
                    plan_steps=plan.steps,
                    schema_context=schema_context,
                    router=router,
                )
                step.tool_params = tool_params

                # ── Step 2: 注入来自上一步的 data（data 不让 LLM 生成，避免幻觉）
                tool_params = self._inject_data(
                    tool_name=step.tool,
                    params=tool_params,
                    completed=completed,
                    plan_steps=plan.steps,
                )

                logger.info(
                    "executor.step_start",
                    step=step.step,
                    tool=step.tool,
                    goal=step.goal[:80],
                    params_preview=str(tool_params)[:120],
                )

                # ── Step 3: 执行工具 ──────────────────────────────────────────
                result: ToolResult = await tool.run(**tool_params)
                step.result = result
                step.done = True

                if result.success:
                    completed[step.step] = result.data
                    logger.info(
                        "executor.step_done",
                        step=step.step,
                        tool=step.tool,
                    )
                else:
                    step.error = result.error
                    logger.warning(
                        "executor.step_failed",
                        step=step.step,
                        tool=step.tool,
                        error=result.error,
                    )

        return plan.steps

    # ── 私有：LLM 生成参数 ────────────────────────────────────────────────────

    async def _generate_params(
        self,
        step: PlanStep,
        tool_schema: dict,
        completed: dict[int, Any],
        plan_steps: list[PlanStep],
        schema_context: str,
        router,
    ) -> dict[str, Any]:
        """
        让 LLM 根据 step.goal + 上下文生成工具参数 JSON。
        """
        previous_context = self._build_previous_context(step, plan_steps)

        prompt = _PARAM_GEN_SYSTEM.format(
            tool_name=step.tool,
            tool_schema=json.dumps(tool_schema, ensure_ascii=False, indent=2),
            goal=step.goal,
            previous_context=previous_context or "(no previous steps)",
            schema_context=schema_context or "(not available)",
        )

        try:
            resp = await router.generate(
                messages=[Message(role="user", content=prompt)],
                task_type=TaskType.CODE,    # 代码/参数生成用 code 路由
                temperature=0.0,
                max_tokens=1024,
            )
            from ai_data_agent.orchestration.planner import _strip_code_fence
            raw = _strip_code_fence(resp.content)
            params = json.loads(raw)
            logger.debug(
                "executor.params_generated",
                step=step.step,
                tool=step.tool,
                params=str(params)[:200],
            )
            return params
        except Exception as e:
            logger.warning(
                "executor.param_gen_failed",
                step=step.step,
                tool=step.tool,
                error=str(e),
            )
            # 降级：返回空参数，工具自己处理缺参情况
            return {}

    def _build_previous_context(
        self,
        current_step: PlanStep,
        all_steps: list[PlanStep],
    ) -> str:
        """将已完成步骤的结果摘要拼成文本，注入给 LLM。"""
        lines = []
        for s in all_steps:
            if s.step >= current_step.step:
                break
            if s.done and s.result and s.result.success:
                # 只取 text 摘要，不传原始 data（避免 token 爆炸）
                preview = (s.result.text or "")[:500]
                lines.append(f"Step {s.step} [{s.tool}] result:\n{preview}")
            elif s.done and s.error:
                lines.append(f"Step {s.step} [{s.tool}] FAILED: {s.error}")
        return "\n\n".join(lines)

    # ── 私有：注入 data 字段 ──────────────────────────────────────────────────

    def _inject_data(
        self,
        tool_name: str,
        params: dict[str, Any],
        completed: dict[int, Any],
        plan_steps: list[PlanStep],
    ) -> dict[str, Any]:
        """
        python_analysis 和 generate_chart 需要前序 SQL 结果作为 data。
        data 由 Executor 注入，不让 LLM 生成（LLM 不知道真实数据）。
        """
        if tool_name not in ("python_analysis", "generate_chart"):
            return params

        sql_data = self._find_latest_result(plan_steps, "sql_query")
        if sql_data is not None:
            params = {**params, "data": sql_data}
            logger.debug(
                "executor.data_injected",
                tool=tool_name,
                rows=len(sql_data) if isinstance(sql_data, list) else "?",
            )
        return params

    def _find_latest_result(
        self,
        plan_steps: list[PlanStep],
        tool_name: str,
    ) -> Any:
        """从已完成步骤中找最近一个指定工具的 data 结果。"""
        for step in reversed(plan_steps):
            if (
                step.tool == tool_name
                and step.done
                and step.result
                and step.result.success
            ):
                return step.result.data
        return None
