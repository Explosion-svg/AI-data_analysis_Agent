"""
orchestration/executor.py — 计划执行器（Plan-and-Execute）

职责：
  接收 Planner 生成的 Plan，按 DAG（有向无环图）拓扑顺序执行每个步骤：
  1. 为每个步骤调用 LLM 动态生成工具的具体参数（SQL、Python 代码、图表配置等）
  2. 执行工具并收集结果
  3. 把依赖步骤的结果传递给后续步骤（如 SQL 结果 → Python 分析的 df 参数）

Planner vs Executor 的分工（核心设计决策）：
  Planner  → 知道"做什么"：tool + goal + depends_on（高层目标，结构简单）
  Executor → 知道"怎么做"：通过 LLM 根据 goal + 上下文动态生成具体参数

这种分工的好处：
  1. Planner 的输出是"类型安全"的（只有工具名和目标，无需验证具体参数格式）
  2. Executor 可以把已完成步骤的结果注入后续步骤的参数生成 prompt，
     形成"上下文感知"的参数生成（如 Python 分析能看到 SQL 返回了哪些列）
  3. 两者可以独立测试：Planner 测试计划质量，Executor 测试参数生成质量

并行执行：
  没有依赖关系的步骤（depends_on=[]）可以并行执行。
  使用 asyncio.gather() + asyncio.Semaphore 控制最大并行步骤数
  （settings.executor_max_parallel_steps，默认 3）。

死锁检测：
  如果 pending 步骤中没有可运行的（所有依赖步骤都未完成），
  检测到可能的循环依赖，标记所有剩余步骤为 "Skipped"。

数据注入：
  python_analysis 和 generate_chart 需要前序 SQL 查询的数据。
  这个数据由 Executor 自动注入（_inject_data()），
  而不是让 LLM 生成（LLM 不知道真实数据内容）。
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from ai_data_agent.config.config import settings
from ai_data_agent.orchestration.planner import Plan, PlanStep
from ai_data_agent.tools.tool_registry import get_registry
from ai_data_agent.tools.base_tool import ToolResult
from ai_data_agent.model_gateway.router import get_router, TaskType
from ai_data_agent.model_gateway.base_model import Message
from ai_data_agent.observability.logger import get_logger
from ai_data_agent.observability.tracer import span

logger = get_logger(__name__)

# ── 参数生成提示词 ────────────────────────────────────────────────────────────

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

    核心执行流程（每步）：
      goal + 上下文  →  [LLM] 生成具体参数  →  执行工具  →  收集结果

    数据流（跨步骤）：
      sql_query 的结果 → 注入 python_analysis/generate_chart 的 data 参数

    失败处理策略：
    - 单个步骤失败（tool_result.success=False）：标记该步骤失败，
      依赖该步骤的所有后续步骤被标记为 "Skipped"
    - 参数生成失败（LLM 调用失败）：返回空参数 {}，工具自己处理缺参情况
    - 工具不存在：立即标记步骤失败，不进入执行
    """

    async def execute(
        self,
        plan: Plan,
        schema_context: str = "",
    ) -> list[PlanStep]:
        """
        按 DAG 拓扑顺序执行 Plan 中的所有步骤。

        调度算法（事件循环驱动）：
        1. 初始化 pending（所有步骤）、completed（已完成步骤的结果）、failed（失败步骤 ID）
        2. 每轮循环：
           a. 先标记所有依赖失败步骤的步骤为 "Skipped"（_mark_blocked_steps）
           b. 找出所有依赖都已完成的可运行步骤（runnable）
           c. 用 asyncio.gather 并行执行所有可运行步骤（受 Semaphore 限流）
           d. 把完成的步骤从 pending 移除
        3. 如果 pending 不为空但 runnable 为空，说明存在循环依赖，直接退出

        这个算法保证：
        - 依赖关系正确（不会在依赖完成前执行步骤）
        - 最大化并行度（无依赖的步骤同时执行）
        - 不会无限等待（检测死锁并退出）

        Args:
            plan: Planner 生成的执行计划
            schema_context: 数据库 schema 上下文（传给参数生成 LLM）

        Returns:
            填充了执行结果的 PlanStep 列表（所有步骤，包括失败和跳过的）
        """
        registry = get_registry()
        router = get_router()

        completed: dict[int, Any] = {}         # step_id → result.data（已完成步骤的数据）
        pending: dict[int, PlanStep] = {step.step: step for step in plan.steps}
        failed: set[int] = set()               # 已失败步骤的 ID 集合
        # 并行度限制：防止同时发起过多 LLM 调用和工具执行
        sem = asyncio.Semaphore(max(1, settings.executor_max_parallel_steps))

        while pending:
            # 先标记被失败步骤阻塞的步骤（级联失败）
            self._mark_blocked_steps(pending, completed, failed)

            # 找出可以立即执行的步骤（所有依赖都已完成）
            runnable = [
                step
                for step in pending.values()
                if all(dep in completed for dep in step.depends_on)
            ]

            if not runnable:
                # pending 中没有可运行的步骤 → 可能存在循环依赖或上面的 _mark_blocked_steps 没有处理干净
                # 直接跳出，标记剩余步骤为死锁
                for step in pending.values():
                    step.done = True
                    step.error = "Skipped: cyclic or unresolved dependencies."
                    logger.warning(
                        "executor.step_deadlocked",
                        step=step.step,
                        deps=step.depends_on,
                    )
                break

            # 并行执行所有可运行步骤（asyncio.gather 不阻塞，等待全部完成）
            await asyncio.gather(
                *[
                    self._execute_step(
                        step=step,
                        completed=completed,
                        plan_steps=plan.steps,
                        schema_context=schema_context,
                        registry=registry,
                        router=router,
                        sem=sem,
                    )
                    for step in runnable
                ]
            )

            # 把刚完成的步骤从 pending 移除
            for step in runnable:
                pending.pop(step.step, None)
                # 失败的步骤加入 failed 集合（触发后续步骤的级联跳过）
                if not (step.result and step.result.success):
                    failed.add(step.step)

        return plan.steps

    async def _execute_step(
        self,
        *,
        step: PlanStep,
        completed: dict[int, Any],
        plan_steps: list[PlanStep],
        schema_context: str,
        registry,
        router,
        sem: asyncio.Semaphore,
    ) -> None:
        """
        执行单个计划步骤（在 Semaphore 保护的并发槽内运行）。

        执行流程：
        1. 检查工具是否存在（不存在立即标记失败）
        2. 调用 LLM 生成工具参数（_generate_params）
        3. 注入依赖数据（_inject_data，python/chart 工具需要 SQL 结果）
        4. 执行工具（tool.run(**tool_params)）
        5. 更新 step 状态和 completed 字典

        使用 OpenTelemetry span 包裹每步执行，方便追踪各步骤的耗时。

        注意：这个方法是 async 的，在 Semaphore 内部执行，
        Semaphore 确保同时运行的步骤数不超过 executor_max_parallel_steps。

        Args:
            step: 要执行的步骤（原地修改其 result、done、error 字段）
            completed: 已完成步骤的结果字典（执行成功后写入）
            plan_steps: 所有步骤列表（用于查找依赖步骤的结果）
            schema_context: 数据库 schema 上下文
            registry: 工具注册中心
            router: 模型路由器（用于参数生成）
            sem: 并发信号量（限制最大并行步骤数）
        """
        async with sem:
            with span("executor.step", {"step": step.step, "tool": step.tool}):
                # Step 1: 检查工具是否存在
                if step.tool not in registry:
                    step.error = f"Tool '{step.tool}' not found."
                    step.done = True
                    logger.warning("executor.tool_not_found", tool=step.tool)
                    return

                tool = registry.get(step.tool)

                # Step 2: LLM 生成工具参数（Code 路由 - DeepSeek 代码更强）
                tool_params = await self._generate_params(
                    step=step,
                    tool_schema=tool.parameters_schema,
                    completed=completed,
                    plan_steps=plan_steps,
                    schema_context=schema_context,
                    router=router,
                )
                step.tool_params = tool_params  # 记录生成的参数（调试用）

                # Step 3: 注入依赖数据（python_analysis/generate_chart 需要 SQL 结果）
                tool_params = self._inject_data(
                    tool_name=step.tool,
                    params=tool_params,
                    completed=completed,
                    plan_steps=plan_steps,
                    depends_on=step.depends_on,
                )

                logger.info(
                    "executor.step_start",
                    step=step.step,
                    tool=step.tool,
                    goal=step.goal[:80],
                    params_preview=str(tool_params)[:120],
                )

                # Step 4: 执行工具
                result: ToolResult = await tool.run(**tool_params)
                step.result = result
                step.done = True

                if result.success:
                    # 成功：把结果数据存入 completed，供后续依赖步骤使用
                    completed[step.step] = result.data
                    logger.info(
                        "executor.step_done",
                        step=step.step,
                        tool=step.tool,
                    )
                    return

                # 失败：记录错误（不抛异常，让 execute() 的 failed 集合处理级联跳过）
                step.error = result.error
                logger.warning(
                    "executor.step_failed",
                    step=step.step,
                    tool=step.tool,
                    error=result.error,
                )

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
        调用 LLM 根据 step.goal 和上下文动态生成工具参数。

        这是 Executor 最核心的功能：把高层目标（goal）转换为具体参数（tool_params）。
        例如：
        - goal: "Query monthly sales from sales table grouped by month for 2026"
        - 生成: {"sql": "SELECT MONTH(date) as month, SUM(amount) FROM sales WHERE YEAR(date)=2026 GROUP BY 1"}

        提示词包含：
        1. tool_schema：工具的 JSON Schema（告诉 LLM 参数格式要求）
        2. goal：当前步骤的目标（来自 Planner）
        3. previous_context：已完成步骤的结果摘要（提供数据上下文）
        4. schema_context：数据库结构（提供可用表/列信息）

        使用 TaskType.CODE 路由（DeepSeek 代码更强），temperature=0.0（确定性输出）。

        失败时返回空字典 {}，让工具自己处理缺参情况（通常是参数验证错误）。

        Args:
            step: 当前步骤（包含 goal 和 depends_on）
            tool_schema: 工具的 JSON Schema 参数定义
            completed: 已完成步骤的结果（用于构建 previous_context）
            plan_steps: 所有步骤列表（用于构建 previous_context）
            schema_context: 数据库 schema 上下文
            router: 模型路由器

        Returns:
            LLM 生成的工具参数字典；解析失败时返回 {}
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
                task_type=TaskType.CODE,    # 代码/参数生成优先 DeepSeek
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
            # 降级：返回空参数，工具自己处理缺参情况（通常会返回错误 ToolResult）
            return {}

    def _build_previous_context(
        self,
        current_step: PlanStep,
        all_steps: list[PlanStep],
    ) -> str:
        """
        将当前步骤之前所有已完成步骤的结果摘要拼接为文本，注入参数生成 prompt。

        为什么注入前序步骤的结果？
        - 后续步骤的参数生成需要了解前步骤的输出（如 SQL 返回了哪些列）
        - 例如：generate_chart 需要知道 SQL 结果有哪些列，才能选择正确的 x/y 轴

        只传 text 摘要，不传原始 data 的原因：
        - 原始 data 可能是大型 DataFrame（数千行数据）
        - 传给 LLM 会导致 token 爆炸
        - text 摘要（如 "返回 5 列: date, revenue, cost..."）足以指导参数生成

        Args:
            current_step: 当前待执行步骤
            all_steps: 所有步骤列表

        Returns:
            已完成步骤的结果摘要文本（多步之间用双换行分隔）
        """
        lines = []
        for s in all_steps:
            if s.step >= current_step.step:
                break  # 只看当前步骤之前的步骤
            if s.done and s.result and s.result.success:
                preview = (s.result.text or "")[:500]  # 最多 500 字符摘要
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
        depends_on: list[int],
    ) -> dict[str, Any]:
        """
        为 python_analysis 和 generate_chart 注入前序 SQL 查询结果数据。

        为什么需要这个注入机制？
        - python_analysis 和 generate_chart 需要实际的数据（list[dict]）才能运行
        - LLM 在参数生成时不知道真实数据的具体内容（只知道结构）
        - 因此数据必须由 Executor 在运行时注入，而不是让 LLM 生成

        为什么 LLM 的参数生成 prompt 明确说 "Do NOT include 'data' field"？
        - 防止 LLM 生成无意义的占位数据（如 {"data": []}）
        - Executor 会在这里自动注入真实数据，覆盖 LLM 可能错误生成的内容

        查找策略（两级 fallback）：
        1. 优先查找 depends_on 中的 sql_query 步骤结果
        2. 如果依赖中没有 sql_query，从所有步骤中查找最近一个成功的 sql_query

        Args:
            tool_name: 当前工具名称（只对 python_analysis 和 generate_chart 有效）
            params: LLM 生成的参数字典（可能被修改）
            completed: 已完成步骤的结果字典
            plan_steps: 所有步骤列表（用于全局查找）
            depends_on: 当前步骤的依赖列表

        Returns:
            注入了 data 字段（如果找到了 SQL 结果）的参数字典
        """
        if tool_name not in ("python_analysis", "generate_chart"):
            return params  # 其他工具不需要注入数据

        # 首先查找直接依赖中的 sql_query 结果
        sql_data = self._find_latest_dependency_result(
            plan_steps=plan_steps,
            completed=completed,
            depends_on=depends_on,
            tool_name="sql_query",
        )
        # 如果依赖中没有 sql_query，全局查找最近的 sql_query 结果
        if sql_data is None:
            sql_data = self._find_latest_result(plan_steps, "sql_query")

        if sql_data is not None:
            params = {**params, "data": sql_data}  # 不修改原字典（使用新字典）
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
        """
        从所有已完成步骤中找最近一个指定工具的执行结果数据。

        "最近"的定义：步骤编号最大（靠后执行）的成功结果。
        使用 reversed() 从后往前遍历，第一个匹配即是最近的。

        Args:
            plan_steps: 所有步骤列表
            tool_name: 要查找的工具名称

        Returns:
            找到的 result.data，如果没找到则返回 None
        """
        for step in reversed(plan_steps):
            if (
                step.tool == tool_name
                and step.done
                and step.result
                and step.result.success
            ):
                return step.result.data
        return None

    @staticmethod
    def _find_latest_dependency_result(
        *,
        plan_steps: list[PlanStep],
        completed: dict[int, Any],
        depends_on: list[int],
        tool_name: str,
    ) -> Any:
        """
        从当前步骤的直接依赖中找最近一个指定工具的执行结果。

        优先使用 completed 字典（比 plan_steps 查找更快），
        通过 depends_on 过滤只看当前步骤依赖的步骤。

        与 _find_latest_result 的区别：
        - _find_latest_result：全局查找，任意历史步骤
        - _find_latest_dependency_result：限定在当前步骤的直接依赖中查找

        Args:
            plan_steps: 所有步骤列表（用于查找 tool 名称）
            completed: 已完成步骤的结果字典
            depends_on: 当前步骤的依赖步骤 ID 列表
            tool_name: 要查找的工具名称

        Returns:
            找到的依赖步骤的 data，如果没找到则返回 None
        """
        # 从最近的依赖开始查找（reversed 保证优先找最近的）
        for dep in reversed(depends_on):
            for step in plan_steps:
                if step.step != dep:
                    continue
                if step.tool == tool_name and dep in completed:
                    return completed[dep]
        return None

    @staticmethod
    def _mark_blocked_steps(
        pending: dict[int, PlanStep],
        completed: dict[int, Any],
        failed: set[int],
    ) -> None:
        """
        标记所有因依赖失败而无法执行的步骤（级联跳过）。

        级联失败原则：
        如果步骤 A 失败，那么所有依赖 A 的步骤（B、C）也应该被跳过。
        类比 CI/CD 管道：某个关键步骤失败后，后续步骤不应继续执行。

        实现：
        1. 找出所有 depends_on 中包含 failed 步骤的步骤
        2. 把它们标记为 done=True（使其从 pending 中被移除）
        3. 设置错误信息（说明是哪些依赖步骤导致了跳过）
        4. 把它们加入 failed 集合（触发进一步的级联）
        5. 从 pending 中移除它们

        注意：这个方法在每轮循环开始时调用，
        确保每轮新的失败都能触发依赖链上的级联跳过。

        Args:
            pending: 待执行步骤字典（原地修改）
            completed: 已完成步骤（只读，用于判断依赖是否满足）
            failed: 已失败步骤 ID 集合（原地修改）
        """
        blocked = [
            step
            for step in pending.values()
            if any(dep in failed for dep in step.depends_on)
        ]
        for step in blocked:
            failed_deps = [dep for dep in step.depends_on if dep in failed]
            step.done = True
            step.error = f"Skipped: dependent steps {failed_deps} failed."
            logger.warning(
                "executor.step_skipped",
                step=step.step,
                missing_deps=failed_deps,
            )
            failed.add(step.step)
            pending.pop(step.step, None)
