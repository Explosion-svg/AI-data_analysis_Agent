"""
orchestration/planner.py — 任务规划器（Planner）

职责：
  评估用户问题的复杂度，并为复杂问题生成高层执行计划。
  Plan 的每一步只定义"用什么工具、要达到什么目标"，
  不生成具体的工具参数（SQL、Python 代码等）——那是 Executor 的职责。

Planner 在 Plan-and-Execute 架构中的位置：
  用户问题 → [Planner] → Plan（高层步骤）
                                ↓
                         [Executor] → 逐步执行（LLM 动态生成参数）
                                ↓
                         [AgentLoop] → 合成最终答案

与 ReAct 模式的关系：
  - 简单问题（complexity="simple"）：Planner 退出，走 ReAct 循环
  - 复杂问题（complexity="moderate"/"complex"）：走 Plan-and-Execute
  - Planner 失败时：降级返回 Plan(complexity="simple")，确保主流程不断

提示词工程（_PLANNER_SYSTEM）：
  - 明确定义"复杂度"标准（simple/moderate/complex 的量化规则）
  - 给出步骤顺序约束（如 get_schema 必须在 sql_query 之前）
  - 要求输出纯 JSON，不含 markdown 和解释
  - 给出 goal 字段的"好/坏示例"（防止模型生成模糊目标）
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

# ── 规划器系统提示词 ──────────────────────────────────────────────────────────

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


# ── 数据类 ─────────────────────────────────────────────────────────────────────

@dataclass
class PlanStep:
    """
    计划中的单个执行步骤（高层定义）。

    Planner 负责填充：step、tool、goal、depends_on。
    Executor 负责填充：tool_params、result、done、error。

    这种职责分离保证了 Planner 的输出是"干净的"（不含执行细节），
    使得 Plan 可以单独测试和验证，独立于 Executor 的实现。

    depends_on 字段：
    - 值为 step 编号的列表（如 [1, 2] 表示依赖第 1、2 步完成）
    - 空列表表示可以并行执行（与其他步骤无依赖关系）
    - Executor 用 DAG（有向无环图）拓扑排序来决定并行执行哪些步骤

    succeeded 属性：
    - done=True 且 result 非 None 且 result.success=True
    - 只有三个条件都满足才算真正成功
    """

    step: int                                      # 步骤编号（1-based）
    tool: str                                      # 工具名称（对应 ToolRegistry 的 key）
    goal: str                                      # 精确目标描述（Executor 用来生成参数）
    depends_on: list[int] = field(default_factory=list)  # 依赖的步骤编号列表
    # 以下字段由 Executor 填充（Planner 阶段为默认值）
    tool_params: dict[str, Any] = field(default_factory=dict)  # Executor 生成的工具参数
    result: Any = None                             # 工具执行结果（ToolResult 对象）
    done: bool = False                             # 是否已执行完成
    error: str = ""                                # 执行失败时的错误信息

    @property
    def succeeded(self) -> bool:
        """
        判断步骤是否成功完成。

        需要同时满足三个条件：
        1. done=True（Executor 已处理过这个步骤）
        2. result 非 None（工具有返回值，不是因为工具不存在而被跳过）
        3. result.success=True（工具自身报告执行成功）

        Returns:
            True 表示步骤成功完成
        """
        return self.done and self.result is not None and self.result.success


@dataclass
class Plan:
    """
    Planner 生成的任务执行计划。

    包含：
    - complexity：问题复杂度评估（驱动 agent_loop 的路由决策）
    - reasoning：复杂度判断的简短理由（用于日志和调试）
    - needs_rag：是否需要知识库检索（AgentLoop 可据此提前触发 RAG）
    - steps：有序步骤列表（由 Executor 按 DAG 拓扑顺序执行）

    is_empty 和 is_simple 属性：
    AgentLoop 用这两个属性判断是否退回 ReAct 循环：
    - is_empty：Planner 没有生成任何步骤（规划失败的降级计划）
    - is_simple：问题太简单，不需要多步骤规划，直接走 ReAct 更高效
    """

    complexity: str = "moderate"                   # simple | moderate | complex
    reasoning: str = ""                            # 一句话理由（日志用）
    needs_rag: bool = False                        # 是否需要 RAG 检索
    steps: list[PlanStep] = field(default_factory=list)  # 步骤列表

    @property
    def is_empty(self) -> bool:
        """
        计划是否为空（没有任何步骤）。

        空计划通常意味着 Planner 调用失败并降级，
        AgentLoop 应该退回 ReAct 循环。

        Returns:
            True 表示计划没有步骤
        """
        return len(self.steps) == 0

    @property
    def is_simple(self) -> bool:
        """
        问题是否被评估为简单级别。

        complexity="simple" 时直接走 ReAct 循环更高效，
        因为简单问题通常只需要 1-2 步工具调用，ReAct 的动态性更灵活。

        Returns:
            True 表示 Planner 认为问题足够简单，不需要多步规划
        """
        return self.complexity == "simple"

    def summary(self) -> str:
        """
        生成计划的可读摘要（用于日志和调试）。

        示例输出：
            Complexity: moderate — User asks for monthly trend with chart
              Step 1: [get_schema] → Get columns from sales table
              Step 2: [sql_query] (after [1]) → Query monthly sales grouped by month
              Step 3: [generate_chart] (after [2]) → Generate line chart of monthly sales

        Returns:
            多行文本格式的计划摘要
        """
        lines = [f"Complexity: {self.complexity} — {self.reasoning}"]
        for s in self.steps:
            deps = f" (after {s.depends_on})" if s.depends_on else ""
            lines.append(f"  Step {s.step}: [{s.tool}]{deps} → {s.goal}")
        return "\n".join(lines)


# ── 规划器 ────────────────────────────────────────────────────────────────────

class Planner:
    """
    将用户问题规划为有序工具调用步骤的规划器。

    设计原则：
    - 只生成"做什么"（tool + goal + depends_on），不生成"怎么做"（具体参数）
    - 参数生成是 Executor 的职责，这样 Planner 的输出更简洁、可验证
    - 使用 TaskType.SIMPLE 路由（fast model），因为规划任务结构简单，节省成本

    失败降级策略：
    - 如果 LLM 调用失败或输出无法解析，返回 Plan(complexity="simple")
    - 这个降级计划会触发 AgentLoop 退回 ReAct 循环
    - 保证 Planner 失败不会影响主流程的可用性
    """

    def __init__(self, router=None) -> None:
        """
        P3-5：模型路由器通过构造函数注入，而不是在内部调用全局 get_router()。

        注入 router 的好处：
        - Planner 的 LLM 调用可以接受熔断保护（assembler 注入已装配的 router）
        - 测试时可以注入 MockLLM，避免打真实全局 router
        - 不传时回退到全局单例，保持向后兼容

        Args:
            router: ModelRouter 实例（可选，默认使用全局单例）

        注意：router 为空时**惰性**解析全局单例（调用时再 get_router()），
        而不是在构造时急切解析。这样组件可以在全局 router 尚未装配时安全构造
        （测试、独立脚本），也保留了 monkeypatch get_router 的测试方式。
        """
        # P3-5：经构造注入的 router 优先；None 时惰性回退全局单例
        self._router = router

    def _resolve_router(self):
        """返回注入的 router，或惰性解析全局单例（P3-5）。"""
        return self._router if self._router is not None else get_router()

    async def plan(
        self,
        query: str,
        available_tools: list[str],
        schema_context: str = "",
    ) -> Plan:
        """
        分析用户问题，生成执行计划。

        流程：
        1. 构建提示词（工具列表 + 用户问题 + schema 上下文）
        2. 调用 fast model 生成 JSON 格式的计划
        3. 解析 JSON，构造 Plan 和 PlanStep 对象
        4. 过滤无效步骤（没有 tool 字段的步骤）
        5. 记录日志（complexity、steps、reasoning）

        schema_context 的作用：
        - 告诉 Planner 数据库中有哪些表
        - 有了 schema，Planner 可以决定是否需要 get_schema 步骤
        - 没有 schema 时，Planner 通常会把 get_schema 放在第一步

        Args:
            query: 用户的自然语言查询
            available_tools: 当前注册的工具名称列表
            schema_context: 已知的数据库 schema 上下文（可选）

        Returns:
            Plan 对象（失败时返回 Plan(complexity="simple") 降级计划）
        """
        tools_desc = "\n".join(f"- {t}" for t in available_tools)

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
            # 规划任务用 SIMPLE/fast model 节省成本（结构简单，不需要最强模型）
            resp = await self._resolve_router().generate(
                messages=messages,
                task_type=TaskType.SIMPLE,
                temperature=0.0,     # 规划需要确定性输出，不要随机性
                max_tokens=1024,     # 计划通常不长
            )
            raw = _strip_code_fence(resp.content)
            parsed = json.loads(raw)
            # P3-3：步骤构造纳入 try 块——模型输出不合规时降级为 ReAct 兜底，
            # 而不是让 KeyError 逃逸导致整请求 500。
            steps = _parse_steps(parsed)
        except Exception as e:
            logger.warning("planner.failed", error=str(e))
            # 任何失败都降级为简单计划，让 AgentLoop 走 ReAct
            return Plan(complexity="simple", reasoning="planning failed, fallback to ReAct")

        plan = Plan(
            complexity=_as_str(parsed.get("complexity"), "moderate"),
            reasoning=_as_str(parsed.get("reasoning"), ""),
            needs_rag=bool(parsed.get("needs_rag", False)),
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

def _as_str(value: Any, default: str) -> str:
    """
    将任意值安全转换为字符串，非法值时返回默认值（P3-3）。

    模型输出不可信，字段可能是 int/bool/None/嵌套结构。
    统一收敛为 str，避免下游消费非字符串字段。

    Args:
        value: 待转换的值
        default: 转换失败或非法时的默认值

    Returns:
        转换后的字符串
    """
    if value is None:
        return default
    if isinstance(value, str):
        return value if value else default
    return str(value)


def _parse_steps(parsed: dict[str, Any]) -> list[PlanStep]:
    """
    从解析后的计划 JSON 中安全构造步骤列表（P3-3）。

    防御性解析要点：
    - 用 .get 取字段并做类型纠正，模型漏 "step" 字段时不抛 KeyError
    - 过滤掉缺少 tool 字段的无效步骤
    - step 编号去重重排：重复编号或非法编号会被重新编号为 1..N，
      避免 executor.py 中重复 step 号静默覆盖（step_id 冲突）
    - depends_on 归一化为 int 列表，且只保留存在的步骤编号
    - goal 兼容 "description" 字段名

    Args:
        parsed: Planner 模型输出的 JSON 字典（可能结构不完整）

    Returns:
        规范化后的 PlanStep 列表（至少为空列表）
    """
    raw_steps = parsed.get("plan", [])
    if not isinstance(raw_steps, list):
        return []

    valid: list[PlanStep] = []
    for i, s in enumerate(raw_steps, start=1):
        if not isinstance(s, dict):
            continue
        tool = s.get("tool")
        if not tool or not isinstance(tool, str):
            continue  # 过滤掉没有 tool 的无效步骤

        goal = s.get("goal", s.get("description", ""))
        depends_raw = s.get("depends_on", [])
        if not isinstance(depends_raw, list):
            depends_raw = []

        valid.append(
            PlanStep(
                step=i,  # P3-3：强制重新编号 1..N，避免重复/缺失编号
                tool=tool,
                goal=_as_str(goal, ""),
                depends_on=[int(d) for d in depends_raw if isinstance(d, (int, str)) and str(d).isdigit()],
            )
        )
    return valid


def _strip_code_fence(text: str) -> str:
    """
    去除 LLM 可能输出的 markdown code fence。

    即使 prompt 要求 "no markdown"，LLM 有时仍然会把 JSON 包在代码块里：
        ```json
        {"key": "value"}
        ```

    这个函数去除首尾的 ``` 行（包括可选的语言标识符如 "json"），
    提取内部的纯 JSON 文本。

    注意：这只处理 triple backtick（```），不处理 single backtick（`）。

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
