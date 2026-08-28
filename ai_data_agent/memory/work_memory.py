"""
memory/work_memory.py — 工作记忆（Work Memory）

职责：
  记录单次分析任务（一次用户请求）在 Agent 执行期间的完整状态轨迹，
  让 Agent 在 ReAct 循环中始终知道"当前做到了哪一步"。

与 ConversationMemory 的边界（清晰区分是关键）：
  ConversationMemory 负责 → "用户和助手聊过什么"（对话语义层）
  WorkMemory        负责 → "当前任务执行到了什么状态"（任务执行层）
  两者桥接         → WorkMemory.build_conversation_bridge() 把执行摘要
                    传递给 ConversationMemory，轻量单向同步

允许进入 WorkMemory 的内容：
  - 查询改写结果（query rewrite）
  - 数据库 schema 摘要和涉及的表名
  - 每次工具调用的参数快照和执行摘要
  - 关键发现（findings）和错误信息
  - 最近执行的 SQL、数据摘要、图表等产物引用

不应该进入 WorkMemory 的内容：
  - 长期知识库文档全文（放 VectorStore）
  - 跨会话共享的用户偏好（放 ConversationMemory.pinned_facts）
  - 作为"对话原文"回放给模型的聊天记录（放 ConversationMemory.recent_turns）

数据模型层次：
  WorkMemory
  └── _store: dict[conversation_id, WorkState]
      └── WorkState（单次运行）
          ├── steps: list[WorkStep]（工具调用轨迹）
          └── artifacts: list[WorkArtifact]（产物引用）

当前版本：
  按"每个 conversation 一个最近运行"建模。
  如需多运行归档，可把 _store 改为 conversation_id -> list[WorkState]。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any
from collections import OrderedDict
import uuid

from ai_data_agent.config.config import settings
from ai_data_agent.observability.logger import get_logger

logger = get_logger(__name__)


def _utcnow() -> datetime:
    """
    统一时间源（UTC）。

    单独封装的原因：
    1. 方便单元测试 mock（可以 patch 这个函数控制时间）
    2. 统一保证所有时间戳都使用 UTC，不受本地时区影响
    3. 将来如需替换为更精确的时间源（如 time.time_ns()），只改这一处

    Returns:
        当前 UTC 时间
    """
    return datetime.utcnow()


@dataclass
class WorkArtifact:
    """
    Agent 执行过程中产生的"产物引用"。

    设计原则：只保存引用（类型 + 预览 + 元数据），不保存原始数据体。
    原因：
    - 完整数据体（如 SQL 结果 DataFrame、图表 JSON）可能很大
    - 放入工作记忆会导致状态无限膨胀，最终撑爆 prompt token 预算
    - 产物的实际内容可以通过 metadata 中的引用（文件路径、URL）按需获取

    典型产物类型：
    - "sql_result"：SQL 查询结果（preview = 首行摘要，metadata = {"rows": N}）
    - "chart"：Plotly 图表（preview = 图表标题，metadata = {"tool": "generate_chart"}）
    - "python_output"：Python 分析输出（preview = 结果摘要）

    artifact_id 使用 uuid4().hex（32 字符十六进制），保证唯一性，
    方便未来构建产物存储系统时通过 ID 检索原始数据。
    """

    artifact_id: str                               # 产物唯一 ID（uuid4 hex）
    type: str                                      # 产物类型（"sql_result", "chart", etc.）
    preview: str = ""                              # 短预览文本（最多 1000 字符）
    metadata: dict[str, Any] = field(default_factory=dict)  # 额外元数据
    created_at: datetime = field(default_factory=_utcnow)   # 创建时间（UTC）


@dataclass
class WorkStep:
    """
    ReAct 循环中的单次工具调用记录。

    与 Planner.PlanStep 的区别：
    - PlanStep 是"计划层"的步骤定义（知道做什么，不知道怎么做）
    - WorkStep 是"执行层"的步骤记录（记录实际发生了什么，结果如何）

    status 状态机：
    - "pending"  → 刚创建，尚未开始执行
    - "running"  → 已开始执行，等待工具返回
    - "done"     → 执行成功（result.success=True）
    - "failed"   → 执行失败（error 非空）

    字段截断策略：
    - observation 截断至 2000 字符（工具返回的原始观测文本可能很长）
    - result_summary 截断至 1000 字符（WorkMemorySummarizer 生成的摘要）
    - error 截断至 1000 字符（防止超长错误栈溢出）
    """

    step_id: str                                   # 步骤唯一 ID（uuid4 hex）
    iteration: int                                 # 所属 ReAct 循环迭代次数（1-based）
    tool: str                                      # 调用的工具名称
    args: dict[str, Any] = field(default_factory=dict)  # 工具参数（原始调用参数）
    status: str = "pending"                        # pending | running | done | failed
    started_at: datetime = field(default_factory=_utcnow)   # 开始时间
    finished_at: datetime | None = None            # 结束时间（None 表示尚未完成）
    observation: str = ""                          # 工具返回的原始观测文本
    result_summary: str = ""                       # WorkMemorySummarizer 生成的压缩摘要
    error: str = ""                                # 错误信息（成功时为空字符串）


@dataclass
class WorkState:
    """
    单次用户请求（一个 run）的完整执行状态。

    生命周期：
    - start_run() 创建新状态 → status="running"
    - 执行过程中通过各 set_xxx() 方法更新字段
    - complete_run() 或 fail_run() 结束状态 → status="completed"/"failed"

    Conversation vs Run 的关系：
    - 同一个 conversation_id 可以跨多次请求（每次请求是一个新的 run）
    - ConversationMemory 横跨多个 run，保留对话历史
    - WorkState 只记录"这一次"的执行轨迹，不跨 run 累积

    字段截断策略（防止 prompt token 爆炸）：
    - original_query / rewritten_query：完整保留（通常不长）
    - schema_context_preview：截断至 1200 字符
    - latest_sql：截断至 4000 字符（一般足够）
    - latest_data_summary：截断至 2000 字符
    - findings：每条 400 字符，最多保留 10 条
    - steps：无限制，但 observation/summary/error 各有截断
    - final_answer：截断至 4000 字符
    """

    conversation_id: str                           # 归属的会话 ID
    run_id: str                                    # 本次运行的唯一 ID（uuid4 hex）
    status: str = "running"                        # running | completed | failed
    original_query: str = ""                       # 用户原始查询
    rewritten_query: str = ""                      # 改写后的查询（QueryRewriter 输出）
    schema_context_preview: str = ""               # schema 摘要（前 1200 字符）
    selected_tables: list[str] = field(default_factory=list)  # 当前任务涉及的表名
    findings: list[str] = field(default_factory=list)  # 关键发现列表（最多 10 条）
    latest_sql: str = ""                           # 最近执行的 SQL（前 4000 字符）
    latest_data_summary: str = ""                  # 最近数据摘要（前 2000 字符）
    latest_error: str = ""                         # 最近的错误信息
    iterations: int = 0                            # ReAct 循环迭代次数
    steps: list[WorkStep] = field(default_factory=list)         # 工具调用步骤记录
    artifacts: list[WorkArtifact] = field(default_factory=list) # 产物引用列表
    final_answer: str = ""                         # 最终答案文本
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)
    completed_at: datetime | None = None

    def touch(self) -> None:
        """
        刷新 updated_at 时间戳。

        每次状态更新后都应调用 touch()，
        方便监控工具了解最近一次状态变更发生在何时（追踪"卡住"的任务）。
        """
        self.updated_at = _utcnow()


class WorkMemory:
    """
    内存版工作记忆存储（WorkState 的管理器）。

    设计原则：
    - 接口尽量简洁，方便未来切换到数据库后端（如 Redis 版 RedisWorkMemory）
    - 当前只存每个 conversation 的"最近一次运行"（覆盖写入）
    - 需要归档历史 run 时，可把 _store 改为 dict[str, list[WorkState]]

    线程安全性：
    - 当前实现在 asyncio 单线程环境中是安全的
    - 如果迁移到多线程，需要添加 threading.Lock
    """

    def __init__(self) -> None:
        """初始化空的工作记忆存储。"""
        # P2-18：会话 ID 维度 LRU 封顶，防止按会话数线性膨胀导致 OOM。
        self._max_conversations = max(1, int(settings.memory_max_conversations))
        self._store: OrderedDict[str, WorkState] = OrderedDict()

    def _set_state(self, conversation_id: str, state: WorkState) -> None:
        """写入会话状态并维护 LRU 上限（P2-18）。Redis 变体本地缓存写入也走这里。"""
        self._store[conversation_id] = state
        self._store.move_to_end(conversation_id)
        while len(self._store) > self._max_conversations:
            _, _ = self._store.popitem(last=False)

    # ── 生命周期管理 ───────────────────────────────────────────────────────────

    def start_run(self, conversation_id: str, query: str) -> WorkState:
        """
        为新的用户请求创建并注册一个全新的工作状态。

        覆盖写入：如果同一 conversation_id 已有旧状态，会直接覆盖。
        原因：
        - 当前系统尚未提供 run 归档查询接口
        - 本轮目标是让 Agent 具备"知道自己正在做什么"的能力
        - 旧 run 的历史已通过 ConversationMemory 的桥接摘要保留

        run_id 使用 uuid4().hex（32 字符十六进制），保证全局唯一，
        方便日志追踪（可以在 Grafana 等工具中按 run_id 过滤）。

        Args:
            conversation_id: 归属的会话标识
            query: 用户的原始查询文本

        Returns:
            新建的 WorkState 对象（status="running"）
        """
        state = WorkState(
            conversation_id=conversation_id,
            run_id=uuid.uuid4().hex,
            original_query=query,
        )
        self._set_state(conversation_id, state)
        logger.debug("work_memory.run_started", conversation_id=conversation_id, run_id=state.run_id)
        return state

    def get_state(self, conversation_id: str) -> WorkState | None:
        """
        获取指定会话的当前工作状态（宽松读取）。

        "宽松"：不存在时返回 None，不抛出异常。
        适合只读场景（查询状态、构建 prompt 上下文）。
        写入场景请用 _require_state()（严格读取，不存在时抛异常）。

        Args:
            conversation_id: 会话标识

        Returns:
            WorkState 对象，如果会话不存在则返回 None
        """
        return self._store.get(conversation_id)

    def clear(self, conversation_id: str) -> None:
        """
        清除指定会话的工作状态。

        使用场景：
        - 用户开始新任务（显式重置）
        - 测试环境清理（确保每个测试用例有干净状态）

        注意：这不会影响 ConversationMemory 中的对话历史，
        两者通过 conversation_id 关联但独立管理生命周期。

        Args:
            conversation_id: 要清除的会话标识
        """
        self._store.pop(conversation_id, None)
        logger.info("work_memory.cleared", conversation_id=conversation_id)

    def complete_run(self, conversation_id: str, final_answer: str) -> None:
        """
        标记运行为成功完成，记录最终答案。

        截断策略：final_answer 截断至 4000 字符。
        原因：最终答案通常较长，但工作记忆不需要保存完整答案，
        桥接摘要和 ConversationMemory 会负责保留关键内容。

        Args:
            conversation_id: 会话标识
            final_answer: Agent 生成的最终回答文本

        Raises:
            RuntimeError: 会话状态不存在时（应先调用 start_run）
        """
        state = self._require_state(conversation_id)
        state.status = "completed"
        state.final_answer = final_answer[:4000]
        state.completed_at = _utcnow()
        state.touch()

    def fail_run(self, conversation_id: str, error: str) -> None:
        """
        标记运行为失败，记录错误信息。

        注意：fail_run 使用宽松读取（不存在时静默忽略），
        因为它通常在异常处理路径中被调用，此时状态可能已经被清除。

        Args:
            conversation_id: 会话标识
            error: 错误描述字符串
        """
        state = self._store.get(conversation_id)
        if state is None:
            return
        state.status = "failed"
        state.latest_error = error[:2000]
        state.completed_at = _utcnow()
        state.touch()

    # ── 基础字段写入 ───────────────────────────────────────────────────────────

    def set_rewritten_query(self, conversation_id: str, rewritten_query: str) -> None:
        """
        更新改写后的查询文本（由 QueryRewriter 写入）。

        改写查询用于 schema 检索和 RAG 检索，
        记录在工作记忆中方便后续步骤引用（如调试时了解 LLM 如何理解用户意图）。

        Args:
            conversation_id: 会话标识
            rewritten_query: QueryRewriter 输出的改写查询
        """
        state = self._require_state(conversation_id)
        state.rewritten_query = rewritten_query
        state.touch()

    def set_schema_context(
        self,
        conversation_id: str,
        schema_context: str,
        selected_tables: list[str] | None = None,
    ) -> None:
        """
        更新 schema 上下文信息（由 SchemaContextBuilder 写入）。

        schema_context 截断至 1200 字符（通常已包含足够的表结构信息）。
        selected_tables 独立保存，方便后续 SQL 安全校验（白名单）。

        Args:
            conversation_id: 会话标识
            schema_context: SchemaContextBuilder 生成的 schema 文本摘要
            selected_tables: 本次查询涉及的表名列表
        """
        state = self._require_state(conversation_id)
        state.schema_context_preview = schema_context[:1200]
        state.selected_tables = list(selected_tables or [])
        state.touch()

    def set_iterations(self, conversation_id: str, iterations: int) -> None:
        """
        更新 ReAct 循环迭代次数（每次循环开始时更新）。

        用于监控（Prometheus 指标）和 prompt 上下文构建（告诉模型已经循环了几次）。

        Args:
            conversation_id: 会话标识
            iterations: 当前迭代次数（1-based）
        """
        state = self._require_state(conversation_id)
        state.iterations = iterations
        state.touch()

    def set_latest_sql(self, conversation_id: str, sql: str) -> None:
        """
        更新最近执行的 SQL 语句。

        在工具执行前（解析 tool_call 参数后）即时更新，
        方便失败时在工作记忆快照中看到 SQL 内容。

        Args:
            conversation_id: 会话标识
            sql: SQL 语句文本（截断至 4000 字符）
        """
        state = self._require_state(conversation_id)
        state.latest_sql = sql[:4000]
        state.touch()

    def set_latest_data_summary(self, conversation_id: str, summary: str) -> None:
        """
        更新最近 SQL 查询结果的数据摘要。

        由 WorkMemorySummarizer.summarize_rows() 生成，
        包含行数、列名、首行预览。用于后续步骤的 prompt 注入。

        Args:
            conversation_id: 会话标识
            summary: WorkMemorySummarizer 生成的数据摘要文本
        """
        state = self._require_state(conversation_id)
        state.latest_data_summary = summary[:2000]
        state.touch()

    # ── 过程记录 ───────────────────────────────────────────────────────────

    def add_finding(self, conversation_id: str, finding: str) -> None:
        """
        追加一条关键发现（关键事实或中间结论）。

        findings 是 ReAct 循环中积累的"已知事实清单"，
        通过 build_prompt_context() 注入到下一次 LLM 调用的 system prompt，
        帮助模型理解"已经知道了什么"，避免重复查询。

        体积控制策略：
        - 单条发现截断至 400 字符（足够表达一个观察点）
        - 最多保留最近 10 条（移除最旧的）
        - 通过 build_prompt_context() 注入时只展示最近 5 条

        Args:
            conversation_id: 会话标识
            finding: 发现文本（会自动去空白和截断）
        """
        finding = finding.strip()
        if not finding:
            return
        state = self._require_state(conversation_id)
        state.findings.append(finding[:400])
        state.findings = state.findings[-10:]  # 保留最近 10 条
        state.touch()

    def start_tool_step(
        self,
        conversation_id: str,
        iteration: int,
        tool: str,
        args: dict[str, Any],
    ) -> WorkStep:
        """
        记录一次工具调用的开始（状态="running"）。

        在工具实际执行前调用，这样即使工具执行超时，
        快照中也能看到"有一个 running 状态的步骤"，而不是什么都没有。

        step_id 使用 uuid4().hex 保证跨步骤唯一性，
        后续调用 finish_tool_step() 时需要传入此 step_id。

        Args:
            conversation_id: 会话标识
            iteration: 当前 ReAct 循环迭代次数
            tool: 工具名称（如 "sql_query", "python_analysis"）
            args: 工具调用参数字典

        Returns:
            新建的 WorkStep 对象（status="running"）
        """
        state = self._require_state(conversation_id)
        step = WorkStep(
            step_id=uuid.uuid4().hex,
            iteration=iteration,
            tool=tool,
            args=args,
            status="running",
        )
        state.steps.append(step)
        state.touch()
        return step

    def finish_tool_step(
        self,
        conversation_id: str,
        step_id: str,
        *,
        success: bool,
        observation: str,
        result_summary: str = "",
        error: str = "",
    ) -> None:
        """
        记录工具调用完成（成功或失败）。

        工具执行结束后调用，更新状态并记录观测结果。
        如果工具失败，同时更新 state.latest_error，
        方便 prompt 上下文中展示最近的错误信息。

        字段说明：
        - observation：工具返回的原始观测文本（ToolResult.to_observation() 的输出）
        - result_summary：WorkMemorySummarizer 生成的压缩摘要（更适合 prompt 注入）
        - error：失败时的错误信息

        Args:
            conversation_id: 会话标识
            step_id: start_tool_step() 返回的步骤 ID
            success: 工具是否成功执行
            observation: 工具返回的原始观测文本
            result_summary: 压缩摘要（可选，默认空字符串）
            error: 错误信息（可选，默认空字符串）

        Raises:
            RuntimeError: 会话状态不存在
            KeyError: step_id 不存在
        """
        state = self._require_state(conversation_id)
        step = self._find_step(state, step_id)
        step.status = "done" if success else "failed"
        step.finished_at = _utcnow()
        step.observation = observation[:2000]
        step.result_summary = result_summary[:1000]
        step.error = error[:1000]
        if error:
            state.latest_error = step.error
        state.touch()

    def add_artifact(
        self,
        conversation_id: str,
        *,
        artifact_type: str,
        preview: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        注册一个执行产物（SQL 结果、图表等）的引用。

        每次工具成功执行后，AgentLoop 调用此方法记录产物的简要信息，
        不保存完整数据体（数据体由工具自己处理）。

        体积控制：只保留最近 10 个产物（超出则移除最旧的）。

        Args:
            conversation_id: 会话标识
            artifact_type: 产物类型（"sql_result", "chart", "python_output" 等）
            preview: 产物的短预览文本（最多 1000 字符）
            metadata: 产物的额外元数据（如 {"rows": 42, "tool": "sql_query"}）
        """
        state = self._require_state(conversation_id)
        state.artifacts.append(
            WorkArtifact(
                artifact_id=uuid.uuid4().hex,
                type=artifact_type,
                preview=preview[:1000],
                metadata=metadata or {},
            )
        )
        # 只保留最近 10 个产物，防止无限增长
        state.artifacts = state.artifacts[-10:]
        state.touch()

    # ── 视图/导出 ───────────────────────────────────────────────────────────

    def build_prompt_context(self, conversation_id: str) -> str:
        """
        将结构化工作状态压缩为适合注入 prompt 的短文本。

        这是 WorkMemory 对外提供"prompt 上下文"的核心接口。
        目标是帮助 LLM 在 ReAct 循环中了解：
        - 当前任务是什么（original_query + rewritten_query）
        - 已经查了哪些表（selected_tables）
        - 最近执行的 SQL 是什么（latest_sql 前 300 字符）
        - 最近数据摘要（latest_data_summary 前 400 字符）
        - 已经发现了什么（findings 最近 5 条）
        - 最近几步工具调用的结果（steps 最近 5 条）
        - 最近错误（latest_error 前 300 字符）

        截断原则：优先提供最近/最相关的信息，
        避免把完整历史倒给模型（token 有限，相关性更重要）。

        Args:
            conversation_id: 会话标识

        Returns:
            多行文本形式的工作状态摘要；会话不存在时返回空字符串
        """
        state = self._store.get(conversation_id)
        if state is None:
            return ""

        lines = [
            "Current task state:",
            f"- run_id: {state.run_id}",
            f"- status: {state.status}",
            f"- original_query: {state.original_query}",
        ]
        if state.rewritten_query:
            lines.append(f"- rewritten_query: {state.rewritten_query}")
        if state.selected_tables:
            # 最多展示 8 个表名，避免过长
            lines.append(f"- selected_tables: {', '.join(state.selected_tables[:8])}")
        if state.latest_sql:
            lines.append(f"- latest_sql: {state.latest_sql[:300]}")
        if state.latest_data_summary:
            lines.append(f"- latest_data_summary: {state.latest_data_summary[:400]}")
        if state.findings:
            lines.append("- findings:")
            for finding in state.findings[-5:]:
                lines.append(f"  * {finding}")
        if state.steps:
            lines.append("- recent_steps:")
            for step in state.steps[-5:]:
                lines.append(
                    f"  * iter={step.iteration} tool={step.tool} "
                    f"status={step.status} summary={step.result_summary[:120]}"
                )
        if state.latest_error:
            lines.append(f"- latest_error: {state.latest_error[:300]}")
        return "\n".join(lines)

    def build_conversation_bridge(self, conversation_id: str) -> dict[str, Any]:
        """
        生成写回 ConversationMemory 的轻量桥接摘要。

        这是 WorkMemory 和 ConversationMemory 之间唯一推荐的单向数据流接口：
        - WorkMemory 不直接操作 ConversationMemory
        - ConversationMemory 也不直接读取 WorkMemory
        - AgentLoop 在 _build_conversation_metadata() 中调用此方法，
          然后把返回值作为 metadata 写入 assistant 消息

        设计为 dict 而不是 prompt 文本的原因：
        - 返回值面向系统内部，不直接给模型消费
        - dict 格式方便后续序列化、日志记录和扩展
        - ConversationMemory 通过 pinned_facts 机制有选择地从中提取长期信息

        Args:
            conversation_id: 会话标识

        Returns:
            包含 run_id、status、iterations、selected_tables、
            latest_sql、latest_data_summary、findings 的字典；
            会话不存在时返回空字典
        """
        state = self._store.get(conversation_id)
        if state is None:
            return {}
        return {
            "run_id": state.run_id,
            "status": state.status,
            "iterations": state.iterations,
            "selected_tables": state.selected_tables[:8],
            "latest_sql": state.latest_sql[:300],
            "latest_data_summary": state.latest_data_summary[:300],
            "findings": state.findings[-3:],  # 只传最近 3 条给 ConversationMemory
        }

    def stats(self) -> dict[str, Any]:
        """
        返回工作记忆的全局统计信息（用于健康检查和监控）。

        Returns:
            包含 active_runs（总数）、running（进行中）、
            completed（已完成）、failed（失败）的计数字典
        """
        return {
            "active_runs": len(self._store),
            "running": sum(1 for s in self._store.values() if s.status == "running"),
            "completed": sum(1 for s in self._store.values() if s.status == "completed"),
            "failed": sum(1 for s in self._store.values() if s.status == "failed"),
        }

    def snapshot(self, conversation_id: str) -> dict[str, Any] | None:
        """
        返回当前工作状态的可序列化快照（用于调试 API）。

        使用 dataclasses.asdict() 递归序列化整个 WorkState 树，
        包括 steps 和 artifacts（datetime 会转换为 datetime 对象，
        序列化为 JSON 时需要自定义 encoder 处理）。

        Args:
            conversation_id: 会话标识

        Returns:
            WorkState 的字典快照；会话不存在时返回 None
        """
        state = self._store.get(conversation_id)
        if state is None:
            return None
        # dataclasses.asdict 递归地把 dataclass 树转成嵌套字典
        return asdict(state)

    # ── 内部 ───────────────────────────────────────────────────────────

    def _require_state(self, conversation_id: str) -> WorkState:
        """
        严格读取会话状态，不存在时抛出异常。

        用于所有写入操作（set_xxx, add_xxx, finish_xxx），
        保证写入操作只能在 start_run() 之后进行。
        这个约束防止"未初始化就写入"导致的数据一致性问题。

        Args:
            conversation_id: 会话标识

        Returns:
            当前 WorkState 对象

        Raises:
            RuntimeError: 会话状态不存在（未调用 start_run）
        """
        state = self._store.get(conversation_id)
        if state is None:
            raise RuntimeError(
                f"Work state not initialized for conversation_id={conversation_id!r}. "
                "Call start_run() first."
            )
        return state

    @staticmethod
    def _find_step(state: WorkState, step_id: str) -> WorkStep:
        """
        在 WorkState 中查找指定 step_id 的步骤。

        线性搜索（O(n)），但步骤数通常很少（< agent_max_iterations），性能可忽略。

        Args:
            state: 工作状态对象
            step_id: 要查找的步骤 ID

        Returns:
            找到的 WorkStep 对象

        Raises:
            KeyError: step_id 不存在（这是编程错误，start_tool_step 必须先调用）
        """
        for step in state.steps:
            if step.step_id == step_id:
                return step
        raise KeyError(f"Work step not found: {step_id}")


# ── 全局单例 ──────────────────────────────────────────────────────────────────

_work_memory: WorkMemory | None = None


def get_work_memory() -> WorkMemory:
    """
    获取全局工作记忆单例（懒加载）。

    与 ConversationMemory 的单例函数不同，WorkMemory 没有依赖注入需求
    （不需要 router/breaker），因此单例函数更简单。

    Returns:
        全局唯一的 WorkMemory 实例
    """
    global _work_memory
    if _work_memory is None:
        _work_memory = WorkMemory()
    return _work_memory
