"""
memory/work_memory.py — 工作记忆（Work Memory）

设计目标：
1. 让 Agent 在一次分析任务执行期间，能够显式记录“当前做到哪一步”
2. 让中间状态结构化，而不是只散落在 prompt 文本里
3. 先提供内存版实现，后续可无缝迁移到 database.py 做持久化

这层记忆和 conversation_memory 的边界要明确：
- conversation_memory 负责“聊过什么”
- work_memory 负责“任务执行到了什么状态”

允许进入 work_memory 的内容：
- query rewrite 结果
- schema 摘要、涉及表
- 工具调用参数和执行摘要
- 当前假设、关键发现、失败原因
- 最近 SQL、最近数据摘要、图表等产物引用

不应该进入 work_memory 的内容：
- 长期知识库文档全文
- 跨会话共享偏好
- 作为“对话原文”回放给模型的聊天记录

当前项目暂未接入 Planner / Executor，因此这里先围绕 ReAct 运行路径建模：
- 记录 query rewrite 结果
- 记录 schema 摘要
- 记录每次工具调用及其结果
- 记录最近 SQL、最近数据摘要、最终答案
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any
# 通用唯一标识符
import uuid

from ai_data_agent.observability.logger import get_logger

logger = get_logger(__name__)


def _utcnow() -> datetime:
    """统一时间源，便于后续替换或测试。"""
    return datetime.utcnow()


@dataclass
class WorkArtifact:
    """
    运行过程中产生的“产物引用”。

    注意这里不直接保存大块原始数据，只保留：
    - 类型
    - 预览
    - 元数据
    这样可以让工作记忆保持轻量，避免越跑越大。
    """

    artifact_id: str
    type: str
    preview: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utcnow)


@dataclass
class WorkStep:
    """
    ReAct 过程中的一个执行步骤。

    这里的 step 不等同于 Planner 的 PlanStep。
    当前版本没有接入 Planner，因此 step 表示“第 N 次工具动作”。
    """

    step_id: str
    iteration: int
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"   # pending | running | done | failed
    started_at: datetime = field(default_factory=_utcnow)
    finished_at: datetime | None = None
    observation: str = ""
    result_summary: str = ""
    error: str = ""


@dataclass
class WorkState:
    """
    单次运行的工作状态。

    一次用户请求对应一个 run_id。
    同一个 conversation_id 可以跨多次请求复用 conversation_memory。
    但 WorkState 关注的是“当前这一次分析任务”的内部执行轨迹。

    这意味着：
    - Conversation 是线程
    - WorkState 是线程中的一次具体运行
    """

    conversation_id: str
    run_id: str
    status: str = "running"   # running | completed | failed
    original_query: str = ""
    rewritten_query: str = ""
    schema_context_preview: str = ""
    selected_tables: list[str] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    latest_sql: str = ""
    latest_data_summary: str = ""
    latest_error: str = ""
    iterations: int = 0
    steps: list[WorkStep] = field(default_factory=list)
    artifacts: list[WorkArtifact] = field(default_factory=list)
    final_answer: str = ""
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)
    completed_at: datetime | None = None

    def touch(self) -> None:
        """每次状态更新时刷新 updated_at。"""
        self.updated_at = _utcnow()


class WorkMemory:
    """
    内存版工作记忆存储。

    设计上刻意保留简洁接口，方便未来切到数据库：
    - 当前只存“每个 conversation 的最近一次运行状态”
    - 后续若要归档历史 run，可把 _store 改成 conversation_id -> [WorkState]
    """

    def __init__(self) -> None:
        self._store: dict[str, WorkState] = {}

# ── 生命周期管理 ───────────────────────────────────────────────────────────

    def start_run(self, conversation_id: str, query: str) -> WorkState:
        """
        为一次新的用户请求创建新的运行状态。

        这里直接覆盖同 conversation_id 的旧工作状态，原因是：
        - 当前系统尚未提供 run 归档查询接口
        - 本轮目标是先让 agent 具备“知道自己正在做什么”的能力
        """
        state = WorkState(
            conversation_id=conversation_id,
            run_id=uuid.uuid4().hex,
            original_query=query,
        )
        self._store[conversation_id] = state
        logger.debug("work_memory.run_started", conversation_id=conversation_id, run_id=state.run_id)
        return state

    def get_state(self, conversation_id: str) -> WorkState | None:
        # 宽松读取
        return self._store.get(conversation_id)

    def clear(self, conversation_id: str) -> None:
        self._store.pop(conversation_id, None)
        logger.info("work_memory.cleared", conversation_id=conversation_id)

    def complete_run(self, conversation_id: str, final_answer: str) -> None:
        state = self._require_state(conversation_id)
        state.status = "completed"
        state.final_answer = final_answer[:4000]
        state.completed_at = _utcnow()
        state.touch()

    def fail_run(self, conversation_id: str, error: str) -> None:
        state = self._store.get(conversation_id)
        if state is None:
            return
        state.status = "failed"
        state.latest_error = error[:2000]
        state.completed_at = _utcnow()
        state.touch()

# ── 基础字段写入 ───────────────────────────────────────────────────────────

    def set_rewritten_query(self, conversation_id: str, rewritten_query: str) -> None:
        state = self._require_state(conversation_id)
        state.rewritten_query = rewritten_query
        state.touch()

    def set_schema_context(
        self,
        conversation_id: str,
        schema_context: str,
        selected_tables: list[str] | None = None,
    ) -> None:
        state = self._require_state(conversation_id)
        state.schema_context_preview = schema_context[:1200]
        state.selected_tables = list(selected_tables or [])
        state.touch()

    def set_iterations(self, conversation_id: str, iterations: int) -> None:
        state = self._require_state(conversation_id)
        state.iterations = iterations
        state.touch()

    def set_latest_sql(self, conversation_id: str, sql: str) -> None:
        state = self._require_state(conversation_id)
        state.latest_sql = sql[:4000]
        state.touch()

    def set_latest_data_summary(self, conversation_id: str, summary: str) -> None:
        state = self._require_state(conversation_id)
        state.latest_data_summary = summary[:2000]
        state.touch()

# ── 过程记录 ───────────────────────────────────────────────────────────

    def add_finding(self, conversation_id: str, finding: str) -> None:
        """
        追加关键发现。

        为避免无限增长：
        - 去空
        - 保留最近 10 条
        - 单条裁剪到适合 prompt 注入的长度
        """
        finding = finding.strip()
        if not finding:
            return
        state = self._require_state(conversation_id)
        state.findings.append(finding[:400])
        state.findings = state.findings[-10:]
        state.touch()

    def start_tool_step(
        self,
        conversation_id: str,
        iteration: int,
        tool: str,
        args: dict[str, Any],
    ) -> WorkStep:
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
        # 添加过程中的人工产物
        state = self._require_state(conversation_id)
        state.artifacts.append(
            WorkArtifact(
                artifact_id=uuid.uuid4().hex,
                type=artifact_type,
                preview=preview[:1000],
                metadata=metadata or {},
            )
        )
        # 控制工作记忆大小，只保留最近 10 个产物
        state.artifacts = state.artifacts[-10:]
        state.touch()

# ── 视图/导出 ───────────────────────────────────────────────────────────

    def build_prompt_context(self, conversation_id: str) -> str:
        """
        将结构化工作状态压缩成适合注入 prompt 的短文本。

        这里的目标不是把整个 WorkState 倒给模型，
        而是提供一份“当前任务状态摘要”，帮助模型知道：
        - 这次任务是什么
        - 已经做过哪些动作
        - 最近一次 SQL / 数据状态是什么
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
                    f"  * iter={step.iteration} tool={step.tool} status={step.status} summary={step.result_summary[:120]}"
                )
        if state.latest_error:
            lines.append(f"- latest_error: {state.latest_error[:300]}")
        return "\n".join(lines)

    def build_conversation_bridge(self, conversation_id: str) -> dict[str, Any]:
        """
        生成一个“写回会话层”的轻量摘要。

        这是 conversation_memory 和 work_memory 之间唯一推荐的桥接方式：
        - conversation_memory 不保存完整运行轨迹
        - 只保存一个足够小的摘要，便于后续排查或展示

        返回值设计为 metadata，而不是 prompt 文本，原因是：
        - 它面向系统内部，不面向模型直接消费
        - 可以避免执行细节污染会话原文
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
            "findings": state.findings[-3:],
        }

    def stats(self) -> dict[str, Any]:
        return {
            "active_runs": len(self._store),
            "running": sum(1 for s in self._store.values() if s.status == "running"),
            "completed": sum(1 for s in self._store.values() if s.status == "completed"),
            "failed": sum(1 for s in self._store.values() if s.status == "failed"),
        }

    def snapshot(self, conversation_id: str) -> dict[str, Any] | None:
        """
        返回当前状态的可序列化快照。

        当前代码暂未暴露 HTTP 调试接口，但保留该能力后续会很有用。
        """
        state = self._store.get(conversation_id)
        if state is None:
            return None
        # state是dataclass，使用 asdict 把 dataclass 转成字典
        return asdict(state)

# ── 内部 ───────────────────────────────────────────────────────────

    def _require_state(self, conversation_id: str) -> WorkState:
        # 强约束读取
        state = self._store.get(conversation_id)
        if state is None:
            raise RuntimeError(
                f"Work state not initialized for conversation_id={conversation_id!r}."
            )
        return state

    @staticmethod
    def _find_step(state: WorkState, step_id: str) -> WorkStep:
        for step in state.steps:
            if step.step_id == step_id:
                return step
        raise KeyError(f"Work step not found: {step_id}")


_work_memory: WorkMemory | None = None


def get_work_memory() -> WorkMemory:
    global _work_memory
    if _work_memory is None:
        _work_memory = WorkMemory()
    return _work_memory
