"""
memory/redis_work_memory.py — Redis 工作记忆

目标：
- 将 Agent 运行态外置，支持多实例共享和进程重启后查看最近状态
- 保持与 WorkMemory 基本一致的调用方式
- Redis 不可用时 fail-open，避免拖垮主链路
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Any
import time

from ai_data_agent.config.config import settings
from ai_data_agent.memory.work_memory import (
    WorkArtifact,
    WorkMemory,
    WorkState,
    WorkStep,
    _utcnow,
)
from ai_data_agent.observability.logger import get_logger

logger = get_logger(__name__)

try:
    from redis import Redis
    from redis.exceptions import RedisError, WatchError
except Exception:  # pragma: no cover - optional dependency
    Redis = None  # type: ignore[assignment]
    RedisError = Exception  # type: ignore[assignment]
    WatchError = Exception  # type: ignore[assignment]


class RedisWorkMemory(WorkMemory):
    def __init__(
        self,
        redis_url: str,
        prefix: str = "ai_data_agent:work",
        *,
        ttl_seconds: int = 86400,
        socket_timeout: float = 2.0,
        socket_connect_timeout: float = 2.0,
        health_check_interval: int = 30,
        retry_on_timeout: bool = True,
        fail_open: bool = True,
        startup_check: bool = True,
    ) -> None:
        super().__init__()
        if Redis is None:
            raise RuntimeError("redis package is required for RedisWorkMemory.")

        self._prefix = prefix.rstrip(":")
        self._ttl_seconds = max(60, int(ttl_seconds))
        self._fail_open = fail_open
        self._health_check_interval = float(health_check_interval)
        self._client = Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_connect_timeout,
            health_check_interval=health_check_interval,
            retry_on_timeout=retry_on_timeout,
        )
        self._available = True
        self._last_ping = time.monotonic()
        self._versions: dict[str, int] = {}

        if startup_check:
            self._ping_on_startup()

    def close(self) -> None:
        """
        关闭 Redis 客户端连接池（P2-20）。

        应用优雅关闭时由 AppContainer.shutdown() 调用。
        幂等：客户端已关闭时直接返回，可重复调用。
        """
        client = getattr(self, "_client", None)
        if client is None:
            return
        try:
            client.close()
        except Exception as e:  # pragma: no cover - 防御性
            logger.warning("redis_work_memory.close_failed", error=str(e))
        self._client = None

    def start_run(self, conversation_id: str, query: str) -> WorkState:
        state = super().start_run(conversation_id, query)
        self._persist_state(conversation_id, state)
        return state

    def get_state(self, conversation_id: str) -> WorkState | None:
        state = self._store.get(conversation_id)
        if state is not None:
            return state
        loaded = self._load_state(conversation_id)
        if loaded is not None:
            self._set_state(conversation_id, loaded)
        return loaded

    def clear(self, conversation_id: str) -> None:
        super().clear(conversation_id)
        self._versions.pop(conversation_id, None)
        self._safe_call("delete", key=conversation_id, fn=lambda: self._client.delete(self._full_key(conversation_id)))

    def complete_run(self, conversation_id: str, final_answer: str) -> None:
        super().complete_run(conversation_id, final_answer)
        self._persist_if_present(conversation_id)

    def fail_run(self, conversation_id: str, error: str) -> None:
        super().fail_run(conversation_id, error)
        self._persist_if_present(conversation_id)

    def set_rewritten_query(self, conversation_id: str, rewritten_query: str) -> None:
        super().set_rewritten_query(conversation_id, rewritten_query)
        self._persist_if_present(conversation_id)

    def set_schema_context(
        self,
        conversation_id: str,
        schema_context: str,
        selected_tables: list[str] | None = None,
    ) -> None:
        super().set_schema_context(conversation_id, schema_context, selected_tables)
        self._persist_if_present(conversation_id)

    def set_iterations(self, conversation_id: str, iterations: int) -> None:
        super().set_iterations(conversation_id, iterations)
        self._persist_if_present(conversation_id)

    def set_latest_sql(self, conversation_id: str, sql: str) -> None:
        super().set_latest_sql(conversation_id, sql)
        self._persist_if_present(conversation_id)

    def set_latest_data_summary(self, conversation_id: str, summary: str) -> None:
        super().set_latest_data_summary(conversation_id, summary)
        self._persist_if_present(conversation_id)

    def add_finding(self, conversation_id: str, finding: str) -> None:
        super().add_finding(conversation_id, finding)
        self._persist_if_present(conversation_id)

    def start_tool_step(
        self,
        conversation_id: str,
        iteration: int,
        tool: str,
        args: dict[str, Any],
    ) -> WorkStep:
        step = super().start_tool_step(conversation_id, iteration, tool, args)
        self._persist_if_present(conversation_id)
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
        super().finish_tool_step(
            conversation_id,
            step_id,
            success=success,
            observation=observation,
            result_summary=result_summary,
            error=error,
        )
        self._persist_if_present(conversation_id)

    def add_artifact(
        self,
        conversation_id: str,
        *,
        artifact_type: str,
        preview: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().add_artifact(
            conversation_id,
            artifact_type=artifact_type,
            preview=preview,
            metadata=metadata,
        )
        self._persist_if_present(conversation_id)

    def stats(self) -> dict[str, Any]:
        base = super().stats()
        size = 0
        available = self._available
        try:
            for idx, _ in enumerate(
                self._client.scan_iter(match=f"{self._prefix}:*", count=200),
                start=1,
            ):
                size += 1
                if idx >= 1000:
                    break
        except RedisError as e:
            available = False
            self._handle_error("stats", e, key=None)

        return {
            **base,
            "backend": "redis",
            "available": available,
            "prefix": self._prefix,
            "ttl_seconds": self._ttl_seconds,
            "redis_keys_estimate": size,
            "fail_open": self._fail_open,
        }

    def snapshot(self, conversation_id: str) -> dict[str, Any] | None:
        state = self.get_state(conversation_id)
        if state is None:
            return None
        return asdict(state)

    def _persist_if_present(self, conversation_id: str) -> None:
        state = self._store.get(conversation_id)
        if state is not None:
            self._persist_state(conversation_id, state)

    def _persist_state(self, conversation_id: str, state: WorkState) -> None:
        self._safe_call(
            "persist",
            key=conversation_id,
            fn=lambda: self._persist_state_with_lock(conversation_id, state),
        )

    def _load_state(self, conversation_id: str) -> WorkState | None:
        raw = self._safe_call(
            "get",
            key=conversation_id,
            fn=lambda: self._client.get(self._full_key(conversation_id)),
        )
        if not raw:
            return None
        try:
            state, version = self._deserialize_state(raw)
            self._versions[conversation_id] = version
            return state
        except Exception as e:
            logger.warning(
                "redis_work_memory.deserialize_failed",
                conversation_id=conversation_id,
                error=str(e),
            )
            return None

    def _full_key(self, conversation_id: str) -> str:
        return f"{self._prefix}:{conversation_id}"

    def _ping_on_startup(self) -> None:
        try:
            self._client.ping()
            logger.info("redis_work_memory.ready", prefix=self._prefix, ttl=self._ttl_seconds)
        except RedisError as e:
            self._available = False
            self._handle_error("startup_ping", e, key=None)
            if not self._fail_open:
                raise RuntimeError(f"Redis work memory startup check failed: {e}") from e

    def _safe_call(self, op: str, *, key: str | None, fn) -> Any:
        if not self._available and self._fail_open:
            # P2-14：瞬时故障后的周期性恢复探测（fail-open 单向翻回问题）。
            if time.monotonic() - self._last_ping >= self._health_check_interval:
                try:
                    self._client.ping()
                    self._available = True
                    logger.info("redis_work_memory.recovered", prefix=self._prefix)
                except RedisError:
                    self._last_ping = time.monotonic()
            return None
        try:
            result = fn()
            self._available = True
            return result
        except RedisError as e:
            self._available = False
            self._last_ping = time.monotonic()
            self._handle_error(op, e, key=key)
            if self._fail_open:
                return None
            raise

    def _handle_error(self, op: str, error: Exception, *, key: str | None) -> None:
        logger.warning(
            "redis_work_memory.error",
            operation=op,
            key=key or "",
            error=str(error),
            fail_open=self._fail_open,
        )

    def _persist_state_with_lock(self, conversation_id: str, state: WorkState) -> bool:
        key = self._full_key(conversation_id)
        expected_version = self._versions.get(conversation_id, 0)
        merged_state = state

        for _ in range(max(1, settings.redis_optimistic_lock_retries)):
            with self._client.pipeline() as pipe:
                try:
                    pipe.watch(key)
                    raw = pipe.get(key)
                    current_state, current_version = self._deserialize_state(raw) if raw else (None, 0)

                    if current_version != expected_version:
                        if current_state is None:
                            expected_version = 0
                        else:
                            merged_state = self._merge_states(current_state, merged_state)
                            expected_version = current_version

                    payload = self._serialize_state(merged_state, expected_version + 1)
                    pipe.multi()
                    pipe.set(key, payload, ex=self._ttl_seconds)
                    pipe.execute()
                    self._versions[conversation_id] = expected_version + 1
                    self._set_state(conversation_id, merged_state)
                    return True
                except WatchError:
                    continue
        logger.warning("redis_work_memory.cas_exhausted", conversation_id=conversation_id)
        return False

    @staticmethod
    def _serialize_state(state: WorkState, version: int) -> str:
        return __import__("json").dumps(
            {"version": version, "state": asdict(state)},
            default=RedisWorkMemory._json_default,
            ensure_ascii=False,
        )

    @staticmethod
    def _deserialize_state(raw: str) -> tuple[WorkState, int]:
        import json

        payload = json.loads(raw)
        version = int(payload.get("version", 0))
        state_payload = payload.get("state", payload)
        steps = [RedisWorkMemory._deserialize_step(item) for item in state_payload.get("steps", [])]
        artifacts = [RedisWorkMemory._deserialize_artifact(item) for item in state_payload.get("artifacts", [])]
        return WorkState(
            conversation_id=state_payload["conversation_id"],
            run_id=state_payload["run_id"],
            status=state_payload.get("status", "running"),
            original_query=state_payload.get("original_query", ""),
            rewritten_query=state_payload.get("rewritten_query", ""),
            schema_context_preview=state_payload.get("schema_context_preview", ""),
            selected_tables=list(state_payload.get("selected_tables", [])),
            findings=list(state_payload.get("findings", [])),
            latest_sql=state_payload.get("latest_sql", ""),
            latest_data_summary=state_payload.get("latest_data_summary", ""),
            latest_error=state_payload.get("latest_error", ""),
            iterations=int(state_payload.get("iterations", 0)),
            steps=steps,
            artifacts=artifacts,
            final_answer=state_payload.get("final_answer", ""),
            created_at=RedisWorkMemory._parse_datetime(state_payload.get("created_at")),
            updated_at=RedisWorkMemory._parse_datetime(state_payload.get("updated_at")),
            completed_at=RedisWorkMemory._parse_datetime(state_payload.get("completed_at")),
        ), version

    @staticmethod
    def _deserialize_step(payload: dict[str, Any]) -> WorkStep:
        return WorkStep(
            step_id=payload["step_id"],
            iteration=int(payload.get("iteration", 0)),
            tool=payload.get("tool", ""),
            args=dict(payload.get("args", {})),
            status=payload.get("status", "pending"),
            started_at=RedisWorkMemory._parse_datetime(payload.get("started_at")),
            finished_at=RedisWorkMemory._parse_datetime(payload.get("finished_at")),
            observation=payload.get("observation", ""),
            result_summary=payload.get("result_summary", ""),
            error=payload.get("error", ""),
        )

    @staticmethod
    def _deserialize_artifact(payload: dict[str, Any]) -> WorkArtifact:
        return WorkArtifact(
            artifact_id=payload["artifact_id"],
            type=payload.get("type", ""),
            preview=payload.get("preview", ""),
            metadata=dict(payload.get("metadata", {})),
            created_at=RedisWorkMemory._parse_datetime(payload.get("created_at")),
        )

    @staticmethod
    def _parse_datetime(value: Any) -> datetime | None:
        if not value:
            return None
        if isinstance(value, datetime):
            return value
        return datetime.fromisoformat(value)

    @staticmethod
    def _json_default(value: Any) -> Any:
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value)

    @staticmethod
    def _merge_states(remote: WorkState, local: WorkState) -> WorkState:
        merged_steps: dict[str, WorkStep] = {step.step_id: step for step in remote.steps}
        for step in local.steps:
            existing = merged_steps.get(step.step_id)
            if existing is None or (existing.finished_at or datetime.min) <= (step.finished_at or datetime.min):
                merged_steps[step.step_id] = step

        merged_artifacts: dict[str, WorkArtifact] = {artifact.artifact_id: artifact for artifact in remote.artifacts}
        for artifact in local.artifacts:
            merged_artifacts[artifact.artifact_id] = artifact

        findings: list[str] = []
        for item in remote.findings + local.findings:
            if item and item not in findings:
                findings.append(item)

        selected_tables: list[str] = []
        for item in remote.selected_tables + local.selected_tables:
            if item and item not in selected_tables:
                selected_tables.append(item)

        merged = WorkState(
            conversation_id=local.conversation_id or remote.conversation_id,
            run_id=local.run_id or remote.run_id,
            status=local.status if local.updated_at >= remote.updated_at else remote.status,
            original_query=local.original_query or remote.original_query,
            rewritten_query=local.rewritten_query or remote.rewritten_query,
            schema_context_preview=local.schema_context_preview or remote.schema_context_preview,
            selected_tables=selected_tables,
            findings=findings[-10:],
            latest_sql=local.latest_sql or remote.latest_sql,
            latest_data_summary=local.latest_data_summary or remote.latest_data_summary,
            latest_error=local.latest_error or remote.latest_error,
            iterations=max(local.iterations, remote.iterations),
            steps=sorted(merged_steps.values(), key=lambda item: (item.iteration, item.started_at)),
            artifacts=sorted(merged_artifacts.values(), key=lambda item: item.created_at)[-10:],
            final_answer=local.final_answer or remote.final_answer,
            created_at=min(remote.created_at, local.created_at),
            updated_at=max(remote.updated_at, local.updated_at),
            completed_at=local.completed_at or remote.completed_at,
        )
        return merged
