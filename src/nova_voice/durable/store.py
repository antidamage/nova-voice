from __future__ import annotations

import asyncio
import shutil
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TypeVar, cast
from uuid import uuid4

from nova_voice.durable.models import (
    AuditRecord,
    AutomationRecord,
    CommitmentRecord,
    ConversationRecord,
    DelegationGrantRecord,
    DurableModel,
    EventRecord,
    ExecutionRecord,
    ExecutionState,
    GoalRecord,
    GoalState,
    IdentityPolicyRecord,
    MemoryReferenceRecord,
    PlanRecord,
    PlanState,
    PlanStepRecord,
    PlanStepState,
    ProactiveInterventionRecord,
    utc_now,
)

Record = (
    AuditRecord
    | AutomationRecord
    | ConversationRecord
    | CommitmentRecord
    | DelegationGrantRecord
    | EventRecord
    | ExecutionRecord
    | GoalRecord
    | IdentityPolicyRecord
    | MemoryReferenceRecord
    | PlanRecord
    | PlanStepRecord
    | ProactiveInterventionRecord
)
RecordT = TypeVar("RecordT", bound=DurableModel)

_RECORD_TYPES: dict[str, type[Record]] = {
    model.__name__: model
    for model in (
        AuditRecord,
        AutomationRecord,
        ConversationRecord,
        CommitmentRecord,
        DelegationGrantRecord,
        EventRecord,
        ExecutionRecord,
        GoalRecord,
        IdentityPolicyRecord,
        MemoryReferenceRecord,
        PlanRecord,
        PlanStepRecord,
        ProactiveInterventionRecord,
    )
}
_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class StoredRecord:
    record: Record
    revision: int


class ConcurrentRecordUpdate(RuntimeError):
    pass


class LeaseUnavailable(RuntimeError):
    pass


class DurableAgentStore:
    """Transactional SQLite store for durable agent lifecycle records."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @contextmanager
    def _connect(self, path: Path | None = None) -> Iterator[sqlite3.Connection]:
        selected = path or self.path
        selected.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(selected, timeout=30)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA synchronous=FULL")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize_sync(self) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                )
                """
            )
            applied = {
                int(row["version"])
                for row in connection.execute("SELECT version FROM schema_migrations")
            }
            if 1 not in applied:
                connection.executescript(
                    """
                    CREATE TABLE durable_records (
                        record_type TEXT NOT NULL,
                        record_id TEXT NOT NULL,
                        schema_version INTEGER NOT NULL,
                        status TEXT,
                        parent_id TEXT,
                        idempotency_key TEXT UNIQUE,
                        payload_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        expires_at TEXT,
                        revision INTEGER NOT NULL,
                        PRIMARY KEY (record_type, record_id)
                    );
                    CREATE INDEX idx_durable_records_status
                        ON durable_records(record_type, status);
                    CREATE INDEX idx_durable_records_parent
                        ON durable_records(record_type, parent_id);
                    CREATE INDEX idx_durable_records_expiry
                        ON durable_records(expires_at);
                    CREATE TABLE durable_audit (
                        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                        audit_id TEXT NOT NULL UNIQUE,
                        payload_json TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );
                    """
                )
                connection.execute(
                    "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                    (1, utc_now().isoformat()),
                )
            connection.commit()

    async def initialize(self) -> None:
        await asyncio.to_thread(self.initialize_sync)

    def migration_version_sync(self) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT MAX(version) AS version FROM schema_migrations"
            ).fetchone()
        return int(row["version"] or 0)

    async def migration_version(self) -> int:
        return await asyncio.to_thread(self.migration_version_sync)

    @staticmethod
    def _metadata(record: Record) -> tuple[str | None, str | None, str | None]:
        status = getattr(record, "status", None)
        status_value = status.value if status is not None else None
        parent = (
            getattr(record, "plan_id", None)
            or getattr(record, "goal_id", None)
            or getattr(record, "conversation_id", None)
        )
        key = record.idempotency_key if isinstance(record, ExecutionRecord) else None
        return status_value, parent, key

    @staticmethod
    def _decode(row: sqlite3.Row) -> StoredRecord:
        model = _RECORD_TYPES[row["record_type"]]
        return StoredRecord(
            record=model.model_validate_json(row["payload_json"]),
            revision=int(row["revision"]),
        )

    @staticmethod
    def _append_audit(
        connection: sqlite3.Connection,
        *,
        action: str,
        record: Record,
        prior_revision: int | None,
        resulting_revision: int,
        actor_id: str,
        detail: dict | None = None,
    ) -> None:
        now = utc_now()
        audit = AuditRecord(
            id=f"audit-{uuid4().hex}",
            created_at=now,
            updated_at=now,
            actor_id=actor_id,
            action=action,
            object_type=type(record).__name__,
            object_id=record.id,
            prior_revision=prior_revision,
            resulting_revision=resulting_revision,
            detail=detail or {},
        )
        connection.execute(
            "INSERT INTO durable_audit(audit_id, payload_json, created_at) VALUES (?, ?, ?)",
            (audit.id, audit.model_dump_json(), now.isoformat()),
        )

    def _insert(
        self,
        connection: sqlite3.Connection,
        record: Record,
        *,
        actor_id: str,
    ) -> StoredRecord:
        status, parent, key = self._metadata(record)
        connection.execute(
            """
            INSERT INTO durable_records(
                record_type, record_id, schema_version, status, parent_id,
                idempotency_key, payload_json, created_at, updated_at, expires_at, revision
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            """,
            (
                type(record).__name__,
                record.id,
                record.schema_version,
                status,
                parent,
                key,
                record.model_dump_json(),
                record.created_at.isoformat(),
                record.updated_at.isoformat(),
                record.expires_at.isoformat() if record.expires_at else None,
            ),
        )
        self._append_audit(
            connection,
            action="create",
            record=record,
            prior_revision=None,
            resulting_revision=1,
            actor_id=actor_id,
        )
        return StoredRecord(record, 1)

    def create_sync(self, record: RecordT, *, actor_id: str = "system") -> StoredRecord:
        with self._connect() as connection:
            stored = self._insert(connection, record, actor_id=actor_id)
        return stored

    async def create(self, record: RecordT, *, actor_id: str = "system") -> StoredRecord:
        return await asyncio.to_thread(self.create_sync, record, actor_id=actor_id)

    def _save(
        self,
        connection: sqlite3.Connection,
        record: Record,
        *,
        expected_revision: int,
        actor_id: str,
        action: str = "update",
    ) -> StoredRecord:
        status, parent, key = self._metadata(record)
        resulting_revision = expected_revision + 1
        cursor = connection.execute(
            """
            UPDATE durable_records
            SET schema_version = ?, status = ?, parent_id = ?, idempotency_key = ?,
                payload_json = ?, updated_at = ?, expires_at = ?, revision = ?
            WHERE record_type = ? AND record_id = ? AND revision = ?
            """,
            (
                record.schema_version,
                status,
                parent,
                key,
                record.model_dump_json(),
                record.updated_at.isoformat(),
                record.expires_at.isoformat() if record.expires_at else None,
                resulting_revision,
                type(record).__name__,
                record.id,
                expected_revision,
            ),
        )
        if cursor.rowcount != 1:
            raise ConcurrentRecordUpdate(
                f"{type(record).__name__}/{record.id} changed from revision {expected_revision}"
            )
        self._append_audit(
            connection,
            action=action,
            record=record,
            prior_revision=expected_revision,
            resulting_revision=resulting_revision,
            actor_id=actor_id,
        )
        return StoredRecord(record, resulting_revision)

    def save_sync(
        self,
        record: RecordT,
        *,
        expected_revision: int,
        actor_id: str = "system",
    ) -> StoredRecord:
        if isinstance(record, AuditRecord):
            raise ValueError("audit records are append-only")
        with self._connect() as connection:
            stored = self._save(
                connection,
                record,
                expected_revision=expected_revision,
                actor_id=actor_id,
            )
        return stored

    async def save(
        self,
        record: RecordT,
        *,
        expected_revision: int,
        actor_id: str = "system",
    ) -> StoredRecord:
        return await asyncio.to_thread(
            self.save_sync,
            record,
            expected_revision=expected_revision,
            actor_id=actor_id,
        )

    def get_sync(self, model: type[RecordT], record_id: str) -> StoredRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM durable_records WHERE record_type = ? AND record_id = ?",
                (model.__name__, record_id),
            ).fetchone()
        return self._decode(row) if row else None

    async def get(self, model: type[RecordT], record_id: str) -> StoredRecord | None:
        return await asyncio.to_thread(self.get_sync, model, record_id)

    def list_sync(
        self,
        model: type[RecordT],
        *,
        status: str | None = None,
        parent_id: str | None = None,
    ) -> tuple[StoredRecord, ...]:
        clauses = ["record_type = ?"]
        arguments: list[object] = [model.__name__]
        if status is not None:
            clauses.append("status = ?")
            arguments.append(status)
        if parent_id is not None:
            clauses.append("parent_id = ?")
            arguments.append(parent_id)
        query = "SELECT * FROM durable_records WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at, record_id"
        with self._connect() as connection:
            rows = connection.execute(query, arguments).fetchall()
        return tuple(self._decode(row) for row in rows)

    async def list(
        self,
        model: type[RecordT],
        *,
        status: str | None = None,
        parent_id: str | None = None,
    ) -> tuple[StoredRecord, ...]:
        return await asyncio.to_thread(
            self.list_sync,
            model,
            status=status,
            parent_id=parent_id,
        )

    def create_plan_bundle_sync(
        self,
        goal: GoalRecord,
        plan: PlanRecord,
        steps: Iterable[PlanStepRecord],
        *,
        actor_id: str = "system",
    ) -> None:
        selected_steps = tuple(steps)
        step_ids = tuple(step.id for step in selected_steps)
        if plan.goal_id != goal.id or plan.step_ids != step_ids:
            raise ValueError("goal, plan, and ordered step ids must agree")
        if any(step.plan_id != plan.id for step in selected_steps):
            raise ValueError("every step must belong to the plan")
        seen: set[str] = set()
        for step in sorted(selected_steps, key=lambda item: item.order):
            if unknown := set(step.depends_on) - seen:
                raise ValueError(
                    f"step dependencies must refer to earlier steps: {sorted(unknown)}"
                )
            seen.add(step.id)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._insert(connection, goal, actor_id=actor_id)
            self._insert(connection, plan, actor_id=actor_id)
            for step in selected_steps:
                self._insert(connection, step, actor_id=actor_id)
            connection.commit()

    async def create_plan_bundle(
        self,
        goal: GoalRecord,
        plan: PlanRecord,
        steps: Iterable[PlanStepRecord],
        *,
        actor_id: str = "system",
    ) -> None:
        await asyncio.to_thread(
            self.create_plan_bundle_sync,
            goal,
            plan,
            tuple(steps),
            actor_id=actor_id,
        )

    @staticmethod
    def _goal_state_for_plan(status: PlanState) -> GoalState:
        return GoalState(status.value)

    def set_plan_state_sync(
        self,
        plan_id: str,
        *,
        status: PlanState,
        reason: str | None,
        actor_id: str,
        now: datetime | None = None,
    ) -> PlanRecord:
        current = (now or utc_now()).astimezone(UTC)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            plan_row = connection.execute(
                "SELECT * FROM durable_records WHERE record_type = ? AND record_id = ?",
                (PlanRecord.__name__, plan_id),
            ).fetchone()
            if plan_row is None:
                raise KeyError(f"unknown plan: {plan_id}")
            stored_plan = self._decode(plan_row)
            plan = cast(PlanRecord, stored_plan.record)
            goal_row = connection.execute(
                "SELECT * FROM durable_records WHERE record_type = ? AND record_id = ?",
                (GoalRecord.__name__, plan.goal_id),
            ).fetchone()
            if goal_row is None:
                raise KeyError(f"unknown goal: {plan.goal_id}")
            stored_goal = self._decode(goal_row)
            goal = cast(GoalRecord, stored_goal.record)
            updated_plan = plan.model_copy(
                update={"status": status, "terminal_reason": reason, "updated_at": current}
            )
            updated_goal = goal.model_copy(
                update={
                    "status": self._goal_state_for_plan(status),
                    "terminal_reason": reason,
                    "updated_at": current,
                }
            )
            self._save(
                connection,
                updated_plan,
                expected_revision=stored_plan.revision,
                actor_id=actor_id,
                action="transition_plan",
            )
            self._save(
                connection,
                updated_goal,
                expected_revision=stored_goal.revision,
                actor_id=actor_id,
                action="transition_goal",
            )
            connection.commit()
            return updated_plan

    async def set_plan_state(self, plan_id: str, **kwargs) -> PlanRecord:
        return await asyncio.to_thread(self.set_plan_state_sync, plan_id, **kwargs)

    def terminate_plan_sync(
        self,
        plan_id: str,
        *,
        status: PlanState,
        reason: str,
        actor_id: str,
        now: datetime | None = None,
    ) -> PlanRecord:
        if status not in {PlanState.CANCELLED, PlanState.EXPIRED}:
            raise ValueError("plan termination status must be cancelled or expired")
        current = (now or utc_now()).astimezone(UTC)
        step_status = (
            PlanStepState.CANCELLED if status == PlanState.CANCELLED else PlanStepState.EXPIRED
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            plan_row = connection.execute(
                "SELECT * FROM durable_records WHERE record_type = ? AND record_id = ?",
                (PlanRecord.__name__, plan_id),
            ).fetchone()
            if plan_row is None:
                raise KeyError(f"unknown plan: {plan_id}")
            stored_plan = self._decode(plan_row)
            plan = cast(PlanRecord, stored_plan.record)
            step_rows = connection.execute(
                "SELECT * FROM durable_records WHERE record_type = ? AND parent_id = ?",
                (PlanStepRecord.__name__, plan_id),
            ).fetchall()
            for row in step_rows:
                stored_step = self._decode(row)
                step = cast(PlanStepRecord, stored_step.record)
                if step.status in {
                    PlanStepState.SATISFIED,
                    PlanStepState.COMPENSATED,
                    PlanStepState.CANCELLED,
                    PlanStepState.EXPIRED,
                    PlanStepState.FAILED,
                }:
                    continue
                updated_step = step.model_copy(
                    update={
                        "status": step_status,
                        "terminal_reason": reason,
                        "updated_at": current,
                    }
                )
                self._save(
                    connection,
                    updated_step,
                    expected_revision=stored_step.revision,
                    actor_id=actor_id,
                    action="terminate_step",
                )
            goal_row = connection.execute(
                "SELECT * FROM durable_records WHERE record_type = ? AND record_id = ?",
                (GoalRecord.__name__, plan.goal_id),
            ).fetchone()
            if goal_row is None:
                raise KeyError(f"unknown goal: {plan.goal_id}")
            stored_goal = self._decode(goal_row)
            goal = cast(GoalRecord, stored_goal.record)
            updated_plan = plan.model_copy(
                update={"status": status, "terminal_reason": reason, "updated_at": current}
            )
            updated_goal = goal.model_copy(
                update={
                    "status": self._goal_state_for_plan(status),
                    "terminal_reason": reason,
                    "updated_at": current,
                }
            )
            self._save(
                connection,
                updated_plan,
                expected_revision=stored_plan.revision,
                actor_id=actor_id,
                action="terminate_plan",
            )
            self._save(
                connection,
                updated_goal,
                expected_revision=stored_goal.revision,
                actor_id=actor_id,
                action="terminate_goal",
            )
            connection.commit()
            return updated_plan

    async def terminate_plan(self, plan_id: str, **kwargs) -> PlanRecord:
        return await asyncio.to_thread(self.terminate_plan_sync, plan_id, **kwargs)

    def acquire_execution_sync(
        self,
        step_id: str,
        *,
        worker_id: str,
        now: datetime | None = None,
        lease_seconds: float = 30,
    ) -> ExecutionRecord | None:
        current = (now or utc_now()).astimezone(UTC)
        lease_expires = current + timedelta(seconds=lease_seconds)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            step_row = connection.execute(
                "SELECT * FROM durable_records WHERE record_type = ? AND record_id = ?",
                (PlanStepRecord.__name__, step_id),
            ).fetchone()
            if step_row is None:
                raise KeyError(f"unknown plan step: {step_id}")
            stored_step = self._decode(step_row)
            step = cast(PlanStepRecord, stored_step.record)
            key = f"{step.plan_id}:{step.id}"
            execution_row = connection.execute(
                "SELECT * FROM durable_records WHERE idempotency_key = ?",
                (key,),
            ).fetchone()
            if execution_row is not None:
                stored_execution = self._decode(execution_row)
                execution = cast(ExecutionRecord, stored_execution.record)
                if execution.status == ExecutionState.SUCCEEDED:
                    connection.commit()
                    return execution
                if (
                    execution.status in {ExecutionState.LEASED, ExecutionState.RUNNING}
                    and execution.lease_expires_at > current
                ):
                    connection.commit()
                    return None
                attempt = execution.attempt + 1
                if attempt > step.max_attempts:
                    connection.commit()
                    return None
                execution = execution.model_copy(
                    update={
                        "status": ExecutionState.LEASED,
                        "attempt": attempt,
                        "lease_owner": worker_id,
                        "lease_token": uuid4().hex,
                        "lease_expires_at": lease_expires,
                        "started_at": None,
                        "completed_at": None,
                        "error_code": None,
                        "updated_at": current,
                    }
                )
                self._save(
                    connection,
                    execution,
                    expected_revision=stored_execution.revision,
                    actor_id=worker_id,
                    action="reclaim_execution",
                )
            else:
                attempt = step.attempt + 1
                if attempt > step.max_attempts:
                    connection.commit()
                    return None
                execution = ExecutionRecord(
                    id=f"execution-{uuid4().hex}",
                    created_at=current,
                    updated_at=current,
                    plan_id=step.plan_id,
                    step_id=step.id,
                    idempotency_key=key,
                    attempt=attempt,
                    lease_owner=worker_id,
                    lease_token=uuid4().hex,
                    lease_expires_at=lease_expires,
                )
                self._insert(connection, execution, actor_id=worker_id)
            leased_step = step.model_copy(
                update={
                    "status": PlanStepState.LEASED,
                    "attempt": execution.attempt,
                    "updated_at": current,
                }
            )
            self._save(
                connection,
                leased_step,
                expected_revision=stored_step.revision,
                actor_id=worker_id,
                action="lease_step",
            )
            connection.commit()
            return execution

    async def acquire_execution(
        self,
        step_id: str,
        *,
        worker_id: str,
        now: datetime | None = None,
        lease_seconds: float = 30,
    ) -> ExecutionRecord | None:
        return await asyncio.to_thread(
            self.acquire_execution_sync,
            step_id,
            worker_id=worker_id,
            now=now,
            lease_seconds=lease_seconds,
        )

    def mark_execution_running_sync(
        self,
        execution_id: str,
        *,
        lease_token: str,
        now: datetime | None = None,
    ) -> ExecutionRecord:
        current = (now or utc_now()).astimezone(UTC)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM durable_records WHERE record_type = ? AND record_id = ?",
                (ExecutionRecord.__name__, execution_id),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown execution: {execution_id}")
            stored = self._decode(row)
            execution = cast(ExecutionRecord, stored.record)
            if execution.lease_token != lease_token or execution.lease_expires_at <= current:
                raise LeaseUnavailable("execution lease is missing, stale, or owned elsewhere")
            running = execution.model_copy(
                update={
                    "status": ExecutionState.RUNNING,
                    "started_at": execution.started_at or current,
                    "updated_at": current,
                }
            )
            self._save(
                connection,
                running,
                expected_revision=stored.revision,
                actor_id=execution.lease_owner,
                action="start_execution",
            )
            step_row = connection.execute(
                "SELECT * FROM durable_records WHERE record_type = ? AND record_id = ?",
                (PlanStepRecord.__name__, execution.step_id),
            ).fetchone()
            if step_row is None:
                raise KeyError(f"unknown plan step: {execution.step_id}")
            stored_step = self._decode(step_row)
            step = cast(PlanStepRecord, stored_step.record)
            running_step = step.model_copy(
                update={"status": PlanStepState.RUNNING, "updated_at": current}
            )
            self._save(
                connection,
                running_step,
                expected_revision=stored_step.revision,
                actor_id=execution.lease_owner,
                action="start_step",
            )
            connection.commit()
            return running

    async def mark_execution_running(
        self,
        execution_id: str,
        *,
        lease_token: str,
        now: datetime | None = None,
    ) -> ExecutionRecord:
        return await asyncio.to_thread(
            self.mark_execution_running_sync,
            execution_id,
            lease_token=lease_token,
            now=now,
        )

    def complete_execution_sync(
        self,
        execution_id: str,
        *,
        lease_token: str,
        succeeded: bool,
        result_revision: str | None,
        error_code: str | None = None,
        retryable: bool = False,
        now: datetime | None = None,
    ) -> ExecutionRecord:
        current = (now or utc_now()).astimezone(UTC)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            execution_row = connection.execute(
                "SELECT * FROM durable_records WHERE record_type = ? AND record_id = ?",
                (ExecutionRecord.__name__, execution_id),
            ).fetchone()
            if execution_row is None:
                raise KeyError(f"unknown execution: {execution_id}")
            stored_execution = self._decode(execution_row)
            execution = cast(ExecutionRecord, stored_execution.record)
            if execution.lease_token != lease_token:
                raise LeaseUnavailable("execution completion does not own the current lease")
            if execution.status == ExecutionState.SUCCEEDED:
                connection.commit()
                return execution
            completed = execution.model_copy(
                update={
                    "status": ExecutionState.SUCCEEDED if succeeded else ExecutionState.FAILED,
                    "completed_at": current,
                    "result_revision": result_revision,
                    "error_code": error_code,
                    "updated_at": current,
                }
            )
            self._save(
                connection,
                completed,
                expected_revision=stored_execution.revision,
                actor_id=execution.lease_owner,
                action="complete_execution",
            )
            step_row = connection.execute(
                "SELECT * FROM durable_records WHERE record_type = ? AND record_id = ?",
                (PlanStepRecord.__name__, execution.step_id),
            ).fetchone()
            if step_row is None:
                raise KeyError(f"unknown plan step: {execution.step_id}")
            stored_step = self._decode(step_row)
            step = cast(PlanStepRecord, stored_step.record)
            if succeeded:
                step_status = (
                    PlanStepState.COMPENSATED
                    if step.kind.value == "compensation"
                    else PlanStepState.SATISFIED
                )
            elif retryable and execution.attempt < step.max_attempts:
                step_status = PlanStepState.PENDING
            else:
                step_status = PlanStepState.FAILED
            updated_step = step.model_copy(
                update={
                    "status": step_status,
                    "result_revision": result_revision,
                    "terminal_reason": error_code,
                    "updated_at": current,
                }
            )
            self._save(
                connection,
                updated_step,
                expected_revision=stored_step.revision,
                actor_id=execution.lease_owner,
                action="complete_step",
            )
            connection.commit()
            return completed

    async def complete_execution(self, execution_id: str, **kwargs) -> ExecutionRecord:
        return await asyncio.to_thread(self.complete_execution_sync, execution_id, **kwargs)

    def list_audit_sync(self) -> tuple[AuditRecord, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM durable_audit ORDER BY sequence"
            ).fetchall()
        return tuple(AuditRecord.model_validate_json(row["payload_json"]) for row in rows)

    async def list_audit(self) -> tuple[AuditRecord, ...]:
        return await asyncio.to_thread(self.list_audit_sync)

    def health_sync(self) -> dict[str, int | bool]:
        with self._connect() as connection:
            version = connection.execute(
                "SELECT MAX(version) AS version FROM schema_migrations"
            ).fetchone()["version"]
            records = connection.execute(
                "SELECT COUNT(*) AS count FROM durable_records"
            ).fetchone()["count"]
        return {
            "ok": int(version or 0) == _SCHEMA_VERSION,
            "enabled": True,
            "schemaVersion": int(version or 0),
            "records": int(records),
        }

    async def health(self) -> dict[str, int | bool]:
        return await asyncio.to_thread(self.health_sync)

    def prune_expired_sync(self, now: datetime | None = None) -> int:
        boundary = (now or utc_now()).astimezone(UTC).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM durable_records WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (boundary,),
            )
            deleted = cursor.rowcount
            connection.commit()
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        return deleted

    async def prune_expired(self, now: datetime | None = None) -> int:
        return await asyncio.to_thread(self.prune_expired_sync, now)

    def backup_to_sync(self, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.resolve() == self.path.resolve():
            raise ValueError("backup destination must differ from the live store")
        with self._connect() as source, closing(sqlite3.connect(destination)) as target:
            source.execute("PRAGMA wal_checkpoint(FULL)").fetchone()
            source.backup(target)
            target.commit()
        self.verify_database_sync(destination)
        return destination

    async def backup_to(self, destination: Path) -> Path:
        return await asyncio.to_thread(self.backup_to_sync, destination)

    @staticmethod
    def verify_database_sync(path: Path) -> None:
        if not path.is_file():
            raise ValueError("database backup does not exist")
        with closing(sqlite3.connect(path)) as connection:
            integrity = connection.execute("PRAGMA quick_check").fetchone()[0]
            if integrity != "ok":
                raise ValueError(f"database integrity check failed: {integrity}")
            row = connection.execute("SELECT MAX(version) FROM schema_migrations").fetchone()
            if row is None or int(row[0] or 0) != _SCHEMA_VERSION:
                raise ValueError("database schema version is not supported")

    def restore_from_sync(self, backup: Path) -> None:
        self.verify_database_sync(backup)
        temporary = self.path.with_suffix(self.path.suffix + ".restore")
        shutil.copy2(backup, temporary)
        self.verify_database_sync(temporary)
        try:
            with (
                closing(sqlite3.connect(temporary)) as source,
                closing(sqlite3.connect(self.path)) as target,
            ):
                source.backup(target)
                target.commit()
        finally:
            temporary.unlink(missing_ok=True)
        self.verify_database_sync(self.path)

    async def restore_from(self, backup: Path) -> None:
        await asyncio.to_thread(self.restore_from_sync, backup)
