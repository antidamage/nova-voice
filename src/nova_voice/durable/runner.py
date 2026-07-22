from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol, cast

from nova_voice.durable.models import (
    ExecutionState,
    PlanRecord,
    PlanState,
    PlanStepKind,
    PlanStepRecord,
    PlanStepState,
    utc_now,
)
from nova_voice.durable.store import DurableAgentStore, StoredRecord


@dataclass(frozen=True)
class StepExecutionResult:
    succeeded: bool
    result_revision: str | None = None
    error_code: str | None = None
    retryable: bool = False


class StepExecutor(Protocol):
    """Provider bridge that must deduplicate the supplied idempotency key."""

    async def execute(
        self,
        step: PlanStepRecord,
        *,
        idempotency_key: str,
    ) -> StepExecutionResult: ...


_EXECUTABLE_KINDS = {
    PlanStepKind.TOOL,
    PlanStepKind.VERIFICATION,
    PlanStepKind.RETRY,
    PlanStepKind.COMPENSATION,
}
_SUCCESS_STATES = {PlanStepState.SATISFIED, PlanStepState.COMPENSATED}
_TERMINAL_STATES = _SUCCESS_STATES | {
    PlanStepState.BLOCKED,
    PlanStepState.CANCELLED,
    PlanStepState.EXPIRED,
    PlanStepState.FAILED,
}


class DurablePlanRunner:
    """Advance persisted plans without repeating a completed side effect."""

    def __init__(
        self,
        store: DurableAgentStore,
        executor: StepExecutor,
        *,
        worker_id: str,
        clock: Callable[[], datetime] = utc_now,
        lease_seconds: float = 30,
    ) -> None:
        self.store = store
        self.executor = executor
        self.worker_id = worker_id
        self.clock = clock
        self.lease_seconds = lease_seconds

    def _now(self) -> datetime:
        value = self.clock()
        if value.utcoffset() is None:
            raise ValueError("durable runner clock must return a timezone-aware datetime")
        return value.astimezone(UTC)

    async def _save_step(
        self,
        stored: StoredRecord,
        *,
        status: PlanStepState,
        reason: str | None = None,
        result_revision: str | None = None,
    ) -> PlanStepRecord:
        step = cast(PlanStepRecord, stored.record)
        updated = step.model_copy(
            update={
                "status": status,
                "terminal_reason": reason,
                "result_revision": result_revision or step.result_revision,
                "updated_at": self._now(),
            }
        )
        result = await self.store.save(
            updated,
            expected_revision=stored.revision,
            actor_id=self.worker_id,
        )
        return cast(PlanStepRecord, result.record)

    async def _prepare_non_tool_steps(
        self,
        steps: tuple[StoredRecord, ...],
    ) -> bool:
        changed = False
        now = self._now()
        by_id = {cast(PlanStepRecord, item.record).id: item for item in steps}
        for stored in steps:
            step = cast(PlanStepRecord, stored.record)
            if step.status not in {
                PlanStepState.PENDING,
                PlanStepState.WAITING,
                PlanStepState.PAUSED,
            }:
                continue
            dependencies = [cast(PlanStepRecord, by_id[item].record) for item in step.depends_on]
            failed_dependencies = [
                dependency
                for dependency in dependencies
                if dependency.status in _TERMINAL_STATES - _SUCCESS_STATES
            ]
            if failed_dependencies and step.kind != PlanStepKind.COMPENSATION:
                await self._save_step(
                    stored,
                    status=PlanStepState.BLOCKED,
                    reason="dependency did not complete successfully",
                )
                changed = True
                continue
            if not all(dependency.status in _SUCCESS_STATES for dependency in dependencies):
                continue
            if step.kind in {PlanStepKind.WAIT, PlanStepKind.TIMER}:
                if step.not_before is not None and now >= step.not_before.astimezone(UTC):
                    await self._save_step(
                        stored,
                        status=PlanStepState.SATISFIED,
                        result_revision=f"time:{step.not_before.astimezone(UTC).isoformat()}",
                    )
                elif step.status != PlanStepState.WAITING:
                    await self._save_step(stored, status=PlanStepState.WAITING)
                else:
                    continue
                changed = True
            elif step.kind in {PlanStepKind.QUESTION, PlanStepKind.APPROVAL, PlanStepKind.EVENT}:
                if step.status != PlanStepState.PAUSED:
                    await self._save_step(stored, status=PlanStepState.PAUSED)
                    changed = True
            elif step.kind == PlanStepKind.COMPENSATION and step.compensates_step_id:
                source_stored = by_id.get(step.compensates_step_id)
                if source_stored is None:
                    await self._save_step(
                        stored,
                        status=PlanStepState.BLOCKED,
                        reason="compensation source is missing",
                    )
                    changed = True
                else:
                    source = cast(PlanStepRecord, source_stored.record)
                    if source.status in _SUCCESS_STATES:
                        await self._save_step(
                            stored,
                            status=PlanStepState.CANCELLED,
                            reason="compensation was not needed",
                        )
                        changed = True
        return changed

    @staticmethod
    def _ready_steps(steps: tuple[StoredRecord, ...]) -> tuple[PlanStepRecord, ...]:
        by_id = {
            cast(PlanStepRecord, item.record).id: cast(PlanStepRecord, item.record)
            for item in steps
        }
        ready: list[PlanStepRecord] = []
        for stored in steps:
            step = cast(PlanStepRecord, stored.record)
            if (
                step.status
                not in {PlanStepState.PENDING, PlanStepState.LEASED, PlanStepState.RUNNING}
                or step.kind not in _EXECUTABLE_KINDS
            ):
                continue
            if not all(by_id[item].status in _SUCCESS_STATES for item in step.depends_on):
                continue
            if step.kind == PlanStepKind.COMPENSATION:
                source = by_id.get(step.compensates_step_id or "")
                if source is None or source.status != PlanStepState.FAILED:
                    continue
            ready.append(step)
        return tuple(sorted(ready, key=lambda item: (item.order, item.id)))

    @staticmethod
    def _next_wave(ready: tuple[PlanStepRecord, ...]) -> tuple[PlanStepRecord, ...]:
        if not ready:
            return ()
        first = ready[0]
        if not first.parallel_safe:
            return (first,)
        selected = [first]
        resources = set(first.resources or (f"plan:{first.plan_id}",))
        for candidate in ready[1:]:
            candidate_resources = set(candidate.resources or (f"plan:{candidate.plan_id}",))
            if candidate.parallel_safe and resources.isdisjoint(candidate_resources):
                selected.append(candidate)
                resources.update(candidate_resources)
        return tuple(selected)

    async def _execute_step(self, step: PlanStepRecord) -> bool:
        execution = await self.store.acquire_execution(
            step.id,
            worker_id=self.worker_id,
            now=self._now(),
            lease_seconds=self.lease_seconds,
        )
        if execution is None:
            return False
        if execution.status == ExecutionState.SUCCEEDED:
            return False
        running = await self.store.mark_execution_running(
            execution.id,
            lease_token=execution.lease_token,
            now=self._now(),
        )
        try:
            outcome = await self.executor.execute(
                step,
                idempotency_key=running.idempotency_key,
            )
        except Exception as error:
            outcome = StepExecutionResult(
                succeeded=False,
                error_code=f"executor:{type(error).__name__}",
                retryable=True,
            )
        await self.store.complete_execution(
            running.id,
            lease_token=running.lease_token,
            succeeded=outcome.succeeded,
            result_revision=outcome.result_revision,
            error_code=outcome.error_code,
            retryable=outcome.retryable,
            now=self._now(),
        )
        return True

    async def _set_plan_state(
        self,
        stored_plan: StoredRecord,
        steps: tuple[StoredRecord, ...],
    ) -> PlanRecord:
        plan = cast(PlanRecord, stored_plan.record)
        states = {cast(PlanStepRecord, item.record).status for item in steps}
        required_steps = [
            cast(PlanStepRecord, item.record)
            for item in steps
            if not (
                cast(PlanStepRecord, item.record).kind == PlanStepKind.COMPENSATION
                and cast(PlanStepRecord, item.record).status == PlanStepState.CANCELLED
            )
        ]
        if required_steps and all(step.status in _SUCCESS_STATES for step in required_steps):
            status = PlanState.SATISFIED
            reason = None
        elif PlanStepState.BLOCKED in states:
            status = PlanState.BLOCKED
            reason = "one or more steps are blocked"
        elif PlanStepState.FAILED in states and not any(
            step.kind == PlanStepKind.COMPENSATION and step.status == PlanStepState.PENDING
            for step in required_steps
        ):
            status = PlanState.FAILED
            reason = "one or more steps failed"
        elif states & {PlanStepState.PAUSED, PlanStepState.WAITING}:
            status = PlanState.PAUSED
            reason = "waiting for time, an event, an answer, or approval"
        else:
            status = PlanState.ACTIVE
            reason = None
        if plan.status == status and plan.terminal_reason == reason:
            return plan
        return await self.store.set_plan_state(
            plan.id,
            status=status,
            reason=reason,
            actor_id=self.worker_id,
            now=self._now(),
        )

    async def run(self, plan_id: str) -> PlanRecord:
        """Advance a plan until it completes or reaches a durable pause."""

        while True:
            stored_plan = await self.store.get(PlanRecord, plan_id)
            if stored_plan is None:
                raise KeyError(f"unknown plan: {plan_id}")
            plan = cast(PlanRecord, stored_plan.record)
            if plan.status in {
                PlanState.SATISFIED,
                PlanState.CANCELLED,
                PlanState.EXPIRED,
                PlanState.FAILED,
            }:
                return plan
            steps = await self.store.list(PlanStepRecord, parent_id=plan_id)
            changed = await self._prepare_non_tool_steps(steps)
            if changed:
                continue
            ready = self._ready_steps(steps)
            wave = self._next_wave(ready)
            if wave:
                outcomes = await asyncio.gather(*(self._execute_step(step) for step in wave))
                if any(outcomes):
                    continue
            refreshed_steps = await self.store.list(PlanStepRecord, parent_id=plan_id)
            refreshed_plan = await self.store.get(PlanRecord, plan_id)
            if refreshed_plan is None:
                raise KeyError(f"unknown plan: {plan_id}")
            result = await self._set_plan_state(refreshed_plan, refreshed_steps)
            return result

    async def resolve_step(
        self,
        step_id: str,
        *,
        accepted: bool,
        result_revision: str,
        reason: str | None = None,
    ) -> PlanStepRecord:
        stored = await self.store.get(PlanStepRecord, step_id)
        if stored is None:
            raise KeyError(f"unknown plan step: {step_id}")
        step = cast(PlanStepRecord, stored.record)
        if step.kind not in {PlanStepKind.QUESTION, PlanStepKind.APPROVAL, PlanStepKind.EVENT}:
            raise ValueError("only question, approval, and event steps accept external resolution")
        if step.status not in {PlanStepState.PAUSED, PlanStepState.WAITING}:
            raise ValueError("step is not waiting for an external resolution")
        return await self._save_step(
            stored,
            status=PlanStepState.SATISFIED if accepted else PlanStepState.FAILED,
            reason=reason,
            result_revision=result_revision,
        )

    async def cancel(self, plan_id: str, *, reason: str) -> PlanRecord:
        return await self.store.terminate_plan(
            plan_id,
            status=PlanState.CANCELLED,
            reason=reason,
            actor_id=self.worker_id,
            now=self._now(),
        )

    async def expire(self, plan_id: str, *, reason: str = "plan expired") -> PlanRecord:
        return await self.store.terminate_plan(
            plan_id,
            status=PlanState.EXPIRED,
            reason=reason,
            actor_id=self.worker_id,
            now=self._now(),
        )
