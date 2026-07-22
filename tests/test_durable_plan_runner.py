from __future__ import annotations

import asyncio
from collections import Counter
from datetime import UTC, datetime, timedelta

import pytest

from nova_voice.durable.models import (
    ExecutionRecord,
    GoalRecord,
    GoalState,
    PlanRecord,
    PlanState,
    PlanStepKind,
    PlanStepRecord,
    PlanStepState,
)
from nova_voice.durable.runner import DurablePlanRunner, StepExecutionResult
from nova_voice.durable.store import DurableAgentStore


class _Clock:
    def __init__(self, value: datetime) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value


class _Executor:
    def __init__(self) -> None:
        self.calls: Counter[str] = Counter()
        self.effects: Counter[str] = Counter()
        self.active = 0
        self.max_active = 0

    async def execute(self, step, *, idempotency_key: str) -> StepExecutionResult:
        self.calls[step.id] += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.1)
        self.active -= 1
        if step.input.get("fail_once") and self.calls[step.id] == 1:
            return StepExecutionResult(False, error_code="temporary", retryable=True)
        if step.input.get("fail"):
            return StepExecutionResult(False, error_code="permanent")
        if not self.effects[idempotency_key]:
            self.effects[idempotency_key] += 1
        return StepExecutionResult(True, result_revision=f"sha256:{step.id}")


async def _create_plan(store, now, steps):
    goal = GoalRecord(
        id="goal",
        summary="durable test goal",
        plan_ids=("plan",),
        created_at=now,
        updated_at=now,
    )
    plan = PlanRecord(
        id="plan",
        goal_id=goal.id,
        step_ids=tuple(step.id for step in steps),
        created_at=now,
        updated_at=now,
    )
    await store.create_plan_bundle(goal, plan, steps)
    return plan


async def test_runner_supports_wait_question_approval_event_tools_and_verification(
    tmp_path,
) -> None:
    now = datetime(2026, 7, 22, tzinfo=UTC)
    store = DurableAgentStore(tmp_path / "durable.sqlite3")
    await store.initialize()
    kinds = (
        PlanStepKind.WAIT,
        PlanStepKind.TIMER,
        PlanStepKind.QUESTION,
        PlanStepKind.APPROVAL,
        PlanStepKind.EVENT,
        PlanStepKind.TOOL,
        PlanStepKind.VERIFICATION,
        PlanStepKind.RETRY,
    )
    steps = []
    for index, kind in enumerate(kinds):
        steps.append(
            PlanStepRecord(
                id=kind.value,
                plan_id="plan",
                kind=kind,
                order=index,
                depends_on=((kinds[index - 1].value,) if index else ()),
                not_before=now if kind in {PlanStepKind.WAIT, PlanStepKind.TIMER} else None,
                event_key="door.opened" if kind == PlanStepKind.EVENT else None,
                created_at=now,
                updated_at=now,
            )
        )
    steps.append(
        PlanStepRecord(
            id="compensation",
            plan_id="plan",
            kind=PlanStepKind.COMPENSATION,
            order=len(steps),
            compensates_step_id="tool",
            created_at=now,
            updated_at=now,
        )
    )
    await _create_plan(store, now, tuple(steps))
    runner = DurablePlanRunner(store, _Executor(), worker_id="worker", clock=_Clock(now))

    assert (await runner.run("plan")).status == PlanState.PAUSED
    await runner.resolve_step("question", accepted=True, result_revision="answer:yes")
    assert (await runner.run("plan")).status == PlanState.PAUSED
    await runner.resolve_step("approval", accepted=True, result_revision="approval:owner")
    assert (await runner.run("plan")).status == PlanState.PAUSED
    await runner.resolve_step("event", accepted=True, result_revision="event:42")

    result = await runner.run("plan")
    assert result.status == PlanState.SATISFIED
    assert (await store.get(GoalRecord, "goal")).record.status == GoalState.SATISFIED
    compensation = await store.get(PlanStepRecord, "compensation")
    assert compensation.record.status == PlanStepState.CANCELLED


async def test_runner_parallelizes_only_disjoint_declared_resources_and_retries_failed_step(
    tmp_path,
) -> None:
    now = datetime(2026, 7, 22, tzinfo=UTC)
    store = DurableAgentStore(tmp_path / "durable.sqlite3")
    await store.initialize()
    steps = (
        PlanStepRecord(
            id="office",
            plan_id="plan",
            kind=PlanStepKind.TOOL,
            order=0,
            resources=("light:office",),
            parallel_safe=True,
            created_at=now,
            updated_at=now,
        ),
        PlanStepRecord(
            id="lounge",
            plan_id="plan",
            kind=PlanStepKind.TOOL,
            order=1,
            resources=("light:lounge",),
            parallel_safe=True,
            max_attempts=2,
            input={"fail_once": True},
            created_at=now,
            updated_at=now,
        ),
        PlanStepRecord(
            id="office-verify",
            plan_id="plan",
            kind=PlanStepKind.VERIFICATION,
            order=2,
            resources=("light:office",),
            parallel_safe=True,
            created_at=now,
            updated_at=now,
        ),
    )
    await _create_plan(store, now, steps)
    executor = _Executor()
    result = await DurablePlanRunner(
        store, executor, worker_id="worker", clock=_Clock(now)
    ).run("plan")

    assert result.status == PlanState.SATISFIED
    assert executor.max_active == 2
    assert executor.calls == Counter({"lounge": 2, "office": 1, "office-verify": 1})
    assert sum(executor.effects.values()) == 3


async def test_runner_preserves_success_and_compensates_only_failed_work(tmp_path) -> None:
    now = datetime(2026, 7, 22, tzinfo=UTC)
    store = DurableAgentStore(tmp_path / "durable.sqlite3")
    await store.initialize()
    steps = (
        PlanStepRecord(
            id="success",
            plan_id="plan",
            kind=PlanStepKind.TOOL,
            order=0,
            created_at=now,
            updated_at=now,
        ),
        PlanStepRecord(
            id="failure",
            plan_id="plan",
            kind=PlanStepKind.TOOL,
            order=1,
            input={"fail": True},
            created_at=now,
            updated_at=now,
        ),
        PlanStepRecord(
            id="compensate-failure",
            plan_id="plan",
            kind=PlanStepKind.COMPENSATION,
            order=2,
            compensates_step_id="failure",
            created_at=now,
            updated_at=now,
        ),
    )
    await _create_plan(store, now, steps)
    executor = _Executor()
    result = await DurablePlanRunner(
        store, executor, worker_id="worker", clock=_Clock(now)
    ).run("plan")

    assert result.status == PlanState.FAILED
    assert executor.calls == Counter({"success": 1, "failure": 1, "compensate-failure": 1})
    assert (await store.get(PlanStepRecord, "success")).record.status == PlanStepState.SATISFIED
    assert (
        await store.get(PlanStepRecord, "compensate-failure")
    ).record.status == PlanStepState.COMPENSATED


class _CrashAfterEffect(BaseException):
    pass


class _CrashSafeExecutor:
    def __init__(self) -> None:
        self.calls = 0
        self.effects: set[str] = set()

    async def execute(self, step, *, idempotency_key: str) -> StepExecutionResult:
        self.calls += 1
        if idempotency_key not in self.effects:
            self.effects.add(idempotency_key)
            raise _CrashAfterEffect
        return StepExecutionResult(True, result_revision="sha256:deduplicated")


async def test_crash_after_side_effect_reuses_key_and_never_duplicates_mutation(tmp_path) -> None:
    now = datetime(2026, 7, 22, tzinfo=UTC)
    clock = _Clock(now)
    store = DurableAgentStore(tmp_path / "durable.sqlite3")
    await store.initialize()
    await _create_plan(
        store,
        now,
        (
            PlanStepRecord(
                id="mutation",
                plan_id="plan",
                kind=PlanStepKind.TOOL,
                order=0,
                max_attempts=2,
                created_at=now,
                updated_at=now,
            ),
        ),
    )
    executor = _CrashSafeExecutor()
    first_runner = DurablePlanRunner(
        store, executor, worker_id="worker-a", clock=clock, lease_seconds=10
    )
    with pytest.raises(_CrashAfterEffect):
        await first_runner.run("plan")

    clock.value += timedelta(seconds=11)
    restarted_store = DurableAgentStore(store.path)
    await restarted_store.initialize()
    result = await DurablePlanRunner(
        restarted_store,
        executor,
        worker_id="worker-b",
        clock=clock,
        lease_seconds=10,
    ).run("plan")

    executions = await restarted_store.list(ExecutionRecord, parent_id="plan")
    assert result.status == PlanState.SATISFIED
    assert executor.calls == 2
    assert len(executor.effects) == 1
    assert len(executions) == 1
    assert executions[0].record.attempt == 2


@pytest.mark.parametrize(
    ("operation", "plan_state", "goal_state", "step_state"),
    (
        ("cancel", PlanState.CANCELLED, GoalState.CANCELLED, PlanStepState.CANCELLED),
        ("expire", PlanState.EXPIRED, GoalState.EXPIRED, PlanStepState.EXPIRED),
    ),
)
async def test_plan_termination_atomically_updates_goal_and_unfinished_steps(
    tmp_path,
    operation,
    plan_state,
    goal_state,
    step_state,
) -> None:
    now = datetime(2026, 7, 22, tzinfo=UTC)
    store = DurableAgentStore(tmp_path / "durable.sqlite3")
    await store.initialize()
    await _create_plan(
        store,
        now,
        (
            PlanStepRecord(
                id="pending",
                plan_id="plan",
                kind=PlanStepKind.APPROVAL,
                order=0,
                created_at=now,
                updated_at=now,
            ),
        ),
    )
    runner = DurablePlanRunner(store, _Executor(), worker_id="owner", clock=_Clock(now))

    result = await getattr(runner, operation)("plan", reason=f"owner {operation}")

    assert result.status == plan_state
    assert (await store.get(GoalRecord, "goal")).record.status == goal_state
    assert (await store.get(PlanStepRecord, "pending")).record.status == step_state
