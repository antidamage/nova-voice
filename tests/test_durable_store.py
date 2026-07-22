from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from nova_voice.durable.models import (
    ExecutionRecord,
    GoalRecord,
    PlanRecord,
    PlanStepKind,
    PlanStepRecord,
)
from nova_voice.durable.store import ConcurrentRecordUpdate, DurableAgentStore


def _bundle(now: datetime):
    goal = GoalRecord(
        id="goal",
        summary="Run a restart-safe plan",
        plan_ids=("plan",),
        created_at=now,
        updated_at=now,
    )
    steps = (
        PlanStepRecord(
            id="first",
            plan_id="plan",
            kind=PlanStepKind.TOOL,
            order=0,
            max_attempts=2,
            created_at=now,
            updated_at=now,
        ),
        PlanStepRecord(
            id="second",
            plan_id="plan",
            kind=PlanStepKind.VERIFICATION,
            order=1,
            depends_on=("first",),
            created_at=now,
            updated_at=now,
        ),
    )
    plan = PlanRecord(
        id="plan",
        goal_id=goal.id,
        step_ids=tuple(step.id for step in steps),
        created_at=now,
        updated_at=now,
    )
    return goal, plan, steps


async def test_store_migrates_and_persists_a_plan_transactionally_across_restart(tmp_path) -> None:
    now = datetime(2026, 7, 22, tzinfo=UTC)
    path = tmp_path / "durable.sqlite3"
    store = DurableAgentStore(path)
    await store.initialize()
    goal, plan, steps = _bundle(now)
    await store.create_plan_bundle(goal, plan, steps, actor_id="owner")

    restarted = DurableAgentStore(path)
    await restarted.initialize()

    assert await restarted.migration_version() == 1
    assert (await restarted.get(GoalRecord, goal.id)).record == goal
    assert [item.record.id for item in await restarted.list(PlanStepRecord, parent_id=plan.id)] == [
        "first",
        "second",
    ]
    assert len(await restarted.list_audit()) == 4


async def test_store_uses_optimistic_revisions_retention_and_verified_restore(tmp_path) -> None:
    now = datetime(2026, 7, 22, tzinfo=UTC)
    store = DurableAgentStore(tmp_path / "live.sqlite3")
    await store.initialize()
    expired = GoalRecord(
        id="expired",
        summary="short lived",
        created_at=now - timedelta(hours=2),
        updated_at=now - timedelta(hours=2),
        expires_at=now - timedelta(hours=1),
    )
    created = await store.create(expired)
    changed = expired.model_copy(update={"summary": "changed", "updated_at": now})
    await store.save(changed, expected_revision=created.revision)
    with pytest.raises(ConcurrentRecordUpdate):
        await store.save(changed, expected_revision=created.revision)

    backup = await store.backup_to(tmp_path / "backup.sqlite3")
    assert await store.prune_expired(now) == 1
    assert await store.get(GoalRecord, expired.id) is None

    await store.restore_from(backup)
    restored = await store.get(GoalRecord, expired.id)
    assert restored is not None
    assert restored.record.summary == "changed"


async def test_execution_lease_reclaims_same_idempotency_key_after_expiry(tmp_path) -> None:
    now = datetime(2026, 7, 22, tzinfo=UTC)
    store = DurableAgentStore(tmp_path / "durable.sqlite3")
    await store.initialize()
    goal, plan, steps = _bundle(now)
    await store.create_plan_bundle(goal, plan, steps)

    first = await store.acquire_execution("first", worker_id="worker-a", now=now, lease_seconds=10)
    assert first is not None
    await store.mark_execution_running(first.id, lease_token=first.lease_token, now=now)
    assert (
        await store.acquire_execution(
            "first", worker_id="worker-b", now=now + timedelta(seconds=5)
        )
        is None
    )

    reclaimed = await store.acquire_execution(
        "first",
        worker_id="worker-b",
        now=now + timedelta(seconds=11),
        lease_seconds=10,
    )
    assert reclaimed is not None
    assert reclaimed.id == first.id
    assert reclaimed.idempotency_key == first.idempotency_key
    assert reclaimed.attempt == 2
    assert len(await store.list(ExecutionRecord, parent_id=plan.id)) == 1
