"""Forgetting companion payloads on schedule, and keeping the audit facts.

NPT-404 and NPT-411. Personal context is permitted to leave the phone in v1, so
"it expires" has to be a mechanism with tests rather than a stated intention.
Every test here uses a synthetic secret and asserts on both halves of the rule:
the content is gone, and the structural history is not.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from nova_voice.companion import jobs
from nova_voice.companion.retention import CompanionRetention, checkpoint_expiry
from nova_voice.durable.models import (
    MAX_CHECKPOINT_BYTES,
    CompanionApprovalRecord,
    CompanionApprovalState,
    CompanionCheckpointRecord,
    CompanionJobRecord,
)
from nova_voice.durable.store import DurableAgentStore

SECRET = "dentist at 3pm with Dr Ngata"


async def _store(tmp_path) -> DurableAgentStore:
    store = DurableAgentStore(tmp_path / "durable.sqlite3")
    await store.initialize()
    return store


def _job(job_id="job-1", *, sensitivity="personal", **overrides) -> CompanionJobRecord:
    return CompanionJobRecord(
        id=job_id,
        workload="research_synthesis",
        input_revision="rev-1",
        idempotency_key="key-1",
        trace_id="trace-1",
        sensitivity=sensitivity,
        **overrides,
    )


def _finished_job(job_id="job-1", *, sensitivity="personal") -> CompanionJobRecord:
    record = _job(job_id, sensitivity=sensitivity)
    record = jobs.offer(record, attempt_id="a1", session_id="s1", lease_seconds=60)
    record = jobs.accept(record, attempt_id="a1")
    record = jobs.start(record, attempt_id="a1")
    record = jobs.progress(record, attempt_id="a1", stage="reading", summary=SECRET)
    return jobs.complete(record, attempt_id="a1", result_ref=f"result://{SECRET}")


def _checkpoint(job_id="job-1", *, sensitivity="personal", now=None):
    moment = now or datetime.now(UTC)
    return CompanionCheckpointRecord(
        id=f"checkpoint-{job_id}",
        job_id=job_id,
        attempt_id="a1",
        stage="reading",
        payload={"notes": SECRET},
        sensitivity=sensitivity,
        expires_at=checkpoint_expiry(sensitivity, now=moment),
    )


# -- bounds (NPT-404) ---------------------------------------------------------


def test_a_checkpoint_larger_than_the_limit_is_refused():
    """An unbounded resumption aid is a storage leak that survives restarts."""

    with pytest.raises(ValueError, match="over the"):
        CompanionCheckpointRecord(
            id="big",
            job_id="job-1",
            attempt_id="a1",
            stage="reading",
            payload={"blob": "x" * (MAX_CHECKPOINT_BYTES + 100)},
            expires_at=datetime.now(UTC) + timedelta(hours=1),
        )


def test_a_checkpoint_must_expire():
    with pytest.raises(ValueError, match="must expire"):
        CompanionCheckpointRecord(
            id="forever", job_id="job-1", attempt_id="a1", stage="reading"
        )


def test_health_checkpoints_expire_far_sooner_than_ordinary_ones():
    now = datetime.now(UTC)
    assert checkpoint_expiry("health", now=now) < checkpoint_expiry("personal", now=now)
    assert checkpoint_expiry("personal", now=now) < checkpoint_expiry("ordinary", now=now)


async def test_a_checkpoint_can_be_listed_for_the_job_it_belongs_to(tmp_path):
    # It is stored under the job as its parent; without that there is no way to
    # find it, which is the only thing anything ever wants to do with it.
    store = await _store(tmp_path)
    await store.create(_job())
    await store.create(_checkpoint())

    found = await store.list(CompanionCheckpointRecord, parent_id="job-1")

    assert [stored.record.id for stored in found] == ["checkpoint-job-1"]


# -- retention (NPT-411) ------------------------------------------------------


async def test_a_finished_job_keeps_its_history_and_loses_its_content(tmp_path):
    store = await _store(tmp_path)
    await store.create(_finished_job())
    retention = CompanionRetention(store)

    result = await retention.sweep(now=datetime.now(UTC) + timedelta(days=1))

    assert result["jobs_redacted"] == 1
    stored = await store.get(CompanionJobRecord, "job-1")
    assert stored is not None
    record: CompanionJobRecord = stored.record

    # Gone: anything that says what the request was about.
    assert record.result_ref is None
    assert record.progress_summary is None
    assert record.checkpoint_ref is None
    # Kept: what happened, which is what audit needs and says nothing.
    assert record.status is jobs.State.COMPLETED
    assert record.workload == "research_synthesis"
    assert record.trace_id == "trace-1"
    assert record.attempts[0].attempt_id == "a1"
    assert record.attempts[0].outcome == "completed"
    assert SECRET not in record.model_dump_json()


async def test_an_unfinished_job_is_never_redacted(tmp_path):
    """Its working state is how it resumes."""

    store = await _store(tmp_path)
    running = jobs.start(
        jobs.accept(
            jobs.offer(_job(), attempt_id="a1", session_id="s1", lease_seconds=60),
            attempt_id="a1",
        ),
        attempt_id="a1",
    )
    await store.create(
        running.model_copy(update={"result_ref": "partial", "progress_summary": SECRET})
    )
    retention = CompanionRetention(store)

    await retention.sweep(now=datetime.now(UTC) + timedelta(days=30))

    stored = await store.get(CompanionJobRecord, "job-1")
    assert stored is not None
    assert stored.record.progress_summary == SECRET


async def test_health_content_goes_long_before_ordinary_content(tmp_path):
    store = await _store(tmp_path)
    await store.create(_finished_job("health-job", sensitivity="health"))
    await store.create(_finished_job("ordinary-job", sensitivity="ordinary"))
    retention = CompanionRetention(store)

    # Ten minutes: past Health's five-minute payload horizon, far short of the
    # ordinary class's day.
    await retention.sweep(now=datetime.now(UTC) + timedelta(minutes=10))

    health = await store.get(CompanionJobRecord, "health-job")
    ordinary = await store.get(CompanionJobRecord, "ordinary-job")
    assert health is not None and ordinary is not None
    assert health.record.result_ref is None
    assert ordinary.record.result_ref is not None


async def test_a_checkpoint_for_a_running_job_survives_its_own_horizon(tmp_path):
    # Dropping it would protect nothing — the running job holds the same state
    # in memory — and would cost the resumption it exists for.
    store = await _store(tmp_path)
    running = jobs.start(
        jobs.accept(
            jobs.offer(_job(), attempt_id="a1", session_id="s1", lease_seconds=60),
            attempt_id="a1",
        ),
        attempt_id="a1",
    )
    await store.create(running)
    await store.create(_checkpoint())
    retention = CompanionRetention(store)

    result = await retention.sweep(now=datetime.now(UTC) + timedelta(days=1))

    assert result["checkpoints_extended"] == 1
    assert await store.get(CompanionCheckpointRecord, "checkpoint-job-1") is not None


async def test_a_checkpoint_for_a_finished_job_is_pruned(tmp_path):
    store = await _store(tmp_path)
    await store.create(_finished_job())
    await store.create(_checkpoint())
    retention = CompanionRetention(store)

    await retention.sweep(now=datetime.now(UTC) + timedelta(days=1))

    assert await store.get(CompanionCheckpointRecord, "checkpoint-job-1") is None


# -- approvals ----------------------------------------------------------------


def _approval(approval_id="approval-1", *, status=CompanionApprovalState.DENIED):
    now = datetime.now(UTC)
    return CompanionApprovalRecord(
        id=approval_id,
        provider="nova",
        tool="calendar_create",
        arguments={"title": SECRET},
        summary="Add an appointment to the calendar",
        status=status,
        idempotency_key=f"key-{approval_id}",
        nonce="nonce-nonce-nonce",
        respond_by=now + timedelta(minutes=15),
        expires_at=now + timedelta(days=30),
    )


async def test_a_settled_approval_loses_its_arguments_but_keeps_its_summary(tmp_path):
    # "The owner was asked to add an appointment and said no" is the audit
    # fact, and it is already written for a human rather than being a payload.
    store = await _store(tmp_path)
    await store.create(_approval())
    retention = CompanionRetention(store)

    result = await retention.sweep(now=datetime.now(UTC) + timedelta(days=8))

    assert result["approvals_redacted"] == 1
    stored = await store.get(CompanionApprovalRecord, "approval-1")
    assert stored is not None
    record: CompanionApprovalRecord = stored.record
    assert record.arguments == {}
    assert record.summary == "Add an appointment to the calendar"
    assert record.status is CompanionApprovalState.DENIED
    assert SECRET not in record.model_dump_json()


@pytest.mark.parametrize(
    "status",
    [
        CompanionApprovalState.PENDING,
        CompanionApprovalState.APPROVED,
        CompanionApprovalState.EXECUTING,
    ],
)
async def test_an_approval_that_can_still_happen_keeps_its_arguments(tmp_path, status):
    """The stored arguments *are* the mutation. Expiring them strands it."""

    store = await _store(tmp_path)
    await store.create(_approval(status=status))
    retention = CompanionRetention(store)

    await retention.sweep(now=datetime.now(UTC) + timedelta(days=8))

    stored = await store.get(CompanionApprovalRecord, "approval-1")
    assert stored is not None
    assert stored.record.arguments == {"title": SECRET}


async def test_a_quiet_sweep_reports_doing_nothing(tmp_path):
    store = await _store(tmp_path)
    await store.create(_finished_job())
    retention = CompanionRetention(store)

    result = await retention.sweep(now=datetime.now(UTC))

    assert result == {
        "checkpoints_extended": 0,
        "jobs_redacted": 0,
        "approvals_redacted": 0,
        "records_pruned": 0,
    }


# -- the timer (wiring) -------------------------------------------------------


async def test_the_sweep_loop_runs_and_can_be_stopped(tmp_path):
    """Retention only matters if something calls it."""

    import asyncio

    store = await _store(tmp_path)
    await store.create(_finished_job())
    retention = CompanionRetention(store)

    task = asyncio.create_task(retention.run(interval_seconds=0.01))
    await asyncio.sleep(0.05)
    retention.stop()
    await asyncio.wait_for(task, timeout=1)

    assert task.done()


async def test_a_failing_sweep_does_not_take_the_loop_down(tmp_path, monkeypatch):
    # Cleanup is the least urgent thing this process does and the least
    # acceptable reason for it to stop.
    import asyncio

    store = await _store(tmp_path)
    retention = CompanionRetention(store)
    attempts = {"count": 0}

    async def exploding_sweep(*, now=None):
        attempts["count"] += 1
        raise RuntimeError("the disk is on fire")

    monkeypatch.setattr(retention, "sweep", exploding_sweep)
    task = asyncio.create_task(retention.run(interval_seconds=0.01))
    await asyncio.sleep(0.05)
    retention.stop()
    await asyncio.wait_for(task, timeout=1)

    assert attempts["count"] > 1, "the loop gave up after the first failure"
