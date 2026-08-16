"""The durable companion job record and the moves it permits.

NPT-401 and NPT-402. These are about what cannot happen: a job completing
without having run, two devices holding one job, a superseded attempt
overwriting the answer that replaced it, or a cancelled job quietly being done
anyway. Each is a transition the machine refuses, so each gets a test that
tries it.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from nova_voice.companion import jobs
from nova_voice.durable.models import CompanionJobRecord
from nova_voice.durable.models import CompanionJobState as State
from nova_voice.durable.store import DurableAgentStore


def _job(**overrides) -> CompanionJobRecord:
    return CompanionJobRecord(
        id="job-1",
        workload="research_synthesis",
        input_revision="rev-1",
        idempotency_key="key-1",
        trace_id="trace-1",
        **overrides,
    )


def _running(job: CompanionJobRecord | None = None) -> tuple[CompanionJobRecord, str]:
    record = job or _job()
    record = jobs.offer(
        record, attempt_id="a1", session_id="session-1", lease_seconds=60
    )
    record = jobs.accept(record, attempt_id="a1")
    return jobs.start(record, attempt_id="a1"), "a1"


# -- the record ---------------------------------------------------------------


def test_a_half_written_lease_is_refused():
    """An owner with no expiry is how two owners come to hold one job."""

    with pytest.raises(ValueError, match="both an owner and an expiry"):
        _job(lease_owner="session-1")
    with pytest.raises(ValueError, match="both an owner and an expiry"):
        _job(lease_expires_at=datetime.now(UTC) + timedelta(seconds=30))


def test_the_input_revision_and_key_survive_a_fallback():
    """The whole point of them: work changing hands must not duplicate a write."""

    record, attempt = _running()
    fallen = jobs.fall_back(record, reason="phone disconnected", attempt_id=attempt)
    done = jobs.complete(fallen, attempt_id=None, result_ref="result-1")

    assert done.input_revision == record.input_revision
    assert done.idempotency_key == record.idempotency_key
    assert done.local_fallback is True


# -- illegal moves ------------------------------------------------------------


def test_a_job_cannot_complete_without_having_run():
    record = _job()
    with pytest.raises(jobs.IllegalTransition):
        jobs.complete(record, attempt_id=None, result_ref="result-1")


def test_a_second_device_cannot_accept_a_job_the_first_holds():
    record = jobs.offer(_job(), attempt_id="a1", session_id="session-1", lease_seconds=60)
    with pytest.raises(jobs.IllegalTransition):
        jobs.accept(record, attempt_id="a2")


def test_a_fallen_back_job_can_never_return_to_the_phone():
    # Re-offering after a fallback is exactly how one job gets done twice.
    record, attempt = _running()
    fallen = jobs.fall_back(record, reason="deadline", attempt_id=attempt)
    with pytest.raises(jobs.IllegalTransition):
        jobs.offer(fallen, attempt_id="a2", session_id="session-2", lease_seconds=60)


def test_every_terminal_state_is_a_dead_end():
    for state in jobs.TERMINAL:
        assert jobs.TRANSITIONS[state] == frozenset(), state


# -- idempotency --------------------------------------------------------------


def test_a_redelivered_result_changes_nothing():
    """Networks redeliver. A repeated job_result is traffic, not an error."""

    record, attempt = _running()
    first = jobs.complete(record, attempt_id=attempt, result_ref="result-1")
    second = jobs.complete(first, attempt_id=attempt, result_ref="result-1")

    assert second is first
    assert second.status is State.COMPLETED


def test_a_redelivered_offer_does_not_add_a_second_attempt():
    record = jobs.offer(_job(), attempt_id="a1", session_id="session-1", lease_seconds=60)
    again = jobs.offer(record, attempt_id="a1", session_id="session-1", lease_seconds=60)

    assert again is record
    assert len(again.attempts) == 1


# -- late results -------------------------------------------------------------


def test_a_superseded_attempt_cannot_overwrite_the_one_that_replaced_it():
    record = jobs.offer(_job(), attempt_id="a1", session_id="session-1", lease_seconds=60)
    record = jobs.reject(record, attempt_id="a1", reason="battery")
    record = jobs.offer(record, attempt_id="a2", session_id="session-2", lease_seconds=60)
    record = jobs.accept(record, attempt_id="a2")
    record = jobs.start(record, attempt_id="a2")

    late = jobs.complete(record, attempt_id="a1", result_ref="stale-result")

    assert late is record
    assert late.result_ref is None
    assert late.status is State.RUNNING


def test_rejection_returns_the_job_to_the_queue_and_releases_the_lease():
    # A phone declining on battery grounds is the system working, so the job
    # should be re-offerable now rather than when the lease happens to expire.
    record = jobs.offer(_job(), attempt_id="a1", session_id="session-1", lease_seconds=60)
    rejected = jobs.reject(record, attempt_id="a1", reason="battery")

    assert rejected.status is State.QUEUED
    assert rejected.lease_owner is None
    assert rejected.lease_expires_at is None
    assert rejected.attempts[0].outcome == "rejected"
    # History is kept: it explains a latency figure nobody could otherwise
    # account for.
    assert len(rejected.attempts) == 1


# -- leases -------------------------------------------------------------------


def test_an_unleased_job_is_not_orphaned():
    """A queued job has no owner. That is not the same as a lost one."""

    queued = _job()
    assert jobs.lease_expired(queued) is False
    # Reclaim must leave it exactly as it found it, or a timer sweeping the
    # table would churn every queued job on every pass.
    assert jobs.reclaim(queued) is queued


def test_an_expired_lease_is_reclaimed_back_to_the_queue():
    record, _ = _running()
    later = datetime.now(UTC) + timedelta(hours=1)

    reclaimed = jobs.reclaim(record, now=later)

    assert reclaimed.status is State.QUEUED
    assert reclaimed.lease_owner is None
    assert reclaimed.attempts[0].outcome == "disconnected"


def test_a_live_lease_is_left_alone():
    record, _ = _running()
    assert jobs.reclaim(record) is record


def test_a_terminal_job_is_never_reclaimed():
    record, attempt = _running()
    done = jobs.complete(record, attempt_id=attempt, result_ref="result-1")
    later = datetime.now(UTC) + timedelta(hours=1)

    assert jobs.reclaim(done, now=later) is done


def test_progress_renews_the_lease_so_visible_work_is_not_reclaimed():
    record, attempt = _running()
    before = record.lease_expires_at
    assert before is not None

    moved = jobs.progress(
        record, attempt_id=attempt, stage="synthesising", fraction=0.4, lease_seconds=600
    )

    assert moved.lease_expires_at is not None
    assert moved.lease_expires_at > before
    assert moved.progress_fraction == 0.4


# -- progress -----------------------------------------------------------------


def test_progress_never_goes_backwards():
    """A bar that retreats is worse than one that stalls."""

    record, attempt = _running()
    record = jobs.progress(record, attempt_id=attempt, stage="a", fraction=0.6)
    record = jobs.progress(record, attempt_id=attempt, stage="b", fraction=0.2)

    assert record.progress_fraction == 0.6
    assert record.progress_stage == "b"


def test_progress_from_a_superseded_attempt_is_ignored():
    record, _ = _running()
    ignored = jobs.progress(record, attempt_id="ghost", stage="nonsense", fraction=0.9)

    assert ignored is record


# -- cancellation and approval ------------------------------------------------


def test_a_cancelled_job_does_not_fall_back():
    # Running it locally instead would be doing the thing that was cancelled.
    record, _ = _running()
    cancelled = jobs.cancel(record, reason="user cancelled")

    assert cancelled.status is State.CANCELLED
    assert cancelled.local_fallback is False
    with pytest.raises(jobs.IllegalTransition):
        jobs.fall_back(cancelled, reason="anything")


def test_an_approval_can_resume_the_job_it_paused():
    record, attempt = _running()
    waiting = jobs.await_approval(record, attempt_id=attempt, approval_id="approval-1")
    assert waiting.status is State.WAITING_APPROVAL
    assert waiting.approval_id == "approval-1"

    resumed = jobs.resume(waiting, attempt_id=attempt)
    assert resumed.status is State.RUNNING


def test_a_tool_wait_returns_to_running():
    record, attempt = _running()
    waiting = jobs.await_tool(record, attempt_id=attempt)
    assert waiting.status is State.WAITING_TOOL
    assert jobs.resume(waiting, attempt_id=attempt).status is State.RUNNING


# -- persistence --------------------------------------------------------------


async def test_a_running_job_survives_a_server_restart(tmp_path):
    """The durable row is authoritative; the socket is a transport detail.

    Note there is no migration to reverse: `durable_records` is a generic typed
    record table, so a new record type is a new model rather than new DDL. That
    is why the roadmap's "migrations are reversible" clause has nothing to
    reverse here.
    """

    path = tmp_path / "durable.sqlite3"
    store = DurableAgentStore(path)
    await store.initialize()

    record, attempt = _running()
    record = jobs.progress(
        record, attempt_id=attempt, stage="synthesising", fraction=0.5, summary="half"
    )
    await store.create(record)

    restarted = DurableAgentStore(path)
    await restarted.initialize()
    stored = await restarted.get(CompanionJobRecord, "job-1")

    assert stored is not None
    loaded = stored.record

    assert loaded.status is State.RUNNING
    assert loaded.progress_fraction == 0.5
    assert loaded.attempts[0].attempt_id == attempt
    assert loaded.lease_owner == "session-1"
    # And the job is still finishable from what was reloaded, which is the
    # property that actually matters after a restart.
    done = jobs.complete(loaded, attempt_id=attempt, result_ref="result-1")
    assert done.status is State.COMPLETED
