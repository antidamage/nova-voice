"""Owning a durable job across crashes, reconnects and fallbacks.

NPT-403 and NPT-407. The property under test throughout is that **no sequence
of failures produces two owners or two executions** — not that the happy path
works, which the state machine's own tests already cover.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from nova_voice.companion import jobs
from nova_voice.companion.ledger import CompanionJobLedger
from nova_voice.durable.models import CompanionJobRecord
from nova_voice.durable.models import CompanionJobState as State
from nova_voice.durable.store import DurableAgentStore


def _job(job_id: str = "job-1", **overrides) -> CompanionJobRecord:
    return CompanionJobRecord(
        id=job_id,
        workload="research_synthesis",
        input_revision=f"rev-{job_id}",
        idempotency_key=f"key-{job_id}",
        trace_id=f"trace-{job_id}",
        **overrides,
    )


async def _ledger(tmp_path, name: str = "durable.sqlite3") -> CompanionJobLedger:
    store = DurableAgentStore(tmp_path / name)
    await store.initialize()
    return CompanionJobLedger(store)


async def _running(ledger: CompanionJobLedger, job_id: str = "job-1", *, session="session-1"):
    await ledger.create(_job(job_id))
    await ledger.apply(
        job_id,
        lambda job: jobs.offer(
            job, attempt_id=f"{job_id}-a1", session_id=session, lease_seconds=60
        ),
    )
    await ledger.apply(job_id, lambda job: jobs.accept(job, attempt_id=f"{job_id}-a1"))
    return await ledger.apply(job_id, lambda job: jobs.start(job, attempt_id=f"{job_id}-a1"))


# -- writing ------------------------------------------------------------------


async def test_a_no_op_transition_is_not_written(tmp_path):
    """Bumping the revision for nothing would cause conflicts for nothing."""

    ledger = await _ledger(tmp_path)
    record = await _running(ledger)
    done = await ledger.apply(
        "job-1", lambda job: jobs.complete(job, attempt_id="job-1-a1", result_ref="r1")
    )
    assert done is not None

    before = await ledger._store.get(CompanionJobRecord, "job-1")
    again = await ledger.apply(
        "job-1", lambda job: jobs.complete(job, attempt_id="job-1-a1", result_ref="r1")
    )
    after = await ledger._store.get(CompanionJobRecord, "job-1")

    assert again is not None
    assert before is not None and after is not None
    assert after.revision == before.revision
    assert record.status is State.RUNNING


async def test_a_missing_job_is_none_rather_than_an_error(tmp_path):
    ledger = await _ledger(tmp_path)
    assert await ledger.apply("nobody", jobs.expire) is None


# -- sweeping -----------------------------------------------------------------


async def test_an_orphaned_lease_returns_to_the_queue(tmp_path):
    # The device crashed or was force-quit. Nothing will ever finish this.
    ledger = await _ledger(tmp_path)
    await _running(ledger)

    touched = await ledger.sweep(now=datetime.now(UTC) + timedelta(hours=1))

    assert touched == ("job-1",)
    reclaimed = await ledger.get("job-1")
    assert reclaimed is not None
    assert reclaimed.status is State.QUEUED
    assert reclaimed.lease_owner is None


async def test_a_job_past_its_deadline_expires_rather_than_requeueing(tmp_path):
    """Requeueing it would only offer another device something to fail at."""

    ledger = await _ledger(tmp_path)
    deadline = datetime.now(UTC) + timedelta(seconds=5)
    await ledger.create(_job(complete_deadline=deadline))
    await ledger.apply(
        "job-1",
        lambda job: jobs.offer(
            job, attempt_id="a1", session_id="session-1", lease_seconds=60
        ),
    )

    await ledger.sweep(now=deadline + timedelta(seconds=1))

    expired = await ledger.get("job-1")
    assert expired is not None
    assert expired.status is State.EXPIRED
    assert expired.failure_code == "deadline"


async def test_a_quiet_sweep_touches_nothing(tmp_path):
    ledger = await _ledger(tmp_path)
    await _running(ledger)
    assert await ledger.sweep() == ()


async def test_sweeping_never_disturbs_a_finished_job(tmp_path):
    ledger = await _ledger(tmp_path)
    await _running(ledger)
    await ledger.apply(
        "job-1", lambda job: jobs.complete(job, attempt_id="job-1-a1", result_ref="r1")
    )

    assert await ledger.sweep(now=datetime.now(UTC) + timedelta(days=1)) == ()
    done = await ledger.get("job-1")
    assert done is not None and done.status is State.COMPLETED


# -- reconnect reconciliation -------------------------------------------------


async def test_a_job_the_reconnected_device_has_forgotten_is_reclaimed(tmp_path):
    # The phone restarted mid-job. Its lease has not expired yet, but it has
    # just told us it is not working on this, so waiting the lease out would
    # only delay the job by whatever is left of it.
    ledger = await _ledger(tmp_path)
    await _running(ledger)

    reclaimed, unknown = await ledger.reconcile(session_id="session-1", device_job_ids=[])

    assert reclaimed == ("job-1",)
    assert unknown == ()
    record = await ledger.get("job-1")
    assert record is not None and record.status is State.QUEUED


async def test_a_job_the_device_still_holds_is_left_running(tmp_path):
    ledger = await _ledger(tmp_path)
    await _running(ledger)

    reclaimed, unknown = await ledger.reconcile(
        session_id="session-1", device_job_ids=["job-1"]
    )

    assert reclaimed == ()
    assert unknown == ()
    record = await ledger.get("job-1")
    assert record is not None and record.status is State.RUNNING


async def test_a_job_the_device_claims_but_we_do_not_own_is_reported(tmp_path):
    """The caller must cancel it on the device, or it will be executed twice."""

    ledger = await _ledger(tmp_path)
    await _running(ledger)

    reclaimed, unknown = await ledger.reconcile(
        session_id="session-1", device_job_ids=["job-1", "ghost-job"]
    )

    assert reclaimed == ()
    assert unknown == ("ghost-job",)


async def test_reconciling_ignores_jobs_leased_to_another_session(tmp_path):
    # Two devices, or one device across a supersede. Reconciling for one must
    # not disturb the other's work.
    ledger = await _ledger(tmp_path)
    await _running(ledger, "job-1", session="session-1")
    await _running(ledger, "job-2", session="session-2")

    reclaimed, unknown = await ledger.reconcile(session_id="session-1", device_job_ids=[])

    assert reclaimed == ("job-1",)
    other = await ledger.get("job-2")
    assert other is not None and other.status is State.RUNNING
    assert other.lease_owner == "session-2"


# -- fallback -----------------------------------------------------------------


async def test_falling_back_releases_the_lease_and_claims_the_work_in_one_write(tmp_path):
    ledger = await _ledger(tmp_path)
    await _running(ledger)

    fallen = await ledger.fall_back("job-1", reason="companion disconnected")

    assert fallen is not None
    assert fallen.status is State.FALLING_BACK
    assert fallen.local_fallback is True
    assert fallen.lease_owner is None
    # Same question, same key: the fallback cannot duplicate a downstream write.
    assert fallen.input_revision == "rev-job-1"
    assert fallen.idempotency_key == "key-job-1"


async def test_a_sweep_cannot_hand_a_fallen_back_job_to_a_device(tmp_path):
    """The window this closes is why fallback is one write rather than two."""

    ledger = await _ledger(tmp_path)
    await _running(ledger)
    await ledger.fall_back("job-1", reason="deadline")

    await ledger.sweep(now=datetime.now(UTC) + timedelta(hours=1))

    record = await ledger.get("job-1")
    assert record is not None
    assert record.status is State.FALLING_BACK
    assert record.lease_owner is None


async def test_a_job_that_already_finished_is_not_dragged_back_for_fallback(tmp_path):
    # It already has an answer. Running it locally would compute a second one.
    ledger = await _ledger(tmp_path)
    await _running(ledger)
    await ledger.apply(
        "job-1", lambda job: jobs.complete(job, attempt_id="job-1-a1", result_ref="r1")
    )

    unchanged = await ledger.fall_back("job-1", reason="too late")

    assert unchanged is not None
    assert unchanged.status is State.COMPLETED
    assert unchanged.local_fallback is False


async def test_a_late_phone_result_cannot_overwrite_the_local_answer(tmp_path):
    ledger = await _ledger(tmp_path)
    await _running(ledger)
    await ledger.fall_back("job-1", reason="companion timed out")
    await ledger.complete_locally("job-1", result_ref="local-result")

    late = await ledger.apply(
        "job-1",
        lambda job: jobs.complete(job, attempt_id="job-1-a1", result_ref="stale-phone-result"),
    )

    assert late is not None
    assert late.result_ref == "local-result"


async def test_a_cancelled_job_is_never_taken_over_locally(tmp_path):
    ledger = await _ledger(tmp_path)
    await _running(ledger)
    await ledger.apply("job-1", lambda job: jobs.cancel(job, reason="user cancelled"))

    unchanged = await ledger.fall_back("job-1", reason="anything")

    assert unchanged is not None
    assert unchanged.status is State.CANCELLED
    assert unchanged.local_fallback is False
