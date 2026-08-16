"""Tool callbacks that outlive their socket.

NPT-405. Two guarantees, and every test is one of them failing if it can:
a retry on either side must not run the tool twice, and a result must only ever
reach the reasoning attempt that asked for it.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from nova_voice.companion.callbacks import CallbackError, CompanionCallbackLedger
from nova_voice.durable.models import CompanionCallbackRecord
from nova_voice.durable.models import CompanionCallbackState as State
from nova_voice.durable.store import DurableAgentStore


class _Tool:
    """Counts real executions, so 'ran twice' is a visible failure."""

    def __init__(self, *, ok: bool = True, raises: bool = False) -> None:
        self.calls: list[str] = []
        self.ok = ok
        self.raises = raises

    async def __call__(self, record: CompanionCallbackRecord):
        self.calls.append(record.idempotency_key)
        if self.raises:
            raise RuntimeError("the provider is unreachable")
        return self.ok, "ok" if self.ok else "blocked", "done", {"isOn": True}


async def _ledger(tmp_path, tool: _Tool, name="durable.sqlite3"):
    store = DurableAgentStore(tmp_path / name)
    await store.initialize()
    return CompanionCallbackLedger(store, execute=tool), store


async def _requested(ledger, *, call_id="call-1", attempt_id="a1"):
    return await ledger.request(
        job_id="job-1",
        attempt_id=attempt_id,
        call_id=call_id,
        provider="nova",
        tool="light_set",
        arguments={"state": "on"},
    )


# -- at-most-once -------------------------------------------------------------


async def test_a_callback_runs_once_and_records_its_answer(tmp_path):
    tool = _Tool()
    ledger, _ = await _ledger(tmp_path, tool)
    await _requested(ledger)

    done = await ledger.execute("call-1")

    assert done.status is State.COMPLETED
    assert done.ok is True
    assert done.observed == {"isOn": True}
    assert tool.calls == ["job-1:call-1"]


async def test_the_device_re_asking_after_a_reconnect_does_not_run_it_again(tmp_path):
    # The device has no way of knowing whether its first request survived, so
    # re-asking is ordinary traffic rather than an error.
    tool = _Tool()
    ledger, _ = await _ledger(tmp_path, tool)
    await _requested(ledger)
    await ledger.execute("call-1")

    again = await _requested(ledger)

    assert again.status is State.COMPLETED
    assert again.observed == {"isOn": True}
    assert tool.calls == ["job-1:call-1"]


async def test_executing_twice_does_not_touch_the_tool_twice(tmp_path):
    tool = _Tool()
    ledger, _ = await _ledger(tmp_path, tool)
    await _requested(ledger)

    await ledger.execute("call-1")
    await ledger.execute("call-1")

    assert tool.calls == ["job-1:call-1"]


async def test_a_callback_already_claimed_by_a_worker_is_not_run_again(tmp_path):
    """Recovery owns the interrupted case; running it here is the duplicate."""

    tool = _Tool()
    ledger, store = await _ledger(tmp_path, tool)
    record = await _requested(ledger)
    stored = await store.get(CompanionCallbackRecord, "call-1")
    assert stored is not None
    await store.save(
        record.model_copy(update={"status": State.EXECUTING}),
        expected_revision=stored.revision,
    )

    result = await ledger.execute("call-1")

    assert result.status is State.EXECUTING
    assert tool.calls == []


async def test_the_key_survives_a_crash_so_recovery_cannot_write_twice(tmp_path):
    # The key is derived from the call, not random, so the retry carries the
    # same key the first attempt did into the executor.
    tool = _Tool()
    ledger, store = await _ledger(tmp_path, tool)
    record = await _requested(ledger)
    stored = await store.get(CompanionCallbackRecord, "call-1")
    assert stored is not None
    await store.save(
        record.model_copy(update={"status": State.EXECUTING}),
        expected_revision=stored.revision,
    )

    resumed = await ledger.recover_interrupted()

    assert resumed == ("call-1",)
    assert tool.calls == ["job-1:call-1"]
    finished = await ledger.get("call-1")
    assert finished is not None and finished.status is State.COMPLETED


async def test_a_failing_tool_is_recorded_rather_than_raised(tmp_path):
    # Silence would strand the reasoning job waiting for an answer that is
    # never coming.
    tool = _Tool(raises=True)
    ledger, _ = await _ledger(tmp_path, tool)
    await _requested(ledger)

    failed = await ledger.execute("call-1")

    assert failed.status is State.FAILED
    assert failed.ok is False
    assert failed.code == "backend_error"


async def test_an_unknown_callback_is_an_error_not_a_silent_no_op(tmp_path):
    ledger, _ = await _ledger(tmp_path, _Tool())
    with pytest.raises(CallbackError, match="unknown callback"):
        await ledger.execute("never-heard-of-it")


# -- delivery to the right attempt --------------------------------------------


async def test_a_result_reaches_the_attempt_that_asked_for_it(tmp_path):
    ledger, _ = await _ledger(tmp_path, _Tool())
    await _requested(ledger, attempt_id="a1")
    await ledger.execute("call-1")

    delivered = await ledger.deliverable("call-1", current_attempt_id="a1")

    assert delivered is not None
    assert delivered.observed == {"isOn": True}


async def test_a_result_is_discarded_when_the_job_has_changed_hands(tmp_path):
    # By the time a slow tool answers, the job may be owned by a different
    # attempt after a reclaim or a supersede. Feeding one reasoning run's
    # working state into another's would be worse than losing the answer.
    ledger, _ = await _ledger(tmp_path, _Tool())
    await _requested(ledger, attempt_id="a1")
    await ledger.execute("call-1")

    delivered = await ledger.deliverable("call-1", current_attempt_id="a2")

    assert delivered is None
    superseded = await ledger.get("call-1")
    # Recorded, not deleted: "we asked for this and threw the answer away"
    # explains why a job took two goes.
    assert superseded is not None and superseded.status is State.SUPERSEDED


async def test_a_result_is_discarded_when_nobody_owns_the_job(tmp_path):
    ledger, _ = await _ledger(tmp_path, _Tool())
    await _requested(ledger, attempt_id="a1")
    await ledger.execute("call-1")

    assert await ledger.deliverable("call-1", current_attempt_id=None) is None


# -- storage ------------------------------------------------------------------


async def test_a_callback_must_carry_an_expiry():
    """Its observation is a household record, not a log line."""

    with pytest.raises(ValueError, match="must expire"):
        CompanionCallbackRecord(
            id="call-1",
            job_id="job-1",
            attempt_id="a1",
            call_id="call-1",
            provider="nova",
            tool="light_set",
            idempotency_key="job-1:call-1",
        )


async def test_health_observations_expire_far_sooner_than_ordinary_ones(tmp_path):
    ledger, _ = await _ledger(tmp_path, _Tool())
    now = datetime.now(UTC)
    health = await ledger.request(
        job_id="job-1",
        attempt_id="a1",
        call_id="health-call",
        provider="nova",
        tool="health_read",
        arguments={},
        sensitivity="health",
        now=now,
    )
    ordinary = await ledger.request(
        job_id="job-1",
        attempt_id="a1",
        call_id="ordinary-call",
        provider="nova",
        tool="light_set",
        arguments={},
        sensitivity="ordinary",
        now=now,
    )

    assert health.expires_at is not None and ordinary.expires_at is not None
    assert health.expires_at < ordinary.expires_at


async def test_callbacks_can_be_listed_for_their_job(tmp_path):
    ledger, _ = await _ledger(tmp_path, _Tool())
    await _requested(ledger, call_id="call-1")
    await _requested(ledger, call_id="call-2")

    found = await ledger.for_job("job-1")

    assert sorted(record.call_id for record in found) == ["call-1", "call-2"]


async def test_a_callback_survives_a_server_restart(tmp_path):
    tool = _Tool()
    path = tmp_path / "durable.sqlite3"
    store = DurableAgentStore(path)
    await store.initialize()
    ledger = CompanionCallbackLedger(store, execute=tool)
    await _requested(ledger)

    restarted = DurableAgentStore(path)
    await restarted.initialize()
    reopened = CompanionCallbackLedger(restarted, execute=tool)
    done = await reopened.execute("call-1")

    assert done.status is State.COMPLETED
    assert tool.calls == ["job-1:call-1"]


async def test_a_result_is_kept_only_as_long_as_its_class_allows(tmp_path):
    ledger, _ = await _ledger(tmp_path, _Tool())
    now = datetime.now(UTC)
    record = await _requested(ledger)

    assert record.expires_at is not None
    # Personal: an hour, not a day. The observation is only useful while the
    # job that asked for it is running.
    assert record.expires_at < now + timedelta(hours=2)
