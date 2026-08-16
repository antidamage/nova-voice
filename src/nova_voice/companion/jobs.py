"""The state machine for a companion job that outlives its socket.

Ordinary routed workloads (``router.py``) live and die inside one voice turn:
if the phone drops, the turn falls back and nobody needs to remember anything.
Durable jobs are the opposite — research synthesis, briefing composition, an
approval waiting on a human — and for those the socket is an implementation
detail. The **durable row is authoritative; frames are transport events.**

Three properties are load-bearing, and they are the reason this is a machine
rather than a set of field writes:

* **Illegal moves fail.** A job cannot complete without having run, and cannot
  be accepted twice by two devices. Enumerating the legal moves makes those
  impossible rather than merely unlikely.
* **Duplicate frames are idempotent.** Networks redeliver. Applying the same
  terminal transition twice returns the record unchanged instead of raising,
  because a retried ``job_result`` is normal traffic and not an error.
* **A crash between transition and send is recoverable.** Every transition is a
  pure function of the record, so the caller writes it with the store's
  optimistic-concurrency check and can safely repeat the whole operation.

``input_revision`` and ``idempotency_key`` never change, *including across the
local fallback*. That is what stops work changing hands from duplicating a
downstream write.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from nova_voice.durable.models import (
    CompanionAttemptRecord,
    CompanionJobRecord,
    CompanionJobState,
)

State = CompanionJobState

# Terminal states have no exits. Anything already here is finished, and a
# duplicate frame naming one is dropped rather than fought over.
TERMINAL: frozenset[State] = frozenset(
    {State.COMPLETED, State.FAILED, State.CANCELLED, State.EXPIRED}
)

TRANSITIONS: dict[State, frozenset[State]] = {
    State.QUEUED: frozenset({State.OFFERED, State.FALLING_BACK, State.CANCELLED, State.EXPIRED}),
    # An offer that is declined or unanswered goes to fallback, not to failure:
    # a phone saying no on battery grounds is the system working.
    State.OFFERED: frozenset(
        {State.ACCEPTED, State.FALLING_BACK, State.CANCELLED, State.EXPIRED}
    ),
    State.ACCEPTED: frozenset(
        {State.RUNNING, State.FALLING_BACK, State.CANCELLED, State.EXPIRED}
    ),
    State.RUNNING: frozenset(
        {
            State.WAITING_TOOL,
            State.WAITING_APPROVAL,
            State.COMPLETED,
            State.FAILED,
            State.FALLING_BACK,
            State.CANCELLED,
            State.EXPIRED,
        }
    ),
    State.WAITING_TOOL: frozenset(
        {State.RUNNING, State.FAILED, State.FALLING_BACK, State.CANCELLED, State.EXPIRED}
    ),
    # An approval can outlive the device that proposed it, so this may return to
    # RUNNING on a different attempt entirely.
    State.WAITING_APPROVAL: frozenset(
        {State.RUNNING, State.FAILED, State.CANCELLED, State.EXPIRED}
    ),
    # Iridium owns the work now. It cannot go back to the phone: re-offering
    # after fallback is how the same job gets done twice.
    State.FALLING_BACK: frozenset({State.COMPLETED, State.FAILED, State.CANCELLED}),
    State.COMPLETED: frozenset(),
    State.FAILED: frozenset(),
    State.CANCELLED: frozenset(),
    State.EXPIRED: frozenset(),
}


class IllegalTransition(RuntimeError):
    """A move the job's current state does not permit."""

    def __init__(self, record: CompanionJobRecord, target: State) -> None:
        super().__init__(
            f"companion job {record.id} cannot move {record.status.value} -> {target.value}"
        )
        self.current = record.status
        self.target = target


def _now() -> datetime:
    return datetime.now(UTC)


def _moved(
    record: CompanionJobRecord, target: State, **changes: object
) -> CompanionJobRecord:
    """Apply a transition, or explain why it is not allowed.

    A job already *in* the target state is returned untouched. That is the
    idempotency rule, and it is deliberately checked before the legality rule:
    a redelivered ``job_result`` naming a completed job is ordinary traffic,
    not an illegal move from COMPLETED.
    """

    if record.status == target:
        return record
    if target not in TRANSITIONS[record.status]:
        raise IllegalTransition(record, target)
    return record.model_copy(
        update={"status": target, "updated_at": _now(), **changes}
    )


def _end_attempt(
    record: CompanionJobRecord, attempt_id: str, outcome: str, detail: str | None = None
) -> tuple[CompanionAttemptRecord, ...]:
    """Close the named attempt in the history, leaving the others alone."""

    closed: list[CompanionAttemptRecord] = []
    for attempt in record.attempts:
        if attempt.attempt_id == attempt_id and attempt.ended_at is None:
            closed.append(
                attempt.model_copy(
                    update={"ended_at": _now(), "outcome": outcome, "detail": detail}
                )
            )
        else:
            closed.append(attempt)
    return tuple(closed)


def current_attempt(record: CompanionJobRecord) -> CompanionAttemptRecord | None:
    """The attempt that still owns the job, if any."""

    for attempt in reversed(record.attempts):
        if attempt.ended_at is None:
            return attempt
    return None


def owns(record: CompanionJobRecord, attempt_id: str) -> bool:
    """Is this frame from the attempt that currently holds the job?

    The guard against late results. A superseded attempt answering after its
    replacement has taken over must be recorded and ignored, never applied.
    """

    attempt = current_attempt(record)
    return attempt is not None and attempt.attempt_id == attempt_id


def lease_expired(record: CompanionJobRecord, *, now: datetime | None = None) -> bool:
    """Has the owner stopped proving it is alive?

    Absence of a lease is not expiry — a queued job has no owner and is not
    orphaned. Only a lease whose deadline has passed is reclaimable.
    """

    if record.lease_expires_at is None:
        return False
    return (now or _now()) >= record.lease_expires_at


# -- transitions --------------------------------------------------------------


def offer(
    record: CompanionJobRecord,
    *,
    attempt_id: str,
    session_id: str,
    lease_seconds: float,
    complete_deadline: datetime | None = None,
) -> CompanionJobRecord:
    if any(attempt.attempt_id == attempt_id for attempt in record.attempts):
        # Redelivered offer for an attempt already on the record. Checked
        # before the status guard: the job is legitimately no longer QUEUED
        # precisely *because* this offer already landed.
        return record
    if record.status is not State.QUEUED:
        raise IllegalTransition(record, State.OFFERED)
    now = _now()
    return _moved(
        record,
        State.OFFERED,
        attempts=(
            *record.attempts,
            CompanionAttemptRecord(
                attempt_id=attempt_id, session_id=session_id, started_at=now
            ),
        ),
        lease_owner=session_id,
        lease_expires_at=now + timedelta(seconds=lease_seconds),
        complete_deadline=complete_deadline or record.complete_deadline,
        progress_stage="offered",
    )


def accept(record: CompanionJobRecord, *, attempt_id: str) -> CompanionJobRecord:
    if not owns(record, attempt_id):
        raise IllegalTransition(record, State.ACCEPTED)
    return _moved(record, State.ACCEPTED, progress_stage="accepted")


def reject(
    record: CompanionJobRecord, *, attempt_id: str, reason: str
) -> CompanionJobRecord:
    """The device declined. Ordinary flow, so the job returns to the queue.

    The lease is released rather than left to expire: a job nobody holds should
    be re-offerable now, not in a minute.
    """

    if not owns(record, attempt_id):
        return record
    closed = _end_attempt(record, attempt_id, "rejected", reason)
    return record.model_copy(
        update={
            "status": State.QUEUED,
            "attempts": closed,
            "lease_owner": None,
            "lease_expires_at": None,
            "progress_stage": "queued",
            "updated_at": _now(),
        }
    )


def start(record: CompanionJobRecord, *, attempt_id: str) -> CompanionJobRecord:
    if not owns(record, attempt_id):
        raise IllegalTransition(record, State.RUNNING)
    return _moved(record, State.RUNNING, progress_stage="running")


def await_tool(record: CompanionJobRecord, *, attempt_id: str) -> CompanionJobRecord:
    if not owns(record, attempt_id):
        raise IllegalTransition(record, State.WAITING_TOOL)
    return _moved(record, State.WAITING_TOOL, progress_stage="waiting_tool")


def await_approval(
    record: CompanionJobRecord, *, attempt_id: str, approval_id: str
) -> CompanionJobRecord:
    if not owns(record, attempt_id):
        raise IllegalTransition(record, State.WAITING_APPROVAL)
    return _moved(
        record,
        State.WAITING_APPROVAL,
        approval_id=approval_id,
        progress_stage="waiting_approval",
    )


def resume(record: CompanionJobRecord, *, attempt_id: str) -> CompanionJobRecord:
    """Back to work after a tool result or an approval."""

    if not owns(record, attempt_id):
        raise IllegalTransition(record, State.RUNNING)
    return _moved(record, State.RUNNING, progress_stage="running")


def progress(
    record: CompanionJobRecord,
    *,
    attempt_id: str,
    stage: str,
    fraction: float | None = None,
    summary: str | None = None,
    lease_seconds: float | None = None,
) -> CompanionJobRecord:
    """Record progress without changing state, and renew the lease with it.

    Progress *is* the liveness signal for a running job, so renewing here means
    a device that is visibly working cannot have its lease reclaimed underneath
    it — while one that has gone quiet still loses it.

    Fraction is monotonic: a report that goes backwards is dropped rather than
    applied, because a progress bar that retreats is worse than one that stalls.
    """

    if not owns(record, attempt_id):
        return record
    if fraction is not None and record.progress_fraction is not None:
        if fraction < record.progress_fraction:
            fraction = record.progress_fraction
    changes: dict[str, object] = {
        "progress_stage": stage,
        "progress_fraction": fraction if fraction is not None else record.progress_fraction,
        "progress_summary": summary,
        "updated_at": _now(),
    }
    if lease_seconds is not None and record.lease_owner is not None:
        changes["lease_expires_at"] = _now() + timedelta(seconds=lease_seconds)
    return record.model_copy(update=changes)


def checkpoint(
    record: CompanionJobRecord, *, attempt_id: str, reference: str
) -> CompanionJobRecord:
    if not owns(record, attempt_id):
        return record
    return record.model_copy(update={"checkpoint_ref": reference, "updated_at": _now()})


def complete(
    record: CompanionJobRecord, *, attempt_id: str | None, result_ref: str
) -> CompanionJobRecord:
    """Finish the job. ``attempt_id`` is None when Iridium finished it locally.

    A late result from a superseded attempt is dropped here rather than
    overwriting a job that has already moved on — including one that fell back
    and was completed by Iridium.
    """

    if attempt_id is not None and not owns(record, attempt_id):
        return record
    attempts = (
        _end_attempt(record, attempt_id, "completed") if attempt_id else record.attempts
    )
    return _moved(
        record,
        State.COMPLETED,
        attempts=attempts,
        result_ref=result_ref,
        lease_owner=None,
        lease_expires_at=None,
        progress_stage="completed",
        progress_fraction=1.0,
    )


def fall_back(
    record: CompanionJobRecord, *, reason: str, attempt_id: str | None = None
) -> CompanionJobRecord:
    """Hand the work to Iridium, under the same input revision and key.

    One-way by design. ``FALLING_BACK`` has no edge back to the phone, because
    re-offering after a fallback is exactly how one job gets done twice.
    """

    attempts = (
        _end_attempt(record, attempt_id, "failed", reason) if attempt_id else record.attempts
    )
    return _moved(
        record,
        State.FALLING_BACK,
        attempts=attempts,
        local_fallback=True,
        lease_owner=None,
        lease_expires_at=None,
        failure_detail=reason,
        progress_stage="falling_back",
    )


def fail(
    record: CompanionJobRecord,
    *,
    code: str,
    detail: str | None = None,
    attempt_id: str | None = None,
) -> CompanionJobRecord:
    attempts = (
        _end_attempt(record, attempt_id, "failed", detail) if attempt_id else record.attempts
    )
    return _moved(
        record,
        State.FAILED,
        attempts=attempts,
        failure_code=code,
        failure_detail=detail,
        lease_owner=None,
        lease_expires_at=None,
        progress_stage="failed",
    )


def cancel(record: CompanionJobRecord, *, reason: str) -> CompanionJobRecord:
    """The user or the system withdrew the work.

    Never falls back: a cancelled job is one nobody wants the answer to, so
    running it locally instead would be doing the thing that was cancelled.
    """

    attempt = current_attempt(record)
    attempts = (
        _end_attempt(record, attempt.attempt_id, "cancelled", reason)
        if attempt
        else record.attempts
    )
    return _moved(
        record,
        State.CANCELLED,
        attempts=attempts,
        failure_detail=reason,
        lease_owner=None,
        lease_expires_at=None,
        progress_stage="cancelled",
    )


def expire(record: CompanionJobRecord) -> CompanionJobRecord:
    attempt = current_attempt(record)
    attempts = (
        _end_attempt(record, attempt.attempt_id, "timeout", "deadline elapsed")
        if attempt
        else record.attempts
    )
    return _moved(
        record,
        State.EXPIRED,
        attempts=attempts,
        failure_code="deadline",
        lease_owner=None,
        lease_expires_at=None,
        progress_stage="expired",
    )


def reclaim(record: CompanionJobRecord, *, now: datetime | None = None) -> CompanionJobRecord:
    """Take an orphaned job back off a device that stopped proving it was alive.

    Returns the record untouched when the lease is still good, so this is safe
    to run over the whole table on a timer.
    """

    if not lease_expired(record, now=now) or record.status in TERMINAL:
        return record
    attempt = current_attempt(record)
    attempts = (
        _end_attempt(record, attempt.attempt_id, "disconnected", "lease expired")
        if attempt
        else record.attempts
    )
    return record.model_copy(
        update={
            "status": State.QUEUED,
            "attempts": attempts,
            "lease_owner": None,
            "lease_expires_at": None,
            "progress_stage": "queued",
            "updated_at": _now(),
        }
    )
