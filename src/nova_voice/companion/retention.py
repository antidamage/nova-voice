"""Forget companion payloads on schedule, and keep the audit facts.

`sensitivity.py` has always held the policy — how long a payload of each class
may be kept, and that the structural record outlives it by a long way. Nothing
applied it. This does.

The split it enforces, in one sentence: **what happened is kept, what it said
is not.** A finished job's workload, timings, attempt history, outcome and
trace id stay for a month; its result reference, checkpoint and progress
summary go on the payload clock for its class — an hour for calendar and
reminder context, five minutes for Health.

Two things are protected from all of it regardless of class:

* a job that has not finished, because its checkpoint is how it resumes; and
* an approval that has been agreed to but not yet executed, because the stored
  arguments *are* the mutation. Expiring those would leave an approval that can
  never be honoured and a house that never gets what it was told it would.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from datetime import UTC, datetime, timedelta

from nova_voice.companion import jobs
from nova_voice.companion.sensitivity import retention_for
from nova_voice.durable.models import (
    CompanionApprovalRecord,
    CompanionApprovalState,
    CompanionCheckpointRecord,
    CompanionJobRecord,
)
from nova_voice.durable.store import ConcurrentRecordUpdate, DurableAgentStore

logger = logging.getLogger(__name__)

# Approvals in these states still need their arguments: the mutation has been
# agreed to and has not happened yet, or is happening now.
_LIVE_APPROVALS = frozenset(
    {
        CompanionApprovalState.PENDING,
        CompanionApprovalState.APPROVED,
        CompanionApprovalState.EXECUTING,
    }
)


def checkpoint_expiry(sensitivity: str, *, now: datetime | None = None) -> datetime:
    """When a checkpoint of this class stops being allowed to exist."""

    policy = retention_for(sensitivity)  # type: ignore[arg-type]
    return (now or datetime.now(UTC)) + timedelta(seconds=policy.payload_seconds)


class CompanionRetention:
    def __init__(self, store: DurableAgentStore) -> None:
        self._store = store
        self._stop = asyncio.Event()

    async def run(self, interval_seconds: float = 300.0) -> None:
        """Sweep on a timer for the life of the process.

        Five minutes rather than something cleverer, because the tightest
        payload horizon is Health's five minutes and a sweep that runs less
        often than the shortest clock would let those payloads outlive it.

        A sweep that fails must not take the loop down with it. Cleanup is the
        least urgent thing this process does and the least acceptable reason
        for it to stop: the next pass will catch whatever this one missed.
        """

        while not self._stop.is_set():
            try:
                outcome = await self.sweep()
            except Exception:
                logger.exception("companion retention sweep failed")
            else:
                if any(outcome.values()):
                    logger.info("companion retention swept %s", outcome)
            with suppress(TimeoutError):
                await asyncio.wait_for(self._stop.wait(), timeout=interval_seconds)

    def stop(self) -> None:
        self._stop.set()

    async def sweep(self, *, now: datetime | None = None) -> dict[str, int]:
        """Run every cleanup rule. Returns what it did, for logging and tests."""

        moment = now or datetime.now(UTC)
        protected = await self._protect_live_checkpoints(moment)
        jobs_redacted = await self._redact_finished_jobs(moment)
        approvals_redacted = await self._redact_settled_approvals(moment)
        # The store already deletes anything past `expires_at`; that field *is*
        # the retention mechanism. So cleanup's job is to get the expiries
        # right first and then let the store's own pass do the deleting, rather
        # than to introduce a second way for records to disappear.
        pruned = await self._store.prune_expired(moment)
        return {
            "checkpoints_extended": protected,
            "jobs_redacted": jobs_redacted,
            "approvals_redacted": approvals_redacted,
            "records_pruned": pruned,
        }

    async def _protect_live_checkpoints(self, now: datetime) -> int:
        """Push out the expiry of checkpoints belonging to unfinished jobs.

        Dropping one of those would protect nothing — the running job holds the
        same working state in memory — and would cost exactly the resumption
        the checkpoint exists for. The extension is deliberately one horizon at
        a time rather than "never", so a job that dies without reaching a
        terminal state still lets its checkpoint go eventually.
        """

        extended = 0
        for stored in await self._store.list(CompanionCheckpointRecord):
            record: CompanionCheckpointRecord = stored.record  # type: ignore[assignment]
            if record.expires_at is None or now < record.expires_at:
                continue
            owner = await self._store.get(CompanionJobRecord, record.job_id)
            if owner is None:
                continue
            job: CompanionJobRecord = owner.record  # type: ignore[assignment]
            if job.status in jobs.TERMINAL:
                continue
            updated = record.model_copy(
                update={
                    "expires_at": checkpoint_expiry(record.sensitivity, now=now),
                    "updated_at": record.updated_at,
                }
            )
            if await self._save(updated, stored.revision):
                extended += 1
        return extended

    async def _redact_finished_jobs(self, now: datetime) -> int:
        """Strip payload references from finished jobs, keeping the history."""

        redacted = 0
        for stored in await self._store.list(CompanionJobRecord):
            record: CompanionJobRecord = stored.record  # type: ignore[assignment]
            if record.status not in jobs.TERMINAL:
                continue
            policy = retention_for(record.sensitivity)
            horizon = record.updated_at + timedelta(seconds=policy.payload_seconds)
            if now < horizon:
                continue
            if record.result_ref is None and record.checkpoint_ref is None:
                if record.progress_summary is None:
                    continue
            # Everything cleared here is a *reference to* or a *summary of*
            # content. The workload, attempt history, outcome, failure code and
            # trace id are untouched, because those are what audit needs and
            # none of them say what the request was about.
            updated = record.model_copy(
                update={
                    "result_ref": None,
                    "checkpoint_ref": None,
                    "progress_summary": None,
                    "updated_at": record.updated_at,
                }
            )
            if await self._save(updated, stored.revision):
                redacted += 1
        return redacted

    async def _redact_settled_approvals(self, now: datetime) -> int:
        """Drop the stored arguments once a mutation can no longer happen."""

        redacted = 0
        for stored in await self._store.list(CompanionApprovalRecord):
            record: CompanionApprovalRecord = stored.record  # type: ignore[assignment]
            if record.status in _LIVE_APPROVALS:
                # Still honourable. Its arguments are the mutation itself.
                continue
            policy = retention_for(record.sensitivity)
            horizon = record.updated_at + timedelta(seconds=policy.payload_seconds)
            if now < horizon or not record.arguments:
                continue
            # The summary stays: "the owner was asked to set the heater to 18
            # and said no" is the audit fact, and it is already plain language
            # written for a human rather than a payload.
            updated = record.model_copy(
                update={"arguments": {}, "updated_at": record.updated_at}
            )
            if await self._save(updated, stored.revision):
                redacted += 1
        return redacted

    async def _save(self, record, revision: int) -> bool:
        try:
            await self._store.save(record, expected_revision=revision)
        except ConcurrentRecordUpdate:
            # Something is actively working on it. Cleanup is never urgent
            # enough to fight a live writer; the next sweep will catch it.
            logger.debug("retention skipped %s, changed under us", record.id)
            return False
        return True
