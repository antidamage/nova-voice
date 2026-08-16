"""Persist companion jobs, and settle who owns what after things go wrong.

``jobs.py`` says which moves are legal; this applies them to the store and
handles the two ways reality diverges from the record:

* **A lease stops being renewed.** The device crashed, was force-quit, lost the
  network, or was suspended by iOS. The job is still marked as owned, and
  nothing will ever finish it. ``sweep`` returns those to the queue.
* **The two sides disagree about what is in flight.** After a reconnect the
  server may think a session owns jobs it has never heard of, and the device
  may still be working on a job the server gave up on. ``reconcile`` resolves
  both directions, because only fixing one leaves the other as a stuck job or a
  duplicate execution.

Every write goes through the store's optimistic-revision check. A conflict
means someone else moved the job while we were deciding, so the transition is
recomputed against the newer record rather than forced over the top of it —
forcing is how two owners come to think they hold one job.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from datetime import UTC, datetime

from nova_voice.companion import jobs
from nova_voice.durable.models import CompanionJobRecord, CompanionJobState
from nova_voice.durable.store import ConcurrentRecordUpdate, DurableAgentStore

logger = logging.getLogger(__name__)

Transition = Callable[[CompanionJobRecord], CompanionJobRecord]


class CompanionJobLedger:
    def __init__(
        self,
        store: DurableAgentStore,
        *,
        lease_seconds: float = 120.0,
        max_conflict_retries: int = 3,
    ) -> None:
        self._store = store
        self._lease_seconds = lease_seconds
        self._max_conflict_retries = max_conflict_retries

    @property
    def lease_seconds(self) -> float:
        return self._lease_seconds

    async def create(self, record: CompanionJobRecord) -> CompanionJobRecord:
        stored = await self._store.create(record)
        return stored.record  # type: ignore[return-value]

    async def get(self, job_id: str) -> CompanionJobRecord | None:
        stored = await self._store.get(CompanionJobRecord, job_id)
        return stored.record if stored else None  # type: ignore[return-value]

    async def apply(self, job_id: str, transition: Transition) -> CompanionJobRecord | None:
        """Read, transition, write — retrying against whoever got there first.

        ``transition`` is called again on each retry rather than its result
        being reused, which is the whole point: the legality of a move depends
        on the state it is applied to, so a move recomputed against the newer
        record may correctly become a no-op (idempotent) or raise (illegal).
        Reusing the first result would write a decision made about a record
        that no longer exists.
        """

        for attempt in range(self._max_conflict_retries):
            stored = await self._store.get(CompanionJobRecord, job_id)
            if stored is None:
                return None
            current: CompanionJobRecord = stored.record  # type: ignore[assignment]
            moved = transition(current)
            if moved is current:
                # A no-op transition: nothing to write, and writing anyway
                # would bump the revision and cause conflicts for no reason.
                return current
            try:
                written = await self._store.save(moved, expected_revision=stored.revision)
            except ConcurrentRecordUpdate:
                logger.debug(
                    "companion job %s changed under us, retrying (%d)", job_id, attempt + 1
                )
                continue
            return written.record  # type: ignore[return-value]
        logger.warning("companion job %s lost every write race; leaving it alone", job_id)
        return await self.get(job_id)

    # -- recovery -------------------------------------------------------------

    async def sweep(self, *, now: datetime | None = None) -> tuple[str, ...]:
        """Return orphaned jobs to the queue and expire ones past their deadline.

        Safe to run on a timer over the whole table: both operations return the
        record untouched when they do not apply, and `apply` does not write a
        no-op, so a quiet sweep costs reads and nothing else.
        """

        moment = now or datetime.now(UTC)
        touched: list[str] = []
        for stored in await self._store.list(CompanionJobRecord):
            record: CompanionJobRecord = stored.record  # type: ignore[assignment]
            if record.status in jobs.TERMINAL:
                continue
            if record.complete_deadline is not None and moment >= record.complete_deadline:
                # Expiry outranks reclaim: putting a job whose deadline has
                # already passed back on the queue would only offer it to
                # another device to fail at.
                await self.apply(record.id, jobs.expire)
                touched.append(record.id)
                continue
            if jobs.lease_expired(record, now=moment):
                await self.apply(record.id, lambda job: jobs.reclaim(job, now=moment))
                touched.append(record.id)
        return tuple(touched)

    async def reconcile(
        self, *, session_id: str, device_job_ids: Iterable[str]
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Settle what is actually in flight after a device reconnects.

        Returns ``(reclaimed, unknown_to_us)``.

        *We think it is running, the device has never heard of it* — the device
        restarted, so nothing will finish it. Reclaimed to the queue.

        *The device thinks it is running, we do not* — we already reclaimed or
        expired it, and it may now be owned by someone else. Returned to the
        caller to cancel **on the device**, because the alternative is the
        device finishing work whose result we will refuse, and, if it was
        reassigned, the same job being executed twice.
        """

        claimed = set(device_job_ids)
        ours: set[str] = set()
        reclaimed: list[str] = []

        for stored in await self._store.list(CompanionJobRecord):
            record: CompanionJobRecord = stored.record  # type: ignore[assignment]
            if record.lease_owner != session_id or record.status in jobs.TERMINAL:
                continue
            ours.add(record.id)
            if record.id in claimed:
                continue
            # Ours on paper, forgotten by the device that held it.
            moved = await self.apply(
                record.id,
                lambda job: jobs.reclaim(
                    job,
                    # Force the reclaim: the lease may not have expired yet, but
                    # the device has just told us it is not working on this, and
                    # waiting out a lease we know to be dead only delays the
                    # job by however long is left on it.
                    now=job.lease_expires_at or datetime.now(UTC),
                ),
            )
            if moved is not None and moved.status is CompanionJobState.QUEUED:
                reclaimed.append(record.id)

        unknown = tuple(sorted(claimed - ours))
        if unknown:
            logger.info(
                "companion session %s claims %d job(s) we do not own", session_id, len(unknown)
            )
        return tuple(reclaimed), unknown

    # -- fallback -------------------------------------------------------------

    async def fall_back(self, job_id: str, *, reason: str) -> CompanionJobRecord | None:
        """Move a job to Iridium under the same input revision and key.

        One step, one write. Doing it as "release the lease, then mark it
        local" would leave a window in which the job is owned by nobody and
        marked for nobody, which is exactly when a sweep or a reconnect would
        hand it back to a device.
        """

        current = await self.get(job_id)
        if current is None:
            return None
        if current.status in jobs.TERMINAL:
            # Already finished — by the phone, or by a cancellation. Falling
            # back now would run work that already has an answer.
            return current
        attempt = jobs.current_attempt(current)
        return await self.apply(
            job_id,
            lambda job: jobs.fall_back(
                job,
                reason=reason,
                attempt_id=attempt.attempt_id if attempt else None,
            ),
        )

    async def complete_locally(
        self, job_id: str, *, result_ref: str
    ) -> CompanionJobRecord | None:
        """Finish a job Iridium took over. A late phone result cannot undo this."""

        return await self.apply(
            job_id, lambda job: jobs.complete(job, attempt_id=None, result_ref=result_ref)
        )
