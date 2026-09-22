"""Tool callbacks that survive the socket they were asked on.

The in-memory loop in `session.py` is right for an ordinary voice turn: if the
phone drops, the turn falls back and nobody needs to remember the callback. A
*durable* job is different — it can outlive several sockets — so its callbacks
need a record, for two reasons that are easy to state and easy to get wrong:

* **A retry on either side must not run the tool twice.** The device reconnects
  and re-asks; Iridium restarts mid-execution. Both are ordinary, and both must
  resolve to one execution.
* **A result must reach the attempt that asked for it.** By the time a slow
  tool answers, the job may be owned by a different attempt entirely — after a
  reclaim, a rejection, or a supersede. Delivering the answer there would be
  feeding one reasoning run's working state into another's.

Policy is untouched by any of this. Callbacks still go through
`CapabilityRegistry` and `ToolPolicy` exactly as a locally planned action does;
this only decides whether a call happens and where its answer goes.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta

from nova_voice.companion.sensitivity import retention_for
from nova_voice.durable.models import CompanionCallbackRecord, CompanionCallbackState
from nova_voice.durable.store import ConcurrentRecordUpdate, DurableAgentStore

logger = logging.getLogger(__name__)

State = CompanionCallbackState


class CallbackError(RuntimeError):
    pass


# Runs the tool and returns (ok, code, message, observed).
Execute = Callable[[CompanionCallbackRecord], Awaitable[tuple[bool, str, str, dict | None]]]


class CompanionCallbackLedger:
    def __init__(self, store: DurableAgentStore, *, execute: Execute) -> None:
        self._store = store
        self._execute = execute

    async def request(
        self,
        *,
        job_id: str,
        attempt_id: str,
        call_id: str,
        provider: str,
        tool: str,
        arguments: dict,
        sensitivity: str = "personal",
        now: datetime | None = None,
    ) -> CompanionCallbackRecord:
        """Record a callback request, or hand back the one already recorded.

        The device re-asking after a reconnect is the common case, not an
        error — it has no way of knowing whether its first request survived. So
        a repeat of the same ``call_id`` returns the existing record, including
        its result if it already has one, rather than starting a second call.
        """

        existing = await self.get(call_id)
        if existing is not None:
            return existing
        moment = now or datetime.now(UTC)
        policy = retention_for(sensitivity)  # type: ignore[arg-type]
        record = CompanionCallbackRecord(
            id=call_id,
            job_id=job_id,
            attempt_id=attempt_id,
            call_id=call_id,
            provider=provider,
            tool=tool,
            arguments=dict(arguments),
            sensitivity=sensitivity,  # type: ignore[arg-type]
            # Derived from the call, not random, so the same call re-requested
            # after a crash carries the same key into the executor.
            idempotency_key=f"{job_id}:{call_id}",
            created_at=moment,
            updated_at=moment,
            expires_at=moment + timedelta(seconds=policy.payload_seconds),
        )
        stored = await self._store.create(record)
        return stored.record  # type: ignore[return-value]

    async def get(self, call_id: str) -> CompanionCallbackRecord | None:
        stored = await self._store.get(CompanionCallbackRecord, call_id)
        return stored.record if stored else None  # type: ignore[return-value]

    async def execute(self, call_id: str) -> CompanionCallbackRecord:
        """Run a pending callback exactly once.

        Same ordering as the approval gate, for the same reason: the record is
        claimed as ``EXECUTING`` before the tool is touched, so a crash in
        between is visible as an interrupted call rather than as one that never
        started. The idempotency key is already stored at that point, so a
        recovery retry cannot become a second write.
        """

        stored = await self._store.get(CompanionCallbackRecord, call_id)
        if stored is None:
            raise CallbackError("unknown callback")
        record: CompanionCallbackRecord = stored.record  # type: ignore[assignment]

        if record.status in (State.COMPLETED, State.FAILED, State.SUPERSEDED):
            return record
        if record.status is State.EXECUTING:
            # Another worker holds it, or a crashed one did. Recovery owns that
            # case; running it here would be the duplicate we are avoiding.
            return record

        claimed = await self._save(record, stored.revision, status=State.EXECUTING)
        try:
            ok, code, message, observed = await self._execute(claimed)
        except Exception as error:  # noqa: BLE001 - recorded, not swallowed
            logger.exception("companion callback failed call=%s", call_id)
            return await self._reload_and_save(
                call_id,
                status=State.FAILED,
                ok=False,
                code="backend_error",
                message=str(error)[:1000],
                completed_at=datetime.now(UTC),
            )
        return await self._reload_and_save(
            call_id,
            status=State.COMPLETED if ok else State.FAILED,
            ok=ok,
            code=code,
            message=message,
            observed=observed,
            completed_at=datetime.now(UTC),
        )

    async def deliverable(
        self, call_id: str, *, current_attempt_id: str | None
    ) -> CompanionCallbackRecord | None:
        """The result, but only if the attempt that asked for it still owns the job.

        Returns None when it does not, having marked the callback superseded —
        an answer computed for one reasoning run must never be fed into
        another's working state.
        """

        record = await self.get(call_id)
        if record is None:
            return None
        if current_attempt_id is not None and record.attempt_id == current_attempt_id:
            return record
        if record.status not in (State.SUPERSEDED,):
            logger.info(
                "companion callback %s belonged to attempt %s, not %s; discarding",
                call_id,
                record.attempt_id,
                current_attempt_id,
            )
            await self._reload_and_save(call_id, status=State.SUPERSEDED)
        return None

    async def recover_interrupted(self) -> tuple[str, ...]:
        """Finish callbacks a restart caught mid-execution."""

        resumed: list[str] = []
        for stored in await self._store.list(
            CompanionCallbackRecord, status=State.EXECUTING.value
        ):
            record: CompanionCallbackRecord = stored.record  # type: ignore[assignment]
            await self._save(record, stored.revision, status=State.PENDING)
            await self.execute(record.id)
            resumed.append(record.id)
        return tuple(resumed)

    async def for_job(self, job_id: str) -> tuple[CompanionCallbackRecord, ...]:
        stored = await self._store.list(CompanionCallbackRecord, parent_id=job_id)
        return tuple(item.record for item in stored)  # type: ignore[misc]

    # -- internals ------------------------------------------------------------

    async def _save(self, record, revision: int, **changes) -> CompanionCallbackRecord:
        updated = record.model_copy(update={**changes, "updated_at": datetime.now(UTC)})
        written = await self._store.save(updated, expected_revision=revision)
        return written.record  # type: ignore[return-value]

    async def _reload_and_save(self, call_id: str, **changes) -> CompanionCallbackRecord:
        """Write against the revision as it is *now*, not as it was before the call.

        A tool can take seconds, and the record may have been touched in the
        meantime. Re-reading first means the write reflects the current record
        rather than failing on a revision that has moved on for reasons that do
        not conflict with what we are recording.
        """

        stored = await self._store.get(CompanionCallbackRecord, call_id)
        if stored is None:
            raise CallbackError("callback vanished mid-execution")
        try:
            return await self._save(stored.record, stored.revision, **changes)
        except ConcurrentRecordUpdate:
            current = await self.get(call_id)
            if current is None:
                raise CallbackError("callback vanished mid-write") from None
            return current
