"""Ask the owner before a companion-proposed mutation happens.

`service.execute_companion_action` already refuses anything a ToolPolicy marks
confirmable, which is the safe half of the contract: a companion can never
silently perform a confirmable mutation. This is the other half — turning that
refusal into a question the owner can actually answer.

Three properties shape the design:

* **The voice turn must not block.** A human may answer in four seconds or
  four hours, and holding a turn open for that would wedge the assistant. The
  turn ends with an acknowledgement and the proposal outlives it.
* **An approval is not an execution.** The decision and the write are separate
  steps with a durable record between them, which is what lets a duplicate
  tap, a reconnect, or a server restart in the gap resolve to one write.
* **The stored target is what runs.** Approving "turn the heater off" executes
  the action described at the moment it was described, not whatever that
  sentence would resolve to later.
"""

from __future__ import annotations

import logging
import secrets
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta

from nova_voice.companion.auth import AuthenticatedIdentity, verify_approval
from nova_voice.durable.models import CompanionApprovalRecord, CompanionApprovalState
from nova_voice.durable.store import ConcurrentRecordUpdate, DurableAgentStore

logger = logging.getLogger(__name__)

State = CompanionApprovalState

# How long a decided approval is kept after its answering deadline, so
# "something was proposed and nobody agreed to it" survives as an audit fact
# rather than being pruned the moment it lapses.
RETENTION_AFTER_DEADLINE = timedelta(days=7)


class ApprovalError(RuntimeError):
    pass


# Something that runs the approved mutation and returns a reference to keep.
Execute = Callable[[CompanionApprovalRecord], Awaitable[str]]


class CompanionApprovalGate:
    def __init__(
        self,
        store: DurableAgentStore,
        *,
        execute: Execute,
        respond_within_seconds: float = 900.0,
    ) -> None:
        self._store = store
        self._execute = execute
        self._respond_within = respond_within_seconds

    # -- proposing ------------------------------------------------------------

    async def propose(
        self,
        *,
        provider: str,
        tool: str,
        arguments: dict,
        summary: str,
        job_id: str | None = None,
        idempotency_key: str | None = None,
        respond_within_seconds: float | None = None,
    ) -> CompanionApprovalRecord:
        """Record a proposed mutation and hand back what to send to the device."""

        window = respond_within_seconds or self._respond_within
        now = datetime.now(UTC)
        respond_by = now + timedelta(seconds=window)
        record = CompanionApprovalRecord(
            id=uuid.uuid4().hex,
            job_id=job_id,
            provider=provider,
            tool=tool,
            arguments=dict(arguments),
            summary=summary,
            idempotency_key=idempotency_key or uuid.uuid4().hex,
            # Single-use and unguessable. Part of the signed material, so one
            # captured "yes" cannot be replayed against a later proposal.
            nonce=secrets.token_urlsafe(24),
            respond_by=respond_by,
            expires_at=respond_by + RETENTION_AFTER_DEADLINE,
            created_at=now,
            updated_at=now,
        )
        stored = await self._store.create(record)
        return stored.record  # type: ignore[return-value]

    async def get(self, approval_id: str) -> CompanionApprovalRecord | None:
        stored = await self._store.get(CompanionApprovalRecord, approval_id)
        return stored.record if stored else None  # type: ignore[return-value]

    # -- deciding -------------------------------------------------------------

    async def decide(
        self,
        approval_id: str,
        *,
        approved: bool,
        identity: AuthenticatedIdentity,
        signature: str,
        now: datetime | None = None,
    ) -> CompanionApprovalRecord:
        """Apply a signed decision, once.

        A second answer — a duplicate tap, a redelivered frame, a reconnect
        that resends — returns the existing record rather than overwriting it.
        Letting a later "no" override an earlier "yes" would mean the answer
        depended on network timing, and letting a later "yes" override a "no"
        is worse.
        """

        stored = await self._store.get(CompanionApprovalRecord, approval_id)
        if stored is None:
            raise ApprovalError("unknown approval")
        record: CompanionApprovalRecord = stored.record  # type: ignore[assignment]
        moment = now or datetime.now(UTC)

        # Verified before the state checks, so a forged signature is reported
        # as forgery rather than as "already decided" — the two need to be
        # distinguishable when someone is looking at an audit trail.
        verify_approval(
            identity,
            approval_id=approval_id,
            approved=approved,
            nonce=record.nonce,
            signature=signature,
        )

        if record.status is not State.PENDING:
            return record
        if moment >= record.respond_by:
            return await self._write(
                stored,
                status=State.EXPIRED,
                failure_detail="the approval expired before it was answered",
            )
        return await self._write(
            stored,
            status=State.APPROVED if approved else State.DENIED,
            decided_at=moment,
            decided_by=identity.identity,
        )

    async def expire_lapsed(self, *, now: datetime | None = None) -> tuple[str, ...]:
        """Mark proposals nobody answered in time.

        Expiry is a decision the system makes, and it has to be *recorded* as
        one: a pending approval that merely stops being actionable would leave
        a caller waiting on an answer that will never come.
        """

        moment = now or datetime.now(UTC)
        lapsed: list[str] = []
        for stored in await self._store.list(
            CompanionApprovalRecord, status=State.PENDING.value
        ):
            record: CompanionApprovalRecord = stored.record  # type: ignore[assignment]
            if moment < record.respond_by:
                continue
            await self._write(
                stored,
                status=State.EXPIRED,
                failure_detail="the approval expired before it was answered",
            )
            lapsed.append(record.id)
        return tuple(lapsed)

    # -- executing ------------------------------------------------------------

    async def execute_approved(self, approval_id: str) -> CompanionApprovalRecord:
        """Run an approved mutation exactly once.

        The record moves to ``EXECUTING`` **before** the mutation runs, and
        that ordering is the whole guarantee. A crash between the two leaves an
        approval visibly mid-execution rather than looking un-run, so recovery
        can retry it knowing the stored idempotency key will collapse a
        duplicate write in the executor. The reverse order — run, then record —
        would make a crash indistinguishable from never having started, and the
        retry would be a genuine second write.
        """

        stored = await self._store.get(CompanionApprovalRecord, approval_id)
        if stored is None:
            raise ApprovalError("unknown approval")
        record: CompanionApprovalRecord = stored.record  # type: ignore[assignment]

        if record.status in (State.EXECUTED, State.FAILED):
            return record
        if record.status is not State.APPROVED:
            raise ApprovalError(f"approval is {record.status.value}, not approved")

        claimed = await self._write(stored, status=State.EXECUTING)
        reclaimed = await self._store.get(CompanionApprovalRecord, approval_id)
        assert reclaimed is not None
        try:
            result_ref = await self._execute(claimed)
        except Exception as error:  # noqa: BLE001 - recorded, not swallowed
            logger.exception("approved mutation failed approval=%s", approval_id)
            return await self._write(
                reclaimed,
                status=State.FAILED,
                failure_detail=str(error)[:400],
            )
        return await self._write(
            reclaimed,
            status=State.EXECUTED,
            executed_at=datetime.now(UTC),
            result_ref=result_ref,
        )

    async def recover_interrupted(self) -> tuple[str, ...]:
        """Finish approvals caught mid-execution by a restart.

        Safe to retry because the idempotency key was stored before the
        mutation was attempted.
        """

        resumed: list[str] = []
        for stored in await self._store.list(
            CompanionApprovalRecord, status=State.EXECUTING.value
        ):
            record: CompanionApprovalRecord = stored.record  # type: ignore[assignment]
            # Put it back to APPROVED so the ordinary path owns the retry, and
            # there is one place where execution happens.
            await self._write(stored, status=State.APPROVED)
            await self.execute_approved(record.id)
            resumed.append(record.id)
        return tuple(resumed)

    # -- internals ------------------------------------------------------------

    async def _write(self, stored, **changes) -> CompanionApprovalRecord:
        record: CompanionApprovalRecord = stored.record
        updated = record.model_copy(update={**changes, "updated_at": datetime.now(UTC)})
        try:
            written = await self._store.save(updated, expected_revision=stored.revision)
        except ConcurrentRecordUpdate:
            # Someone answered at the same moment. Theirs stands: a decision is
            # first-writer-wins by design, so re-reading is the right answer
            # rather than retrying our own write over the top.
            current = await self.get(record.id)
            if current is None:
                raise ApprovalError("approval vanished mid-write") from None
            return current
        return written.record  # type: ignore[return-value]
