"""Own the live companion session and everything waiting on it.

Callers never hold a WebSocket. They ask this manager to offer a job, await a
result, or dispatch a personal call, and every one of those waits is registered
here so a disconnect can terminate all of them at once. That is the property
that matters: a phone that drops off mid-job must not leave a voice turn, a
durable plan or a background pass waiting forever on a future nobody will ever
resolve.

One session is current at a time. A newer authenticated connection supersedes
the older one, because a phone that has just changed network is far more likely
to be the live one than a socket that has not yet noticed it is dead.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from nova_voice.companion.auth import AuthenticatedIdentity
from nova_voice.companion.protocol import (
    ApprovalRequest,
    CompanionHello,
    CompanionTelemetry,
    CompanionWorkload,
    JobCancel,
    JobEnvelope,
    JobOffer,
    JobProgress,
    Locality,
    PersonalCall,
    PersonalResult,
    Sensitivity,
    ToolCall,
    ToolResultMessage,
    serialize,
)
from nova_voice.companion.tiers import CompanionTier, TierThresholds, TierTracker

logger = logging.getLogger(__name__)

SendJson = Callable[[dict], Awaitable[None]]
# Runs one companion-requested action through Iridium's own execution gate.
ExecuteToolCall = Callable[[ToolCall], Awaitable[ToolResultMessage]]


@dataclass(frozen=True)
class OfferOutcome:
    accepted: bool
    attempt_id: str
    # Machine-readable when the companion declined, so the router can classify
    # a rejection (ordinary flow) apart from a failure (health-affecting).
    reason: str | None = None
    retry_after_seconds: float | None = None


@dataclass(frozen=True)
class JobOutcome:
    ok: bool
    result: dict | None = None
    # "rejected"/"timeout"/"disconnected"/"failed"/"cancelled"/"invalid"
    failure: str | None = None
    detail: str | None = None


@dataclass
class _Attempt:
    job_id: str
    attempt_id: str
    workload: CompanionWorkload
    accept: asyncio.Future[OfferOutcome]
    result: asyncio.Future[JobOutcome]
    callbacks_remaining: int
    started_at: float
    last_progress_sequence: int = -1
    stage: str = "offered"


@dataclass
class CompanionSession:
    session_id: str
    identity: AuthenticatedIdentity
    locality: Locality
    send: SendJson
    hello: CompanionHello
    tiers: TierTracker
    telemetry: CompanionTelemetry
    connected_at: float
    last_heartbeat: float
    attempts: dict[str, _Attempt] = field(default_factory=dict)
    # None resolves these when the session goes away, so an awaiting caller
    # gets "no answer" rather than a cancellation it did not ask for.
    personal_calls: dict[str, asyncio.Future[PersonalResult | None]] = field(default_factory=dict)
    approvals: dict[str, asyncio.Future[bool | None]] = field(default_factory=dict)

    @property
    def roles(self) -> tuple[str, ...]:
        return self.identity.roles

    def attempt_for_job(self, job_id: str, attempt_id: str) -> _Attempt | None:
        attempt = self.attempts.get(attempt_id)
        if attempt is None or attempt.job_id != job_id:
            return None
        return attempt


@dataclass(frozen=True)
class SessionSnapshot:
    """Everything eligibility and the dashboard need, with no personal data."""

    connected: bool
    session_id: str | None
    identity: str | None
    roles: tuple[str, ...]
    locality: Locality
    tier: CompanionTier
    tier_reason: str
    workloads: frozenset[CompanionWorkload]
    personal_tools: frozenset[str]
    schema_versions: tuple[int, ...]
    app_version: str | None
    os_version: str | None
    telemetry_age_seconds: float | None
    last_heartbeat_age_seconds: float | None
    active_attempts: int
    hot_context_tokens: int


class CompanionSessionManager:
    def __init__(
        self,
        *,
        execute_tool_call: ExecuteToolCall | None = None,
        thresholds: TierThresholds | None = None,
        callback_cap: int = 12,
        accept_timeout_seconds: float = 1.5,
    ) -> None:
        self._execute_tool_call = execute_tool_call
        self._thresholds = thresholds or TierThresholds()
        self._callback_cap = callback_cap
        self._accept_timeout = accept_timeout_seconds
        self._session: CompanionSession | None = None
        self._tool_tasks: set[asyncio.Task] = set()

    def bind_tool_executor(self, execute_tool_call: ExecuteToolCall) -> None:
        """Late-bind execution: the service owns it and is constructed last."""

        self._execute_tool_call = execute_tool_call

    # -- lifecycle ------------------------------------------------------------

    def register(
        self,
        *,
        identity: AuthenticatedIdentity,
        hello: CompanionHello,
        locality: Locality,
        send: SendJson,
    ) -> CompanionSession:
        now = time.monotonic()
        tiers = TierTracker(thresholds=self._thresholds)
        tiers.update(hello.telemetry, now=now)
        session = CompanionSession(
            session_id=uuid.uuid4().hex,
            identity=identity,
            locality=locality,
            send=send,
            hello=hello,
            tiers=tiers,
            telemetry=hello.telemetry,
            connected_at=now,
            last_heartbeat=now,
        )
        previous = self._session
        self._session = session
        if previous is not None:
            self._abandon(previous, "disconnected", "a newer companion session superseded this one")
        logger.info(
            "companion session registered id=%s identity=%s locality=%s roles=%s",
            session.session_id,
            identity.identity,
            locality,
            ",".join(identity.roles),
        )
        return session

    def release(self, session: CompanionSession) -> bool:
        """Drop a session and resolve everything waiting on it."""

        self._abandon(session, "disconnected", "the companion disconnected")
        if self._session is session:
            self._session = None
            logger.info("companion session released id=%s", session.session_id)
            return True
        return False

    def is_current(self, session: CompanionSession) -> bool:
        return self._session is session

    @property
    def current(self) -> CompanionSession | None:
        return self._session

    def snapshot(self, *, now: float | None = None) -> SessionSnapshot:
        moment = time.monotonic() if now is None else now
        session = self._session
        if session is None:
            return SessionSnapshot(
                connected=False,
                session_id=None,
                identity=None,
                roles=(),
                locality="other",
                tier=CompanionTier.OFF,
                tier_reason="no companion session",
                workloads=frozenset(),
                personal_tools=frozenset(),
                schema_versions=(),
                app_version=None,
                os_version=None,
                telemetry_age_seconds=None,
                last_heartbeat_age_seconds=None,
                active_attempts=0,
                hot_context_tokens=4096,
            )
        decision = session.tiers.current(now=moment)
        return SessionSnapshot(
            connected=True,
            session_id=session.session_id,
            identity=session.identity.identity,
            roles=session.roles,
            locality=session.locality,
            tier=decision.tier,
            tier_reason=decision.reason,
            workloads=frozenset(session.hello.workloads),
            personal_tools=frozenset(session.hello.personal_tools),
            schema_versions=tuple(session.hello.schema_versions),
            app_version=session.hello.app_version,
            os_version=session.hello.os_version,
            telemetry_age_seconds=(
                moment - session.tiers.last_update if session.tiers.last_update else None
            ),
            last_heartbeat_age_seconds=moment - session.last_heartbeat,
            active_attempts=len(session.attempts),
            hot_context_tokens=session.telemetry.models.hot_context_tokens,
        )

    # -- inbound --------------------------------------------------------------

    def handle_telemetry(self, session: CompanionSession, telemetry: CompanionTelemetry) -> None:
        session.telemetry = telemetry
        decision = session.tiers.update(telemetry, now=time.monotonic())
        if decision.changed:
            logger.info(
                "companion tier changed session=%s tier=%s reason=%s",
                session.session_id,
                decision.tier.value,
                decision.reason,
            )

    def handle_heartbeat(self, session: CompanionSession) -> None:
        session.last_heartbeat = time.monotonic()

    def handle_accept(self, session: CompanionSession, job_id: str, attempt_id: str) -> None:
        attempt = session.attempt_for_job(job_id, attempt_id)
        if attempt is None or attempt.accept.done():
            return
        attempt.stage = "accepted"
        attempt.accept.set_result(OfferOutcome(accepted=True, attempt_id=attempt_id))

    def handle_reject(
        self,
        session: CompanionSession,
        job_id: str,
        attempt_id: str,
        reason: str,
        retry_after_seconds: float | None = None,
    ) -> None:
        attempt = session.attempt_for_job(job_id, attempt_id)
        if attempt is None or attempt.accept.done():
            return
        attempt.stage = "rejected"
        attempt.accept.set_result(
            OfferOutcome(
                accepted=False,
                attempt_id=attempt_id,
                reason=reason,
                retry_after_seconds=retry_after_seconds,
            )
        )
        self._resolve(attempt, JobOutcome(ok=False, failure="rejected", detail=reason))
        session.attempts.pop(attempt_id, None)

    def handle_progress(self, session: CompanionSession, progress: JobProgress) -> None:
        attempt = session.attempt_for_job(progress.job_id, progress.attempt_id)
        if attempt is None:
            return
        if progress.sequence <= attempt.last_progress_sequence:
            # Progress is monotonic within an attempt; a lower sequence is a
            # duplicate or reordered frame carrying nothing new.
            return
        attempt.last_progress_sequence = progress.sequence
        attempt.stage = progress.stage

    def handle_result(
        self, session: CompanionSession, job_id: str, attempt_id: str, result: dict
    ) -> None:
        attempt = session.attempt_for_job(job_id, attempt_id)
        if attempt is None:
            # A superseded attempt answering late. Recorded and ignored — it
            # must never overwrite whatever replaced it.
            logger.info(
                "companion late result ignored session=%s job=%s attempt=%s",
                session.session_id,
                job_id,
                attempt_id,
            )
            return
        self._resolve(attempt, JobOutcome(ok=True, result=result))
        session.attempts.pop(attempt_id, None)

    def handle_failed(
        self,
        session: CompanionSession,
        job_id: str,
        attempt_id: str,
        reason: str,
        detail: str | None,
    ) -> None:
        attempt = session.attempt_for_job(job_id, attempt_id)
        if attempt is None:
            return
        if not attempt.accept.done():
            attempt.accept.set_result(
                OfferOutcome(accepted=False, attempt_id=attempt_id, reason=reason)
            )
        self._resolve(attempt, JobOutcome(ok=False, failure="failed", detail=detail or reason))
        session.attempts.pop(attempt_id, None)

    def handle_personal_result(self, session: CompanionSession, result: PersonalResult) -> None:
        future = session.personal_calls.pop(result.call_id, None)
        if future is not None and not future.done():
            future.set_result(result)

    def handle_approval_response(
        self, session: CompanionSession, approval_id: str, approved: bool
    ) -> None:
        future = session.approvals.pop(approval_id, None)
        if future is not None and not future.done():
            future.set_result(approved)

    def dispatch_tool_call(self, session: CompanionSession, call: ToolCall) -> None:
        """Run a companion tool request without blocking the receive loop."""

        attempt = session.attempt_for_job(call.job_id, call.attempt_id)
        if attempt is None:
            task = asyncio.create_task(
                self._send_tool_error(session, call, "invalid", "no such running attempt")
            )
        elif attempt.callbacks_remaining <= 0:
            task = asyncio.create_task(
                self._send_tool_error(session, call, "blocked", "callback budget exhausted")
            )
        elif self._execute_tool_call is None:
            task = asyncio.create_task(
                self._send_tool_error(session, call, "blocked", "tool execution is not wired up")
            )
        else:
            attempt.callbacks_remaining -= 1
            task = asyncio.create_task(self._run_tool_call(session, call))
        self._tool_tasks.add(task)
        task.add_done_callback(self._tool_tasks.discard)

    # -- outbound -------------------------------------------------------------

    async def offer(
        self,
        *,
        workload: CompanionWorkload,
        payload: dict,
        idempotency_key: str,
        input_revision: str,
        result_schema: str,
        trace_id: str,
        complete_deadline_seconds: float,
        sensitivity: Sensitivity = "ordinary",
        callback_budget: int | None = None,
    ) -> tuple[CompanionSession, _Attempt, OfferOutcome] | None:
        """Offer one job. None means there was no session to offer it to."""

        session = self._session
        if session is None:
            return None
        loop = asyncio.get_running_loop()
        job_id = uuid.uuid4().hex
        attempt_id = uuid.uuid4().hex
        now = datetime.now(UTC)
        envelope = JobEnvelope(
            jobId=job_id,
            attemptId=attempt_id,
            idempotencyKey=idempotency_key,
            workload=workload,
            inputRevision=input_revision,
            resultSchema=result_schema,
            sensitivity=sensitivity,
            locality=session.locality,
            traceId=trace_id,
            createdAt=now,
            acceptDeadline=now + timedelta(seconds=self._accept_timeout),
            completeDeadline=now + timedelta(seconds=complete_deadline_seconds),
        )
        budget = min(callback_budget or self._callback_cap, self._callback_cap)
        attempt = _Attempt(
            job_id=job_id,
            attempt_id=attempt_id,
            workload=workload,
            accept=loop.create_future(),
            result=loop.create_future(),
            callbacks_remaining=budget,
            started_at=time.monotonic(),
        )
        session.attempts[attempt_id] = attempt
        offer = JobOffer(
            envelope=envelope,
            payload=payload,
            callbackBudget=budget,
            contextTokens=session.telemetry.models.hot_context_tokens,
        )
        try:
            await session.send(serialize(offer))
        except Exception:
            session.attempts.pop(attempt_id, None)
            logger.debug("companion offer could not be sent workload=%s", workload)
            return None
        try:
            outcome = await asyncio.wait_for(attempt.accept, timeout=self._accept_timeout)
        except TimeoutError:
            session.attempts.pop(attempt_id, None)
            self._resolve(attempt, JobOutcome(ok=False, failure="timeout", detail="accept timeout"))
            await self._cancel_quietly(session, job_id, attempt_id, "deadline")
            return session, attempt, OfferOutcome(
                accepted=False, attempt_id=attempt_id, reason="accept_timeout"
            )
        except asyncio.CancelledError:
            session.attempts.pop(attempt_id, None)
            await self._cancel_quietly(session, job_id, attempt_id, "user_cancelled")
            raise
        return session, attempt, outcome

    async def await_result(
        self, session: CompanionSession, attempt: _Attempt, *, timeout_seconds: float
    ) -> JobOutcome:
        """Wait for an accepted attempt to finish, bounded by its deadline."""

        try:
            return await asyncio.wait_for(
                asyncio.shield(attempt.result), timeout=timeout_seconds
            )
        except TimeoutError:
            session.attempts.pop(attempt.attempt_id, None)
            await self._cancel_quietly(session, attempt.job_id, attempt.attempt_id, "deadline")
            return JobOutcome(ok=False, failure="timeout", detail="workload deadline elapsed")
        except asyncio.CancelledError:
            session.attempts.pop(attempt.attempt_id, None)
            await self._cancel_quietly(
                session, attempt.job_id, attempt.attempt_id, "user_cancelled"
            )
            raise

    async def cancel(
        self, session: CompanionSession, attempt: _Attempt, reason: str = "superseded"
    ) -> None:
        session.attempts.pop(attempt.attempt_id, None)
        self._resolve(attempt, JobOutcome(ok=False, failure="cancelled", detail=reason))
        await self._cancel_quietly(session, attempt.job_id, attempt.attempt_id, reason)

    async def personal_call(
        self,
        tool: str,
        arguments: dict,
        *,
        max_items: int = 50,
        deadline_seconds: float = 15.0,
    ) -> PersonalResult | None:
        session = self._session
        if session is None:
            return None
        call_id = uuid.uuid4().hex
        loop = asyncio.get_running_loop()
        future: asyncio.Future[PersonalResult | None] = loop.create_future()
        session.personal_calls[call_id] = future
        message = PersonalCall(
            callId=call_id,
            tool=tool,
            arguments=arguments,
            maxItems=max_items,
            deadlineSeconds=deadline_seconds,
        )
        try:
            await session.send(serialize(message))
            return await asyncio.wait_for(future, timeout=deadline_seconds)
        except Exception:
            session.personal_calls.pop(call_id, None)
            return None

    async def request_approval(
        self, request: ApprovalRequest, *, timeout_seconds: float
    ) -> bool | None:
        """Ask the owner's device to approve a mutation. None means no answer."""

        session = self._session
        if session is None:
            return None
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool | None] = loop.create_future()
        session.approvals[request.approval_id] = future
        try:
            await session.send(serialize(request))
            return await asyncio.wait_for(future, timeout=timeout_seconds)
        except Exception:
            session.approvals.pop(request.approval_id, None)
            return None

    # -- internals ------------------------------------------------------------

    async def _run_tool_call(self, session: CompanionSession, call: ToolCall) -> None:
        assert self._execute_tool_call is not None
        try:
            result = await self._execute_tool_call(call)
        except Exception:
            logger.exception(
                "companion tool call failed provider=%s tool=%s", call.provider, call.tool
            )
            await self._send_tool_error(session, call, "backend_error", "execution failed")
            return
        try:
            await session.send(serialize(result))
        except Exception:
            logger.debug("companion tool result delivery failed call=%s", call.call_id)

    async def _send_tool_error(
        self, session: CompanionSession, call: ToolCall, code: str, message: str
    ) -> None:
        payload = ToolResultMessage(
            jobId=call.job_id,
            attemptId=call.attempt_id,
            callId=call.call_id,
            ok=False,
            code=code,
            message=message,
        )
        try:
            await session.send(serialize(payload))
        except Exception:
            logger.debug("companion tool error delivery failed call=%s", call.call_id)

    async def _cancel_quietly(
        self, session: CompanionSession, job_id: str, attempt_id: str, reason: str
    ) -> None:
        """Tell the companion to stop burning battery on work nobody wants."""

        try:
            await session.send(
                serialize(JobCancel(jobId=job_id, attemptId=attempt_id, reason=reason))
            )
        except Exception:
            pass

    @staticmethod
    def _resolve(attempt: _Attempt, outcome: JobOutcome) -> None:
        if not attempt.result.done():
            attempt.result.set_result(outcome)

    def _abandon(self, session: CompanionSession, failure: str, detail: str) -> None:
        for attempt in list(session.attempts.values()):
            if not attempt.accept.done():
                attempt.accept.set_result(
                    OfferOutcome(accepted=False, attempt_id=attempt.attempt_id, reason=failure)
                )
            self._resolve(attempt, JobOutcome(ok=False, failure=failure, detail=detail))
        session.attempts.clear()
        for future in list(session.personal_calls.values()):
            if not future.done():
                future.set_result(None)
        session.personal_calls.clear()
        for approval in list(session.approvals.values()):
            if not approval.done():
                approval.set_result(None)
        session.approvals.clear()
