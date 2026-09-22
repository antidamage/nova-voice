from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now() -> datetime:
    return datetime.now(UTC)


class DurableModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    id: str = Field(min_length=1, max_length=160)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime | None = None

    @model_validator(mode="after")
    def require_utc_ordered_timestamps(self) -> DurableModel:
        values = (self.created_at, self.updated_at, self.expires_at)
        if any(value is not None and value.utcoffset() is None for value in values):
            raise ValueError("durable timestamps must be timezone-aware")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot precede created_at")
        if self.expires_at is not None and self.expires_at < self.created_at:
            raise ValueError("expires_at cannot precede created_at")
        return self


class ConversationState(StrEnum):
    ACTIVE = "active"
    CLOSED = "closed"
    EXPIRED = "expired"


class GoalState(StrEnum):
    PLANNED = "planned"
    ACTIVE = "active"
    SATISFIED = "satisfied"
    PAUSED = "paused"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    FAILED = "failed"


class PlanState(StrEnum):
    PLANNED = "planned"
    ACTIVE = "active"
    SATISFIED = "satisfied"
    PAUSED = "paused"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    FAILED = "failed"


class PlanStepKind(StrEnum):
    TOOL = "tool"
    QUESTION = "question"
    APPROVAL = "approval"
    WAIT = "wait"
    TIMER = "timer"
    EVENT = "event"
    VERIFICATION = "verification"
    RETRY = "retry"
    COMPENSATION = "compensation"


class PlanStepState(StrEnum):
    PENDING = "pending"
    LEASED = "leased"
    RUNNING = "running"
    WAITING = "waiting"
    SATISFIED = "satisfied"
    PAUSED = "paused"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    FAILED = "failed"
    COMPENSATED = "compensated"


class ExecutionState(StrEnum):
    LEASED = "leased"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class HouseholdRole(StrEnum):
    OWNER = "owner"
    RECOGNIZED_HOUSEHOLD = "recognized_household"
    GUEST = "guest"


class ProactiveInterventionState(StrEnum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    DELIVERED = "delivered"
    DISMISSED = "dismissed"
    CANCELLED = "cancelled"


class AutomationState(StrEnum):
    DRAFT = "draft"
    SIMULATED = "simulated"
    APPROVED = "approved"
    ACTIVE = "active"
    PAUSED = "paused"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


class RolloutStage(StrEnum):
    FIXTURE = "fixture"
    REPLAY = "replay"
    SHADOW = "shadow"
    OWNER_CANARY = "owner_canary"
    HOUSEHOLD = "household"
    STANDING_AUTONOMY = "standing_autonomy"


class RolloutStatus(StrEnum):
    ACTIVE = "active"
    REVOKED = "revoked"


class CommitmentState(StrEnum):
    ACTIVE = "active"
    DUE = "due"
    DELIVERED = "delivered"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    MISSED = "missed"


class ResearchState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DialogueMessageState(StrEnum):
    PENDING = "pending"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


class ConversationRecord(DurableModel):
    status: ConversationState = ConversationState.ACTIVE
    room_id: str = Field(min_length=1, max_length=120)
    participant_ids: tuple[str, ...] = ()
    active_goal_ids: tuple[str, ...] = ()
    last_event_id: str | None = None


class ConversationTopicRecord(DurableModel):
    room_id: str = Field(min_length=1, max_length=120)
    participant_ids: tuple[str, ...] = ()
    topic_stack: tuple[str, ...] = ()
    summary: str = Field(default="", max_length=3000)
    unresolved_references: tuple[str, ...] = ()
    open_questions: tuple[str, ...] = ()
    linked_goal_ids: tuple[str, ...] = ()
    discussion_depth: Literal["brief", "normal", "deep"] = "normal"
    deliberate_pauses: bool = False
    reflective_listening: bool = False
    disagreement_style: Literal["supportive", "candid"] = "supportive"
    humour_enabled: bool = True
    storytelling_enabled: bool = False
    last_turn_at: datetime

    @model_validator(mode="after")
    def validate_topic_record(self) -> ConversationTopicRecord:
        if self.last_turn_at.utcoffset() is None:
            raise ValueError("last turn time must be timezone-aware")
        for values in (
            self.participant_ids,
            self.topic_stack,
            self.unresolved_references,
            self.open_questions,
            self.linked_goal_ids,
        ):
            if len(set(values)) != len(values):
                raise ValueError("conversation continuity fields must be unique")
        return self


class RelationshipContinuityRecord(DurableModel):
    person_id: str = Field(min_length=1, max_length=160)
    narrative_summary: str = Field(default="", max_length=3000)
    explicit_preferences: dict[str, str] = Field(default_factory=dict)
    speaking_style: Literal["default", "brief", "detailed", "slow", "direct"] = "default"
    callback_topic_ids: tuple[str, ...] = ()
    provenance_conversation_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_relationship_continuity(self) -> RelationshipContinuityRecord:
        if len(set(self.callback_topic_ids)) != len(self.callback_topic_ids):
            raise ValueError("callback topic ids must be unique")
        if len(set(self.provenance_conversation_ids)) != len(self.provenance_conversation_ids):
            raise ValueError("relationship provenance ids must be unique")
        return self


class DialogueMessageRecord(DurableModel):
    sender_id: str
    recipient_scope: Literal["person", "household"]
    recipient_id: str | None = None
    recipient_name: str | None = None
    speech_act: Literal["tell", "ask"]
    content: str = Field(min_length=1, max_length=2000)
    source_conversation_id: str | None = None
    status: DialogueMessageState = DialogueMessageState.PENDING
    delivered_to: tuple[str, ...] = ()
    delivered_at: datetime | None = None

    @model_validator(mode="after")
    def validate_dialogue_message(self) -> DialogueMessageRecord:
        if self.recipient_scope == "person" and not (self.recipient_id or self.recipient_name):
            raise ValueError("person relay requires a recipient")
        if self.delivered_at is not None and self.delivered_at.utcoffset() is None:
            raise ValueError("dialogue delivery time must be timezone-aware")
        if len(set(self.delivered_to)) != len(self.delivered_to):
            raise ValueError("dialogue delivery recipients must be unique")
        return self


class EventRecord(DurableModel):
    conversation_id: str | None = None
    source: str = Field(min_length=1, max_length=120)
    kind: str = Field(min_length=1, max_length=120)
    cursor: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    payload_revision: str = Field(min_length=1, max_length=160)


class GoalRecord(DurableModel):
    conversation_id: str | None = None
    status: GoalState = GoalState.PLANNED
    summary: str = Field(min_length=1, max_length=500)
    owner_id: str | None = None
    plan_ids: tuple[str, ...] = ()
    terminal_reason: str | None = Field(default=None, max_length=500)


class PlanRecord(DurableModel):
    goal_id: str
    status: PlanState = PlanState.PLANNED
    step_ids: tuple[str, ...] = ()
    terminal_reason: str | None = Field(default=None, max_length=500)


class PlanStepRecord(DurableModel):
    plan_id: str
    kind: PlanStepKind
    status: PlanStepState = PlanStepState.PENDING
    order: int = Field(ge=0)
    depends_on: tuple[str, ...] = ()
    input: dict[str, Any] = Field(default_factory=dict)
    resources: tuple[str, ...] = ()
    parallel_safe: bool = False
    not_before: datetime | None = None
    max_attempts: int = Field(default=1, ge=1, le=20)
    attempt: int = Field(default=0, ge=0)
    event_key: str | None = Field(default=None, min_length=1, max_length=200)
    compensates_step_id: str | None = None
    result_revision: str | None = None
    terminal_reason: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def validate_step(self) -> PlanStepRecord:
        if self.id in self.depends_on:
            raise ValueError("a plan step cannot depend on itself")
        if len(set(self.depends_on)) != len(self.depends_on):
            raise ValueError("plan step dependencies must be unique")
        if len(set(self.resources)) != len(self.resources):
            raise ValueError("plan step resources must be unique")
        if self.not_before is not None and self.not_before.utcoffset() is None:
            raise ValueError("not_before must be timezone-aware")
        if self.kind in {PlanStepKind.WAIT, PlanStepKind.TIMER} and self.not_before is None:
            raise ValueError("wait and timer steps require not_before")
        if self.kind == PlanStepKind.EVENT and self.event_key is None:
            raise ValueError("event steps require event_key")
        if self.kind == PlanStepKind.COMPENSATION and self.compensates_step_id is None:
            raise ValueError("compensation steps require compensates_step_id")
        if self.attempt > self.max_attempts:
            raise ValueError("attempt cannot exceed max_attempts")
        return self


class ExecutionRecord(DurableModel):
    plan_id: str
    step_id: str
    status: ExecutionState = ExecutionState.LEASED
    idempotency_key: str = Field(min_length=1, max_length=240)
    attempt: int = Field(ge=1)
    lease_owner: str = Field(min_length=1, max_length=160)
    lease_token: str = Field(min_length=1, max_length=160)
    lease_expires_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result_revision: str | None = None
    error_code: str | None = Field(default=None, max_length=120)

    @model_validator(mode="after")
    def validate_execution(self) -> ExecutionRecord:
        if self.lease_expires_at.utcoffset() is None:
            raise ValueError("lease_expires_at must be timezone-aware")
        return self


class IdentityPolicyRecord(DurableModel):
    person_id: str = Field(min_length=1, max_length=160)
    role: HouseholdRole
    active: bool = True


class GrantSchedule(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    weekdays: tuple[int, ...] = Field(default=(), max_length=7)
    start_time: str | None = Field(default=None, pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    end_time: str | None = Field(default=None, pattern=r"^([01]\d|2[0-3]):[0-5]\d$")

    @model_validator(mode="after")
    def validate_window(self) -> GrantSchedule:
        if any(day < 0 or day > 6 for day in self.weekdays):
            raise ValueError("grant weekdays must be between 0 and 6")
        if len(set(self.weekdays)) != len(self.weekdays):
            raise ValueError("grant weekdays must be unique")
        if (self.start_time is None) != (self.end_time is None):
            raise ValueError("grant schedule requires both start_time and end_time")
        return self


class DelegationGrantRecord(DurableModel):
    grantor_id: str
    grantee_id: str
    capability: str
    target_scope: tuple[str, ...] = ()
    recipients: tuple[str, ...] = ()
    locations: tuple[str, ...] = ()
    schedule: GrantSchedule | None = None
    max_uses: int | None = Field(default=None, ge=1)
    uses: int = Field(default=0, ge=0)
    max_amount: float | None = Field(default=None, ge=0)
    spent_amount: float = Field(default=0, ge=0)
    currency: str | None = Field(default=None, min_length=3, max_length=3)
    notify_on_use: bool = True
    active: bool = True
    revoked_at: datetime | None = None

    @model_validator(mode="after")
    def validate_grant(self) -> DelegationGrantRecord:
        if self.revoked_at is not None and self.revoked_at.utcoffset() is None:
            raise ValueError("revoked_at must be timezone-aware")
        if self.uses and self.max_uses is not None and self.uses > self.max_uses:
            raise ValueError("grant uses cannot exceed max_uses")
        if (
            self.spent_amount
            and self.max_amount is not None
            and self.spent_amount > self.max_amount
        ):
            raise ValueError("grant spend cannot exceed max_amount")
        if self.currency is not None and self.max_amount is None:
            raise ValueError("grant currency requires max_amount")
        return self


class ProactiveInterventionRecord(DurableModel):
    goal_id: str | None = None
    event_id: str | None = None
    reason_code: str
    reason_detail: str = ""
    channel: Literal["voice", "dashboard", "notification"]
    status: ProactiveInterventionState
    deduplication_key: str
    room_id: str | None = None
    feedback: Literal["accepted", "dismissed", "redundant", "annoying"] | None = None
    delivered_at: datetime | None = None
    feedback_at: datetime | None = None


class AutomationRecord(DurableModel):
    owner_id: str
    summary: str
    trigger: dict[str, Any] = Field(default_factory=dict)
    proposed_actions: tuple[dict[str, Any], ...] = ()
    simulation: dict[str, Any] | None = None
    state: AutomationState = AutomationState.DRAFT
    approval_id: str | None = None
    activated_at: datetime | None = None
    rolled_back_at: datetime | None = None
    monitor_failures: int = Field(default=0, ge=0)


class RolloutEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: RolloutStage
    artifact_revision: str = Field(min_length=1, max_length=240)
    pins_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    eligible: bool
    scenario_runs: dict[str, str] = Field(default_factory=dict)
    reasons: tuple[str, ...] = ()


class RolloutEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    at: datetime = Field(default_factory=utc_now)
    actor_id: str = Field(min_length=1, max_length=160)
    action: Literal["created", "promoted", "revoked", "rolled_back"]
    from_stage: RolloutStage | None = None
    to_stage: RolloutStage
    evidence_revision: str | None = None
    reason: str | None = Field(default=None, max_length=500)


class RolloutRecord(DurableModel):
    owner_id: str = Field(min_length=1, max_length=160)
    component: str = Field(min_length=1, max_length=160)
    pins_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    stage: RolloutStage = RolloutStage.FIXTURE
    status: RolloutStatus = RolloutStatus.ACTIVE
    authority_scope: tuple[str, ...] = ()
    history: tuple[RolloutEvent, ...] = ()
    revoked_at: datetime | None = None

    @model_validator(mode="after")
    def validate_rollout(self) -> RolloutRecord:
        if self.revoked_at is not None and self.revoked_at.utcoffset() is None:
            raise ValueError("revoked_at must be timezone-aware")
        if self.status == RolloutStatus.REVOKED and self.revoked_at is None:
            raise ValueError("revoked rollout requires revoked_at")
        if self.stage == RolloutStage.STANDING_AUTONOMY and not self.authority_scope:
            raise ValueError("standing autonomy requires a bounded authority scope")
        return self


class CommitmentRecord(DurableModel):
    owner_id: str
    summary: str = Field(min_length=1, max_length=1000)
    status: CommitmentState = CommitmentState.ACTIVE
    due_at: datetime | None = None
    deadline: datetime | None = None
    recurrence: str | None = Field(default=None, max_length=500)
    wait_event_key: str | None = Field(default=None, max_length=200)
    channels: tuple[Literal["voice", "dashboard", "notification"], ...] = Field(
        default=("dashboard",), min_length=1
    )
    occurrence: int = Field(default=1, ge=1)
    missed_count: int = Field(default=0, ge=0)
    delivered_at: datetime | None = None
    completed_at: datetime | None = None
    continuation_device: str | None = Field(default=None, max_length=160)

    @model_validator(mode="after")
    def validate_commitment(self) -> CommitmentRecord:
        if self.due_at is None and self.wait_event_key is None:
            raise ValueError("commitment requires a due time or wait event")
        for value in (self.due_at, self.deadline, self.delivered_at, self.completed_at):
            if value is not None and value.utcoffset() is None:
                raise ValueError("commitment timestamps must be timezone-aware")
        if self.deadline and self.due_at and self.deadline < self.due_at:
            raise ValueError("commitment deadline cannot precede due time")
        if len(set(self.channels)) != len(self.channels):
            raise ValueError("commitment channels must be unique")
        return self


class ResearchRecord(DurableModel):
    owner_id: str
    query: str = Field(min_length=1, max_length=1000)
    status: ResearchState = ResearchState.QUEUED
    spoken_summary: str | None = Field(default=None, max_length=2000)
    detail: dict[str, Any] = Field(default_factory=dict)
    citations: tuple[str, ...] = ()
    uncertainty: Literal["low", "medium", "high"] = "high"
    backend: str | None = Field(default=None, max_length=80)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def validate_research(self) -> ResearchRecord:
        for value in (self.started_at, self.completed_at):
            if value is not None and value.utcoffset() is None:
                raise ValueError("research timestamps must be timezone-aware")
        if len(set(self.citations)) != len(self.citations):
            raise ValueError("research citations must be unique")
        return self


class CompanionJobState(StrEnum):
    """Every state a durable companion job can be in.

    Named rather than derived so an illegal move is a rejected transition
    instead of an unnoticed field write. The two that look redundant are not:
    ``FALLING_BACK`` records that the phone is finished with and Iridium has
    taken the work over — which is different from ``FAILED`` (nobody has it) and
    from ``RUNNING`` (the phone still holds it) — and ``EXPIRED`` distinguishes
    a deadline nobody was waiting on from a failure someone should look at.
    """

    QUEUED = "queued"
    OFFERED = "offered"
    ACCEPTED = "accepted"
    RUNNING = "running"
    WAITING_TOOL = "waiting_tool"
    WAITING_APPROVAL = "waiting_approval"
    FALLING_BACK = "falling_back"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class CompanionAttemptRecord(BaseModel):
    """One offer of a job to one device, kept even after it ends.

    History rather than current state: a job that succeeded on its third
    attempt should still show the two that did not, because "the phone rejected
    this twice on battery grounds before taking it" is the fact that explains a
    latency figure nobody can otherwise account for.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    attempt_id: str = Field(min_length=1, max_length=160)
    session_id: str | None = Field(default=None, max_length=160)
    started_at: datetime = Field(default_factory=utc_now)
    ended_at: datetime | None = None
    outcome: (
        Literal[
            "accepted",
            "rejected",
            "completed",
            "failed",
            "timeout",
            "disconnected",
            "cancelled",
            "invalid",
        ]
        | None
    ) = None
    detail: str | None = Field(default=None, max_length=400)

    @model_validator(mode="after")
    def validate_attempt(self) -> CompanionAttemptRecord:
        for value in (self.started_at, self.ended_at):
            if value is not None and value.utcoffset() is None:
                raise ValueError("companion attempt timestamps must be timezone-aware")
        if self.ended_at is not None and self.ended_at < self.started_at:
            raise ValueError("companion attempt cannot end before it started")
        return self


class CompanionJobRecord(DurableModel):
    """A reasoning job that outlives the socket it was offered on.

    The durable row is authoritative and the WebSocket frames are transport
    events, not state. That ordering is what lets a job survive a phone
    restart, a server restart, or both.

    ``input_revision`` and ``idempotency_key`` are immutable for the life of
    the job **including its local fallback**. A retry, a second attempt and the
    fallback all answer the same question under the same key, so no downstream
    write can be duplicated by the work changing hands.
    """

    workload: str = Field(min_length=1, max_length=64)
    status: CompanionJobState = CompanionJobState.QUEUED
    input_revision: str = Field(min_length=1, max_length=128)
    idempotency_key: str = Field(min_length=1, max_length=240)
    # Safe to log and to put in a metrics label; carries nothing about content.
    trace_id: str = Field(min_length=1, max_length=64)
    sensitivity: Literal["ordinary", "personal", "health", "location", "mutation"] = (
        "ordinary"
    )
    locality: Literal["home_lan", "tailnet", "other"] = "home_lan"

    attempts: tuple[CompanionAttemptRecord, ...] = ()
    # Which session currently owns the work, and until when. A lease that has
    # expired is reclaimable regardless of what the device believes.
    lease_owner: str | None = Field(default=None, max_length=160)
    lease_expires_at: datetime | None = None
    complete_deadline: datetime | None = None

    progress_stage: str = Field(default="queued", max_length=64)
    progress_fraction: float | None = Field(default=None, ge=0, le=1)
    # A summary, never the model's working text.
    progress_summary: str | None = Field(default=None, max_length=400)
    checkpoint_ref: str | None = Field(default=None, max_length=240)
    result_ref: str | None = Field(default=None, max_length=240)
    approval_id: str | None = Field(default=None, max_length=160)

    # True once the work has moved to Iridium. A late phone result for a job
    # that has fallen back is recorded and ignored, never applied.
    local_fallback: bool = False
    failure_code: str | None = Field(default=None, max_length=64)
    failure_detail: str | None = Field(default=None, max_length=400)

    @model_validator(mode="after")
    def validate_companion_job(self) -> CompanionJobRecord:
        for value in (self.lease_expires_at, self.complete_deadline):
            if value is not None and value.utcoffset() is None:
                raise ValueError("companion job timestamps must be timezone-aware")
        seen = [attempt.attempt_id for attempt in self.attempts]
        if len(set(seen)) != len(seen):
            raise ValueError("companion job attempts must be unique")
        # A lease with no owner, or an owner with no expiry, is a half-written
        # lease — the exact shape that lets two owners think they hold one job.
        if (self.lease_owner is None) != (self.lease_expires_at is None):
            raise ValueError("a companion lease needs both an owner and an expiry")
        return self


MAX_CHECKPOINT_BYTES = 64 * 1024


class CompanionCheckpointRecord(DurableModel):
    """Enough state to resume a long job without starting it over.

    Kept as its own record rather than a field on the job so it can expire on
    the payload clock while the job's structural history stays for audit — a
    checkpoint is working state, and working state for a personal-context job
    is exactly the thing that must not linger.

    Bounded hard. A checkpoint is a resumption aid, and one that grows without
    limit turns an optional optimisation into a storage leak that survives
    every restart.
    """

    job_id: str = Field(min_length=1, max_length=160)
    attempt_id: str = Field(min_length=1, max_length=160)
    stage: str = Field(min_length=1, max_length=64)
    payload: dict[str, Any] = Field(default_factory=dict)
    sensitivity: Literal["ordinary", "personal", "health", "location", "mutation"] = (
        "ordinary"
    )

    @model_validator(mode="after")
    def validate_checkpoint(self) -> CompanionCheckpointRecord:
        try:
            size = len(json.dumps(self.payload, default=str).encode("utf-8"))
        except (TypeError, ValueError) as error:
            raise ValueError("checkpoint payload must be serialisable") from error
        if size > MAX_CHECKPOINT_BYTES:
            raise ValueError(
                f"checkpoint payload is {size} bytes, over the {MAX_CHECKPOINT_BYTES} limit"
            )
        # A checkpoint with no expiry is a personal payload kept forever.
        if self.expires_at is None:
            raise ValueError("a companion checkpoint must expire")
        return self


class CompanionCallbackState(StrEnum):
    PENDING = "pending"
    # Handed to the registry. Recoverable because the idempotency key is
    # already stored before the tool is touched.
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    # The attempt that asked for this is no longer the one that owns the job,
    # so its answer is worthless. Recorded rather than deleted: "the phone
    # asked for this and we threw the answer away" is a fact worth having when
    # someone is working out why a job took two goes.
    SUPERSEDED = "superseded"


class CompanionCallbackRecord(DurableModel):
    """One tool call a reasoning job asked Iridium to make on its behalf.

    Persisted because the socket is not the system of record. A callback issued
    just before a disconnect must not run twice when the device reconnects and
    asks again, and its result must not be delivered to whichever attempt
    happens to hold the job by then.
    """

    job_id: str = Field(min_length=1, max_length=160)
    # What makes a late result detectable. A result is only ever applied to the
    # attempt that asked for it.
    attempt_id: str = Field(min_length=1, max_length=160)
    call_id: str = Field(min_length=1, max_length=64)
    provider: str = Field(min_length=1, max_length=64)
    tool: str = Field(min_length=1, max_length=128)
    arguments: dict[str, Any] = Field(default_factory=dict)
    status: CompanionCallbackState = CompanionCallbackState.PENDING
    idempotency_key: str = Field(min_length=1, max_length=240)
    sensitivity: Literal["ordinary", "personal", "health", "location", "mutation"] = (
        "personal"
    )
    # The observation the tool returned. Marked with its class and expiring on
    # that class's clock, because a calendar read's contents are exactly the
    # thing that must not outlive the job that needed them.
    observed: dict[str, Any] | None = None
    ok: bool | None = None
    code: str | None = Field(default=None, max_length=64)
    message: str | None = Field(default=None, max_length=1000)
    completed_at: datetime | None = None

    @model_validator(mode="after")
    def validate_companion_callback(self) -> CompanionCallbackRecord:
        if self.completed_at is not None and self.completed_at.utcoffset() is None:
            raise ValueError("companion callback timestamps must be timezone-aware")
        if self.expires_at is None:
            raise ValueError("a companion callback must expire")
        return self


class CompanionApprovalState(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"
    # Handed to the executor. A crash here is recoverable precisely because the
    # idempotency key is already stored: the retry cannot write twice.
    EXECUTING = "executing"
    EXECUTED = "executed"
    FAILED = "failed"


class CompanionApprovalRecord(DurableModel):
    """A mutation a companion proposed, and the owner's answer to it.

    Durable because the voice turn must not block waiting for a human. The turn
    ends with an acknowledgement, the phone shows a prompt, and the answer
    arrives whenever it arrives — possibly after a reconnect, a server restart,
    or both.

    The exact target is stored, not a description of it. Approving "turn off
    the lights" must execute the action that was described at the moment it was
    described, not whatever that phrase would resolve to later.
    """

    job_id: str | None = Field(default=None, max_length=160)
    provider: str = Field(min_length=1, max_length=64)
    tool: str = Field(min_length=1, max_length=128)
    arguments: dict[str, Any] = Field(default_factory=dict)
    # Plain language, checkable at a glance. This is what the owner actually
    # agrees to, so it has to describe the stored arguments and not paraphrase.
    summary: str = Field(min_length=1, max_length=400)
    status: CompanionApprovalState = CompanionApprovalState.PENDING
    idempotency_key: str = Field(min_length=1, max_length=240)
    # Single-use, and part of the signed material, so one "yes" cannot be
    # replayed against a later approval.
    nonce: str = Field(min_length=8, max_length=128)
    sensitivity: Literal["ordinary", "personal", "health", "location", "mutation"] = (
        "mutation"
    )
    # The answering deadline, which is NOT the store's `expires_at`.
    #
    # `expires_at` is retention: the store prunes past it. If the two were the
    # same field, an approval nobody answered would be *deleted* at the moment
    # it lapsed rather than recorded as expired — losing the audit fact that
    # something was proposed and never agreed to. So the proposal ages out at
    # `respond_by` and the record is kept for a while after.
    respond_by: datetime
    decided_at: datetime | None = None
    decided_by: str | None = Field(default=None, max_length=160)
    executed_at: datetime | None = None
    result_ref: str | None = Field(default=None, max_length=240)
    failure_detail: str | None = Field(default=None, max_length=400)

    @model_validator(mode="after")
    def validate_companion_approval(self) -> CompanionApprovalRecord:
        for value in (self.respond_by, self.decided_at, self.executed_at):
            if value is not None and value.utcoffset() is None:
                raise ValueError("companion approval timestamps must be timezone-aware")
        # An approval with no deadline is one that can be executed years later
        # against a house that has changed. Every proposal must age out.
        if self.expires_at is None:
            raise ValueError("a companion approval must have a retention horizon")
        if self.expires_at < self.respond_by:
            raise ValueError("a companion approval must outlive its answering deadline")
        return self


class BriefingScheduleRecord(DurableModel):
    owner_id: str
    period: Literal["morning", "evening"]
    local_time: str = Field(pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    timezone: str = Field(min_length=1, max_length=100)
    channels: tuple[Literal["voice", "dashboard", "notification"], ...] = ("dashboard",)
    enabled: bool = True
    last_local_date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")


class BriefingRecord(DurableModel):
    schedule_id: str
    owner_id: str
    period: Literal["morning", "evening"]
    local_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    summary: str = Field(min_length=1, max_length=3000)
    agenda: tuple[dict[str, Any], ...] = ()
    conflicts: tuple[dict[str, Any], ...] = ()
    preparation_prompts: tuple[str, ...] = ()


class EventSubscriptionRecord(DurableModel):
    owner_id: str
    summary: str = Field(min_length=1, max_length=1000)
    event_kind: str = Field(min_length=1, max_length=120)
    match: dict[str, Any] = Field(default_factory=dict)
    channels: tuple[Literal["voice", "dashboard", "notification"], ...] = ("dashboard",)
    one_shot: bool = True
    active: bool = True
    trigger_count: int = Field(default=0, ge=0)
    last_event_id: str | None = None
    triggered_at: datetime | None = None

    @model_validator(mode="after")
    def validate_subscription(self) -> EventSubscriptionRecord:
        if self.triggered_at is not None and self.triggered_at.utcoffset() is None:
            raise ValueError("subscription trigger time must be timezone-aware")
        return self


class MemoryReferenceRecord(DurableModel):
    memory_id: str
    memory_type: str
    provider: str
    provenance_revision: str
    audience: tuple[str, ...] = ()
    sensitivity: Literal["normal", "sensitive", "restricted"] = "normal"


class VisualContextRecord(DurableModel):
    owner_id: str = Field(min_length=1, max_length=160)
    audience: tuple[str, ...] = ()
    asset_id: str = Field(min_length=1, max_length=160)
    source_revision: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    kind: Literal["object_location", "walkthrough", "cross_device"]
    label: str = Field(min_length=1, max_length=240)
    summary: str = Field(min_length=1, max_length=4000)
    location: str | None = Field(default=None, max_length=240)
    device_id: str | None = Field(default=None, max_length=160)
    sensitivity: Literal["normal", "sensitive", "restricted"] = "normal"
    explicit_save: bool = False

    @model_validator(mode="after")
    def require_explicit_object_location(self) -> VisualContextRecord:
        if self.kind == "object_location" and not self.explicit_save:
            raise ValueError("object locations require an explicit save")
        return self


class AuditRecord(DurableModel):
    actor_id: str
    action: str
    object_type: str
    object_id: str
    prior_revision: int | None = None
    resulting_revision: int
    detail: dict[str, Any] = Field(default_factory=dict)
