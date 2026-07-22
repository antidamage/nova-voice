from __future__ import annotations

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


class ConversationRecord(DurableModel):
    status: ConversationState = ConversationState.ACTIVE
    room_id: str = Field(min_length=1, max_length=120)
    participant_ids: tuple[str, ...] = ()
    active_goal_ids: tuple[str, ...] = ()
    last_event_id: str | None = None


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


class MemoryReferenceRecord(DurableModel):
    memory_id: str
    memory_type: str
    provider: str
    provenance_revision: str
    audience: tuple[str, ...] = ()
    sensitivity: Literal["normal", "sensitive", "restricted"] = "normal"


class AuditRecord(DurableModel):
    actor_id: str
    action: str
    object_type: str
    object_id: str
    prior_revision: int | None = None
    resulting_revision: int
    detail: dict[str, Any] = Field(default_factory=dict)
