"""Strict, versioned wire contract for the companion channel (protocol v1).

Everything crossing this socket is parsed into one of these models before it is
looked at. ``extra="forbid"`` and explicit bounds are the point: an unknown
message type, an unknown field, an out-of-range deadline or an oversized
payload is rejected deterministically at the edge and never reaches a model, a
capability provider or a durable job row.

Two invariants shape the message set:

* **Offer, then accept.** Iridium *offers* a job; the companion accepts or
  rejects it. Rejection is ordinary flow, not a failure. Once an attempt is
  accepted, Iridium does not run the same workload locally — the whole purpose
  is to free the single llama.cpp slot, and an eager local hedge would defeat
  it. The local path starts only on a terminal outcome for that attempt.
* **The companion plans; Iridium executes.** A ``tool_call`` is a *request*.
  It re-enters Iridium's capability registry, tool policy and household
  authority unchanged, and mutations become durable approval steps. The
  companion holds no dashboard credentials and cannot name a tool that was not
  advertised to it.

Frames are transport events. The durable job row is the authority on state;
progress is structured and monotonic, never parsed back out of log text.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

PROTOCOL_VERSION = 1
# Protocol versions this server will negotiate. A client outside the range is
# closed at the handshake rather than being half-supported.
SUPPORTED_PROTOCOL_VERSIONS = (1,)

# Hard ceilings enforced before parsing, so a payload bomb costs one length
# check rather than a full JSON parse and model validation.
MAX_FRAME_BYTES = 256 * 1024
MAX_RESULT_BYTES = 128 * 1024
MAX_TOOL_ARGUMENT_BYTES = 32 * 1024

# Workloads that may be routed. Closed so a configured route cannot name a
# workload that has no handler and no schema (NPT-003).
#
# These are exactly the five passes that occupy llama.cpp's single slot today.
# Research synthesis, briefing composition and automation drafting are absent
# on purpose: they are deterministic in this codebase, so there is no local
# implementation for a route to fall back to. Adding them means building that
# local stage first. See nova_voice.companion.workloads and
# docs/COMPANION-BASELINE.md.
CompanionWorkload = Literal[
    "interpret",
    "render_response",
    "confirm_objective",
    "extract_self_profile_update",
    "classify_icon",
]

# How much care a payload needs in logs, storage and retention (NPT-005).
Sensitivity = Literal["ordinary", "personal", "location", "health", "mutation"]

# Where the peer actually connected from, as classified by the server. The
# phone's own opinion is telemetry; this is the security-relevant value.
Locality = Literal["home_lan", "tailnet", "other"]

CompanionRole = Literal["companion", "satellite"]

RejectReason = Literal[
    "battery",
    "thermal",
    "memory",
    "model_unavailable",
    "busy",
    "unsupported_workload",
    "unsupported_schema",
    "deadline_too_short",
    "disabled",
]

FailureReason = Literal[
    "model_error",
    "invalid_result",
    "tool_limit",
    "timeout",
    "cancelled",
    "internal",
]


class CompanionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# -- envelope -----------------------------------------------------------------


class JobEnvelope(CompanionModel):
    """Identity and lifetime carried by every job-related frame.

    ``attempt_id`` is what makes a late result from a superseded attempt
    detectable: the job may be retried, but each offer creates a new attempt,
    and a frame naming a stale one is recorded as late and ignored.
    ``idempotency_key`` is stable across attempts and across a local fallback,
    so downstream writes cannot be duplicated by a retry.
    """

    job_id: str = Field(alias="jobId", min_length=1, max_length=64)
    attempt_id: str = Field(alias="attemptId", min_length=1, max_length=64)
    idempotency_key: str = Field(alias="idempotencyKey", min_length=1, max_length=128)
    workload: CompanionWorkload
    # Hash/revision of the exact input. A fallback reuses it unchanged.
    input_revision: str = Field(alias="inputRevision", min_length=1, max_length=128)
    schema_version: int = Field(default=1, ge=1, le=1000, alias="schemaVersion")
    result_schema: str = Field(alias="resultSchema", min_length=1, max_length=128)
    sensitivity: Sensitivity = "ordinary"
    locality: Locality = "home_lan"
    # Safe to log: correlates frames, job rows and audit entries without
    # carrying anything about what the request was.
    trace_id: str = Field(alias="traceId", min_length=1, max_length=64)
    created_at: datetime = Field(alias="createdAt")
    accept_deadline: datetime = Field(alias="acceptDeadline")
    complete_deadline: datetime = Field(alias="completeDeadline")


# -- authentication -----------------------------------------------------------


class AuthChallenge(CompanionModel):
    """Server to client: prove you hold the key for the identity you claim."""

    type: Literal["auth_challenge"] = "auth_challenge"
    nonce: str = Field(min_length=32, max_length=128)
    supported_versions: list[int] = Field(alias="supportedVersions")
    expires_at: datetime = Field(alias="expiresAt")


class AuthResponse(CompanionModel):
    type: Literal["auth_response"] = "auth_response"
    protocol_version: int = Field(alias="protocolVersion", ge=1, le=1000)
    announced_id: str = Field(alias="announcedId", min_length=1, max_length=64)
    roles: list[CompanionRole] = Field(min_length=1)
    # PEM leaf first, then any intermediates, validated against the household CA.
    certificate_chain: list[str] = Field(alias="certificateChain", min_length=1, max_length=4)
    # Base64 signature over the canonical challenge string. The signed material
    # covers the nonce, version, id and roles together, so none of them can be
    # swapped after the fact.
    signature: str = Field(min_length=1, max_length=2048)


class HelloAck(CompanionModel):
    type: Literal["hello_ack"] = "hello_ack"
    protocol_version: int = Field(alias="protocolVersion")
    authenticated_id: str = Field(alias="authenticatedId")
    locality: Locality
    session_id: str = Field(alias="sessionId")
    heartbeat_seconds: float = Field(default=20.0, gt=0, le=300, alias="heartbeatSeconds")


# -- capability and runtime state ---------------------------------------------


class ModelAvailability(CompanionModel):
    hot_available: bool = Field(default=False, alias="hotAvailable")
    hot_context_tokens: int = Field(
        default=4096, ge=512, le=1_000_000, alias="hotContextTokens"
    )
    # Optional local weights. Never a prerequisite for reachability.
    deep_available: bool = Field(default=False, alias="deepAvailable")
    deep_model_id: str | None = Field(default=None, max_length=128, alias="deepModelId")


class PermissionState(CompanionModel):
    """Per-capability permission, so one denial degrades only its own tools."""

    calendar: Literal["available", "denied", "restricted", "partial", "unknown"] = "unknown"
    reminders: Literal["available", "denied", "restricted", "partial", "unknown"] = "unknown"
    location: Literal["available", "denied", "restricted", "partial", "unknown"] = "unknown"
    health: Literal["available", "denied", "restricted", "partial", "unknown"] = "unknown"


class CompanionTelemetry(CompanionModel):
    battery: float = Field(default=1.0, ge=0, le=1)
    charging: bool = False
    low_power_mode: bool = Field(default=False, alias="lowPowerMode")
    thermal_state: Literal["nominal", "fair", "serious", "critical"] = Field(
        default="nominal", alias="thermalState"
    )
    app_state: Literal["foreground", "background", "inactive"] = Field(
        default="background", alias="appState"
    )
    models: ModelAvailability = Field(default_factory=ModelAvailability)
    permissions: PermissionState = Field(default_factory=PermissionState)
    # Diagnostic only. The server classifies locality from the actual peer
    # address; a phone claiming to be at home cannot unlock a home-LAN route.
    reported_network: str | None = Field(default=None, max_length=64, alias="reportedNetwork")


class CompanionHello(CompanionModel):
    type: Literal["hello"] = "hello"
    protocol_version: int = Field(alias="protocolVersion")
    schema_versions: list[int] = Field(default_factory=lambda: [1], alias="schemaVersions")
    display_name: str = Field(alias="displayName", min_length=1, max_length=64)
    roles: list[CompanionRole] = Field(min_length=1)
    app_version: str = Field(alias="appVersion", max_length=32)
    os_version: str = Field(alias="osVersion", max_length=32)
    workloads: list[CompanionWorkload] = Field(default_factory=list)
    personal_tools: list[str] = Field(
        default_factory=list, max_length=64, alias="personalTools"
    )
    telemetry: CompanionTelemetry = Field(default_factory=CompanionTelemetry)
    # Durable jobs this device believes it is still working on.
    #
    # Sent on every hello, including the first, where it is empty. Its absence
    # and its emptiness must mean the same thing — "I hold nothing" — because
    # an older client that never sends it has, in fact, restarted and holds
    # nothing. Treating absence as "unknown" instead would leave those jobs
    # leased to a device that will never finish them.
    active_jobs: list[str] = Field(
        default_factory=list, max_length=64, alias="activeJobs"
    )


class TelemetryMessage(CompanionModel):
    type: Literal["telemetry"] = "telemetry"
    telemetry: CompanionTelemetry


class Heartbeat(CompanionModel):
    type: Literal["heartbeat"] = "heartbeat"
    sent_at: datetime = Field(alias="sentAt")


class Ping(CompanionModel):
    type: Literal["ping"] = "ping"
    sent_at: datetime = Field(alias="sentAt")


class ConfigurationChanged(CompanionModel):
    type: Literal["configuration_changed"] = "configuration_changed"
    # Named settings only — never the values, which may be secret.
    changed: list[str] = Field(default_factory=list, max_length=64)


# -- job lifecycle ------------------------------------------------------------


class JobOffer(CompanionModel):
    type: Literal["job_offer"] = "job_offer"
    envelope: JobEnvelope
    payload: dict[str, Any] = Field(default_factory=dict)
    callback_budget: int = Field(default=12, ge=0, le=64, alias="callbackBudget")
    context_tokens: int = Field(default=4096, ge=512, le=1_000_000, alias="contextTokens")
    # Exactly the tools this attempt may call back for, as ``provider.tool``.
    # The companion is told the catalogue rather than left to guess it: a call
    # naming anything outside this list is refused in the session manager,
    # before the registry ever sees it. An empty catalogue means no callbacks.
    tool_catalogue: list[str] = Field(
        default_factory=list, max_length=64, alias="toolCatalogue"
    )
    # Wall-clock ceiling for one callback, and for every callback this attempt
    # makes put together. Count alone is not a bound: twelve calls that each
    # hang for the workload deadline are twelve times the latency the voice
    # turn had budgeted for.
    callback_deadline_seconds: float = Field(
        default=8.0, gt=0, le=120, alias="callbackDeadlineSeconds"
    )
    callback_budget_seconds: float = Field(
        default=30.0, gt=0, le=600, alias="callbackBudgetSeconds"
    )
    max_concurrent_callbacks: int = Field(
        default=2, ge=1, le=8, alias="maxConcurrentCallbacks"
    )


class JobAccept(CompanionModel):
    type: Literal["job_accept"] = "job_accept"
    job_id: str = Field(alias="jobId")
    attempt_id: str = Field(alias="attemptId")
    # Which tier the companion intends to run it on, for later measurement.
    tier: Literal["hot", "deep"] = "hot"


class JobReject(CompanionModel):
    type: Literal["job_reject"] = "job_reject"
    job_id: str = Field(alias="jobId")
    attempt_id: str = Field(alias="attemptId")
    reason: RejectReason
    detail: str | None = Field(default=None, max_length=200)
    # When set, the router holds off re-offering to this session for a while.
    retry_after_seconds: float | None = Field(
        default=None, ge=0, le=3600, alias="retryAfterSeconds"
    )


class JobProgress(CompanionModel):
    type: Literal["job_progress"] = "job_progress"
    job_id: str = Field(alias="jobId")
    attempt_id: str = Field(alias="attemptId")
    # Monotonic within an attempt; a lower sequence than one already seen is a
    # duplicate or reordered frame and is dropped.
    sequence: int = Field(ge=0)
    stage: str = Field(min_length=1, max_length=64)
    fraction: float | None = Field(default=None, ge=0, le=1)
    # Deliberately a summary, not the model's working text.
    summary: str | None = Field(default=None, max_length=400)


class JobResult(CompanionModel):
    type: Literal["job_result"] = "job_result"
    job_id: str = Field(alias="jobId")
    attempt_id: str = Field(alias="attemptId")
    result: dict[str, Any]
    tokens_used: int | None = Field(default=None, ge=0, le=10_000_000, alias="tokensUsed")


class JobFailed(CompanionModel):
    type: Literal["job_failed"] = "job_failed"
    job_id: str = Field(alias="jobId")
    attempt_id: str = Field(alias="attemptId")
    reason: FailureReason
    detail: str | None = Field(default=None, max_length=400)


class JobCancel(CompanionModel):
    type: Literal["job_cancel"] = "job_cancel"
    job_id: str = Field(alias="jobId")
    attempt_id: str = Field(alias="attemptId")
    reason: Literal["superseded", "user_cancelled", "deadline", "shutdown", "locality_lost"]


# -- tool callbacks -----------------------------------------------------------


class ToolCall(CompanionModel):
    type: Literal["tool_call"] = "tool_call"
    job_id: str = Field(alias="jobId")
    attempt_id: str = Field(alias="attemptId")
    call_id: str = Field(alias="callId", min_length=1, max_length=64)
    provider: str = Field(min_length=1, max_length=64)
    tool: str = Field(min_length=1, max_length=128)
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResultMessage(CompanionModel):
    type: Literal["tool_result"] = "tool_result"
    job_id: str = Field(alias="jobId")
    attempt_id: str = Field(alias="attemptId")
    call_id: str = Field(alias="callId")
    ok: bool
    code: str = Field(max_length=32)
    message: str = Field(max_length=1000)
    observed: dict[str, Any] | None = None
    sensitivity: Sensitivity = "ordinary"


# -- approvals ----------------------------------------------------------------


class ApprovalRequest(CompanionModel):
    """Server to client: a mutation is proposed and needs an explicit yes.

    An approval is not an execution. The mutation runs once, afterwards, under
    the stored idempotency key — so a duplicate tap, a reconnect or a server
    restart between approval and execution cannot produce a second write.
    """

    type: Literal["approval_request"] = "approval_request"
    approval_id: str = Field(alias="approvalId", min_length=1, max_length=64)
    job_id: str | None = Field(default=None, alias="jobId")
    # Exactly what will happen, in language the owner can check at a glance.
    summary: str = Field(min_length=1, max_length=400)
    provider: str = Field(min_length=1, max_length=64)
    tool: str = Field(min_length=1, max_length=128)
    sensitivity: Sensitivity = "mutation"
    expires_at: datetime = Field(alias="expiresAt")


class ApprovalResponse(CompanionModel):
    type: Literal["approval_response"] = "approval_response"
    approval_id: str = Field(alias="approvalId")
    approved: bool
    # Signature over the approval id and decision, so an approval cannot be
    # replayed or forged by anything that merely reached the socket.
    signature: str = Field(min_length=1, max_length=2048)


# -- personal context ---------------------------------------------------------


class PersonalCall(CompanionModel):
    type: Literal["personal_call"] = "personal_call"
    call_id: str = Field(alias="callId", min_length=1, max_length=64)
    tool: str = Field(min_length=1, max_length=128)
    arguments: dict[str, Any] = Field(default_factory=dict)
    # Bounded by construction: a personal read names its window and its cap.
    max_items: int = Field(default=50, ge=1, le=500, alias="maxItems")
    deadline_seconds: float = Field(default=15.0, gt=0, le=120, alias="deadlineSeconds")


class PersonalResult(CompanionModel):
    type: Literal["personal_result"] = "personal_result"
    call_id: str = Field(alias="callId")
    ok: bool
    code: str = Field(max_length=32)
    message: str = Field(max_length=1000)
    # Scoped records may cross to Iridium in v1. They are marked so logging,
    # retention and redaction can treat them differently from ordinary payloads.
    sensitivity: Sensitivity = "personal"
    items: list[dict[str, Any]] = Field(default_factory=list, max_length=500)
    truncated: bool = False


class LogEvent(CompanionModel):
    type: Literal["log_event"] = "log_event"
    level: Literal["debug", "info", "warning", "error"] = "info"
    event: str = Field(min_length=1, max_length=64)
    # Structural only. Raw event titles, reminder notes, locations and Health
    # values must never be put here.
    detail: str | None = Field(default=None, max_length=1000)
    trace_id: str | None = Field(default=None, alias="traceId", max_length=64)


ClientMessage = Annotated[
    AuthResponse
    | CompanionHello
    | TelemetryMessage
    | Heartbeat
    | JobAccept
    | JobReject
    | JobProgress
    | JobResult
    | JobFailed
    | ToolCall
    | ApprovalResponse
    | PersonalResult
    | LogEvent,
    Field(discriminator="type"),
]

ServerMessage = Annotated[
    AuthChallenge
    | HelloAck
    | JobOffer
    | JobCancel
    | ToolResultMessage
    | ApprovalRequest
    | PersonalCall
    | ConfigurationChanged
    | Ping,
    Field(discriminator="type"),
]


_CLIENT_MESSAGES = TypeAdapter(ClientMessage)


def parse_client_message(raw: str | bytes) -> ClientMessage:
    """Parse one client frame, rejecting oversized input before validation.

    The size check comes first deliberately: a payload bomb should cost one
    length comparison, not a full JSON parse followed by model validation.
    """

    if len(raw) > MAX_FRAME_BYTES:
        raise ValueError(f"companion frame exceeds {MAX_FRAME_BYTES} bytes")
    return _CLIENT_MESSAGES.validate_json(raw)


def serialize(message) -> dict:
    return message.model_dump(mode="json", by_alias=True)
