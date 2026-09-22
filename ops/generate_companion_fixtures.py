"""Regenerate the frozen companion protocol v1 fixtures.

The generated files under ``docs/companion-protocol/`` are the artifact; this
script exists so they can be refreshed deliberately after a protocol change,
not so they are built at test time. ``tests/test_companion_fixtures.py`` reads
the committed files, which is what makes an incompatible model change fail.

Run from the repository root:

    python ops/generate_companion_fixtures.py
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nova_voice.companion.protocol import (  # noqa: E402
    SUPPORTED_PROTOCOL_VERSIONS,
    ApprovalRequest,
    ApprovalResponse,
    AuthChallenge,
    AuthResponse,
    CompanionHello,
    CompanionTelemetry,
    ConfigurationChanged,
    Heartbeat,
    HelloAck,
    JobAccept,
    JobCancel,
    JobEnvelope,
    JobFailed,
    JobOffer,
    JobProgress,
    JobReject,
    JobResult,
    LogEvent,
    ModelAvailability,
    PermissionState,
    PersonalCall,
    PersonalResult,
    Ping,
    TelemetryMessage,
    ToolCall,
    ToolResultMessage,
    serialize,
)

OUTPUT = Path(__file__).resolve().parent.parent / "docs" / "companion-protocol"

CREATED = datetime(2026, 8, 13, 9, 0, tzinfo=UTC)
JOB_ID = "5f2b9c1d4e8a47c0"
ATTEMPT_ID = "a41d7e93b0f5426d"
TRACE_ID = "trace-9c81f0a2"

# A placeholder identity throughout. No deployment name belongs in a tracked
# file, and these are read by anyone setting the topology up.
IDENTITY = "companion-1"

ENVELOPE = JobEnvelope(
    jobId=JOB_ID,
    attemptId=ATTEMPT_ID,
    idempotencyKey="a1f0c3d29b6e4f7583c1d0e2a4b6c8d9",
    workload="interpret",
    inputRevision="8c1f0a2b3d4e5f60",
    resultSchema="interpret.v1",
    sensitivity="ordinary",
    locality="home_lan",
    traceId=TRACE_ID,
    createdAt=CREATED,
    acceptDeadline=CREATED + timedelta(seconds=2),
    completeDeadline=CREATED + timedelta(seconds=6),
)

TELEMETRY = CompanionTelemetry(
    battery=0.82,
    charging=False,
    lowPowerMode=False,
    thermalState="nominal",
    appState="background",
    models=ModelAvailability(hotAvailable=True, hotContextTokens=4096, deepAvailable=False),
    permissions=PermissionState(
        calendar="available", reminders="available", location="partial", health="denied"
    ),
    reportedNetwork="home-wifi",
)

MESSAGES = {
    "auth_challenge": AuthChallenge(
        nonce="Yk9wS3ZQdV9tSFhyM0pMYmNEZ0FzTnhFdDJRaFcxUGo",
        supportedVersions=list(SUPPORTED_PROTOCOL_VERSIONS),
        expiresAt=CREATED + timedelta(seconds=30),
    ),
    "auth_response": AuthResponse(
        protocolVersion=1,
        announcedId=IDENTITY,
        roles=["companion", "satellite"],
        certificateChain=[
            "-----BEGIN CERTIFICATE-----\nPLACEHOLDER\n-----END CERTIFICATE-----\n"
        ],
        signature="MEUCIQDPLACEHOLDERsignatureBASE64==",
    ),
    "hello": CompanionHello(
        protocolVersion=1,
        schemaVersions=[1],
        displayName="Companion",
        roles=["companion", "satellite"],
        appVersion="1.0.0",
        osVersion="26.0",
        workloads=[
            "interpret",
            "render_response",
            "confirm_objective",
            "extract_self_profile_update",
            "classify_icon",
        ],
        personalTools=["calendar.events", "reminders.list"],
        telemetry=TELEMETRY,
    ),
    "hello_ack": HelloAck(
        protocolVersion=1,
        authenticatedId=IDENTITY,
        locality="home_lan",
        sessionId="0d5b7c1a9e2f4836",
        heartbeatSeconds=20.0,
    ),
    "telemetry": TelemetryMessage(telemetry=TELEMETRY),
    "heartbeat": Heartbeat(sentAt=CREATED),
    "ping": Ping(sentAt=CREATED),
    "configuration_changed": ConfigurationChanged(changed=["routes", "companion_enabled"]),
    "job_offer": JobOffer(
        envelope=ENVELOPE,
        payload={"transcript": "turn the lounge lights down a bit", "roomId": "lounge"},
        callbackBudget=12,
        contextTokens=4096,
    ),
    "job_accept": JobAccept(jobId=JOB_ID, attemptId=ATTEMPT_ID, tier="hot"),
    "job_reject": JobReject(
        jobId=JOB_ID,
        attemptId=ATTEMPT_ID,
        reason="thermal",
        detail="device is in a serious thermal state",
        retryAfterSeconds=120.0,
    ),
    "job_progress": JobProgress(
        jobId=JOB_ID,
        attemptId=ATTEMPT_ID,
        sequence=1,
        stage="reasoning",
        fraction=0.5,
        summary="resolving the referenced room",
    ),
    "job_result": JobResult(
        jobId=JOB_ID,
        attemptId=ATTEMPT_ID,
        # Validated separately against the workload's resultSchema; snake_case
        # because these shapes come from the existing domain models.
        result={
            "emotion": {"label": "neutral", "confidence": 0.9, "intensity": 0.1, "evidence": []},
            "speech_act": "directive",
            "addressed_probability": 0.97,
            "decision": "execute",
            "confidence": 0.93,
            "active_goal": {"summary": "dim the lounge", "status": "in_progress", "pending": []},
            "actions": [],
            "response_plan": {
                "acknowledgement_style": "concise",
                "pre_action_speech": None,
                "requires_post_tool_rendering": False,
            },
            "self_profile_update": None,
        },
        tokensUsed=412,
    ),
    "job_failed": JobFailed(
        jobId=JOB_ID,
        attemptId=ATTEMPT_ID,
        reason="invalid_result",
        detail="generated output did not match interpret.v1",
    ),
    "job_cancel": JobCancel(jobId=JOB_ID, attemptId=ATTEMPT_ID, reason="superseded"),
    "tool_call": ToolCall(
        jobId=JOB_ID,
        attemptId=ATTEMPT_ID,
        callId="c0a71f2e",
        provider="nova",
        tool="nova.lighting_set",
        arguments={"zoneId": "lounge", "brightness": 40},
    ),
    "tool_result": ToolResultMessage(
        jobId=JOB_ID,
        attemptId=ATTEMPT_ID,
        callId="c0a71f2e",
        ok=True,
        code="ok",
        message="Lounge set to 40%.",
        observed={"zoneId": "lounge", "brightness": 40},
        sensitivity="ordinary",
    ),
    "approval_request": ApprovalRequest(
        approvalId="ap-3f7c9d21",
        jobId=JOB_ID,
        summary='Create a reminder "Collect prescription" due tomorrow at 9:00 am.',
        provider="companion_personal",
        tool="reminders.create",
        sensitivity="mutation",
        expiresAt=CREATED + timedelta(minutes=5),
    ),
    "approval_response": ApprovalResponse(
        approvalId="ap-3f7c9d21",
        approved=True,
        signature="MEQCIEPLACEHOLDERapprovalSIGNATURE==",
    ),
    "personal_call": PersonalCall(
        callId="pc-8b2e15d0",
        tool="calendar.events",
        arguments={"from": "2026-08-13", "to": "2026-08-14", "fields": ["id", "startsAt"]},
        maxItems=50,
        deadlineSeconds=15.0,
    ),
    "personal_result": PersonalResult(
        callId="pc-8b2e15d0",
        ok=True,
        code="ok",
        message="2 events",
        sensitivity="personal",
        items=[
            {"id": "opaque-1", "startsAt": "2026-08-13T09:00:00Z", "allDay": False},
            {"id": "opaque-2", "startsAt": "2026-08-13T14:30:00Z", "allDay": False},
        ],
        truncated=False,
    ),
    "log_event": LogEvent(
        level="info",
        event="tier_changed",
        detail="reduced: battery 43% is below half",
        traceId=TRACE_ID,
    ),
}

INVALID = {
    "unknown_type": {"type": "take_over_the_house"},
    "unknown_field": {"type": "heartbeat", "sentAt": "2026-08-13T09:00:00Z", "extra": 1},
    "missing_required_field": {"type": "job_accept", "jobId": JOB_ID},
    "out_of_range_fraction": {
        "type": "job_progress",
        "jobId": JOB_ID,
        "attemptId": ATTEMPT_ID,
        "sequence": 1,
        "stage": "reasoning",
        "fraction": 4.2,
    },
}


def _challenge_vectors() -> list[dict]:
    """Cross-language vectors for the signed challenge material.

    A mismatch between the Python and Swift constructions is not a decoding
    error — it is a signature that silently never verifies, which is a
    miserable thing to debug from a phone. Pinning the exact bytes makes a
    divergence a failing test on whichever side moved.
    """

    from nova_voice.companion.auth import challenge_material

    cases = [
        {"nonce": "Yk9wS3ZQdV9tSFhyM0pMYmNEZ0FzTnhFdDJRaFcxUGo", "protocolVersion": 1,
         "announcedId": IDENTITY, "roles": ["companion"]},
        # Role order and case must not change the signed bytes: both sides sort
        # and lowercase before joining.
        {"nonce": "bm9uY2UtdHdvLXNhbXBsZS12YWx1ZS1mb3ItdGVzdHM", "protocolVersion": 1,
         "announcedId": IDENTITY, "roles": ["Satellite", "companion"]},
        {"nonce": "dGhpcmQtbm9uY2UtZm9yLWNyb3NzLWxhbmd1YWdlLXQ", "protocolVersion": 1,
         "announcedId": "  Companion-1  ", "roles": ["companion", "satellite"]},
    ]
    for case in cases:
        case["material"] = challenge_material(
            nonce=case["nonce"],
            protocol_version=case["protocolVersion"],
            announced_id=case["announcedId"],
            roles=list(case["roles"]),
        ).decode("utf-8")
    return cases


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "invalid").mkdir(exist_ok=True)
    (OUTPUT / "challenge-material.json").write_text(
        json.dumps(_challenge_vectors(), indent=2) + "\n", encoding="utf-8"
    )
    for name, message in MESSAGES.items():
        path = OUTPUT / f"{name}.json"
        path.write_text(json.dumps(serialize(message), indent=2) + "\n", encoding="utf-8")
    for name, payload in INVALID.items():
        path = OUTPUT / "invalid" / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(MESSAGES)} valid and {len(INVALID)} invalid fixtures to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
