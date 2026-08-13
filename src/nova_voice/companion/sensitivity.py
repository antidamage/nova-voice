"""Sensitivity classification, log redaction and retention for companion data.

Personal context is permitted to leave the phone in v1 — that is what makes the
Calendar and Reminders path possible at all. The honest consequence is that
event titles, reminder notes, coordinates and Health values now flow through
Iridium, so "don't log the payload" has to be a mechanism rather than a
convention.

The rule this module enforces: **structural facts are loggable, values are
not.** How many items came back, of what kind, how long it took, under which
correlation id — all fine, and all that operators actually need to debug a
stuck job. The contents are never fine.

Retention follows the same split. A completed job's structural record is worth
keeping for audit; its personal payload is not, and expires on a much shorter
clock. Records still needed for correctness — an approval that has not yet
executed — are protected from that expiry regardless of class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nova_voice.companion.protocol import Sensitivity

# Keys whose *values* are never safe to log or persist past a job, regardless of
# which payload they turn up in. Matched case-insensitively on substrings so a
# nested "eventTitle" or "reminder_notes" is caught without enumerating shapes.
_UNSAFE_KEY_FRAGMENTS = (
    "title",
    "note",
    "summary",
    "body",
    "content",
    "text",
    "transcript",
    "location",
    "latitude",
    "longitude",
    "coordinate",
    "address",
    "value",
    "sample",
    "heart",
    "sleep",
    "name",
    "email",
    "phone",
    # An SSID is diagnostic-only and never a security gate, but it identifies a
    # household, and nothing is lost by keeping it out of logs.
    "network",
    "ssid",
)

REDACTED = "[redacted]"


@dataclass(frozen=True)
class RetentionPolicy:
    """How long a completed job's payload may be kept, by class."""

    payload_seconds: float
    # The structural record — counts, timings, outcome, correlation id — is
    # what audit actually needs, and it outlives the payload by a long way.
    structural_seconds: float


RETENTION: dict[Sensitivity, RetentionPolicy] = {
    "ordinary": RetentionPolicy(payload_seconds=86_400, structural_seconds=30 * 86_400),
    # Calendar and reminder records: useful only while the job that fetched
    # them is running or awaiting an approval.
    "personal": RetentionPolicy(payload_seconds=3_600, structural_seconds=30 * 86_400),
    "location": RetentionPolicy(payload_seconds=900, structural_seconds=30 * 86_400),
    # Health is the tightest: it is the most sensitive and the least useful to
    # retain once the request that needed it has been answered.
    "health": RetentionPolicy(payload_seconds=300, structural_seconds=30 * 86_400),
    # A proposed mutation must survive long enough to be approved and executed.
    "mutation": RetentionPolicy(payload_seconds=7 * 86_400, structural_seconds=90 * 86_400),
}


# Every message type on the wire, classified. Messages whose sensitivity is
# carried per-instance (a job's envelope declares its own class, and its
# payload inherits it) are recorded here as their *floor* — the least sensitive
# they can ever be — so a lookup can never under-classify.
MESSAGE_SENSITIVITY: dict[str, Sensitivity] = {
    # Handshake and liveness: no household content at all.
    "auth_challenge": "ordinary",
    "auth_response": "ordinary",
    "hello": "ordinary",
    "hello_ack": "ordinary",
    "heartbeat": "ordinary",
    "ping": "ordinary",
    # Battery, thermal and permission state. No household content; the SSID it
    # may carry is redacted by key rather than by class.
    "telemetry": "ordinary",
    "configuration_changed": "ordinary",
    "log_event": "ordinary",
    # Job traffic. The envelope's own `sensitivity` field overrides this floor
    # for the payload it carries.
    "job_offer": "ordinary",
    "job_accept": "ordinary",
    "job_reject": "ordinary",
    "job_progress": "ordinary",
    "job_result": "ordinary",
    "job_failed": "ordinary",
    "job_cancel": "ordinary",
    # A tool call names a provider and arguments, which routinely include
    # entity ids and free text.
    "tool_call": "personal",
    "tool_result": "personal",
    # A mutation the owner is being asked to approve, described in plain
    # language — by construction it states exactly what will change.
    "approval_request": "mutation",
    "approval_response": "mutation",
    # Scoped Calendar/Reminders/location/Health records.
    "personal_call": "personal",
    "personal_result": "personal",
}


def sensitivity_for(message_type: str) -> Sensitivity:
    """Classification floor for a message type. Unknown types are treated as
    the most sensitive class rather than the least."""

    return MESSAGE_SENSITIVITY.get(message_type, "health")


def retention_for(sensitivity: Sensitivity) -> RetentionPolicy:
    return RETENTION[sensitivity]


def is_personal(sensitivity: Sensitivity) -> bool:
    return sensitivity in ("personal", "location", "health", "mutation")


def _unsafe_key(key: str) -> bool:
    lowered = key.lower()
    return any(fragment in lowered for fragment in _UNSAFE_KEY_FRAGMENTS)


def redact(payload: Any, *, sensitivity: Sensitivity = "personal", _keyed: bool = False) -> Any:
    """Return a copy safe to put in a log line or a metrics label.

    Structure is preserved so the shape of a payload stays debuggable: a list
    of ten reminders still looks like a list of ten reminders, with every field
    that could carry content replaced. Ordinary payloads keep their scalars —
    they are prompts and verdicts, not household records — but are still walked
    so a personal value that leaked into an ordinary payload is caught.

    ``_keyed`` records whether a string arrived under a key that was already
    judged safe. Without it the "assume bare strings are content" rule below
    would also destroy opaque ids and timestamps, which are exactly the
    structural fields an operator needs to correlate a stuck job.
    """

    if isinstance(payload, dict):
        return {
            key: (
                REDACTED
                if _unsafe_key(str(key)) and value is not None
                else redact(value, sensitivity=sensitivity, _keyed=True)
            )
            for key, value in payload.items()
        }
    if isinstance(payload, (list, tuple)):
        # List items have no key of their own, so they are judged on their own.
        return [redact(item, sensitivity=sensitivity, _keyed=False) for item in payload]
    if isinstance(payload, str) and is_personal(sensitivity) and not _keyed:
        # A bare string with no key to judge it by is assumed to be content.
        return REDACTED
    return payload


def structural_summary(payload: Any, *, sensitivity: Sensitivity) -> dict:
    """The loggable facts about a payload: shape and size, never content."""

    if isinstance(payload, dict):
        kind, count = "object", len(payload)
    elif isinstance(payload, (list, tuple)):
        kind, count = "array", len(payload)
    elif payload is None:
        kind, count = "null", 0
    else:
        kind, count = type(payload).__name__, 1
    return {"kind": kind, "count": count, "sensitivity": sensitivity}
