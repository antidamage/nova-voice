"""Redaction and retention for personal context crossing from the phone.

v1 lets Calendar, Reminders, location and Health records reach Iridium. These
tests are the mechanism that keeps "and they never turn up in a log" true, so
they use synthetic records that look exactly like the real thing.
"""

from __future__ import annotations

import json

import pytest

from nova_voice.companion.protocol import ClientMessage, ServerMessage
from nova_voice.companion.sensitivity import (
    MESSAGE_SENSITIVITY,
    REDACTED,
    RETENTION,
    is_personal,
    redact,
    retention_for,
    sensitivity_for,
    structural_summary,
)

# Deliberately shaped like a real EventKit payload, with a value in every field
# that must not survive redaction.
CALENDAR_PAYLOAD = {
    "items": [
        {
            "id": "opaque-1",
            "title": "Endocrinologist appointment",
            "notes": "bring blood test results",
            "startsAt": "2026-08-14T09:00:00+12:00",
            "allDay": False,
            "location": "12 Example Street",
        }
    ],
    "truncated": False,
    "count": 1,
}

HEALTH_PAYLOAD = {
    "samples": [{"type": "heartRate", "value": 148, "unit": "count/min"}],
    "count": 1,
}


def _message_types(union) -> set[str]:
    types = set()
    for member in union.__args__[0].__args__:
        annotation = member.model_fields["type"].annotation
        types.update(annotation.__args__)
    return types


def test_every_protocol_message_type_is_classified():
    """NPT-005: no payload type may reach the wire unclassified."""

    declared = _message_types(ClientMessage) | _message_types(ServerMessage)
    assert declared <= set(MESSAGE_SENSITIVITY), (
        f"unclassified message types: {sorted(declared - set(MESSAGE_SENSITIVITY))}"
    )


def test_unknown_message_type_is_treated_as_most_sensitive():
    """An unrecognised type must fail closed, not open."""

    assert sensitivity_for("something_new") == "health"


def test_calendar_titles_notes_and_locations_do_not_survive_redaction():
    redacted = redact(CALENDAR_PAYLOAD, sensitivity="personal")
    serialized = json.dumps(redacted)

    assert "Endocrinologist appointment" not in serialized
    assert "blood test results" not in serialized
    assert "12 Example Street" not in serialized


def test_redaction_preserves_shape_so_payloads_stay_debuggable():
    redacted = redact(CALENDAR_PAYLOAD, sensitivity="personal")

    assert len(redacted["items"]) == 1
    assert redacted["count"] == 1
    assert redacted["truncated"] is False
    # Structural fields survive; content fields do not.
    assert redacted["items"][0]["id"] == "opaque-1"
    assert redacted["items"][0]["startsAt"] == "2026-08-14T09:00:00+12:00"
    assert redacted["items"][0]["title"] == REDACTED


def test_health_values_do_not_survive_redaction():
    serialized = json.dumps(redact(HEALTH_PAYLOAD, sensitivity="health"))
    assert "148" not in serialized


def test_bare_strings_in_a_personal_payload_are_redacted():
    """With no key to judge by, content is assumed."""

    assert redact(["Dentist", "Pharmacy"], sensitivity="personal") == [REDACTED, REDACTED]


def test_ordinary_payloads_keep_their_scalars():
    payload = {"attempts": 2, "confirmed": True}
    assert redact(payload, sensitivity="ordinary") == payload


def test_a_personal_value_leaking_into_an_ordinary_payload_is_still_caught():
    leaked = redact({"title": "Endocrinologist appointment"}, sensitivity="ordinary")
    assert leaked["title"] == REDACTED


def test_structural_summary_carries_no_content():
    summary = structural_summary(CALENDAR_PAYLOAD, sensitivity="personal")
    assert summary == {"kind": "object", "count": 3, "sensitivity": "personal"}
    assert "Endocrinologist appointment" not in json.dumps(summary)


@pytest.mark.parametrize("sensitivity", sorted(RETENTION))
def test_payload_retention_never_outlives_structural_retention(sensitivity):
    policy = retention_for(sensitivity)
    assert policy.payload_seconds <= policy.structural_seconds


def test_health_and_location_expire_faster_than_ordinary_payloads():
    assert retention_for("health").payload_seconds < retention_for("ordinary").payload_seconds
    assert retention_for("location").payload_seconds < retention_for("ordinary").payload_seconds


def test_a_pending_mutation_outlives_the_context_that_proposed_it():
    """An approval must survive long enough to be approved and executed."""

    assert retention_for("mutation").payload_seconds > retention_for("personal").payload_seconds


def test_personal_classes_are_identified():
    assert is_personal("personal") and is_personal("health") and is_personal("location")
    assert not is_personal("ordinary")
