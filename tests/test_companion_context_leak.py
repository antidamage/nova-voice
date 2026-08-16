"""A synthetic secret, followed everywhere it could escape.

NPT-511. Personal context is permitted to leave the phone in v1 — that is the
whole reason calendar and reminder integration is possible — so "it does not
leak" has to be a property with a test rather than an intention with a comment.

Every test here puts the same recognisable string into a personal payload and
then asserts it is absent from a place it must never reach: a log line, a
structural summary, a metrics label, a durable record after retention, or an
unrelated job. The string is distinctive on purpose, so a failure names the
exact escape route rather than reporting a mismatch.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta

import pytest

from nova_voice.companion.retention import CompanionRetention
from nova_voice.companion.sensitivity import (
    REDACTED,
    is_personal,
    redact,
    retention_for,
    sensitivity_for,
    structural_summary,
)
from nova_voice.durable.models import (
    CompanionCallbackRecord,
    CompanionCallbackState,
    CompanionJobRecord,
    CompanionJobState,
)
from nova_voice.durable.store import DurableAgentStore
from nova_voice.providers.companion.provider import validate_health_write

# Distinctive enough that a failure names the escape route rather than reporting
# a mismatch between two plausible strings.
SECRET = "Colonoscopy-with-Dr-Ngata-at-0730"
CALENDAR_PAYLOAD = {
    "items": [
        {"id": "event-1", "title": SECRET, "start": "2026-08-20T07:30:00Z"},
        {"id": "event-2", "summary": SECRET, "notes": SECRET},
    ],
    "truncated": False,
}


# -- redaction ----------------------------------------------------------------


def test_a_calendar_payload_is_redacted_field_by_field():
    safe = redact(CALENDAR_PAYLOAD, sensitivity="personal")
    encoded = json.dumps(safe)

    assert SECRET not in encoded
    # Structure survives so the shape stays debuggable: a list of two events
    # still looks like a list of two events.
    assert len(safe["items"]) == 2
    assert safe["items"][0]["title"] == REDACTED
    assert safe["items"][0]["id"] == "event-1"


def test_the_structural_summary_carries_no_content_at_all():
    summary = structural_summary(CALENDAR_PAYLOAD, sensitivity="personal")

    assert SECRET not in json.dumps(summary)
    assert summary["kind"] == "object"
    assert summary["sensitivity"] == "personal"


def test_a_bare_string_in_a_personal_payload_is_assumed_to_be_content():
    # It has no key to be judged by, so the safe reading is that it is a title.
    safe = redact([SECRET, "another"], sensitivity="personal")

    assert safe == [REDACTED, REDACTED]


def test_an_opaque_id_survives_redaction():
    # Otherwise the redacted form is useless: correlating a stuck job needs the
    # ids, and stripping them protects nothing.
    safe = redact({"id": "event-1", "traceId": "trace-9", "title": SECRET})

    assert safe["id"] == "event-1"
    assert safe["traceId"] == "trace-9"
    assert safe["title"] == REDACTED


def test_an_unknown_message_type_is_treated_as_the_most_sensitive_class():
    """A type nobody classified must not default to 'ordinary'."""

    assert sensitivity_for("some_future_frame") == "health"
    assert is_personal(sensitivity_for("some_future_frame"))


def test_health_is_retained_far_more_briefly_than_ordinary_reasoning():
    assert retention_for("health").payload_seconds < retention_for("personal").payload_seconds
    assert retention_for("personal").payload_seconds < retention_for("ordinary").payload_seconds
    # And the structural record outlives all of them, because that is the part
    # audit actually needs.
    for sensitivity in ("ordinary", "personal", "health"):
        policy = retention_for(sensitivity)
        assert policy.structural_seconds > policy.payload_seconds


# -- logging ------------------------------------------------------------------


def test_logging_a_redacted_payload_puts_nothing_in_the_log(caplog):
    with caplog.at_level(logging.INFO):
        logging.getLogger("nova_voice.test").info(
            "personal read returned %s", redact(CALENDAR_PAYLOAD, sensitivity="personal")
        )

    assert SECRET not in caplog.text
    # The line is still worth having: it says something happened and what shape.
    assert "personal read returned" in caplog.text


def test_logging_the_structural_summary_is_the_safe_default(caplog):
    with caplog.at_level(logging.INFO):
        logging.getLogger("nova_voice.test").info(
            "personal read %s", structural_summary(CALENDAR_PAYLOAD, sensitivity="personal")
        )

    assert SECRET not in caplog.text
    assert "count" in caplog.text


# -- persistence and cleanup --------------------------------------------------


async def test_a_secret_in_a_callback_result_is_gone_after_retention(tmp_path):
    store = DurableAgentStore(tmp_path / "durable.sqlite3")
    await store.initialize()
    now = datetime.now(UTC)

    job = CompanionJobRecord(
        id="job-1",
        workload="interpret",
        input_revision="rev-1",
        idempotency_key="key-1",
        trace_id="trace-1",
        sensitivity="personal",
        status=CompanionJobState.COMPLETED,
        result_ref=f"result://{SECRET}",
        progress_summary=SECRET,
        created_at=now,
        updated_at=now,
    )
    callback = CompanionCallbackRecord(
        id="call-1",
        job_id="job-1",
        attempt_id="a1",
        call_id="call-1",
        provider="companion",
        tool="companion.calendar.list",
        arguments={"start": "2026-08-20T00:00:00Z"},
        status=CompanionCallbackState.COMPLETED,
        idempotency_key="job-1:call-1",
        sensitivity="personal",
        observed={"items": [{"title": SECRET}]},
        created_at=now,
        updated_at=now,
        # Personal payloads expire on the hour clock.
        expires_at=now + timedelta(seconds=retention_for("personal").payload_seconds),
    )
    await store.create(job)
    await store.create(callback)

    await CompanionRetention(store).sweep(now=now + timedelta(days=1))

    remaining = await store.get(CompanionCallbackRecord, "call-1")
    assert remaining is None, "the callback's observation outlived its retention window"

    stored_job = await store.get(CompanionJobRecord, "job-1")
    assert stored_job is not None
    assert SECRET not in stored_job.record.model_dump_json()
    # And the audit facts are still there.
    assert stored_job.record.workload == "interpret"
    assert stored_job.record.trace_id == "trace-1"


async def test_one_jobs_secret_never_appears_in_another(tmp_path):
    """Records are keyed per job; nothing is shared between them."""

    store = DurableAgentStore(tmp_path / "durable.sqlite3")
    await store.initialize()
    now = datetime.now(UTC)

    for index, payload in ((1, SECRET), (2, "nothing sensitive")):
        await store.create(
            CompanionCallbackRecord(
                id=f"call-{index}",
                job_id=f"job-{index}",
                attempt_id="a1",
                call_id=f"call-{index}",
                provider="companion",
                tool="companion.calendar.list",
                arguments={},
                idempotency_key=f"job-{index}:call-{index}",
                observed={"items": [{"title": payload}]},
                created_at=now,
                updated_at=now,
                expires_at=now + timedelta(hours=1),
            )
        )

    other = await store.get(CompanionCallbackRecord, "call-2")
    assert other is not None
    assert SECRET not in other.record.model_dump_json()


# -- Health values (NPT-510) --------------------------------------------------


@pytest.mark.parametrize(
    ("metric", "value", "unit"),
    [
        ("bodyMass", 72.5, "kg"),
        ("dietaryWater", 250.0, "mL"),
        ("mindfulSession", 15.0, "min"),
    ],
)
def test_an_allowlisted_health_write_passes_validation(metric, value, unit):
    assert validate_health_write(metric, value, unit) is None


@pytest.mark.parametrize(
    ("metric", "value", "unit", "expected"),
    [
        # The wrong unit is the dangerous one: 160 lb read as 160 kg is a
        # plausible-looking number and a completely wrong record.
        ("bodyMass", 160.0, "lb", "must be written in kg"),
        ("bodyMass", 0.5, "kg", "outside the plausible range"),
        ("dietaryWater", 99_999.0, "mL", "outside the plausible range"),
        ("bloodGlucose", 5.5, "mmol/L", "not an allowlisted"),
        ("insulinDelivery", 2.0, "IU", "not an allowlisted"),
    ],
)
def test_a_dangerous_health_write_is_refused_before_anyone_is_asked(
    metric, value, unit, expected
):
    problem = validate_health_write(metric, value, unit)

    assert problem is not None
    assert expected in problem


def test_the_health_allowlist_contains_nothing_clinical():
    """A wrong number in any of these is embarrassing; a wrong number in a
    clinical one could inform a real decision."""

    from nova_voice.providers.companion.provider import HEALTH_WRITE_ALLOWLIST

    clinical = {"bloodGlucose", "insulinDelivery", "bloodPressure", "heartRate", "oxygenSaturation"}
    assert set(HEALTH_WRITE_ALLOWLIST).isdisjoint(clinical)
