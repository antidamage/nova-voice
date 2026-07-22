from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from nova_voice.api import create_app
from nova_voice.config import Settings
from nova_voice.telemetry import StructuralTelemetry, StructuralTelemetryRecord


def test_monitor_events_become_content_free_structural_telemetry(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.jsonl"
    telemetry = StructuralTelemetry(path)
    record = telemetry.ingest_monitor(
        "response_completed",
        {
            "transcript": "private words",
            "responseText": "also private",
            "timingsMs": {"stt": 12.5, "service": 8},
            "turnTrace": {
                "trace_id": "trace-1",
                "input_revision": "input-hash",
                "context_revision": "context-hash",
                "terminal_status": "completed",
                "policy": {"reason": "allowed", "prompt": "private"},
                "stages": [{"stage": "capture", "status": "completed"}],
                "tool_journal": [
                    {
                        "provider": "nova",
                        "tool": "nova.device_control",
                        "status": "completed",
                        "result_code": "ok",
                        "arguments": {"private": "value"},
                    }
                ],
            },
        },
    )

    assert record is not None
    assert record.trace_id == "trace-1"
    assert record.latencies_ms == {"stt": 12.5, "service": 8.0}
    assert record.tool_outcomes[0].result_code == "ok"
    serialized = path.read_text(encoding="utf-8")
    assert "private words" not in serialized
    assert "also private" not in serialized
    assert "arguments" not in serialized
    assert "prompt" not in serialized


def test_structural_telemetry_covers_queue_memory_proactivity_and_interruptions() -> None:
    telemetry = StructuralTelemetry()
    telemetry.record_queue("audio", 4, capacity=750)
    telemetry.record_memory(retrieved=3, accepted=2, precision=0.66)
    telemetry.record_proactive("suppressed_quiet_hours", decisionMs=2.5)
    telemetry.ingest_monitor(
        "interruption_classified",
        {"classification": "backchannel", "confidence": 0.9, "transcript": "yeah"},
    )

    events = telemetry.snapshot()
    assert [event.kind for event in events] == [
        "queue",
        "memory",
        "proactive",
        "interruption",
    ]
    assert events[-1].interruption_kind == "backchannel"


def test_structural_schema_rejects_conversation_content() -> None:
    with pytest.raises(ValidationError, match="transcript"):
        StructuralTelemetryRecord(
            event_id="event",
            at="2026-07-22T00:00:00Z",
            kind="turn",
            transcript="must not fit the schema",
        )


def test_app_exposes_separate_structural_telemetry_sink() -> None:
    app = create_app(Settings(), service=object(), audio_runtime=None)

    assert isinstance(app.state.structural_telemetry, StructuralTelemetry)
