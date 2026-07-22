from __future__ import annotations

from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class StructuralToolOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str
    tool: str
    status: str
    result_code: str | None = None


class StructuralTelemetryRecord(BaseModel):
    """Content-free operational evidence safe for persistent evaluation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str
    at: datetime
    kind: Literal[
        "turn",
        "interruption",
        "queue",
        "memory",
        "proactive",
        "error",
        "tts_pacing",
    ]
    trace_id: str | None = None
    input_revision: str | None = None
    context_revision: str | None = None
    terminal_status: str | None = None
    stage_statuses: tuple[str, ...] = ()
    latencies_ms: dict[str, float] = Field(default_factory=dict)
    queue_depths: dict[str, int] = Field(default_factory=dict)
    policy_reason: str | None = None
    tool_outcomes: tuple[StructuralToolOutcome, ...] = ()
    memory_metrics: dict[str, float] = Field(default_factory=dict)
    proactive_outcome: str | None = None
    interruption_kind: str | None = None
    interruption_confidence: float | None = Field(default=None, ge=0, le=1)
    error_stage: str | None = None
    error_type: str | None = None


class StructuralTelemetry:
    """Bounded structural telemetry with optional content-free JSONL output."""

    def __init__(self, path: Path | None = None, *, max_events: int = 2_000) -> None:
        self.path = path
        self._events: deque[StructuralTelemetryRecord] = deque(maxlen=max_events)

    def ingest_monitor(self, kind: str, detail: dict[str, Any]) -> StructuralTelemetryRecord | None:
        record: StructuralTelemetryRecord | None = None
        if kind == "response_completed":
            trace = detail.get("turnTrace") or {}
            policy = trace.get("policy") or {}
            record = self._record(
                kind="turn",
                trace_id=_optional_string(trace.get("trace_id")),
                input_revision=_optional_string(trace.get("input_revision")),
                context_revision=_optional_string(trace.get("context_revision")),
                terminal_status=_optional_string(trace.get("terminal_status")),
                stage_statuses=tuple(
                    f"{stage.get('stage')}:{stage.get('status')}"
                    for stage in trace.get("stages", ())
                ),
                latencies_ms=_numeric_map(detail.get("timingsMs")),
                policy_reason=_optional_string(policy.get("reason")),
                tool_outcomes=tuple(
                    StructuralToolOutcome(
                        provider=str(item.get("provider", "")),
                        tool=str(item.get("tool", "")),
                        status=str(item.get("status", "")),
                        result_code=_optional_string(item.get("result_code")),
                    )
                    for item in trace.get("tool_journal", ())
                ),
            )
        elif kind == "interruption_classified":
            record = self._record(
                kind="interruption",
                interruption_kind=_optional_string(detail.get("classification")),
                interruption_confidence=_optional_float(detail.get("confidence")),
            )
        elif kind == "tts_pacing":
            record = self._record(
                kind="tts_pacing",
                latencies_ms={
                    "worstDeficit": float(detail.get("worstDeficitMs", 0)),
                },
            )
        elif kind == "processing_error":
            record = self._record(
                kind="error",
                error_stage=_optional_string(detail.get("stage")),
                error_type=_optional_string(detail.get("errorType")),
            )
        return record

    def record_queue(self, name: str, depth: int, *, capacity: int) -> StructuralTelemetryRecord:
        return self._record(
            kind="queue",
            queue_depths={name: max(0, int(depth)), f"{name}Capacity": max(0, int(capacity))},
        )

    def record_memory(self, **metrics: float) -> StructuralTelemetryRecord:
        return self._record(kind="memory", memory_metrics=_numeric_map(metrics))

    def record_proactive(self, outcome: str, **metrics: float) -> StructuralTelemetryRecord:
        return self._record(
            kind="proactive",
            proactive_outcome=outcome,
            latencies_ms=_numeric_map(metrics),
        )

    def snapshot(self) -> tuple[StructuralTelemetryRecord, ...]:
        return tuple(self._events)

    def _record(self, **values: Any) -> StructuralTelemetryRecord:
        record = StructuralTelemetryRecord(
            event_id=uuid4().hex,
            at=datetime.now(UTC),
            **values,
        )
        self._events.append(record)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as output:
                output.write(record.model_dump_json() + "\n")
        return record


def _numeric_map(value: object) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): float(item)
        for key, item in value.items()
        if isinstance(item, int | float) and not isinstance(item, bool)
    }


def _optional_string(value: object) -> str | None:
    return str(value) if value is not None else None


def _optional_float(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) else None
