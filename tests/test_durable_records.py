from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from nova_voice.durable.models import (
    AuditRecord,
    ConversationRecord,
    DelegationGrantRecord,
    EventRecord,
    ExecutionRecord,
    GoalRecord,
    MemoryReferenceRecord,
    PlanRecord,
    PlanStepKind,
    PlanStepRecord,
    ProactiveInterventionRecord,
)


def test_all_durable_record_contracts_are_versioned_and_round_trip() -> None:
    now = datetime(2026, 7, 22, tzinfo=UTC)
    records = (
        ConversationRecord(id="conversation", room_id="office", created_at=now, updated_at=now),
        EventRecord(
            id="event",
            source="nova",
            kind="state_changed",
            payload_revision="sha256:event",
            created_at=now,
            updated_at=now,
        ),
        GoalRecord(
            id="goal",
            summary="Warm the office",
            created_at=now,
            updated_at=now,
        ),
        PlanRecord(id="plan", goal_id="goal", created_at=now, updated_at=now),
        PlanStepRecord(
            id="step",
            plan_id="plan",
            kind=PlanStepKind.TOOL,
            order=0,
            created_at=now,
            updated_at=now,
        ),
        ExecutionRecord(
            id="execution",
            plan_id="plan",
            step_id="step",
            idempotency_key="plan:step",
            attempt=1,
            lease_owner="worker",
            lease_token="token",
            lease_expires_at=now + timedelta(seconds=30),
            created_at=now,
            updated_at=now,
        ),
        DelegationGrantRecord(
            id="grant",
            grantor_id="owner",
            grantee_id="household",
            capability="lights",
            created_at=now,
            updated_at=now,
        ),
        ProactiveInterventionRecord(
            id="intervention",
            reason_code="window_open",
            channel="voice",
            status="proposed",
            deduplication_key="window:office",
            created_at=now,
            updated_at=now,
        ),
        MemoryReferenceRecord(
            id="memory-reference",
            memory_id="memory",
            memory_type="preference",
            provider="mempalace",
            provenance_revision="sha256:memory",
            created_at=now,
            updated_at=now,
        ),
        AuditRecord(
            id="audit",
            actor_id="owner",
            action="create",
            object_type="GoalRecord",
            object_id="goal",
            resulting_revision=1,
            created_at=now,
            updated_at=now,
        ),
    )

    for record in records:
        assert record.schema_version == 1
        assert type(record).model_validate_json(record.model_dump_json()) == record


def test_durable_records_reject_unknown_fields_naive_times_and_invalid_waits() -> None:
    now = datetime(2026, 7, 22, tzinfo=UTC)
    with pytest.raises(ValidationError, match="Extra inputs"):
        GoalRecord(id="goal", summary="test", invented=True)
    with pytest.raises(ValidationError, match="timezone-aware"):
        GoalRecord(
            id="goal",
            summary="test",
            created_at=datetime(2026, 7, 22),
            updated_at=datetime(2026, 7, 22),
        )
    with pytest.raises(ValidationError, match="require not_before"):
        PlanStepRecord(
            id="wait",
            plan_id="plan",
            kind=PlanStepKind.WAIT,
            order=0,
            created_at=now,
            updated_at=now,
        )
