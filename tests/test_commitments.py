from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import httpx

from nova_voice.api import create_app
from nova_voice.capabilities.registry import CapabilityRegistry
from nova_voice.commitments import CommitmentManager, next_recurrence
from nova_voice.config import Settings
from nova_voice.domain import CapabilityToolCall, PlannedAction
from nova_voice.durable.models import (
    CommitmentRecord,
    CommitmentState,
    ProactiveInterventionRecord,
)
from nova_voice.durable.store import DurableAgentStore
from nova_voice.providers.commitments.provider import CommitmentsProvider


async def _manager(tmp_path) -> tuple[CommitmentManager, DurableAgentStore]:
    store = DurableAgentStore(tmp_path / "durable.sqlite3")
    await store.initialize()
    return CommitmentManager(store, poll_seconds=60), store


async def test_commitment_provider_contract_and_timezone_creation(tmp_path) -> None:
    manager, _ = await _manager(tmp_path)
    provider = CommitmentsProvider(manager)
    registry = CapabilityRegistry(allowlist={"commitments"})
    registry.register(provider)
    action = PlannedAction(
        id="create-1",
        order=0,
        call=CapabilityToolCall(
            provider="commitments",
            tool="commitments.create",
            arguments={
                "summary": "Call Sam",
                "due": "2026-08-03T09:00:00",
                "timezone": "Pacific/Auckland",
                "recurrence": "FREQ=WEEKLY",
            },
        ),
    )

    result = await provider.execute(action)

    assert len(registry.tool_catalog()) == 5
    assert result.ok
    assert result.observed["commitment"]["due_at"].endswith("+12:00")


def test_recurrence_preserves_schedule_and_clamps_month_end() -> None:
    value = datetime(2027, 1, 31, 9, tzinfo=UTC)
    assert next_recurrence(value, "FREQ=DAILY;INTERVAL=2") == value + timedelta(days=2)
    assert next_recurrence(value, "FREQ=WEEKLY") == value + timedelta(weeks=1)
    assert next_recurrence(value, "FREQ=MONTHLY") == datetime(2027, 2, 28, 9, tzinfo=UTC)


async def test_due_occurrence_is_claimed_exactly_once_across_restart(tmp_path) -> None:
    manager, store = await _manager(tmp_path)
    now = datetime(2026, 8, 1, 9, tzinfo=UTC)
    await manager.create(
        CommitmentRecord(
            id="commitment-1",
            owner_id="owner",
            summary="Call Sam",
            due_at=now,
        ),
        actor_id="owner",
    )

    first = await manager.poll(now=now)
    restarted = CommitmentManager(store)
    second = await restarted.poll(now=now + timedelta(minutes=1))
    interventions = await store.list(ProactiveInterventionRecord)

    assert [record.id for record in first] == ["commitment-1"]
    assert second == ()
    assert len(interventions) == 1


async def test_missed_one_time_and_recurring_commitments_recover_differently(tmp_path) -> None:
    manager, _ = await _manager(tmp_path)
    due = datetime(2026, 8, 1, 9, tzinfo=UTC)
    deadline = due + timedelta(minutes=10)
    for record_id, recurrence in (("once", None), ("daily", "FREQ=DAILY")):
        await manager.create(
            CommitmentRecord(
                id=record_id,
                owner_id="owner",
                summary=record_id,
                due_at=due,
                deadline=deadline,
                recurrence=recurrence,
            ),
            actor_id="owner",
        )

    await manager.poll(now=due + timedelta(days=2, minutes=20))
    records = {record.id: record for record in await manager.list()}

    assert records["once"].status == CommitmentState.MISSED
    assert records["once"].missed_count == 1
    assert records["daily"].status == CommitmentState.ACTIVE
    assert records["daily"].due_at > due + timedelta(days=2)
    assert records["daily"].missed_count == 1


async def test_wait_event_and_cross_device_continuation(tmp_path) -> None:
    manager, _ = await _manager(tmp_path)
    now = datetime.now(UTC)
    await manager.create(
        CommitmentRecord(
            id="event-1",
            owner_id="owner",
            summary="Check washing",
            wait_event_key="washer.finished",
        ),
        actor_id="owner",
    )
    await manager.satisfy_event("washer.finished", now=now)
    await manager.poll(now=now)

    snoozed = await manager.snooze("event-1", until=now + timedelta(hours=1), device="ipad")

    assert snoozed.status == CommitmentState.ACTIVE
    assert snoozed.continuation_device == "ipad"


async def test_commitment_api_lists_and_completes_on_another_device(tmp_path) -> None:
    manager, _ = await _manager(tmp_path)
    now = datetime.now(UTC)
    await manager.create(
        CommitmentRecord(id="api-1", owner_id="owner", summary="Test", due_at=now),
        actor_id="owner",
    )
    await manager.poll(now=now)
    app = create_app(Settings(), service=SimpleNamespace(commitments=manager))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="https://voice.test") as client:
        listed = await client.get("/v1/commitments")
        completed = await client.post("/v1/commitments/api-1/complete", json={"device": "browser"})

    assert listed.json()["commitments"][0]["id"] == "api-1"
    assert completed.json()["commitment"]["status"] == "completed"
    assert completed.json()["commitment"]["continuation_device"] == "browser"
