from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import httpx

from nova_voice.api import create_app
from nova_voice.briefings import BriefingManager
from nova_voice.config import Settings
from nova_voice.domain import CapabilityToolCall, PlannedAction
from nova_voice.durable.models import (
    BriefingScheduleRecord,
    CommitmentRecord,
    EventRecord,
    EventSubscriptionRecord,
    ProactiveInterventionRecord,
)
from nova_voice.durable.store import DurableAgentStore
from nova_voice.providers.briefings.provider import BriefingsProvider
from nova_voice.providers.icloud.client import PersonalItem


class FakeCalendar:
    async def list_items(self, kind, *, start=None, end=None):
        return (
            PersonalItem(
                uid="calendar-1",
                kind="calendar",
                title="Project meeting",
                starts_at=start + timedelta(minutes=30),
                ends_at=start + timedelta(minutes=90),
                timezone="Pacific/Auckland",
            ),
        )


async def _manager(tmp_path, *, calendar=None):
    store = DurableAgentStore(tmp_path / "durable.sqlite3")
    await store.initialize()
    return BriefingManager(store, calendar=calendar, poll_seconds=60), store


async def test_daily_briefing_combines_agenda_conflicts_and_preparation(tmp_path) -> None:
    manager, store = await _manager(tmp_path, calendar=FakeCalendar())
    now = datetime(2026, 7, 21, 20, 5, tzinfo=UTC)  # 08:05 in Auckland
    await store.create(
        CommitmentRecord(
            id="commitment-briefing",
            owner_id="owner",
            summary="Call the builder",
            due_at=now + timedelta(minutes=45),
        ),
        actor_id="owner",
    )
    await manager.create_schedule(
        BriefingScheduleRecord(
            id="morning-owner",
            owner_id="owner",
            period="morning",
            local_time="08:00",
            timezone="Pacific/Auckland",
            channels=("voice",),
        )
    )

    first = await manager.poll(now=now)
    second = await manager.poll(now=now + timedelta(hours=1))
    interventions = await store.list(ProactiveInterventionRecord)

    assert len(first) == 1
    assert second == ()
    assert len(first[0].agenda) == 2
    assert first[0].conflicts[0]["first"] == "Project meeting"
    assert len(first[0].preparation_prompts) == 2
    assert len(interventions) == 1


async def test_event_subscription_matches_payload_and_is_one_shot(tmp_path) -> None:
    manager, store = await _manager(tmp_path)
    subscription = await manager.create_subscription(
        EventSubscriptionRecord(
            id="washer-done",
            owner_id="owner",
            summary="The washing machine is finished.",
            event_kind="ha_state",
            match={"entity_id": "sensor.washer", "state": "done"},
            channels=("voice",),
        )
    )
    event = EventRecord(
        id="event-washer",
        source="dashboard",
        kind="ha_state",
        payload={"entity_id": "sensor.washer", "state": "done"},
        payload_revision="r1",
    )

    first = await manager.handle_event(event)
    second = await manager.handle_event(event)

    assert first[0].id == subscription.id
    assert first[0].active is False
    assert first[0].trigger_count == 1
    assert second == ()
    assert len(await store.list(ProactiveInterventionRecord)) == 1


async def test_briefing_provider_validates_timezone_and_creates_subscription(tmp_path) -> None:
    manager, _ = await _manager(tmp_path)
    provider = BriefingsProvider(manager)
    bad = await provider.execute(
        PlannedAction(
            id="bad-zone",
            order=0,
            call=CapabilityToolCall(
                provider="briefings",
                tool="briefings.schedule",
                arguments={
                    "ownerId": "owner",
                    "period": "morning",
                    "localTime": "08:00",
                    "timezone": "Not/AZone",
                },
            ),
        )
    )
    created = await provider.execute(
        PlannedAction(
            id="subscription-create",
            order=0,
            call=CapabilityToolCall(
                provider="briefings",
                tool="subscriptions.create",
                arguments={
                    "ownerId": "owner",
                    "summary": "Tell me when it rains",
                    "eventKind": "weather",
                    "match": {"condition": "rain"},
                },
            ),
        )
    )

    assert bad.ok is False
    assert bad.code == "invalid"
    assert created.ok
    assert created.observed["subscription"]["active"] is True


async def test_briefing_and_subscription_api_expose_records(tmp_path) -> None:
    manager, _ = await _manager(tmp_path)
    await manager.create_subscription(
        EventSubscriptionRecord(
            id="api-subscription",
            owner_id="owner",
            summary="API test",
            event_kind="energy",
        )
    )
    app = create_app(Settings(), service=SimpleNamespace(briefings=manager))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://voice.test"
    ) as client:
        listed = await client.get("/v1/subscriptions")
        cancelled = await client.post(
            "/v1/subscriptions/api-subscription/cancel", json={"actor": "owner"}
        )

    assert listed.status_code == 200
    assert listed.json()["subscriptions"][0]["id"] == "api-subscription"
    assert cancelled.json()["subscription"]["active"] is False
