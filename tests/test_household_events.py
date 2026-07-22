from __future__ import annotations

import asyncio

import httpx
import pytest

from nova_voice.durable.models import EventRecord
from nova_voice.durable.store import DurableAgentStore
from nova_voice.events import HouseholdEventConsumer
from nova_voice.providers.nova.client import NovaDashboardClient


def _event(cursor: int, kind: str = "ha_state") -> dict:
    return {
        "version": 1,
        "cursor": cursor,
        "id": f"dashboard-{cursor}",
        "occurredAt": f"2026-07-22T10:00:{cursor:02d}+00:00",
        "source": "home_assistant",
        "kind": kind,
        "deduplicationKey": f"ha:event-{cursor}",
        "payload": {"entityId": f"light.fixture_{cursor}", "state": "on"},
    }


def _batch(after: int, events: list[dict], *, first: int = 1, reset: bool = False) -> dict:
    return {
        "version": 1,
        "after": after,
        "firstAvailableCursor": first,
        "nextCursor": events[-1]["cursor"] if events else max(after, first - 1),
        "resetRequired": reset,
        "events": events,
    }


class _Client:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = responses
        self.calls: list[tuple[int, int]] = []

    async def household_events(self, after: int, limit: int = 200) -> dict:
        self.calls.append((after, limit))
        return self.responses.pop(0)


async def test_consumer_persists_events_checkpoint_and_resumes_without_duplicates(tmp_path) -> None:
    store = DurableAgentStore(tmp_path / "durable.sqlite3")
    await store.initialize()
    client = _Client([_batch(0, [_event(1), _event(2, "occupancy")])])
    consumer = HouseholdEventConsumer(client, store)
    await consumer.initialize()

    assert await consumer.poll_once() == 2
    assert consumer.cursor == 2

    restarted = HouseholdEventConsumer(_Client([_batch(2, [])]), DurableAgentStore(store.path))
    await restarted.store.initialize()
    await restarted.initialize()
    assert restarted.cursor == 2
    assert await restarted.poll_once() == 0
    records = await restarted.store.list(EventRecord)
    event_records = [
        record.record for record in records if record.record.kind != "cursor_checkpoint"
    ]
    assert [record.cursor for record in event_records] == ["1", "2"]


async def test_consumer_records_retention_gap_then_continues_from_first_available(tmp_path) -> None:
    store = DurableAgentStore(tmp_path / "durable.sqlite3")
    await store.initialize()
    client = _Client([_batch(0, [_event(3)], first=3, reset=True)])
    consumer = HouseholdEventConsumer(client, store)
    await consumer.initialize()

    assert await consumer.poll_once() == 1
    assert consumer.cursor == 3
    gaps = [
        record.record
        for record in await store.list(EventRecord)
        if record.record.kind == "cursor_gap"
    ]
    assert len(gaps) == 1
    assert gaps[0].payload == {"missingFrom": 1, "missingThrough": 2}


async def test_consumer_rejects_unannounced_cursor_gaps(tmp_path) -> None:
    store = DurableAgentStore(tmp_path / "durable.sqlite3")
    await store.initialize()
    consumer = HouseholdEventConsumer(_Client([_batch(0, [_event(2)])]), store)
    await consumer.initialize()

    with pytest.raises(ValueError, match="cursor gap"):
        await consumer.poll_once()
    assert consumer.cursor == 0


async def test_dashboard_client_authenticates_cursor_feed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer service-token"
        assert dict(request.url.params) == {"after": "7", "limit": "25"}
        return httpx.Response(200, json=_batch(7, []))

    client = NovaDashboardClient(
        "http://nova.local",
        mcp_token="service-token",
        transport=httpx.MockTransport(handler),
    )
    try:
        assert (await client.household_events(7, 25))["after"] == 7
    finally:
        await client.close()


async def test_consumer_background_health_is_content_free(tmp_path) -> None:
    store = DurableAgentStore(tmp_path / "durable.sqlite3")
    await store.initialize()
    consumer = HouseholdEventConsumer(_Client([_batch(0, [])]), store, poll_seconds=60)
    await consumer.initialize()
    consumer.start()
    try:
        for _ in range(20):
            if consumer.health()["lastSuccessAt"] is not None:
                break
            await asyncio.sleep(0)
        health = consumer.health()
        assert health["ok"]
        assert health["running"]
        assert health["cursor"] == 0
        assert health["lastError"] is None
    finally:
        await consumer.close()
