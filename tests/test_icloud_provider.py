from __future__ import annotations

from datetime import UTC, datetime

import httpx

from nova_voice.authority import action_capability
from nova_voice.capabilities.registry import CapabilityRegistry
from nova_voice.config import Settings
from nova_voice.domain import CapabilityToolCall, PlannedAction
from nova_voice.providers.icloud.client import (
    ICloudCalDAVClient,
    PersonalItem,
    parse_calendar_data,
    serialize_item,
)
from nova_voice.providers.icloud.provider import ICloudProvider


class _Client:
    def __init__(self) -> None:
        self.items: dict[tuple[str, str], PersonalItem] = {}
        self.closed = False

    async def list_items(self, kind, *, start=None, end=None):
        values = [item for (item_kind, _), item in self.items.items() if item_kind == kind]
        return tuple(
            item
            for item in values
            if (
                start is None
                or (item.starts_at or item.due_at) is None
                or (item.starts_at or item.due_at) >= start
            )
            and (
                end is None
                or (item.starts_at or item.due_at) is None
                or (item.starts_at or item.due_at) <= end
            )
        )

    async def get_item(self, kind, uid):
        return self.items.get((kind, uid))

    async def put_item(self, item, *, create_only=False):
        key = (item.kind, item.uid)
        if create_only and key in self.items:
            return
        self.items[key] = item.model_copy(update={"revision": '"2"', "href": f"/{item.uid}.ics"})

    async def delete_item(self, item):
        self.items.pop((item.kind, item.uid), None)

    async def health(self):
        return {"ok": True, "collections": 2}

    async def close(self):
        self.closed = True


def _action(tool: str, arguments: dict, *, action_id: str = "turn-1") -> PlannedAction:
    return PlannedAction(
        id=action_id,
        order=0,
        call=CapabilityToolCall(provider="icloud", tool=tool, arguments=arguments),
    )


def test_icloud_manifest_has_explicit_read_and_authority_safe_write_policies() -> None:
    provider = ICloudProvider(_Client())
    registry = CapabilityRegistry(allowlist={"icloud"})
    registry.register(provider)

    assert len(registry.tool_catalog()) == 9
    assert registry.policy_for("icloud", "icloud.calendar.list").cancellation == "anytime"
    create = registry.policy_for("icloud", "icloud.calendar.create")
    assert create.risk == "low"
    assert not create.requires_confirmation
    assert create.cancellation == "before_side_effects"
    assert (
        action_capability(_action("icloud.calendar.create", {"uid": "a"}))
        == "icloud.calendar.create"
    )


async def test_calendar_create_is_timezone_aware_idempotent_and_verified() -> None:
    client = _Client()
    provider = ICloudProvider(client)
    action = _action(
        "icloud.calendar.create",
        {
            "title": "Dentist",
            "start": "2026-08-02T10:00:00",
            "end": "2026-08-02T10:30:00",
            "timezone": "Pacific/Auckland",
            "recurrence": "FREQ=MONTHLY;COUNT=2",
        },
    )

    first = await provider.execute(action)
    second = await provider.execute(action)

    assert first.ok and second.ok
    assert first.target == "nova-turn-1"
    item = await client.get_item("calendar", "nova-turn-1")
    assert item is not None
    assert item.starts_at.utcoffset().total_seconds() == 12 * 3600
    assert item.recurrence == "FREQ=MONTHLY;COUNT=2"
    assert len(client.items) == 1


async def test_reminder_create_complete_list_and_cancel_are_verified() -> None:
    client = _Client()
    provider = ICloudProvider(client)
    created = await provider.execute(
        _action(
            "icloud.reminders.create",
            {
                "uid": "reminder-1",
                "title": "Water plants",
                "due": "2026-08-03T18:00:00+12:00",
                "timezone": "Pacific/Auckland",
            },
        )
    )
    completed = await provider.execute(_action("icloud.reminders.complete", {"uid": "reminder-1"}))
    listed = await provider.execute(_action("icloud.reminders.list", {}))
    cancelled = await provider.execute(_action("icloud.reminders.cancel", {"uid": "reminder-1"}))

    assert created.ok and completed.ok and listed.ok and cancelled.ok
    assert listed.observed["items"][0]["completed"] is True
    assert await client.get_item("reminder", "reminder-1") is None


async def test_calendar_and_reminder_updates_preserve_identity_and_verify() -> None:
    client = _Client()
    provider = ICloudProvider(client)
    await provider.execute(
        _action(
            "icloud.calendar.create",
            {
                "uid": "event-1",
                "title": "Planning",
                "start": "2026-08-02T10:00:00+12:00",
                "end": "2026-08-02T11:00:00+12:00",
                "timezone": "Pacific/Auckland",
            },
        )
    )
    updated_event = await provider.execute(
        _action(
            "icloud.calendar.update",
            {"uid": "event-1", "title": "Revised planning"},
        )
    )
    await provider.execute(
        _action(
            "icloud.reminders.create",
            {
                "uid": "reminder-2",
                "title": "Original",
                "timezone": "Pacific/Auckland",
            },
        )
    )
    updated_reminder = await provider.execute(
        _action(
            "icloud.reminders.update",
            {
                "uid": "reminder-2",
                "title": "Revised",
                "due": "2026-08-04T09:00:00+12:00",
            },
        )
    )

    assert updated_event.ok and updated_event.observed["title"] == "Revised planning"
    assert updated_reminder.ok and updated_reminder.observed["title"] == "Revised"
    assert len(client.items) == 2


async def test_calendar_rejects_invalid_range_before_side_effects() -> None:
    client = _Client()
    result = await ICloudProvider(client).execute(
        _action(
            "icloud.calendar.create",
            {
                "title": "Backwards",
                "start": "2026-08-02T11:00:00+12:00",
                "end": "2026-08-02T10:00:00+12:00",
                "timezone": "Pacific/Auckland",
            },
        )
    )
    assert not result.ok and result.code == "invalid"
    assert not client.items


def test_ical_round_trip_preserves_timezone_recurrence_and_completion() -> None:
    item = PersonalItem(
        uid="event-1",
        kind="calendar",
        title="Planning, session",
        starts_at=datetime(2026, 8, 2, 10, tzinfo=UTC),
        ends_at=datetime(2026, 8, 2, 11, tzinfo=UTC),
        recurrence="FREQ=WEEKLY;COUNT=3",
    )
    parsed = parse_calendar_data(
        serialize_item(item), href="https://example/event-1.ics", revision='"1"'
    )
    assert parsed.title == "Planning, session"
    assert parsed.starts_at == item.starts_at
    assert parsed.recurrence == item.recurrence


async def test_caldav_client_reads_collection_and_uses_conditional_writes() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "REPORT":
            payload = serialize_item(
                PersonalItem(
                    uid="event-1",
                    kind="calendar",
                    title="Dentist",
                    starts_at=datetime(2026, 8, 2, 10, tzinfo=UTC),
                    ends_at=datetime(2026, 8, 2, 11, tzinfo=UTC),
                )
            )
            xml = (
                '<D:multistatus xmlns:D="DAV:" '
                'xmlns:C="urn:ietf:params:xml:ns:caldav">'
                "<D:response><D:href>/calendar/event-1.ics</D:href>"
                '<D:propstat><D:prop><D:getetag>"1"</D:getetag>'
                f"<C:calendar-data>{payload}</C:calendar-data>"
                "</D:prop></D:propstat></D:response></D:multistatus>"
            )
            return httpx.Response(207, text=xml)
        return httpx.Response(204)

    client = ICloudCalDAVClient(
        username="owner@example.test",
        app_password="secret",
        calendar_url="https://caldav.example/calendar/",
        reminders_url="https://caldav.example/reminders/",
        transport=httpx.MockTransport(handler),
    )
    items = await client.list_items("calendar")
    await client.put_item(items[0].model_copy(update={"title": "Updated"}))
    await client.delete_item(items[0])
    await client.close()

    assert items[0].href == "https://caldav.example/calendar/event-1.ics"
    assert [request.method for request in requests] == ["REPORT", "PUT", "DELETE"]
    assert requests[1].headers["if-match"] == '"1"'


def test_icloud_is_registered_only_when_all_secrets_and_collections_exist() -> None:
    assert not Settings().icloud_configured
    configured = Settings(
        icloud_username="owner@example.test",
        icloud_app_password="app-password",
        icloud_calendar_url="https://caldav.example/calendar/",
        icloud_reminders_url="https://caldav.example/reminders/",
    )
    assert configured.icloud_configured
