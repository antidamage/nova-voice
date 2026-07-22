from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from nova_voice.durable.models import (
    BriefingRecord,
    BriefingScheduleRecord,
    CommitmentRecord,
    CommitmentState,
    EventRecord,
    EventSubscriptionRecord,
    ProactiveInterventionRecord,
    ProactiveInterventionState,
    utc_now,
)
from nova_voice.durable.store import ConcurrentRecordUpdate, DurableAgentStore
from nova_voice.providers.icloud.client import PersonalItem


class CalendarSource(Protocol):
    async def list_items(
        self, kind: str, *, start: datetime | None = None, end: datetime | None = None
    ) -> tuple[PersonalItem, ...]: ...


def _time_value(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _conflicts(agenda: list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    timed = []
    for item in agenda:
        start = item.get("start")
        if not start:
            continue
        start_at = datetime.fromisoformat(str(start))
        end_at = datetime.fromisoformat(str(item.get("end") or start))
        if end_at <= start_at:
            end_at = start_at + timedelta(minutes=30)
        timed.append((start_at.astimezone(UTC), end_at.astimezone(UTC), item))
    timed.sort(key=lambda row: row[0])
    found = []
    for index, (start, end, item) in enumerate(timed):
        for other_start, other_end, other in timed[index + 1 :]:
            if other_start >= end:
                break
            if start < other_end and other_start < end:
                found.append(
                    {
                        "first": item["title"],
                        "second": other["title"],
                        "overlapStart": max(start, other_start).isoformat(),
                    }
                )
    return tuple(found)


class BriefingManager:
    def __init__(
        self,
        store: DurableAgentStore,
        *,
        calendar: CalendarSource | None = None,
        poll_seconds: float = 30,
    ) -> None:
        self.store = store
        self.calendar = calendar
        self.poll_seconds = max(5, poll_seconds)
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    async def _save(self, record, *, actor_id: str):
        stored = await self.store.get(type(record), record.id)
        if stored is None:
            raise KeyError(record.id)
        saved = await self.store.save(record, expected_revision=stored.revision, actor_id=actor_id)
        return saved.record

    async def create_schedule(self, record: BriefingScheduleRecord) -> BriefingScheduleRecord:
        try:
            ZoneInfo(record.timezone)
        except ZoneInfoNotFoundError as error:
            raise ValueError("unknown IANA timezone") from error
        stored = await self.store.create(record, actor_id=record.owner_id)
        return cast(BriefingScheduleRecord, stored.record)

    async def create_subscription(self, record: EventSubscriptionRecord) -> EventSubscriptionRecord:
        stored = await self.store.create(record, actor_id=record.owner_id)
        return cast(EventSubscriptionRecord, stored.record)

    async def schedules(self) -> tuple[BriefingScheduleRecord, ...]:
        return tuple(
            cast(BriefingScheduleRecord, row.record)
            for row in await self.store.list(BriefingScheduleRecord)
        )

    async def briefings(self) -> tuple[BriefingRecord, ...]:
        return tuple(
            cast(BriefingRecord, row.record) for row in await self.store.list(BriefingRecord)
        )

    async def subscriptions(self) -> tuple[EventSubscriptionRecord, ...]:
        return tuple(
            cast(EventSubscriptionRecord, row.record)
            for row in await self.store.list(EventSubscriptionRecord)
        )

    async def cancel_subscription(
        self, subscription_id: str, *, actor_id: str
    ) -> EventSubscriptionRecord:
        stored = await self.store.get(EventSubscriptionRecord, subscription_id)
        if stored is None:
            raise KeyError(subscription_id)
        record = cast(EventSubscriptionRecord, stored.record)
        return cast(
            EventSubscriptionRecord,
            await self._save(
                record.model_copy(update={"active": False, "updated_at": utc_now()}),
                actor_id=actor_id,
            ),
        )

    async def handle_event(self, event: EventRecord) -> tuple[EventSubscriptionRecord, ...]:
        triggered = []
        for subscription in await self.subscriptions():
            if not subscription.active or subscription.event_kind != event.kind:
                continue
            if any(event.payload.get(key) != value for key, value in subscription.match.items()):
                continue
            intervention = ProactiveInterventionRecord(
                id=f"intervention:subscription:{subscription.id}:{event.id}",
                event_id=event.id,
                reason_code="event_subscription",
                reason_detail=subscription.summary,
                channel=subscription.channels[0],
                status=ProactiveInterventionState.PROPOSED,
                deduplication_key=f"subscription:{subscription.id}:{event.id}",
            )
            try:
                await self.store.create(intervention, actor_id="subscription-worker")
            except sqlite3.IntegrityError:
                if await self.store.get(ProactiveInterventionRecord, intervention.id) is None:
                    raise
            try:
                saved = await self._save(
                    subscription.model_copy(
                        update={
                            "active": not subscription.one_shot,
                            "trigger_count": subscription.trigger_count + 1,
                            "last_event_id": event.id,
                            "triggered_at": event.created_at,
                            "updated_at": utc_now(),
                        }
                    ),
                    actor_id="subscription-worker",
                )
            except ConcurrentRecordUpdate:
                continue
            triggered.append(cast(EventSubscriptionRecord, saved))
        return tuple(triggered)

    async def _agenda(self, start: datetime, end: datetime) -> list[dict[str, Any]]:
        agenda = []
        for commitment in (
            cast(CommitmentRecord, row.record) for row in await self.store.list(CommitmentRecord)
        ):
            if (
                commitment.status in {CommitmentState.ACTIVE, CommitmentState.DUE}
                and commitment.due_at
                and start <= commitment.due_at.astimezone(UTC) < end
            ):
                agenda.append(
                    {
                        "id": commitment.id,
                        "source": "commitment",
                        "title": commitment.summary,
                        "start": _time_value(commitment.due_at),
                        "end": _time_value(commitment.deadline),
                    }
                )
        if self.calendar is not None:
            try:
                items = await self.calendar.list_items("calendar", start=start, end=end)
            except Exception:
                items = ()
            for item in items:
                agenda.append(
                    {
                        "id": item.uid,
                        "source": "icloud",
                        "title": item.title,
                        "start": _time_value(item.starts_at),
                        "end": _time_value(item.ends_at),
                    }
                )
        agenda.sort(key=lambda item: str(item.get("start") or ""))
        return agenda

    async def poll(self, *, now: datetime | None = None) -> tuple[BriefingRecord, ...]:
        current = (now or utc_now()).astimezone(UTC)
        created = []
        for schedule in await self.schedules():
            if not schedule.enabled:
                continue
            local = current.astimezone(ZoneInfo(schedule.timezone))
            local_date = local.date().isoformat()
            if (
                local.strftime("%H:%M") < schedule.local_time
                or schedule.last_local_date == local_date
            ):
                continue
            briefing_id = f"briefing:{schedule.id}:{local_date}"
            existing = await self.store.get(BriefingRecord, briefing_id)
            if existing is None:
                agenda = await self._agenda(current, current + timedelta(hours=24))
                conflicts = _conflicts(agenda)
                soon = [
                    item["title"]
                    for item in agenda
                    if item.get("start")
                    and datetime.fromisoformat(str(item["start"])).astimezone(UTC)
                    <= current + timedelta(hours=4)
                ]
                prompts = tuple(f"Prepare for {title}." for title in soon[:4])
                summary = (
                    f"Your {schedule.period} briefing has {len(agenda)} upcoming item"
                    f"{'s' if len(agenda) != 1 else ''}, {len(conflicts)} schedule conflict"
                    f"{'s' if len(conflicts) != 1 else ''}, and {len(prompts)} preparation prompt"
                    f"{'s' if len(prompts) != 1 else ''}."
                )
                briefing = BriefingRecord(
                    id=briefing_id,
                    schedule_id=schedule.id,
                    owner_id=schedule.owner_id,
                    period=schedule.period,
                    local_date=local_date,
                    summary=summary,
                    agenda=tuple(agenda),
                    conflicts=conflicts,
                    preparation_prompts=prompts,
                )
                try:
                    stored = await self.store.create(briefing, actor_id="briefing-worker")
                except sqlite3.IntegrityError:
                    stored = await self.store.get(BriefingRecord, briefing_id)
                    if stored is None:
                        raise
                created.append(cast(BriefingRecord, stored.record))
                intervention = ProactiveInterventionRecord(
                    id=f"intervention:{briefing_id}",
                    reason_code="scheduled_briefing",
                    reason_detail=summary,
                    channel=schedule.channels[0],
                    status=ProactiveInterventionState.PROPOSED,
                    deduplication_key=briefing_id,
                )
                try:
                    await self.store.create(intervention, actor_id="briefing-worker")
                except sqlite3.IntegrityError:
                    if await self.store.get(ProactiveInterventionRecord, intervention.id) is None:
                        raise
            try:
                await self._save(
                    schedule.model_copy(
                        update={"last_local_date": local_date, "updated_at": utc_now()}
                    ),
                    actor_id="briefing-worker",
                )
            except ConcurrentRecordUpdate:
                continue
        return tuple(created)

    async def run_forever(self) -> None:
        while not self._stop.is_set():
            await self.poll()
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.poll_seconds)
            except TimeoutError:
                pass

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._stop.clear()
            self._task = asyncio.create_task(self.run_forever(), name="briefing-worker")

    async def close(self) -> None:
        self._stop.set()
        if self._task is not None:
            await self._task
