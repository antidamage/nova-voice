from __future__ import annotations

import asyncio
import logging
import sqlite3
from datetime import UTC, datetime, timedelta
from typing import Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nova_voice.durable.models import EventRecord, utc_now
from nova_voice.durable.store import ConcurrentRecordUpdate, DurableAgentStore
from nova_voice.turns import revision

logger = logging.getLogger(__name__)

HouseholdEventKind = Literal[
    "ha_state",
    "occupancy",
    "device_health",
    "weather",
    "energy",
    "reminder",
    "calendar",
    "agent_task",
]
HouseholdEventSource = Literal[
    "home_assistant",
    "dashboard",
    "calendar",
    "reminder",
    "agent_task",
]


class DashboardHouseholdEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    version: Literal[1]
    cursor: int = Field(gt=0)
    id: str = Field(min_length=1, max_length=160)
    occurred_at: datetime = Field(alias="occurredAt")
    source: HouseholdEventSource
    kind: HouseholdEventKind
    deduplication_key: str = Field(alias="deduplicationKey", min_length=1, max_length=240)
    payload: dict

    @model_validator(mode="after")
    def require_aware_time(self) -> DashboardHouseholdEvent:
        if self.occurred_at.utcoffset() is None:
            raise ValueError("household event time must be timezone-aware")
        return self


class DashboardEventBatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    version: Literal[1]
    after: int = Field(ge=0)
    first_available_cursor: int = Field(alias="firstAvailableCursor", gt=0)
    next_cursor: int = Field(alias="nextCursor", ge=0)
    reset_required: bool = Field(alias="resetRequired")
    events: tuple[DashboardHouseholdEvent, ...]

    @model_validator(mode="after")
    def require_ordered_cursors(self) -> DashboardEventBatch:
        cursors = tuple(event.cursor for event in self.events)
        if cursors != tuple(sorted(set(cursors))):
            raise ValueError("dashboard household event cursors must be unique and ordered")
        if cursors and self.next_cursor != cursors[-1]:
            raise ValueError("nextCursor must identify the final returned event")
        return self


class HouseholdEventClient(Protocol):
    async def household_events(self, after: int, limit: int = 200) -> dict: ...


class HouseholdEventConsumer:
    """Poll an authenticated cursor feed into the durable event store exactly once."""

    _CHECKPOINT_ID = "dashboard-household-event-cursor"

    def __init__(
        self,
        client: HouseholdEventClient,
        store: DurableAgentStore,
        *,
        poll_seconds: float = 1,
        batch_size: int = 200,
        retention_days: float = 30,
    ) -> None:
        if poll_seconds <= 0 or not 1 <= batch_size <= 1_000 or retention_days <= 0:
            raise ValueError("household event polling settings must be positive and bounded")
        self.client = client
        self.store = store
        self.poll_seconds = poll_seconds
        self.batch_size = batch_size
        self.retention = timedelta(days=retention_days)
        self.cursor = 0
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._last_success_at: datetime | None = None
        self._last_error: str | None = None

    async def initialize(self) -> None:
        checkpoint = await self.store.get(EventRecord, self._CHECKPOINT_ID)
        if checkpoint is not None:
            value = cast(EventRecord, checkpoint.record).cursor
            self.cursor = int(value or 0)

    async def _checkpoint(self, cursor: int) -> None:
        now = utc_now()
        existing = await self.store.get(EventRecord, self._CHECKPOINT_ID)
        if existing is None:
            record = EventRecord(
                id=self._CHECKPOINT_ID,
                created_at=now,
                updated_at=now,
                source="dashboard",
                kind="cursor_checkpoint",
                cursor=str(cursor),
                payload={},
                payload_revision=revision({"cursor": cursor}),
            )
            try:
                await self.store.create(record, actor_id="dashboard-event-consumer")
            except sqlite3.IntegrityError:
                # A second consumer may have created the singleton checkpoint.
                existing = await self.store.get(EventRecord, self._CHECKPOINT_ID)
                if existing is None:
                    raise
            else:
                self.cursor = cursor
                return
        if existing is None:
            existing = await self.store.get(EventRecord, self._CHECKPOINT_ID)
        if existing is None:
            raise RuntimeError("household event cursor checkpoint disappeared")
        record = cast(EventRecord, existing.record).model_copy(
            update={
                "cursor": str(cursor),
                "payload_revision": revision({"cursor": cursor}),
                "updated_at": now,
            }
        )
        try:
            await self.store.save(
                record,
                expected_revision=existing.revision,
                actor_id="dashboard-event-consumer",
            )
        except ConcurrentRecordUpdate:
            latest = await self.store.get(EventRecord, self._CHECKPOINT_ID)
            if latest is None or int(cast(EventRecord, latest.record).cursor or 0) < cursor:
                raise
        self.cursor = cursor

    async def _record_gap(self, first_available_cursor: int) -> None:
        gap_cursor = first_available_cursor - 1
        gap_id = f"dashboard-household-event-gap:{self.cursor + 1}:{gap_cursor}"
        if await self.store.get(EventRecord, gap_id) is None:
            now = utc_now()
            await self.store.create(
                EventRecord(
                    id=gap_id,
                    created_at=now,
                    updated_at=now,
                    source="dashboard",
                    kind="cursor_gap",
                    cursor=str(gap_cursor),
                    payload={
                        "missingFrom": self.cursor + 1,
                        "missingThrough": gap_cursor,
                    },
                    payload_revision=revision(
                        {"missingFrom": self.cursor + 1, "missingThrough": gap_cursor}
                    ),
                ),
                actor_id="dashboard-event-consumer",
            )
        await self._checkpoint(gap_cursor)

    async def poll_once(self) -> int:
        payload = await self.client.household_events(self.cursor, self.batch_size)
        batch = DashboardEventBatch.model_validate(payload)
        if batch.after != self.cursor:
            raise ValueError("dashboard household event response cursor does not match request")
        if batch.reset_required:
            await self._record_gap(batch.first_available_cursor)
        expected = self.cursor + 1
        accepted = 0
        for event in batch.events:
            if event.cursor < expected:
                continue
            if event.cursor != expected:
                raise ValueError(
                    f"dashboard household event cursor gap: expected {expected}, got {event.cursor}"
                )
            record_id = f"dashboard-household-event:{event.id}"
            if await self.store.get(EventRecord, record_id) is None:
                occurred_at = event.occurred_at.astimezone(UTC)
                await self.store.create(
                    EventRecord(
                        id=record_id,
                        created_at=occurred_at,
                        updated_at=occurred_at,
                        expires_at=occurred_at + self.retention,
                        source=f"dashboard:{event.source}",
                        kind=event.kind,
                        cursor=str(event.cursor),
                        payload=event.payload,
                        payload_revision=revision(event.payload),
                    ),
                    actor_id="dashboard-event-consumer",
                )
                accepted += 1
            await self._checkpoint(event.cursor)
            expected = event.cursor + 1
        self._last_success_at = utc_now()
        self._last_error = None
        return accepted

    async def run_forever(self) -> None:
        backoff = self.poll_seconds
        while not self._stop.is_set():
            try:
                await self.poll_once()
                backoff = self.poll_seconds
            except Exception as error:
                self._last_error = type(error).__name__
                logger.warning("household event polling failed error=%s", type(error).__name__)
                backoff = min(30, max(self.poll_seconds, backoff * 2))
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=backoff)
            except TimeoutError:
                pass

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._stop.clear()
            self._task = asyncio.create_task(
                self.run_forever(),
                name="household-event-consumer",
            )

    async def close(self) -> None:
        self._stop.set()
        if self._task is not None:
            await self._task
        self._task = None

    def health(self) -> dict:
        return {
            "ok": self._last_error is None,
            "enabled": True,
            "running": self._task is not None and not self._task.done(),
            "cursor": self.cursor,
            "lastSuccessAt": (
                self._last_success_at.isoformat() if self._last_success_at is not None else None
            ),
            "lastError": self._last_error,
        }
