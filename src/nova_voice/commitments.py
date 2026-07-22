from __future__ import annotations

import asyncio
import calendar
import sqlite3
from datetime import UTC, datetime, timedelta
from typing import cast
from uuid import uuid4

from nova_voice.durable.models import (
    CommitmentRecord,
    CommitmentState,
    ProactiveInterventionRecord,
    ProactiveInterventionState,
    utc_now,
)
from nova_voice.durable.store import ConcurrentRecordUpdate, DurableAgentStore


def next_recurrence(value: datetime, rule: str) -> datetime:
    fields = dict(item.split("=", 1) for item in rule.upper().split(";") if "=" in item)
    frequency = fields.get("FREQ")
    interval = int(fields.get("INTERVAL", "1"))
    if interval < 1 or interval > 365:
        raise ValueError("recurrence interval is out of range")
    if frequency == "DAILY":
        return value + timedelta(days=interval)
    if frequency == "WEEKLY":
        return value + timedelta(weeks=interval)
    if frequency == "MONTHLY":
        month_index = value.year * 12 + value.month - 1 + interval
        year, month_zero = divmod(month_index, 12)
        month = month_zero + 1
        day = min(value.day, calendar.monthrange(year, month)[1])
        return value.replace(year=year, month=month, day=day)
    raise ValueError("recurrence must use DAILY, WEEKLY, or MONTHLY")


class CommitmentManager:
    """Restart-safe reminders, conditions, recurrence, and missed recovery."""

    def __init__(
        self,
        store: DurableAgentStore,
        *,
        poll_seconds: float = 5,
    ) -> None:
        self.store = store
        self.poll_seconds = max(1, poll_seconds)
        self._task: asyncio.Task | None = None

    async def create(self, record: CommitmentRecord, *, actor_id: str) -> CommitmentRecord:
        if record.recurrence:
            next_recurrence(record.due_at or utc_now(), record.recurrence)
        stored = await self.store.create(record, actor_id=actor_id)
        return cast(CommitmentRecord, stored.record)

    async def list(self) -> tuple[CommitmentRecord, ...]:
        return tuple(
            cast(CommitmentRecord, item.record) for item in await self.store.list(CommitmentRecord)
        )

    async def _save(self, record: CommitmentRecord, *, actor_id: str) -> CommitmentRecord:
        stored = await self.store.get(CommitmentRecord, record.id)
        if stored is None:
            raise KeyError(record.id)
        result = await self.store.save(
            record,
            expected_revision=stored.revision,
            actor_id=actor_id,
        )
        return cast(CommitmentRecord, result.record)

    async def poll(self, *, now: datetime | None = None) -> tuple[CommitmentRecord, ...]:
        current = (now or utc_now()).astimezone(UTC)
        due: list[CommitmentRecord] = []
        for record in await self.list():
            if record.status != CommitmentState.ACTIVE or record.due_at is None:
                continue
            if record.due_at.astimezone(UTC) > current:
                continue
            if record.deadline and record.deadline.astimezone(UTC) < current:
                update = {
                    "missed_count": record.missed_count + 1,
                    "updated_at": current,
                }
                if record.recurrence:
                    next_due = record.due_at
                    while next_due.astimezone(UTC) <= current:
                        next_due = next_recurrence(next_due, record.recurrence)
                    update.update(
                        {
                            "status": CommitmentState.ACTIVE,
                            "due_at": next_due,
                            "deadline": None,
                            "occurrence": record.occurrence + 1,
                        }
                    )
                else:
                    update["status"] = CommitmentState.MISSED
                try:
                    await self._save(record.model_copy(update=update), actor_id="commitment-worker")
                except ConcurrentRecordUpdate:
                    continue
                continue
            intervention = ProactiveInterventionRecord(
                id=f"intervention:commitment:{record.id}:{record.occurrence}",
                reason_code="commitment_due",
                reason_detail=record.summary,
                channel=record.channels[0],
                status=ProactiveInterventionState.PROPOSED,
                deduplication_key=f"commitment:{record.id}:{record.occurrence}",
            )
            try:
                await self.store.create(intervention, actor_id="commitment-worker")
                saved = await self._save(
                    record.model_copy(
                        update={"status": CommitmentState.DUE, "updated_at": current}
                    ),
                    actor_id="commitment-worker",
                )
            except (ConcurrentRecordUpdate, sqlite3.IntegrityError) as error:
                # A duplicate intervention means a prior process already claimed
                # this occurrence; do not emit it twice after restart.
                if isinstance(error, ConcurrentRecordUpdate):
                    continue
                existing = await self.store.get(ProactiveInterventionRecord, intervention.id)
                if existing is None:
                    raise
                saved = await self._save(
                    record.model_copy(
                        update={"status": CommitmentState.DUE, "updated_at": current}
                    ),
                    actor_id="commitment-worker",
                )
            due.append(saved)
        return tuple(due)

    async def satisfy_event(
        self, event_key: str, *, now: datetime | None = None
    ) -> tuple[CommitmentRecord, ...]:
        current = now or utc_now()
        changed = []
        for record in await self.list():
            if record.status == CommitmentState.ACTIVE and record.wait_event_key == event_key:
                changed.append(
                    await self._save(
                        record.model_copy(
                            update={
                                "due_at": current,
                                "wait_event_key": None,
                                "updated_at": current,
                            }
                        ),
                        actor_id="household-event",
                    )
                )
        return tuple(changed)

    async def acknowledge(
        self, commitment_id: str, *, device: str, now: datetime | None = None
    ) -> CommitmentRecord:
        current = now or utc_now()
        stored = await self.store.get(CommitmentRecord, commitment_id)
        if stored is None:
            raise KeyError(commitment_id)
        record = cast(CommitmentRecord, stored.record)
        if record.status not in {CommitmentState.DUE, CommitmentState.DELIVERED}:
            raise ValueError("commitment is not awaiting acknowledgement")
        if record.recurrence and record.due_at:
            update = {
                "status": CommitmentState.ACTIVE,
                "due_at": next_recurrence(record.due_at, record.recurrence),
                "deadline": None,
                "occurrence": record.occurrence + 1,
                "delivered_at": None,
                "completed_at": None,
                "continuation_device": device,
                "updated_at": current,
            }
        else:
            update = {
                "status": CommitmentState.COMPLETED,
                "delivered_at": current,
                "completed_at": current,
                "continuation_device": device,
                "updated_at": current,
            }
        return await self._save(record.model_copy(update=update), actor_id=device)

    async def snooze(self, commitment_id: str, *, until: datetime, device: str) -> CommitmentRecord:
        stored = await self.store.get(CommitmentRecord, commitment_id)
        if stored is None:
            raise KeyError(commitment_id)
        record = cast(CommitmentRecord, stored.record)
        if until.utcoffset() is None or until <= utc_now():
            raise ValueError("snooze time must be a future timezone-aware datetime")
        return await self._save(
            record.model_copy(
                update={
                    "status": CommitmentState.ACTIVE,
                    "due_at": until,
                    "continuation_device": device,
                    "updated_at": utc_now(),
                }
            ),
            actor_id=device,
        )

    async def cancel(self, commitment_id: str, *, device: str) -> CommitmentRecord:
        stored = await self.store.get(CommitmentRecord, commitment_id)
        if stored is None:
            raise KeyError(commitment_id)
        record = cast(CommitmentRecord, stored.record)
        return await self._save(
            record.model_copy(
                update={
                    "status": CommitmentState.CANCELLED,
                    "continuation_device": device,
                    "updated_at": utc_now(),
                }
            ),
            actor_id=device,
        )

    async def _run(self) -> None:
        while True:
            await self.poll()
            await asyncio.sleep(self.poll_seconds)

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())

    async def close(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)
        self._task = None

    @staticmethod
    def new_id() -> str:
        return f"commitment-{uuid4().hex}"
