"""Conservative durable proactive-intervention policy for household events."""
# ruff: noqa: E501

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from nova_voice.durable.models import (
    EventRecord,
    ProactiveInterventionRecord,
    ProactiveInterventionState,
    utc_now,
)
from nova_voice.durable.store import DurableAgentStore


@dataclass(frozen=True)
class ProactiveDecision:
    reason_code: str
    detail: str
    channel: Literal["voice", "dashboard", "notification"]
    room_id: str | None
    deduplication_key: str


class ProactivePolicy:
    """Maps normalized events to reviewable suggestions, never side effects."""

    def __init__(self, *, quiet_start_hour: int = 22, quiet_end_hour: int = 7) -> None:
        self.quiet_start_hour = quiet_start_hour
        self.quiet_end_hour = quiet_end_hour

    def quiet_hours(self, now: datetime) -> bool:
        hour = now.hour
        if self.quiet_start_hour == self.quiet_end_hour:
            return False
        if self.quiet_start_hour < self.quiet_end_hour:
            return self.quiet_start_hour <= hour < self.quiet_end_hour
        return hour >= self.quiet_start_hour or hour < self.quiet_end_hour

    def evaluate(self, event: EventRecord, *, occupied_rooms: set[str], now: datetime) -> ProactiveDecision | None:
        payload = event.payload
        target = str(payload.get("entity_id") or payload.get("device_id") or payload.get("target") or "household")
        room = str(payload.get("room") or payload.get("area") or "") or None
        if event.kind == "device_health" and str(payload.get("status") or payload.get("state") or "").casefold() in {"offline", "unavailable", "failed"}:
            return ProactiveDecision("device_offline", f"{target} needs attention", "dashboard", room, f"device-offline:{target}")
        if event.kind == "energy" and bool(payload.get("unusual") or payload.get("alert")):
            return ProactiveDecision("unusual_energy", f"Unusual energy use from {target}", "dashboard", room, f"energy:{target}:{payload.get('period', 'current')}")
        if event.kind == "ha_state" and str(payload.get("state") or "").casefold() in {"open", "on"} and bool(payload.get("risk")):
            channel: Literal["voice", "dashboard", "notification"] = "voice"
            if self.quiet_hours(now) or (room is not None and room not in occupied_rooms):
                channel = "dashboard"
            return ProactiveDecision("household_risk", f"{target} needs a household check", channel, room, f"risk:{target}:{payload.get('state')}")
        return None


class ProactiveInterventionEngine:
    def __init__(self, store: DurableAgentStore, policy: ProactivePolicy | None = None) -> None:
        self.store = store
        self.policy = policy or ProactivePolicy()
        self.occupied_rooms: set[str] = set()

    async def handle_event(self, event: EventRecord) -> None:
        """Update local occupancy and persist a proposal without blocking ingestion."""

        if event.kind == "occupancy":
            room = str(event.payload.get("room") or event.payload.get("area") or "")
            person = str(event.payload.get("person") or event.payload.get("person_id") or "")
            if room and str(event.payload.get("state") or "present").casefold() not in {"away", "absent"}:
                self.occupied_rooms.add(room)
            elif person:
                self.occupied_rooms.discard(room)
            return
        await self.consider(event, occupied_rooms=self.occupied_rooms)

    async def consider(self, event: EventRecord, *, occupied_rooms: set[str], now: datetime | None = None) -> ProactiveInterventionRecord | None:
        current = now or utc_now()
        decision = self.policy.evaluate(event, occupied_rooms=occupied_rooms, now=current)
        if decision is None:
            return None
        existing = await self.store.list(ProactiveInterventionRecord)
        if any(item.record.deduplication_key == decision.deduplication_key and item.record.status in {ProactiveInterventionState.PROPOSED, ProactiveInterventionState.APPROVED, ProactiveInterventionState.DELIVERED} for item in existing):
            return None
        record = ProactiveInterventionRecord(
            id=f"proactive:{decision.deduplication_key}", created_at=current, updated_at=current,
            event_id=event.id, reason_code=decision.reason_code, reason_detail=decision.detail,
            channel=decision.channel, status=ProactiveInterventionState.PROPOSED, deduplication_key=decision.deduplication_key, room_id=decision.room_id,
        )
        await self.store.create(record, actor_id="proactive-policy")
        return record

    async def feedback(
        self,
        intervention_id: str,
        *,
        outcome: Literal["accepted", "dismissed", "redundant", "annoying"],
        actor_id: str,
        now: datetime | None = None,
    ) -> ProactiveInterventionRecord:
        """Persist household feedback used to tune, rather than bypass, policy."""

        stored = await self.store.get(ProactiveInterventionRecord, intervention_id)
        if stored is None:
            raise KeyError(intervention_id)
        record = stored.record
        current = now or utc_now()
        status = (
            ProactiveInterventionState.DELIVERED
            if outcome == "accepted"
            else ProactiveInterventionState.DISMISSED
        )
        updated = record.model_copy(
            update={
                "status": status,
                "feedback": outcome,
                "feedback_at": current,
                "delivered_at": current if outcome == "accepted" else record.delivered_at,
                "updated_at": current,
            }
        )
        return (
            await self.store.save(
                updated,
                expected_revision=stored.revision,
                actor_id=actor_id,
            )
        ).record
