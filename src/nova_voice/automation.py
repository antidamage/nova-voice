"""Owner-governed durable automation lifecycle with simulation before activation."""
# ruff: noqa: E501

from __future__ import annotations

from datetime import datetime
from typing import Any

from nova_voice.durable.models import AutomationRecord, AutomationState, utc_now
from nova_voice.durable.store import DurableAgentStore, StoredRecord


class AutomationLifecycleError(ValueError):
    pass


class AutomationManager:
    def __init__(self, store: DurableAgentStore) -> None:
        self.store = store

    async def draft(
        self, *, automation_id: str, owner_id: str, summary: str, trigger: dict[str, Any], actions: list[dict[str, Any]]
    ) -> AutomationRecord:
        if not summary.strip() or not trigger or not actions:
            raise AutomationLifecycleError("a draft requires summary, trigger, and proposed actions")
        now = utc_now()
        record = AutomationRecord(
            id=automation_id, owner_id=owner_id, summary=summary.strip(), trigger=trigger,
            proposed_actions=tuple(actions), created_at=now, updated_at=now,
        )
        await self.store.create(record, actor_id=owner_id)
        return record

    async def _load(self, automation_id: str) -> StoredRecord:
        stored = await self.store.get(AutomationRecord, automation_id)
        if stored is None:
            raise KeyError(automation_id)
        return stored

    async def simulate(self, automation_id: str, *, state: dict[str, Any]) -> AutomationRecord:
        stored = await self._load(automation_id)
        record = stored.record
        if record.state not in {AutomationState.DRAFT, AutomationState.SIMULATED}:
            raise AutomationLifecycleError("only a draft can be simulated")
        # The simulation is intentionally side-effect free and fully retained
        # as approval evidence. Provider-specific historical replay can enrich
        # this payload without weakening the activation boundary.
        simulation = {
            "stateRevision": str(state.get("generatedAt") or "unknown"),
            "triggerMatched": bool(record.trigger),
            "proposedActionCount": len(record.proposed_actions),
            "safe": True,
        }
        updated = record.model_copy(
            update={"state": AutomationState.SIMULATED, "simulation": simulation, "updated_at": utc_now()}
        )
        return (await self.store.save(updated, expected_revision=stored.revision, actor_id="automation-simulator")).record

    async def approve(self, automation_id: str, *, actor_id: str) -> AutomationRecord:
        stored = await self._load(automation_id)
        record = stored.record
        if record.owner_id != actor_id:
            raise PermissionError("only the automation owner can approve it")
        if record.state != AutomationState.SIMULATED or not record.simulation or not record.simulation.get("safe"):
            raise AutomationLifecycleError("a safe simulation is required before approval")
        updated = record.model_copy(
            update={"state": AutomationState.APPROVED, "approval_id": actor_id, "updated_at": utc_now()}
        )
        return (await self.store.save(updated, expected_revision=stored.revision, actor_id=actor_id)).record

    async def activate(self, automation_id: str, *, actor_id: str) -> AutomationRecord:
        stored = await self._load(automation_id)
        record = stored.record
        if record.owner_id != actor_id or record.state != AutomationState.APPROVED:
            raise AutomationLifecycleError("only an approved owner automation can activate")
        now = utc_now()
        updated = record.model_copy(
            update={"state": AutomationState.ACTIVE, "activated_at": now, "updated_at": now}
        )
        return (await self.store.save(updated, expected_revision=stored.revision, actor_id=actor_id)).record

    async def rollback(self, automation_id: str, *, actor_id: str, now: datetime | None = None) -> AutomationRecord:
        stored = await self._load(automation_id)
        record = stored.record
        if record.owner_id != actor_id or record.state not in {AutomationState.ACTIVE, AutomationState.PAUSED, AutomationState.FAILED}:
            raise AutomationLifecycleError("only the owner can roll back an active automation")
        current = now or utc_now()
        updated = record.model_copy(
            update={"state": AutomationState.ROLLED_BACK, "rolled_back_at": current, "updated_at": current}
        )
        return (await self.store.save(updated, expected_revision=stored.revision, actor_id=actor_id)).record
