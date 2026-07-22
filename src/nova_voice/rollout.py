from __future__ import annotations

from dataclasses import dataclass

from nova_voice.durable.models import (
    RolloutEvent,
    RolloutEvidence,
    RolloutRecord,
    RolloutStage,
    RolloutStatus,
    utc_now,
)
from nova_voice.durable.store import DurableAgentStore


class RolloutLifecycleError(RuntimeError):
    pass


STAGES = (
    RolloutStage.FIXTURE,
    RolloutStage.REPLAY,
    RolloutStage.SHADOW,
    RolloutStage.OWNER_CANARY,
    RolloutStage.HOUSEHOLD,
    RolloutStage.STANDING_AUTONOMY,
)


@dataclass(frozen=True)
class RolloutDecision:
    allowed: bool
    reason: str
    stage: RolloutStage


class RolloutManager:
    """Durable fail-closed promotion, revocation, and rollback control plane."""

    def __init__(self, store: DurableAgentStore) -> None:
        self.store = store

    async def create(
        self,
        *,
        rollout_id: str,
        owner_id: str,
        component: str,
        pins_digest: str,
    ) -> RolloutRecord:
        now = utc_now()
        record = RolloutRecord(
            id=rollout_id,
            created_at=now,
            updated_at=now,
            owner_id=owner_id,
            component=component,
            pins_digest=pins_digest,
            history=(
                RolloutEvent(
                    at=now,
                    actor_id=owner_id,
                    action="created",
                    to_stage=RolloutStage.FIXTURE,
                ),
            ),
        )
        await self.store.create(record, actor_id=owner_id)
        return record

    async def list(self) -> tuple[RolloutRecord, ...]:
        return tuple(row.record for row in await self.store.list(RolloutRecord))

    async def _stored(self, rollout_id: str):
        stored = await self.store.get(RolloutRecord, rollout_id)
        if stored is None:
            raise KeyError(rollout_id)
        return stored

    async def promote(
        self,
        rollout_id: str,
        *,
        actor_id: str,
        evidence: RolloutEvidence,
        authority_scope: tuple[str, ...] = (),
    ) -> RolloutRecord:
        stored = await self._stored(rollout_id)
        record = stored.record
        if record.status == RolloutStatus.REVOKED:
            raise RolloutLifecycleError("revoked rollout must be rolled back before promotion")
        if actor_id != record.owner_id:
            raise PermissionError("rollout promotion requires its household owner")
        index = STAGES.index(record.stage)
        if index == len(STAGES) - 1:
            raise RolloutLifecycleError("rollout is already at standing autonomy")
        if evidence.stage != record.stage:
            raise RolloutLifecycleError("evidence does not match the current rollout stage")
        if evidence.pins_digest != record.pins_digest:
            raise RolloutLifecycleError("evaluation pins do not match the rollout")
        if not evidence.eligible or evidence.reasons or not evidence.scenario_runs:
            raise RolloutLifecycleError("a passing evaluation gate is required for promotion")
        target = STAGES[index + 1]
        selected_scope = authority_scope or record.authority_scope
        if target == RolloutStage.STANDING_AUTONOMY and not selected_scope:
            raise RolloutLifecycleError("standing autonomy requires a bounded authority scope")
        now = utc_now()
        updated = record.model_copy(
            update={
                "stage": target,
                "authority_scope": selected_scope,
                "updated_at": now,
                "history": (
                    *record.history,
                    RolloutEvent(
                        at=now,
                        actor_id=actor_id,
                        action="promoted",
                        from_stage=record.stage,
                        to_stage=target,
                        evidence_revision=evidence.artifact_revision,
                    ),
                ),
            }
        )
        await self.store.save(updated, expected_revision=stored.revision, actor_id=actor_id)
        return updated

    async def revoke(self, rollout_id: str, *, actor_id: str, reason: str) -> RolloutRecord:
        stored = await self._stored(rollout_id)
        record = stored.record
        if actor_id != record.owner_id:
            raise PermissionError("rollout revocation requires its household owner")
        if record.status == RolloutStatus.REVOKED:
            return record
        now = utc_now()
        updated = record.model_copy(
            update={
                "status": RolloutStatus.REVOKED,
                "revoked_at": now,
                "updated_at": now,
                "history": (
                    *record.history,
                    RolloutEvent(
                        at=now,
                        actor_id=actor_id,
                        action="revoked",
                        from_stage=record.stage,
                        to_stage=record.stage,
                        reason=reason,
                    ),
                ),
            }
        )
        await self.store.save(updated, expected_revision=stored.revision, actor_id=actor_id)
        return updated

    async def rollback(
        self,
        rollout_id: str,
        *,
        actor_id: str,
        target_stage: RolloutStage,
        reason: str,
    ) -> RolloutRecord:
        stored = await self._stored(rollout_id)
        record = stored.record
        if actor_id != record.owner_id:
            raise PermissionError("rollout rollback requires its household owner")
        if STAGES.index(target_stage) >= STAGES.index(record.stage):
            raise RolloutLifecycleError("rollback target must be an earlier stage")
        now = utc_now()
        updated = record.model_copy(
            update={
                "stage": target_stage,
                "status": RolloutStatus.ACTIVE,
                "authority_scope": (),
                "revoked_at": None,
                "updated_at": now,
                "history": (
                    *record.history,
                    RolloutEvent(
                        at=now,
                        actor_id=actor_id,
                        action="rolled_back",
                        from_stage=record.stage,
                        to_stage=target_stage,
                        reason=reason,
                    ),
                ),
            }
        )
        await self.store.save(updated, expected_revision=stored.revision, actor_id=actor_id)
        return updated

    async def allows(self, rollout_id: str, required_stage: RolloutStage) -> RolloutDecision:
        stored = await self._stored(rollout_id)
        record = stored.record
        if record.status == RolloutStatus.REVOKED:
            return RolloutDecision(False, "rollout_revoked", record.stage)
        if STAGES.index(record.stage) < STAGES.index(required_stage):
            return RolloutDecision(False, "stage_not_reached", record.stage)
        return RolloutDecision(True, "stage_reached", record.stage)
