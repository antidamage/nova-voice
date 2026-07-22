from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, time, tzinfo
from typing import cast
from uuid import uuid4

from nova_voice.domain import PlannedAction, SpeakerIdentity
from nova_voice.durable.models import (
    DelegationGrantRecord,
    HouseholdRole,
    IdentityPolicyRecord,
    utc_now,
)
from nova_voice.durable.store import ConcurrentRecordUpdate, DurableAgentStore


@dataclass(frozen=True)
class AuthorityOutcome:
    allowed: bool
    role: HouseholdRole
    reason: str
    grant_ids: tuple[str, ...] = ()


def action_capability(action: PlannedAction) -> str:
    call = action.call
    if call.provider == "web":
        return "knowledge.read"
    if call.provider != "nova":
        return f"{call.provider}.{call.tool}"
    if call.tool == "nova.query":
        return "tasks.read" if call.arguments.get("scope") == "tasks" else "home.read"
    if call.tool == "nova.task":
        return "tasks.read" if call.arguments.get("operation") == "list" else "tasks.manage"
    if call.tool in {"nova.control", "nova.lighting_shortcut"}:
        return "home.control"
    return call.tool


def _normalized(value: object) -> str | None:
    return value.strip().casefold() if isinstance(value, str) and value.strip() else None


class HouseholdAuthority:
    """Cached deterministic household roles and standing delegation grants."""

    _BASE_CAPABILITIES = {
        HouseholdRole.GUEST: frozenset({"knowledge.read"}),
        HouseholdRole.RECOGNIZED_HOUSEHOLD: frozenset(
            {"knowledge.read", "home.read", "home.control", "tasks.read", "tasks.manage"}
        ),
        HouseholdRole.OWNER: frozenset({"*"}),
    }

    def __init__(self, store: DurableAgentStore, household_timezone: tzinfo) -> None:
        self.store = store
        self.household_timezone = household_timezone
        self._identities: dict[str, IdentityPolicyRecord] = {}
        self._grants: dict[str, DelegationGrantRecord] = {}
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        identities, grants = await asyncio.gather(
            self.store.list(IdentityPolicyRecord),
            self.store.list(DelegationGrantRecord),
        )
        self._identities = {
            cast(IdentityPolicyRecord, row.record).person_id: cast(
                IdentityPolicyRecord, row.record
            )
            for row in identities
        }
        self._grants = {
            cast(DelegationGrantRecord, row.record).id: cast(
                DelegationGrantRecord, row.record
            )
            for row in grants
        }

    def classify(self, speaker: SpeakerIdentity) -> HouseholdRole:
        if speaker.status == "recognized" and speaker.person_id:
            assigned = self._identities.get(speaker.person_id)
            if assigned is not None and assigned.active:
                return assigned.role
            return HouseholdRole.RECOGNIZED_HOUSEHOLD
        return HouseholdRole.GUEST

    def role_for_person(self, person_id: str) -> HouseholdRole | None:
        """Return an explicitly assigned active role for an administrative actor.

        A dashboard/API caller is not a speaker-recognition result, so it must
        never inherit the permissive recognised-household default.  Sensitive
        administration flows use this method to require a persisted owner
        assignment.
        """

        assigned = self._identities.get(person_id)
        if assigned is None or not assigned.active:
            return None
        return assigned.role

    def is_owner(self, person_id: str) -> bool:
        return self.role_for_person(person_id) == HouseholdRole.OWNER

    def _schedule_allows(self, grant: DelegationGrantRecord, now: datetime) -> bool:
        schedule = grant.schedule
        if schedule is None:
            return True
        local = now.astimezone(self.household_timezone)
        if schedule.weekdays and local.weekday() not in schedule.weekdays:
            return False
        if schedule.start_time is None or schedule.end_time is None:
            return True
        start = time.fromisoformat(schedule.start_time)
        end = time.fromisoformat(schedule.end_time)
        current = local.timetz().replace(tzinfo=None)
        return start <= current < end if start <= end else current >= start or current < end

    def _grant_allows(
        self,
        grant: DelegationGrantRecord,
        action: PlannedAction,
        capability: str,
        now: datetime,
    ) -> bool:
        if not grant.active or grant.revoked_at is not None:
            return False
        if grant.expires_at is not None and grant.expires_at <= now:
            return False
        if grant.capability not in {"*", capability}:
            return False
        if grant.max_uses is not None and grant.uses >= grant.max_uses:
            return False
        amount = action.call.arguments.get("amount")
        if grant.max_amount is not None:
            if not isinstance(amount, (int, float)):
                return False
            if grant.spent_amount + float(amount) > grant.max_amount:
                return False
        if not self._schedule_allows(grant, now):
            return False
        arguments = action.call.arguments
        target = _normalized(arguments.get("target"))
        recipient = _normalized(arguments.get("recipient"))
        location = _normalized(arguments.get("location") or arguments.get("room"))
        if grant.target_scope and target not in {_normalized(item) for item in grant.target_scope}:
            return False
        if grant.recipients and recipient not in {_normalized(item) for item in grant.recipients}:
            return False
        if grant.locations and location not in {_normalized(item) for item in grant.locations}:
            return False
        return True

    def authorize(
        self,
        speaker: SpeakerIdentity,
        actions: Iterable[PlannedAction],
        *,
        now: datetime | None = None,
    ) -> AuthorityOutcome:
        selected = tuple(actions)
        role = self.classify(speaker)
        base = self._BASE_CAPABILITIES[role]
        if not selected:
            return AuthorityOutcome(True, role, "no capability required")
        current = (now or utc_now()).astimezone(UTC)
        grants_used: list[str] = []
        for action in selected:
            capability = action_capability(action)
            if "*" in base or capability in base:
                continue
            person_id = speaker.person_id if speaker.status == "recognized" else None
            matching = next(
                (
                    grant
                    for grant in self._grants.values()
                    if person_id is not None
                    and grant.grantee_id == person_id
                    and self._grant_allows(grant, action, capability, current)
                ),
                None,
            )
            if matching is None:
                return AuthorityOutcome(
                    False,
                    role,
                    f"{role.value} identity lacks {capability}",
                )
            grants_used.append(matching.id)
        return AuthorityOutcome(
            True,
            role,
            "allowed by household role or standing grant",
            tuple(dict.fromkeys(grants_used)),
        )

    async def set_role(
        self,
        person_id: str,
        role: HouseholdRole,
        *,
        actor_id: str,
    ) -> IdentityPolicyRecord:
        async with self._lock:
            now = utc_now()
            record_id = f"identity-policy:{person_id}"
            stored = await self.store.get(IdentityPolicyRecord, record_id)
            if stored is None:
                record = IdentityPolicyRecord(
                    id=record_id,
                    person_id=person_id,
                    role=role,
                    created_at=now,
                    updated_at=now,
                )
                await self.store.create(record, actor_id=actor_id)
            else:
                current = cast(IdentityPolicyRecord, stored.record)
                record = current.model_copy(
                    update={"role": role, "active": True, "updated_at": now}
                )
                await self.store.save(record, expected_revision=stored.revision, actor_id=actor_id)
            self._identities[person_id] = record
            return record

    async def create_grant(
        self,
        grant: DelegationGrantRecord,
        *,
        actor_id: str,
    ) -> DelegationGrantRecord:
        async with self._lock:
            await self.store.create(grant, actor_id=actor_id)
            self._grants[grant.id] = grant
            return grant

    async def revoke_grant(self, grant_id: str, *, actor_id: str) -> DelegationGrantRecord:
        async with self._lock:
            stored = await self.store.get(DelegationGrantRecord, grant_id)
            if stored is None:
                raise KeyError(grant_id)
            current = cast(DelegationGrantRecord, stored.record)
            now = utc_now()
            revoked = current.model_copy(
                update={"active": False, "revoked_at": now, "updated_at": now}
            )
            await self.store.save(revoked, expected_revision=stored.revision, actor_id=actor_id)
            self._grants[grant_id] = revoked
            return revoked

    async def record_use(
        self,
        grant_ids: Iterable[str],
        actions: Iterable[PlannedAction],
        *,
        actor_id: str,
    ) -> None:
        amount = sum(
            float(action.call.arguments.get("amount", 0))
            for action in actions
            if isinstance(action.call.arguments.get("amount", 0), (int, float))
        )
        for grant_id in dict.fromkeys(grant_ids):
            async with self._lock:
                stored = await self.store.get(DelegationGrantRecord, grant_id)
                if stored is None:
                    continue
                current = cast(DelegationGrantRecord, stored.record)
                updated = current.model_copy(
                    update={
                        "uses": current.uses + 1,
                        "spent_amount": current.spent_amount + amount,
                        "updated_at": utc_now(),
                    }
                )
                try:
                    await self.store.save(
                        updated,
                        expected_revision=stored.revision,
                        actor_id=actor_id,
                    )
                except ConcurrentRecordUpdate:
                    await self.initialize()
                    continue
                self._grants[grant_id] = updated

    @staticmethod
    def new_grant_id() -> str:
        return f"grant-{uuid4().hex}"
