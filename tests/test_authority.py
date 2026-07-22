from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from nova_voice.authority import HouseholdAuthority
from nova_voice.domain import CapabilityToolCall, PlannedAction, SpeakerIdentity
from nova_voice.durable.models import (
    DelegationGrantRecord,
    GrantSchedule,
    HouseholdRole,
)
from nova_voice.durable.store import DurableAgentStore


def action(
    tool: str,
    arguments: dict,
    *,
    provider: str = "nova",
) -> PlannedAction:
    return PlannedAction(
        id=f"{provider}:{tool}",
        order=0,
        call=CapabilityToolCall(provider=provider, tool=tool, arguments=arguments),
    )


def recognized(person_id: str = "person-1") -> SpeakerIdentity:
    return SpeakerIdentity(status="recognized", person_id=person_id, confidence=0.9)


async def authority(tmp_path) -> HouseholdAuthority:
    store = DurableAgentStore(tmp_path / "durable.sqlite3")
    await store.initialize()
    value = HouseholdAuthority(store, ZoneInfo("Pacific/Auckland"))
    await value.initialize()
    return value


async def test_identity_classes_enforce_deterministic_base_capabilities(tmp_path) -> None:
    value = await authority(tmp_path)
    control = action("nova.control", {"target": "lounge light", "action": "turn_on"})
    web = action("web.ask", {"query": "weather"}, provider="web")

    assert not value.authorize(SpeakerIdentity(), [control]).allowed
    assert value.authorize(SpeakerIdentity(), [web]).allowed
    assert value.authorize(recognized(), [control]).allowed

    await value.set_role("person-1", HouseholdRole.GUEST, actor_id="owner")
    assert not value.authorize(recognized(), [control]).allowed
    await value.set_role("person-1", HouseholdRole.OWNER, actor_id="owner")
    assert value.authorize(recognized(), [control]).allowed


async def test_scoped_grant_honors_target_schedule_budget_expiry_and_revocation(tmp_path) -> None:
    value = await authority(tmp_path)
    await value.set_role("person-1", HouseholdRole.GUEST, actor_id="owner")
    now = datetime(2026, 7, 22, 10, 0, tzinfo=UTC)
    local = now.astimezone(ZoneInfo("Pacific/Auckland"))
    grant = DelegationGrantRecord(
        id="grant-1",
        created_at=now,
        updated_at=now,
        grantor_id="owner",
        grantee_id="person-1",
        capability="home.control",
        target_scope=("lounge light",),
        locations=("lounge",),
        schedule=GrantSchedule(
            weekdays=(local.weekday(),),
            start_time="00:00",
            end_time="23:59",
        ),
        max_uses=1,
        expires_at=now + timedelta(hours=1),
    )
    await value.create_grant(grant, actor_id="owner")
    allowed = action(
        "nova.control",
        {"target": "lounge light", "room": "lounge", "action": "turn_on"},
    )
    wrong_target = action(
        "nova.control",
        {"target": "bedroom light", "room": "lounge", "action": "turn_on"},
    )

    outcome = value.authorize(recognized(), [allowed], now=now)
    assert outcome.allowed
    assert outcome.grant_ids == ("grant-1",)
    assert not value.authorize(recognized(), [wrong_target], now=now).allowed

    await value.record_use(outcome.grant_ids, [allowed], actor_id="person-1")
    assert not value.authorize(recognized(), [allowed], now=now).allowed
    await value.revoke_grant("grant-1", actor_id="owner")
    assert not value.authorize(recognized(), [allowed], now=now).allowed


async def test_roles_and_grants_reload_from_durable_store(tmp_path) -> None:
    value = await authority(tmp_path)
    await value.set_role("person-1", HouseholdRole.GUEST, actor_id="owner")
    grant = DelegationGrantRecord(
        id="grant-1",
        grantor_id="owner",
        grantee_id="person-1",
        capability="home.control",
    )
    await value.create_grant(grant, actor_id="owner")

    restarted = HouseholdAuthority(value.store, ZoneInfo("Pacific/Auckland"))
    await restarted.initialize()
    outcome = restarted.authorize(
        recognized(),
        [action("nova.control", {"target": "light", "action": "turn_on"})],
    )
    assert outcome.allowed
    assert outcome.grant_ids == ("grant-1",)
