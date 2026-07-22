from __future__ import annotations

from types import SimpleNamespace
from zoneinfo import ZoneInfo

import httpx

from nova_voice.api import create_app
from nova_voice.authority import HouseholdAuthority
from nova_voice.automation import AutomationManager
from nova_voice.config import Settings
from nova_voice.durable.models import EventRecord, HouseholdRole, utc_now
from nova_voice.durable.store import DurableAgentStore
from nova_voice.proactive import ProactiveInterventionEngine


async def test_authenticated_transport_administration_round_trip(tmp_path) -> None:
    store = DurableAgentStore(tmp_path / "durable.sqlite3")
    await store.initialize()
    authority = HouseholdAuthority(store, ZoneInfo("Pacific/Auckland"))
    await authority.initialize()
    service = SimpleNamespace(durable_store=store, authority=authority)
    app = create_app(Settings(), service=service)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="https://voice.test") as client:
        role = await client.put(
            "/v1/agent/identities/person-1",
            json={"role": "guest"},
        )
        assert role.status_code == 200

        created = await client.post(
            "/v1/agent/grants",
            json={
                "grantee_id": "person-1",
                "capability": "home.control",
                "target_scope": ["lounge light"],
                "max_uses": 2,
                "notify_on_use": True,
            },
        )
        assert created.status_code == 201
        grant_id = created.json()["grant"]["id"]

        summary = await client.get("/v1/agent/administration")
        assert summary.status_code == 200
        assert summary.json()["identities"][0]["role"] == "guest"
        assert summary.json()["grants"][0]["target_scope"] == ["lounge light"]

        revoked = await client.delete(f"/v1/agent/grants/{grant_id}")
        assert revoked.status_code == 200
        assert revoked.json()["grant"]["active"] is False

        audit = await client.get(
            "/v1/agent/audit",
            params={"object_type": "DelegationGrantRecord"},
        )
        assert audit.status_code == 200
        assert [event["action"] for event in audit.json()["events"]] == [
            "create",
            "update",
        ]


async def test_automation_api_requires_assigned_owner_and_records_feedback(tmp_path) -> None:
    store = DurableAgentStore(tmp_path / "durable.sqlite3")
    await store.initialize()
    authority = HouseholdAuthority(store, ZoneInfo("Pacific/Auckland"))
    await authority.initialize()
    proactive = ProactiveInterventionEngine(store)
    service = SimpleNamespace(
        durable_store=store,
        authority=authority,
        automations=AutomationManager(store),
        proactive=proactive,
    )
    app = create_app(Settings(), service=service)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="https://voice.test") as client:
        draft = await client.post(
            "/v1/agent/automations",
            params={"owner_id": "addie"},
            json={
                "id": "automation:door",
                "summary": "Warn about an open door",
                "trigger": {"kind": "ha_state"},
                "proposed_actions": [{"channel": "dashboard"}],
            },
        )
        assert draft.status_code == 403

        await authority.set_role("addie", HouseholdRole.OWNER, actor_id="dashboard-admin")
        draft = await client.post(
            "/v1/agent/automations",
            params={"owner_id": "addie"},
            json={
                "id": "automation:door",
                "summary": "Warn about an open door",
                "trigger": {"kind": "ha_state"},
                "proposed_actions": [{"channel": "dashboard"}],
            },
        )
        assert draft.status_code == 201

        now = utc_now()
        intervention = await proactive.consider(
            EventRecord(
                id="event:door",
                created_at=now,
                updated_at=now,
                source="dashboard:home_assistant",
                kind="device_health",
                payload={"device_id": "door", "status": "offline"},
                payload_revision="test",
            ),
            occupied_rooms=set(),
        )
        assert intervention is not None

        feedback = await client.post(
            f"/v1/agent/proactive-interventions/{intervention.id}/feedback",
            json={"owner_id": "addie", "outcome": "annoying"},
        )
        assert feedback.status_code == 200
        assert feedback.json()["intervention"]["feedback"] == "annoying"
        assert feedback.json()["intervention"]["status"] == "dismissed"
