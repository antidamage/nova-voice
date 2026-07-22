from __future__ import annotations

from types import SimpleNamespace
from zoneinfo import ZoneInfo

import httpx

from nova_voice.api import create_app
from nova_voice.authority import HouseholdAuthority
from nova_voice.config import Settings
from nova_voice.durable.store import DurableAgentStore


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
