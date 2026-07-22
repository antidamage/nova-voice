from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from nova_voice.api import create_app
from nova_voice.capabilities.registry import CapabilityRegistry
from nova_voice.communications import CommunicationManager
from nova_voice.config import Settings
from nova_voice.domain import CapabilityToolCall, PlannedAction
from nova_voice.providers.communications.provider import CommunicationsProvider
from nova_voice.providers.personal.store import PersonalDataStore


class _Transport:
    def __init__(self) -> None:
        self.sent = []
        self.cancelled = []

    async def send(self, draft):
        self.sent.append(draft.id)
        return {"delivered": True, "receipt": f"receipt-{draft.id}"}

    async def cancel(self, draft):
        self.cancelled.append(draft.delivery_receipt)
        return {"cancelled": True}

    async def health(self):
        return {"ok": True, "configured": True}

    async def close(self):
        return None


def _action(tool: str, arguments: dict, action_id: str) -> PlannedAction:
    return PlannedAction(
        id=action_id,
        order=0,
        call=CapabilityToolCall(provider="communications", tool=tool, arguments=arguments),
    )


async def _contact(store: PersonalDataStore, record_id: str, name: str, **data) -> None:
    await store.mutate(
        record_id=record_id,
        kind="contact",
        label=name,
        data={"phones": data.get("phones", []), "emails": data.get("emails", [])},
        undo_token=f"seed-{record_id}",
    )


async def test_provider_drafts_only_after_unique_recipient_validation(tmp_path) -> None:
    contacts = PersonalDataStore(tmp_path / "personal.sqlite3")
    await _contact(contacts, "sam-one", "Sam Lee", emails=["sam@example.test"])
    await _contact(contacts, "sam-two", "Sam Lee", emails=["other@example.test"])
    manager = CommunicationManager(tmp_path / "communications.sqlite3", contacts, _Transport())
    provider = CommunicationsProvider(manager)

    ambiguous = await provider.execute(
        _action(
            "communications.draft",
            {
                "channel": "email",
                "recipient": "Sam Lee",
                "subject": "Hello",
                "body": "Checking in.",
            },
            "ambiguous",
        )
    )
    drafted = await provider.execute(
        _action(
            "communications.draft",
            {
                "channel": "email",
                "recipient": "sam-one",
                "subject": "Hello",
                "body": "Checking in.",
            },
            "unique",
        )
    )

    assert ambiguous.code == "ambiguous" and not ambiguous.ok
    assert len(ambiguous.observed["candidates"]) == 2
    assert drafted.ok and drafted.observed["approvalRequired"]
    assert drafted.observed["draft"]["recipient_address"] == "sam@example.test"


async def test_preview_token_is_revision_bound_one_time_and_delivery_verified(tmp_path) -> None:
    contacts = PersonalDataStore(tmp_path / "personal.sqlite3")
    await _contact(contacts, "sam", "Sam", phones=["+6421000000"])
    transport = _Transport()
    manager = CommunicationManager(tmp_path / "communications.sqlite3", contacts, transport)
    draft, _ = await manager.create_draft(
        draft_id="draft-1",
        channel="message",
        recipient="sam",
        body="On my way",
    )
    preview, token = await manager.preview("draft-1", actor="dashboard-admin")

    with pytest.raises(PermissionError):
        await manager.send_approved("draft-1", "wrong", actor="dashboard-admin")
    sent = await manager.send_approved("draft-1", token, actor="dashboard-admin")
    with pytest.raises(PermissionError):
        await manager.send_approved("draft-1", token, actor="dashboard-admin")

    assert preview.state == "draft"
    assert sent.state == "sent"
    assert sent.delivery_receipt == "receipt-draft-1"
    assert transport.sent == ["draft-1"]
    actions = [event["action"] for event in await manager.audit("draft-1")]
    assert actions == ["drafted", "previewed", "send_authorized", "delivery_verified"]


async def test_sent_invitation_cancellation_requires_transport_verification(tmp_path) -> None:
    contacts = PersonalDataStore(tmp_path / "personal.sqlite3")
    await _contact(contacts, "sam", "Sam", emails=["sam@example.test"])
    transport = _Transport()
    manager = CommunicationManager(tmp_path / "communications.sqlite3", contacts, transport)
    await manager.create_draft(
        draft_id="invite-1",
        channel="invitation",
        recipient="sam",
        subject="Dinner",
        body="Dinner on Friday?",
        invitation={"start": "2026-08-07T18:00:00+12:00"},
    )
    _, token = await manager.preview("invite-1", actor="dashboard-admin")
    await manager.send_approved("invite-1", token, actor="dashboard-admin")

    cancelled = await manager.cancel("invite-1", actor="dashboard-admin")

    assert cancelled.state == "cancelled"
    assert transport.cancelled == ["receipt-invite-1"]


def test_send_tool_is_impossible_in_immediate_voice_executor(tmp_path) -> None:
    contacts = PersonalDataStore(tmp_path / "personal.sqlite3")
    provider = CommunicationsProvider(
        CommunicationManager(tmp_path / "communications.sqlite3", contacts)
    )
    registry = CapabilityRegistry(allowlist={"communications"})
    registry.register(provider)

    policy = registry.policy_for("communications", "communications.send")

    assert policy.risk == "confirmation"
    assert policy.requires_confirmation
    assert policy.cancellation == "never"


async def test_authenticated_api_preview_send_and_audit_round_trip(tmp_path) -> None:
    contacts = PersonalDataStore(tmp_path / "personal.sqlite3")
    await _contact(contacts, "sam", "Sam", emails=["sam@example.test"])
    manager = CommunicationManager(tmp_path / "communications.sqlite3", contacts, _Transport())
    await manager.create_draft(
        draft_id="draft-api",
        channel="email",
        recipient="sam",
        subject="Test",
        body="Hello",
    )
    app = create_app(Settings(), service=SimpleNamespace(communications=manager))
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="https://voice.test") as client:
        preview = await client.post(
            "/v1/communications/draft-api/preview",
            json={"actor": "dashboard-admin"},
        )
        rejected = await client.post(
            "/v1/communications/draft-api/send",
            json={"approval_token": "wrong"},
        )
        sent = await client.post(
            "/v1/communications/draft-api/send",
            json={"approval_token": preview.json()["approvalToken"]},
        )
        audit = await client.get("/v1/communications/draft-api/audit")

    assert preview.status_code == 200
    assert rejected.status_code == 403
    assert sent.json()["draft"]["state"] == "sent"
    assert audit.json()["events"][-1]["action"] == "delivery_verified"
