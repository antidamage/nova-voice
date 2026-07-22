from __future__ import annotations

from types import SimpleNamespace

import httpx
from nova_voice.dialogue import MultiPartyDialogueManager, detect_dialogue_routing

from nova_voice.api import create_app
from nova_voice.config import Settings
from nova_voice.domain import CapabilityToolCall, PlannedAction
from nova_voice.durable.store import DurableAgentStore
from nova_voice.providers.dialogue.provider import DialogueProvider


async def _manager(tmp_path):
    store = DurableAgentStore(tmp_path / "durable.sqlite3")
    await store.initialize()
    return MultiPartyDialogueManager(store)


def test_dialogue_routing_distinguishes_assistant_person_and_relays() -> None:
    values = {
        "agent_names": ("Football", "Agent"),
        "participant_names": ("Addie", "Sam"),
    }

    assert detect_dialogue_routing("Football, help me", **values).addressee == "assistant"
    assert detect_dialogue_routing("Addie, what do you think?", **values).addressee == "person"
    ask = detect_dialogue_routing("Ask Addie should we leave at eight?", **values)
    tell = detect_dialogue_routing("Tell the household dinner is ready", **values)

    assert (ask.addressee, ask.relay_act, ask.target_name) == ("person", "ask", "Addie")
    assert ask.relay_content == "should we leave at eight?"
    assert (tell.addressee, tell.relay_act) == ("household", "tell")


async def test_person_and_household_relays_enforce_audience_and_acknowledgement(tmp_path) -> None:
    manager = await _manager(tmp_path)
    person = await manager.create(
        sender_id="sam",
        recipient_scope="person",
        recipient_name="Addie",
        speech_act="ask",
        content="Should we leave at eight?",
    )
    household = await manager.create(
        sender_id="addie",
        recipient_scope="household",
        speech_act="tell",
        content="Dinner is ready.",
    )

    assert await manager.pending_for(person_id="alex", display_name="Alex") == (household,)
    addie_pending = await manager.pending_for(person_id="person-addie", display_name="Addie")
    assert {item.id for item in addie_pending} == {person.id, household.id}
    delivered = await manager.acknowledge(person.id, person_id="person-addie", display_name="Addie")
    await manager.acknowledge(household.id, person_id="person-addie")

    assert delivered.status == "delivered"
    assert await manager.pending_for(person_id="person-addie", display_name="Addie") == ()


async def test_dialogue_provider_and_api_preserve_attribution(tmp_path) -> None:
    manager = await _manager(tmp_path)
    provider = DialogueProvider(manager)
    created = await provider.execute(
        PlannedAction(
            id="relay-1",
            order=0,
            call=CapabilityToolCall(
                provider="dialogue",
                tool="dialogue.relay",
                arguments={
                    "senderId": "addie",
                    "recipientScope": "person",
                    "recipientId": "sam",
                    "speechAct": "tell",
                    "content": "The parcel arrived.",
                },
            ),
        )
    )
    message_id = created.observed["message"]["id"]
    app = create_app(Settings(), service=SimpleNamespace(dialogue=manager))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://voice.test"
    ) as client:
        listed = await client.get("/v1/dialogue/messages")
        acknowledged = await client.post(
            f"/v1/dialogue/messages/{message_id}/acknowledge",
            json={"person_id": "sam"},
        )

    assert listed.json()["messages"][0]["sender_id"] == "addie"
    assert acknowledged.json()["message"]["delivered_to"] == ["sam"]
