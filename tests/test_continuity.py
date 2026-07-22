from __future__ import annotations

from types import SimpleNamespace

import httpx

from nova_voice.api import create_app
from nova_voice.config import Settings
from nova_voice.continuity import ConversationContinuityManager
from nova_voice.durable.store import DurableAgentStore


async def _manager(tmp_path):
    store = DurableAgentStore(tmp_path / "durable.sqlite3")
    await store.initialize()
    return ConversationContinuityManager(store)


async def test_continuity_persists_structure_without_full_assistant_transcript(tmp_path) -> None:
    manager = await _manager(tmp_path)
    first = await manager.observe(
        conversation_id="conversation-1",
        room_id="lounge",
        participant_id="addie",
        topic_summary="Planning the garden renovation",
        user_text="Should we ask Sam about that?",
        linked_goal_ids=("goal-garden",),
    )
    second = await manager.observe(
        conversation_id="conversation-1",
        room_id="lounge",
        participant_id="sam",
        topic_summary="Choosing garden materials",
        user_text="They might prefer timber.",
        linked_goal_ids=("goal-materials",),
    )

    assert first.open_questions == ("Should we ask Sam about that?",)
    assert second.participant_ids == ("addie", "sam")
    assert second.topic_stack == (
        "Planning the garden renovation",
        "Choosing garden materials",
    )
    assert second.unresolved_references == ("that", "they")
    assert second.linked_goal_ids == ("goal-garden", "goal-materials")
    assert "prefer timber" not in second.summary


async def test_continuity_survives_manager_restart_and_resolves_question(tmp_path) -> None:
    store = DurableAgentStore(tmp_path / "durable.sqlite3")
    await store.initialize()
    manager = ConversationContinuityManager(store)
    created = await manager.observe(
        conversation_id="conversation-restart",
        room_id="study",
        participant_id="addie",
        topic_summary="Trip planning",
        user_text="Which train should we take?",
    )
    restarted = ConversationContinuityManager(store)

    resolved = await restarted.resolve_question(
        created.id, "Which train should we take?", actor_id="addie"
    )

    assert resolved.open_questions == ()
    assert (await restarted.get(created.id)).summary == "Trip planning"


async def test_continuity_api_lists_and_resolves_open_questions(tmp_path) -> None:
    manager = await _manager(tmp_path)
    record = await manager.observe(
        conversation_id="conversation-api",
        room_id="lounge",
        participant_id="addie",
        topic_summary="Dinner",
        user_text="What should we cook?",
    )
    app = create_app(Settings(), service=SimpleNamespace(continuity=manager))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://voice.test"
    ) as client:
        listed = await client.get("/v1/conversation-topics")
        resolved = await client.post(
            f"/v1/conversation-topics/{record.id}/resolve",
            json={"question": "What should we cook?", "actor": "addie"},
        )

    assert listed.json()["conversations"][0]["topic_stack"] == ["Dinner"]
    assert resolved.json()["conversation"]["open_questions"] == []
