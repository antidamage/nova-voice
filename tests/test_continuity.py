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


async def test_relationship_continuity_uses_only_explicit_preferences_and_provenance(
    tmp_path,
) -> None:
    manager = await _manager(tmp_path)
    await manager.observe(
        conversation_id="conversation-garden",
        room_id="lounge",
        participant_id="addie",
        topic_summary="Planning the garden renovation",
        user_text="I prefer native plants, and please keep it brief. Which nursery should we use?",
    )
    await manager.observe(
        conversation_id="conversation-unrelated",
        room_id="study",
        participant_id="addie",
        topic_summary="Choosing a new laptop",
        user_text="Maybe that one.",
    )

    context = await manager.context_for("addie", "What was our garden plan?")
    relationship = (await manager.relationships())[0]

    assert relationship.speaking_style == "brief"
    assert list(relationship.explicit_preferences.values()) == [
        "native plants, and please keep it brief"
    ]
    assert context["callbacks"] == [
        {
            "summary": "Planning the garden renovation",
            "sourceConversationId": "conversation-garden",
        }
    ]
    assert context["openThreads"][0]["sourceConversationId"] == "conversation-garden"
    assert "laptop" not in str(context["callbacks"])


async def test_relationship_api_exposes_provenance_bound_records(tmp_path) -> None:
    manager = await _manager(tmp_path)
    await manager.observe(
        conversation_id="conversation-relationship-api",
        room_id="lounge",
        participant_id="addie",
        topic_summary="Book discussion",
        user_text="I prefer short chapters.",
    )
    app = create_app(Settings(), service=SimpleNamespace(continuity=manager))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://voice.test"
    ) as client:
        response = await client.get("/v1/relationship-continuity")

    record = response.json()["relationships"][0]
    assert record["person_id"] == "addie"
    assert record["provenance_conversation_ids"] == ["conversation-relationship-api"]


async def test_discussion_mode_is_user_controlled_and_persists_per_conversation(tmp_path) -> None:
    manager = await _manager(tmp_path)
    current = await manager.discussion_mode_for(
        "conversation-depth", "Go deeper, reflect that back, challenge me, and no jokes."
    )
    await manager.observe(
        conversation_id="conversation-depth",
        room_id="lounge",
        participant_id="addie",
        topic_summary="Career decision",
        user_text=(
            "Go deeper, take your time, reflect that back, challenge me, no jokes, "
            "and tell it as a story."
        ),
    )
    persisted = await manager.discussion_mode_for("conversation-depth", "Continue.")

    assert current["discussion_depth"] == "deep"
    assert current["reflective_listening"] is True
    assert current["disagreement_style"] == "candid"
    assert current["humour_enabled"] is False
    assert persisted == {
        "discussion_depth": "deep",
        "deliberate_pauses": True,
        "reflective_listening": True,
        "disagreement_style": "candid",
        "humour_enabled": False,
        "storytelling_enabled": True,
    }
