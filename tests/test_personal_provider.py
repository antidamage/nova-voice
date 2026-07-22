from __future__ import annotations

from nova_voice.authority import action_capability
from nova_voice.capabilities.registry import CapabilityRegistry
from nova_voice.domain import CapabilityToolCall, PlannedAction
from nova_voice.providers.personal.provider import PersonalDataProvider
from nova_voice.providers.personal.store import PersonalDataStore


def _action(tool: str, arguments: dict, action_id: str) -> PlannedAction:
    return PlannedAction(
        id=action_id,
        order=0,
        call=CapabilityToolCall(provider="personal", tool=tool, arguments=arguments),
    )


def test_personal_manifest_is_allowlisted_auditable_and_resource_locked(tmp_path) -> None:
    registry = CapabilityRegistry(allowlist={"personal"})
    registry.register(PersonalDataProvider(PersonalDataStore(tmp_path / "personal.sqlite3")))

    assert len(registry.tool_catalog()) == 14
    policy = registry.policy_for("personal", "personal.notes.create")
    assert policy.risk == "low" and policy.reversible and policy.idempotent
    assert registry.resources_for(
        _action("personal.notes.create", {"title": "A", "body": "B"}, "create")
    ) == ("personal:data",)
    assert (
        action_capability(_action("personal.contacts.lookup", {"query": "Sam"}, "lookup"))
        == "personal.contacts.lookup"
    )


async def test_note_create_retry_update_delete_and_undo(tmp_path) -> None:
    provider = PersonalDataProvider(PersonalDataStore(tmp_path / "personal.sqlite3"))
    create = _action(
        "personal.notes.create", {"title": "Trip", "body": "Pack boots"}, "note-create"
    )
    first = await provider.execute(create)
    retry = await provider.execute(create)
    note_id = first.target
    updated = await provider.execute(
        _action(
            "personal.notes.update",
            {"selector": note_id, "body": "Pack boots and coat"},
            "note-update",
        )
    )
    deleted = await provider.execute(
        _action("personal.notes.delete", {"selector": note_id}, "note-delete")
    )
    restored = await provider.execute(
        _action(
            "personal.undo",
            {"token": deleted.observed["undoToken"]},
            "note-undo",
        )
    )

    assert first.ok and retry.ok and not retry.observed["changed"]
    assert updated.observed["record"]["data"]["body"] == "Pack boots and coat"
    assert deleted.observed["record"] is None
    assert restored.observed["record"]["id"] == note_id


async def test_ambiguous_contact_mutation_returns_candidates_without_writing(tmp_path) -> None:
    store = PersonalDataStore(tmp_path / "personal.sqlite3")
    provider = PersonalDataProvider(store)
    await provider.execute(_action("personal.contacts.create", {"name": "Sam Lee"}, "sam-one"))
    await provider.execute(_action("personal.contacts.create", {"name": "Sam Lee"}, "sam-two"))

    result = await provider.execute(
        _action(
            "personal.contacts.update",
            {"selector": "Sam Lee", "phones": ["0212345678"]},
            "sam-update",
        )
    )

    assert not result.ok and result.code == "ambiguous"
    assert len(result.observed["candidates"]) == 2
    assert all(not record.data["phones"] for record in await store.search("contact", "Sam Lee"))


async def test_list_item_identity_and_undo_are_deterministic(tmp_path) -> None:
    provider = PersonalDataProvider(PersonalDataStore(tmp_path / "personal.sqlite3"))
    created = await provider.execute(
        _action("personal.lists.create", {"name": "Groceries"}, "list-create")
    )
    list_id = created.target
    added = await provider.execute(
        _action(
            "personal.lists.add",
            {"selector": list_id, "text": "Oat milk"},
            "item-add",
        )
    )
    completed = await provider.execute(
        _action(
            "personal.lists.complete",
            {"selector": list_id, "item": "Oat milk"},
            "item-complete",
        )
    )
    undone = await provider.execute(
        _action(
            "personal.undo",
            {"token": completed.observed["undoToken"]},
            "item-undo",
        )
    )

    assert added.observed["record"]["data"]["items"][0]["id"] == "item-item-add"
    assert completed.observed["record"]["data"]["items"][0]["completed"] is True
    assert undone.observed["record"]["data"]["items"][0]["completed"] is False


async def test_undo_refuses_to_overwrite_a_newer_revision(tmp_path) -> None:
    provider = PersonalDataProvider(PersonalDataStore(tmp_path / "personal.sqlite3"))
    created = await provider.execute(
        _action("personal.notes.create", {"title": "Draft", "body": "one"}, "create")
    )
    first_update = await provider.execute(
        _action(
            "personal.notes.update",
            {"selector": created.target, "body": "two"},
            "update-one",
        )
    )
    await provider.execute(
        _action(
            "personal.notes.update",
            {"selector": created.target, "body": "three"},
            "update-two",
        )
    )

    undo = await provider.execute(
        _action(
            "personal.undo",
            {"token": first_update.observed["undoToken"]},
            "stale-undo",
        )
    )

    assert not undo.ok and undo.code == "blocked"
    current = await provider.store.get(created.target)
    assert current.data["body"] == "three"
