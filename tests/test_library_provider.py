from __future__ import annotations

from nova_voice.authority import action_capability
from nova_voice.capabilities.registry import CapabilityRegistry
from nova_voice.domain import CapabilityToolCall, PlannedAction
from nova_voice.providers.library.provider import HouseholdLibraryProvider
from nova_voice.providers.nova.provider import NovaProvider
from nova_voice.providers.personal.store import PersonalDataStore


def _action(tool: str, arguments: dict, action_id: str) -> PlannedAction:
    return PlannedAction(
        id=action_id,
        order=0,
        call=CapabilityToolCall(provider="library", tool=tool, arguments=arguments),
    )


def test_library_contracts_classify_reads_writes_and_capabilities(tmp_path) -> None:
    registry = CapabilityRegistry(allowlist={"library"})
    registry.register(HouseholdLibraryProvider(PersonalDataStore(tmp_path / "personal.sqlite3")))

    assert len(registry.tool_catalog()) == 5
    assert registry.policy_for("library", "library.search_shared").parallel_safe
    assert registry.policy_for("library", "library.add").resource_templates == ("library:data",)
    assert (
        action_capability(_action("library.search_shared", {"query": "wifi"}, "read"))
        == "library.search_shared"
    )


async def test_shared_search_excludes_private_content_and_returns_citations(tmp_path) -> None:
    provider = HouseholdLibraryProvider(PersonalDataStore(tmp_path / "personal.sqlite3"))
    shared = await provider.execute(
        _action(
            "library.add",
            {
                "kind": "recipe",
                "title": "Soup",
                "content": "Use pumpkin and stock.",
                "sourceUri": "https://example.test/soup",
                "audience": "household",
            },
            "shared",
        )
    )
    await provider.execute(
        _action(
            "library.add",
            {
                "kind": "document",
                "title": "Private plan",
                "content": "Pumpkin budget notes.",
                "audience": "owner",
            },
            "private",
        )
    )

    shared_results = await provider.execute(
        _action("library.search_shared", {"query": "pumpkin"}, "shared-search")
    )
    private_results = await provider.execute(
        _action("library.search_private", {"query": "pumpkin"}, "private-search")
    )

    assert shared.ok
    assert len(shared_results.observed["results"]) == 1
    result = shared_results.observed["results"][0]
    assert result["citation"] == "https://example.test/soup"
    assert result["contentRevision"].startswith("sha256:")
    assert result["contentTrust"] == "untrusted_data"
    assert len(private_results.observed["results"]) == 2


async def test_library_mutations_require_unique_identity_and_return_undo(tmp_path) -> None:
    provider = HouseholdLibraryProvider(PersonalDataStore(tmp_path / "personal.sqlite3"))
    for action_id in ("one", "two"):
        await provider.execute(
            _action(
                "library.add",
                {
                    "kind": "household",
                    "title": "Wi-Fi",
                    "content": f"revision {action_id}",
                    "audience": "household",
                },
                action_id,
            )
        )

    ambiguous = await provider.execute(
        _action(
            "library.update",
            {"selector": "Wi-Fi", "content": "wrong"},
            "ambiguous-update",
        )
    )
    selected_id = ambiguous.observed["candidates"][0]["id"]
    updated = await provider.execute(
        _action(
            "library.update",
            {"selector": selected_id, "content": "correct"},
            "selected-update",
        )
    )

    assert ambiguous.code == "ambiguous" and not ambiguous.ok
    assert updated.ok and updated.observed["undoToken"] == "undo-selected-update"


def test_weather_and_media_observations_are_cited_replaceable_contracts() -> None:
    state = {
        "generatedAt": "2026-07-23T02:00:00+12:00",
        "weather": {"condition": "rainy", "temperature": 11},
        "entities": [
            {
                "entity_id": "media_player.lounge",
                "name": "Lounge TV",
                "domain": "media_player",
                "state": "playing",
                "attributes": {},
            }
        ],
    }

    weather = NovaProvider._extended_household_observed(state, "weather")
    media = NovaProvider._extended_household_observed(state, "media")

    assert weather["weather"]["temperatureC"] == 11
    assert weather["citation"].startswith("nova://weather@")
    assert media["players"][0]["id"] == "media_player.lounge"
    assert media["citation"].startswith("nova://media@")
