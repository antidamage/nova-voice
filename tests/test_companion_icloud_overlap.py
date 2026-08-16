"""One semantic path per question, without breaking the other one.

NPT-508. Two providers can answer "what is on my calendar" — the phone and the
household iCloud account — and offering both to a planner is a genuinely
ambiguous choice. It picks one, sometimes the wrong one, and the result is a
reminder created somewhere the owner does not look.

The awkward half is that hiding the loser must not *disable* it: a durable plan
made an hour ago may already name an iCloud action, and it has to keep working.
So these tests check both directions at once — hidden from new catalogues,
still valid and still executable.
"""

from __future__ import annotations

from nova_voice.capabilities.base import CapabilityManifest, CapabilityProvider, ToolPolicy
from nova_voice.capabilities.registry import CapabilityRegistry
from nova_voice.domain import CapabilityToolCall, PlannedAction, ToolResult
from nova_voice.providers.companion.provider import (
    SUPERSEDED_ICLOUD_TOOLS,
    CompanionPersonalProvider,
)


def _tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    }


class _ICloud(CapabilityProvider):
    """Just enough of the iCloud provider to be superseded."""

    def manifest(self) -> CapabilityManifest:
        names = ["icloud.calendar.list", "icloud.reminders.list", "icloud.calendar.create"]
        return CapabilityManifest(
            id="icloud",
            version="0.1.0",
            contract_version="1",
            execution_class="iridium_local",
            tools=[_tool(name) for name in names],
            skill_files=[],
            tool_policies={name: ToolPolicy() for name in names},
        )

    async def execute(self, action: PlannedAction) -> ToolResult:
        return ToolResult(
            action_id=action.id, ok=True, code="ok", requested={}, message="icloud ran it"
        )

    async def health(self) -> dict:
        return {"ok": True}


class _Snapshot:
    def __init__(self, connected: bool) -> None:
        self.connected = connected
        self.personal_tools = frozenset({"companion.calendar.list"}) if connected else frozenset()
        self.locality = "home_lan"


class _Sessions:
    def __init__(self, connected: bool) -> None:
        self._snapshot = _Snapshot(connected)

    def snapshot(self, *, now=None):
        return self._snapshot

    async def personal_call(self, tool, arguments, *, max_items=50, deadline_seconds=15.0):
        return None


def _registry(*, companion_connected: bool) -> CapabilityRegistry:
    registry = CapabilityRegistry()
    registry.register(_ICloud())
    registry.register(CompanionPersonalProvider(_Sessions(companion_connected)))
    return registry


def _names(catalog: list[dict]) -> set[str]:
    return {tool["function"]["name"] for tool in catalog}


async def test_a_live_companion_hides_the_equivalent_icloud_reads():
    names = _names(await _registry(companion_connected=True).planner_tool_catalog())

    assert "companion.calendar.list" in names
    assert "icloud.calendar.list" not in names
    assert "icloud.reminders.list" not in names


async def test_a_disconnected_companion_leaves_icloud_in_the_catalogue():
    """The phone going away restores the fallback to the next catalogue."""

    names = _names(await _registry(companion_connected=False).planner_tool_catalog())

    assert "icloud.calendar.list" in names
    assert "icloud.reminders.list" in names


async def test_icloud_mutations_are_never_hidden():
    # Those go through CalDAV and work whether or not a phone is present, so
    # an approval already agreed to should execute by whichever route works.
    names = _names(await _registry(companion_connected=True).planner_tool_catalog())

    assert "icloud.calendar.create" in names
    assert "icloud.calendar.create" not in SUPERSEDED_ICLOUD_TOOLS


async def test_a_hidden_tool_is_still_valid_and_still_executable():
    """The property that makes this safe rather than destructive.

    A durable plan made before the phone connected already names an iCloud
    action. Hiding it from new catalogues must not retroactively break it.
    """

    registry = _registry(companion_connected=True)
    action = PlannedAction(
        id="action-1",
        order=0,
        call=CapabilityToolCall(provider="icloud", tool="icloud.calendar.list", arguments={}),
    )

    canonical = registry.validate_action(action)
    result = await registry.provider("icloud").execute(canonical)

    assert result.ok is True
    assert "icloud.calendar.list" in _names(registry.tool_catalog())


async def test_the_full_catalogue_is_never_filtered():
    # Validation and execution read this one, and a plan holding a tool whose
    # provider has gone quiet must still validate when it comes back.
    names = _names(_registry(companion_connected=True).tool_catalog())

    assert "icloud.calendar.list" in names
    assert "companion.calendar.list" in names


async def test_a_registered_but_unhealthy_superseder_hides_nothing():
    """Superseding on the strength of a provider that cannot answer would hide
    the working one and leave the planner with neither."""

    registry = _registry(companion_connected=False)
    health = await registry.health()

    assert health["companion"]["ok"] is False
    assert "icloud.calendar.list" in _names(await registry.planner_tool_catalog())


async def test_a_provider_that_supersedes_nothing_changes_nothing():
    registry = CapabilityRegistry()
    registry.register(_ICloud())

    assert _names(await registry.planner_tool_catalog()) == _names(registry.tool_catalog())
