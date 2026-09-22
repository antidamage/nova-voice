"""Calendar, reminders, location and Health as an ordinary Nova capability.

NPT-501 and NPT-507. The manifest is the contract, so most of these tests are
about what the manifest *refuses to allow*: an unbounded query, a mutation that
looks like a read, or a Health type nobody vetted. The provider tests are about
the one thing that differs from a local provider — what happens when the device
is not there.
"""

from __future__ import annotations

import pytest

from nova_voice.capabilities.base import ToolPolicy
from nova_voice.domain import CapabilityToolCall, PlannedAction
from nova_voice.providers.companion.provider import (
    COMPANION_PERSONAL_TOOLS,
    HEALTH_READ_ALLOWLIST,
    HEALTH_WRITE_ALLOWLIST,
    CompanionPersonalProvider,
    validate_health_write,
)


class _Snapshot:
    def __init__(self, *, connected=True, tools=("companion.calendar.list",)) -> None:
        self.connected = connected
        self.personal_tools = frozenset(tools)
        self.locality = "home_lan"


class _Result:
    def __init__(self, *, ok=True, items=None, truncated=False, code="ok", message="done"):
        self.ok = ok
        self.items = items or []
        self.truncated = truncated
        self.code = code
        self.message = message


class _Sessions:
    def __init__(self, *, snapshot=None, result=None) -> None:
        self._snapshot = snapshot or _Snapshot()
        self._result = result
        self.calls: list[tuple[str, dict, int]] = []

    def snapshot(self, *, now=None):
        return self._snapshot

    async def personal_call(self, tool, arguments, *, max_items=50, deadline_seconds=15.0):
        self.calls.append((tool, arguments, max_items))
        return self._result


def _action(tool: str, **arguments) -> PlannedAction:
    return PlannedAction(
        id="action-1",
        order=0,
        call=CapabilityToolCall(provider="companion", tool=tool, arguments=arguments),
    )


def _tool_names() -> list[str]:
    return [entry["function"]["name"] for entry in COMPANION_PERSONAL_TOOLS]


# -- the manifest (NPT-501) ---------------------------------------------------


def test_every_read_is_bounded_by_a_required_argument():
    """No tool here can ask for "my calendar"."""

    provider = CompanionPersonalProvider(_Sessions())
    policies = provider.manifest().tool_policies

    for entry in COMPANION_PERSONAL_TOOLS:
        function = entry["function"]
        policy = policies[function["name"]]
        if policy.risk != "low":
            continue
        required = set(function["parameters"]["required"])
        assert required, f"{function['name']} takes no required bound"
        # Either a count cap or an explicit window. A read with neither is the
        # shape that turns one careless plan into a large personal payload.
        assert "maxItems" in required or {"start", "end"} <= required or {
            "accuracy",
            "maxAgeSeconds",
        } <= required, function["name"]


def test_every_read_is_a_different_tool_from_every_mutation():
    provider = CompanionPersonalProvider(_Sessions())
    policies = provider.manifest().tool_policies
    reads = {name for name, policy in policies.items() if policy.risk == "low"}
    writes = {name for name, policy in policies.items() if policy.risk != "low"}

    assert reads and writes
    assert reads.isdisjoint(writes)
    # Nothing is left unclassified: an unpoliced tool would execute as low-risk.
    assert reads | writes == set(_tool_names())


def test_every_mutation_requires_confirmation():
    """This is what makes the approval gate reachable at all."""

    provider = CompanionPersonalProvider(_Sessions())
    for name, policy in provider.manifest().tool_policies.items():
        if policy.risk == "low":
            continue
        assert policy.requires_confirmation, name
        assert isinstance(policy, ToolPolicy)


def test_a_deletion_is_declared_irreversible():
    provider = CompanionPersonalProvider(_Sessions())
    policies = provider.manifest().tool_policies

    assert policies["companion.calendar.delete"].reversible is False
    assert policies["companion.reminders.delete"].reversible is False
    # A Health sample written and then deleted is not the same as one never
    # written, and the owner should be told that before agreeing.
    assert policies["companion.health.write"].reversible is False


def test_there_is_no_generic_health_writer():
    write = next(
        entry
        for entry in COMPANION_PERSONAL_TOOLS
        if entry["function"]["name"] == "companion.health.write"
    )
    metric = write["function"]["parameters"]["properties"]["metric"]

    assert metric["enum"] == list(HEALTH_WRITE_ALLOWLIST)
    assert len(HEALTH_WRITE_ALLOWLIST) < 10


def test_location_makes_the_planner_state_the_precision_it_wants():
    # Coarse-by-default would hide a request for street-level precision behind
    # an omitted argument.
    location = next(
        entry
        for entry in COMPANION_PERSONAL_TOOLS
        if entry["function"]["name"] == "companion.location.current"
    )
    required = location["function"]["parameters"]["required"]

    assert "accuracy" in required
    assert "maxAgeSeconds" in required


def test_the_provider_declares_itself_as_running_off_this_box():
    provider = CompanionPersonalProvider(_Sessions())
    assert provider.manifest().execution_class == "household_lan_service"


# -- Health validation --------------------------------------------------------


@pytest.mark.parametrize(
    ("metric", "value", "unit", "expected"),
    [
        ("bodyMass", 72.0, "kg", None),
        ("bodyMass", 72.0, "lb", "must be written in kg"),
        ("bodyMass", 900.0, "kg", "outside the plausible range"),
        ("heartRate", 60.0, "bpm", "not an allowlisted"),
        ("dietaryWater", 250.0, "mL", None),
        ("mindfulSession", 0.0, "min", "outside the plausible range"),
    ],
)
def test_a_health_write_is_checked_before_the_owner_is_asked(metric, value, unit, expected):
    """Nobody should be asked to approve a write that could not have run."""

    problem = validate_health_write(metric, value, unit)
    if expected is None:
        assert problem is None
    else:
        assert problem is not None and expected in problem


# -- the proxy (NPT-507) ------------------------------------------------------


async def test_a_read_is_dispatched_to_the_device():
    sessions = _Sessions(result=_Result(items=[{"id": "event-1"}]))
    provider = CompanionPersonalProvider(sessions)

    result = await provider.execute(
        _action("companion.calendar.list", start="a", end="b", maxItems=10)
    )

    assert result.ok is True
    assert result.observed == {"items": [{"id": "event-1"}], "truncated": False}
    assert sessions.calls[0][0] == "companion.calendar.list"
    # The declared cap is what is enforced, not the provider's default.
    assert sessions.calls[0][2] == 10


async def test_a_disconnected_phone_is_unavailable_rather_than_an_error():
    """A phone in a pocket is a normal state of the world, not a fault."""

    sessions = _Sessions(snapshot=_Snapshot(connected=False))
    provider = CompanionPersonalProvider(sessions)

    result = await provider.execute(
        _action("companion.calendar.list", start="a", end="b", maxItems=10)
    )

    assert result.ok is False
    assert result.code == "unavailable"
    assert sessions.calls == []


async def test_a_silent_phone_is_unavailable_rather_than_hanging():
    sessions = _Sessions(result=None)
    provider = CompanionPersonalProvider(sessions)

    result = await provider.execute(
        _action("companion.reminders.list", maxItems=5)
    )

    assert result.ok is False
    assert result.code == "unavailable"


async def test_a_mutation_never_reaches_the_phone_through_this_path():
    """The approval gate owns writes. A second route would bypass it."""

    sessions = _Sessions(result=_Result())
    provider = CompanionPersonalProvider(sessions)

    result = await provider.execute(
        _action("companion.reminders.delete", reminderId="reminder-1")
    )

    assert result.ok is False
    assert result.code == "blocked"
    assert "approval" in result.message
    assert sessions.calls == []


async def test_an_unvetted_health_metric_is_refused_before_dispatch():
    sessions = _Sessions(result=_Result())
    provider = CompanionPersonalProvider(sessions)

    result = await provider.execute(
        _action("companion.health.summary", metrics=["bloodGlucose"], start="a", end="b")
    )

    assert result.ok is False
    assert result.code == "invalid"
    assert sessions.calls == []
    assert "bloodGlucose" in result.message


async def test_an_unknown_tool_is_refused():
    provider = CompanionPersonalProvider(_Sessions(result=_Result()))
    result = await provider.execute(_action("companion.calendar.obliterate"))

    assert result.ok is False
    assert result.code == "invalid"


async def test_a_device_error_nova_understands_keeps_its_code():
    sessions = _Sessions(
        result=_Result(ok=False, code="not_found", message="no such reminder")
    )
    provider = CompanionPersonalProvider(sessions)

    result = await provider.execute(_action("companion.reminders.list", maxItems=5))

    assert result.ok is False
    assert result.code == "not_found"


async def test_an_unrecognised_device_code_is_carried_in_the_message_not_the_code():
    # `ToolResult.code` is a closed set. A device inventing a code must not
    # turn a handled refusal into a validation crash, and the original must not
    # be lost either.
    sessions = _Sessions(
        result=_Result(ok=False, code="permission_denied", message="calendar access is off")
    )
    provider = CompanionPersonalProvider(sessions)

    result = await provider.execute(
        _action("companion.calendar.list", start="a", end="b", maxItems=10)
    )

    assert result.ok is False
    assert result.code == "backend_error"
    assert "calendar access is off" in result.message
    assert "permission_denied" in result.message


# -- health reporting ---------------------------------------------------------


async def test_health_reflects_the_live_session():
    provider = CompanionPersonalProvider(_Sessions())
    assert (await provider.health())["ok"] is True

    disconnected = CompanionPersonalProvider(_Sessions(snapshot=_Snapshot(connected=False)))
    report = await disconnected.health()
    assert report["ok"] is False
    assert report["detail"] == "no companion session"


async def test_denying_every_permission_is_reported_distinctly_from_being_offline():
    # These need different answers: one is "open the app", the other is
    # "change a setting", and an operator should not have to guess which.
    provider = CompanionPersonalProvider(_Sessions(snapshot=_Snapshot(tools=())))
    report = await provider.health()

    assert report["connected"] is True
    assert report["ok"] is False
    assert "no personal permissions" in report["detail"]


def test_the_read_allowlist_is_reflected_in_the_summary_schema():
    summary = next(
        entry
        for entry in COMPANION_PERSONAL_TOOLS
        if entry["function"]["name"] == "companion.health.summary"
    )
    metrics = summary["function"]["parameters"]["properties"]["metrics"]["items"]

    assert metrics["enum"] == list(HEALTH_READ_ALLOWLIST)
