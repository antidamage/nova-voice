"""Relative control verbs: brighten, dim, warm_up, cool_down.

These run against a mock dashboard in a dry run, so they assert on the exact
request the provider builds without waiting for a device to publish anything.
That is deliberate: the bug these verbs exist for was only ever reproducible by
speaking at a live host and reading its logs, which made a one-line change cost
a deploy. The arithmetic is the provider's, so it can be tested here in
milliseconds.
"""

from __future__ import annotations

import httpx
import pytest

from nova_voice.domain import CapabilityToolCall, PlannedAction
from nova_voice.dry_run import begin_dry_run, current_dry_run, end_dry_run
from nova_voice.providers.nova.client import NovaDashboardClient
from nova_voice.providers.nova.provider import NovaProvider

STATE = {
    "zones": [
        {"id": "lounge", "name": "Lounge", "isOn": True, "brightnessPct": 40},
        {"id": "hallway", "name": "Hallway", "isOn": False, "brightnessPct": 0},
        {"id": "study", "name": "Study", "isOn": True, "brightnessPct": 95},
    ],
    "preferences": {"aircon": {"autoMode": True, "temperature": 22}},
    "entities": [
        {
            "entity_id": "light.lamp",
            "name": "Reading lamp",
            "domain": "light",
            "state": "on",
            "area_id": "lounge",
            "attributes": {"friendly_name": "Reading lamp", "brightness": 128},
        },
        {
            "entity_id": "climate.aircon",
            "name": "Air Conditioner",
            "domain": "climate",
            "state": "cool",
            "area_id": "lounge",
            "attributes": {"temperature": 22, "current_temperature": 24},
        },
        {
            "entity_id": "climate.panel_heater",
            "name": "Panel Heater",
            "domain": "climate",
            "state": "heat",
            "area_id": "bedroom",
            "attributes": {"temperature": 18, "current_temperature": 17},
        },
        # A climate device that publishes no target at all. Nothing can be
        # "two degrees warmer" than an unknown number.
        {
            "entity_id": "climate.spare_heater",
            "name": "Spare Heater",
            "domain": "climate",
            "state": "heat",
            "area_id": "bedroom",
            "attributes": {"current_temperature": 19},
        },
    ],
}


@pytest.fixture
def dry_run():
    token = begin_dry_run()
    try:
        yield current_dry_run()
    finally:
        end_dry_run(token)


def _provider() -> NovaProvider:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/state":
            return httpx.Response(200, json=STATE)
        return httpx.Response(200, json={"ok": True})

    return NovaProvider(
        NovaDashboardClient("http://nova.test", transport=httpx.MockTransport(handler))
    )


async def _control(arguments: dict):
    provider = _provider()
    try:
        return await provider.execute(
            PlannedAction(
                id="a1",
                order=0,
                call=CapabilityToolCall(
                    provider="nova", tool="nova.control", arguments=arguments
                ),
            )
        )
    finally:
        await provider.close()


@pytest.mark.parametrize(
    ("target", "action", "expected"),
    [
        ("Lounge", "brighten", 65),
        ("Lounge", "dim", 15),
        # Clamped at the ends of the scale rather than refused: asking for
        # brighter at 95 means "all the way up", not "error".
        ("Study", "brighten", 100),
        ("Hallway", "dim", 0),
        # Off is a real starting point, and brighter from off lights the room.
        ("Hallway", "brighten", 25),
    ],
)
async def test_relative_brightness_resolves_against_the_zone_reading(
    dry_run, target: str, action: str, expected: int
) -> None:
    result = await _control({"target": target, "action": action})

    assert result.ok, result.message
    assert result.requested == {
        "zoneId": target.casefold(),
        "action": "brightness",
        "brightnessPct": expected,
    }


async def test_relative_brightness_reads_an_entity_in_home_assistant_units(
    dry_run,
) -> None:
    """A bulb reports 0-255, and the step is in percent, so one has to convert."""

    result = await _control({"target": "Reading lamp", "action": "brighten"})

    # 128/255 is 50%, so one step up is 75.
    assert result.requested["data"] == {"brightness_pct": 75}


@pytest.mark.parametrize(
    ("target", "action", "expected"),
    [
        ("Air Conditioner", "warm_up", 24.0),
        ("Air Conditioner", "cool_down", 20.0),
        ("Panel Heater", "warm_up", 20.0),
        ("Panel Heater", "cool_down", 16.0),
    ],
)
async def test_relative_temperature_moves_two_degrees_from_the_current_target(
    dry_run, target: str, action: str, expected: float
) -> None:
    result = await _control({"target": target, "action": action})

    assert result.ok, result.message
    # The aircon goes through the dashboard's own climate routine and carries a
    # top-level temperature; every other climate device is an ordinary entity
    # service call. Both are correct, and the resolved value is what is on test.
    requested = result.requested
    assert requested.get("temperature", requested.get("data", {}).get("temperature")) == expected


async def test_a_relative_climate_step_stays_inside_the_configured_bounds(
    dry_run,
) -> None:
    provider = _provider()
    await provider.refresh()
    target = provider.aliases.resolve("Panel Heater")[0]

    # Directly, because driving a heater to 35 through the state fixture would
    # only be testing the fixture.
    provider._state["entities"][2]["attributes"]["temperature"] = 34
    assert provider._resolve_relative(target, "warm_up", {}) == (
        "set_temperature",
        {"value": 35.0},
    )
    provider._state["entities"][2]["attributes"]["temperature"] = 6
    assert provider._resolve_relative(target, "cool_down", {}) == (
        "set_temperature",
        {"value": 5.0},
    )
    await provider.close()


async def test_the_speaker_may_size_the_step_but_never_flips_its_direction(
    dry_run,
) -> None:
    up = await _control({"target": "Lounge", "action": "brighten", "value": 10})
    assert up.requested["brightnessPct"] == 50

    # The verb already said which way. A negative magnitude is a model slip, and
    # honouring its sign would brighten a light the household asked to dim.
    down = await _control({"target": "Lounge", "action": "dim", "value": -10})
    assert down.requested["brightnessPct"] == 30


async def test_an_unknown_reading_is_refused_rather_than_guessed(dry_run) -> None:
    result = await _control({"target": "Spare Heater", "action": "warm_up"})

    assert not result.ok
    assert result.code == "blocked"
    assert "no current target temperature" in result.message
    # Nothing was built, so nothing could have been sent.
    assert dry_run.requests == []


async def test_the_relative_verbs_are_offered_to_the_planning_model() -> None:
    """The model can only emit a verb the manifest and the manifest hint name."""

    provider = NovaProvider(object())
    control = next(
        tool for tool in provider.manifest().tools if tool["function"]["name"] == "nova.control"
    )

    actions = control["function"]["parameters"]["properties"]["action"]["enum"]
    assert {"brighten", "dim", "warm_up", "cool_down"} <= set(actions)

    climate = NovaProvider._climate_controls(STATE)
    assert climate, "the fixture should expose at least one climate control"
    for control_row in climate:
        assert {"warm_up", "cool_down"} <= set(control_row["supportedActions"])
