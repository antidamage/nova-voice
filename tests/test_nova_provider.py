from __future__ import annotations

import json

import httpx
import pytest

from nova_voice.capabilities.registry import CapabilityRegistry
from nova_voice.domain import CapabilityToolCall, PlannedAction
from nova_voice.providers.nova.client import NovaDashboardClient
from nova_voice.providers.nova.provider import AliasIndex, AliasTarget, NovaProvider

STATE = {
    "generatedAt": "2026-07-14T00:00:00Z",
    "zones": [
        {"id": "everything", "name": "Home", "isOn": False, "brightnessPct": 0},
        {"id": "lounge", "name": "Lounge", "isOn": False, "brightnessPct": 0},
    ],
    "entities": [
        {
            "entity_id": "light.lounge_light",
            "domain": "light",
            "state": "off",
            "name": "Lounge light",
            "area_id": "lounge",
            "attributes": {"friendly_name": "Lounge light", "brightness": None},
        },
        {
            "entity_id": "light.kitchen_light_a",
            "domain": "light",
            "state": "off",
            "name": "Kitchen light",
            "area_id": "kitchen",
            "attributes": {"friendly_name": "Kitchen light", "brightness": None},
        },
        {
            "entity_id": "light.kitchen_light_b",
            "domain": "light",
            "state": "off",
            "name": "Kitchen light",
            "area_id": "kitchen",
            "attributes": {"friendly_name": "Kitchen light", "brightness": None},
        },
    ],
}


def test_alias_index_detects_ambiguity() -> None:
    index = AliasIndex()
    index.rebuild(STATE)
    assert len(index.resolve("kitchen light")) == 2
    target = index.resolve("lounge light")
    assert len(target) == 1
    assert target[0].id == "light.lounge_light"


@pytest.mark.asyncio
async def test_dashboard_client_collects_voice_settings_contract() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/api/voice"
        return httpx.Response(200, json={"voice": {"speaker": "Ryan"}})

    client = NovaDashboardClient(
        "http://nova.test",
        transport=httpx.MockTransport(handler),
    )
    assert await client.voice_settings() == {"voice": {"speaker": "Ryan"}}
    await client.close()


@pytest.mark.asyncio
async def test_device_action_admission_uses_the_full_dashboard_tree() -> None:
    state = {
        "zones": [
            {
                "id": "lounge",
                "name": "Lounge",
                "entities": [
                    {
                        "entity_id": "media_player.lounge_tv",
                        "domain": "media_player",
                        "name": "Lounge TV",
                        "state": "off",
                    }
                ],
            }
        ],
        # Deliberately no top-level entities: this proves admission follows the
        # tree genealogy rather than a light/climate-only catalogue.
        "entities": [],
    }

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/state"
        return httpx.Response(200, json=state)

    client = NovaDashboardClient(
        "http://nova.test", transport=httpx.MockTransport(handler)
    )
    provider = NovaProvider(client)
    real_tv = PlannedAction(
        id="tv",
        order=0,
        call=CapabilityToolCall(
            provider="nova",
            tool="nova.control",
            arguments={"target": "lounge tv", "action": "turn_on"},
        ),
    )
    invented_video = real_tv.model_copy(
        update={
            "id": "video",
            "call": real_tv.call.model_copy(
                update={"arguments": {"target": "video", "action": "turn_on"}}
            ),
        }
    )

    assert await provider.invalid_device_action_ids([real_tv]) == set()
    assert await provider.invalid_device_action_ids([invented_video]) == {"video"}
    await provider.close()


def test_prompt_context_carries_a_compact_weather_brief() -> None:
    assert NovaProvider._weather_brief(
        {
            "weather": {
                "entity_id": "weather.home",
                "condition": "rainy",
                "temperature": 11.4,
                "high": 14,
                "low": 8,
                "humidity": 87,
            }
        }
    ) == {
        "condition": "rainy",
        "temperatureC": 11.4,
        "todayHighC": 14,
        "todayLowC": 8,
        "humidityPct": 87,
    }
    assert NovaProvider._weather_brief({"weather": None}) is None
    assert NovaProvider._weather_brief({"weather": {"condition": None}}) is None


def test_indoor_temperatures_prefer_the_configured_room_sensor() -> None:
    state = {
        "zones": [
            {
                "id": "lounge",
                "name": "Lounge",
                "environment": {"temperatureEntityId": "sensor.tuya_lounge"},
            },
            {"id": "bedroom", "name": "Bedroom"},
            {"id": "hall", "name": "Hall"},
        ],
        "entities": [
            # Lounge: the trusted Tuya sensor wins over the aircon's own reading.
            {
                "entity_id": "sensor.tuya_lounge", "area_id": "lounge",
                "domain": "sensor", "state": "21.5",
            },
            {
                "entity_id": "climate.lounge_aircon", "area_id": "climate", "domain": "climate",
                "state": "cool", "attributes": {"current_temperature": 26, "temperature": 22},
            },
            # Both appliances live in the organisational Climate area. The
            # panel heater still measures the bedroom.
            {
                "entity_id": "climate.panel_heater", "name": "Panel Heater",
                "area_id": "climate", "domain": "climate",
                "state": "heat", "attributes": {"current_temperature": 18.5},
            },
            # Hall: nothing measures it.
            {"entity_id": "light.hall", "area_id": "hall", "domain": "light", "state": "off"},
        ],
    }
    temps = NovaProvider._indoor_temperatures(state)
    assert temps["lounge"] == 21.5
    assert temps["bedroom"] == 18.5
    assert temps["hall"] is None


def test_entity_body_routes_aircon_on_off_through_the_wired_routine() -> None:
    aircon = AliasTarget("entity", "climate.gree_lounge", "Air Conditioner", domain="climate")

    off = NovaProvider._entity_body(aircon, "turn_off", {})
    assert off["service"] == "turn_off"
    assert off["remember"] == {"aircon": {"autoMode": False, "offTimerEndsAt": None}}

    on = NovaProvider._entity_body(aircon, "turn_on", {})
    assert on["remember"] == {"aircon": {"autoMode": True}}

    temperature = NovaProvider._entity_body(aircon, "set_temperature", {"value": 24})
    assert temperature["remember"] == {"aircon": {"temperature": 24.0}}

    # A panel heater is a climate device but keeps its own controls, not the aircon routine.
    heater = AliasTarget("entity", "climate.panel_heater", "Panel Heater", domain="climate")
    assert "remember" not in NovaProvider._entity_body(heater, "turn_off", {})

    # A light off is a plain turn_off with no aircon preferences attached.
    light = AliasTarget("entity", "light.lounge", "Lounge Lamp", domain="light")
    assert "remember" not in NovaProvider._entity_body(light, "turn_off", {})


def test_climate_aliases_use_logical_rooms_not_the_organisational_area() -> None:
    index = AliasIndex()
    index.rebuild(
        {
            "zones": [{"id": "bedroom", "name": "Bedroom"}],
            "entities": [
                {
                    "entity_id": "climate.panel_heater_2",
                    "name": "Panel Heater",
                    "area_id": "climate",
                    "domain": "climate",
                    "state": "off",
                    "attributes": {},
                }
            ],
        }
    )

    target = index.resolve("bedroom heater", room="bedroom")
    assert len(target) == 1
    assert target[0].id == "climate.panel_heater_2"
    assert target[0].room == "bedroom"


def test_climate_power_verification_accepts_active_hvac_modes() -> None:
    heater = {
        "entity_id": "climate.panel_heater_2",
        "domain": "climate",
        "state": "heat",
        "attributes": {},
    }
    assert NovaProvider._verify("turn_on", {}, heater)
    assert not NovaProvider._verify("turn_off", {}, heater)
    assert NovaProvider._verify("turn_off", {}, {**heater, "state": "off"})


def test_device_states_separate_raw_devices_from_canonical_climate_controls() -> None:
    state = {
        "zones": [
            {"id": "lounge", "name": "Lounge"},
            {"id": "bedroom", "name": "Bedroom"},
            {"id": "climate", "name": "Climate"},
            {"id": "outside", "name": "Outside"},
            {"id": "network", "name": "Network"},
        ],
        "preferences": {"aircon": {"autoMode": True, "temperature": 22}},
        "entities": [
            {
                "entity_id": "light.lounge", "name": "Lounge Lamp", "area_id": "lounge",
                "domain": "light", "state": "on",
            },
            {
                "entity_id": "climate.aircon", "name": "Air Conditioner", "area_id": "lounge",
                "domain": "climate", "state": "cool",
                "attributes": {"temperature": 22, "current_temperature": 24},
            },
            # A diagnostic sensor is not a controllable device and is excluded.
            {
                "entity_id": "sensor.power", "name": "Power", "area_id": "lounge",
                "domain": "sensor", "state": "812",
            },
        ],
    }
    rows = NovaProvider._device_states(state)
    assert {row["name"] for row in rows} == {"Lounge Lamp"}

    controls = NovaProvider._climate_controls(state)
    assert controls == [
        {
            "name": "Air Conditioner",
            "room": "lounge",
            "power": "on",
            "targetTemperatureC": 22.0,
            "roomTemperatureC": 24.0,
            "supportedActions": ["turn_on", "turn_off", "set_temperature"],
        }
    ]
    assert NovaProvider._indoor_rooms(state) == ["lounge", "bedroom"]


@pytest.mark.asyncio
async def test_prompt_context_labels_indoor_rooms_and_bedroom_heater_temperature() -> None:
    state = {
        "zones": [
            {
                "id": "lounge",
                "name": "Lounge",
                "environment": {"temperatureEntityId": "sensor.lounge_temperature"},
            },
            {"id": "bedroom", "name": "Bedroom"},
            {"id": "climate", "name": "Climate"},
            {"id": "outside", "name": "Outside"},
            {"id": "network", "name": "Network"},
        ],
        "entities": [
            {
                "entity_id": "sensor.lounge_temperature",
                "name": "Temperature",
                "area_id": "lounge",
                "domain": "sensor",
                "state": "23.8",
                "attributes": {"device_class": "temperature"},
            },
            {
                "entity_id": "climate.c6780cad",
                "name": "Air Conditioner",
                "area_id": "climate",
                "domain": "climate",
                "state": "heat",
                "attributes": {"temperature": 25, "current_temperature": 23.8},
            },
            {
                "entity_id": "climate.panel_heater_2",
                "name": "Panel Heater",
                "area_id": "climate",
                "domain": "climate",
                "state": "off",
                "attributes": {"temperature": 22, "current_temperature": 21},
            },
        ],
        "preferences": {"aircon": {"autoMode": True, "temperature": 25}},
        "weather": {"condition": "clear-night", "temperature": 8.7},
    }

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/state"
        return httpx.Response(200, json=state)

    provider = NovaProvider(
        NovaDashboardClient("http://nova.test", transport=httpx.MockTransport(handler))
    )
    context = await provider.prompt_context("bedroom")

    assert context["indoorRooms"] == ["lounge", "bedroom"]
    assert context["indoorTemperatureC"] == 21.0
    assert context["weather"]["temperatureC"] == 8.7
    assert {control["room"] for control in context["climateControls"]} == {
        "lounge",
        "bedroom",
    }
    assert all(
        control["power"] in {"on", "off", "unavailable"}
        for control in context["climateControls"]
    )
    assert all(
        control.get("state") not in {"heat", "cool"}
        for control in context["climateControls"]
    )
    await provider.close()


def test_llm_control_contract_exposes_only_idempotent_state_changes() -> None:
    provider = NovaProvider(object())
    manifest = provider.manifest()
    control = next(tool for tool in manifest.tools if tool["function"]["name"] == "nova.control")

    actions = control["function"]["parameters"]["properties"]["action"]["enum"]
    assert "toggle" not in actions
    assert "set_mode" not in actions
    assert manifest.tool_policies["nova.control"].idempotent is True


@pytest.mark.parametrize(
    ("transcript", "scope", "action"),
    [
        ("turn all of the lights on", "indoors", "on"),
        ("Bandit, turn all lights on", "indoors", "on"),
        ("turn everything off", "indoors", "off"),
        ("please turn the outside lights on", "outside", "on"),
        ("turn all lights including outside off", "all", "off"),
    ],
)
def test_broad_lighting_phrases_use_deterministic_dashboard_shortcuts(
    transcript: str,
    scope: str,
    action: str,
) -> None:
    planned = NovaProvider.deterministic_lighting_shortcut(transcript)

    assert planned is not None
    assert planned.call.tool == "nova.lighting_shortcut"
    assert planned.call.arguments == {"scope": scope, "action": action}


@pytest.mark.parametrize(
    "transcript",
    [
        "turn the lounge lights on",
        "the television said turn everything off",
        "don't turn all the lights on",
    ],
)
def test_lighting_shortcut_does_not_capture_other_phrases(transcript: str) -> None:
    assert NovaProvider.deterministic_lighting_shortcut(transcript) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scope", "action", "path"),
    [
        ("indoors", "on", "/api/lights/on"),
        ("indoors", "off", "/api/lights/off"),
        ("all", "off", "/api/all-lights/off"),
        ("outside", "on", "/api/outside-light/on"),
    ],
)
async def test_lighting_shortcut_calls_the_dashboard_button_endpoint(
    scope: str,
    action: str,
    path: str,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == path
        return httpx.Response(200, text=action)

    client = NovaDashboardClient("http://nova.test", transport=httpx.MockTransport(handler))

    assert await client.lighting_shortcut(scope, action) == action

    await client.close()


@pytest.mark.asyncio
async def test_provider_verifies_adaptive_indoor_lighting_shortcut() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/api/lights/on":
            return httpx.Response(200, text="on")
        if request.url.path == "/api/state":
            return httpx.Response(
                200,
                json={
                    **STATE,
                    "zones": [
                        {
                            **STATE["zones"][0],
                            "isOn": True,
                            "entities": [
                                {**entity, "state": "on"} for entity in STATE["entities"]
                            ],
                        },
                        *STATE["zones"][1:],
                    ],
                },
            )
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    client = NovaDashboardClient("http://nova.test", transport=httpx.MockTransport(handler))
    provider = NovaProvider(client)
    planned = NovaProvider.deterministic_lighting_shortcut("turn all of the lights on")
    assert planned is not None

    result = await provider.execute(planned)

    assert result.ok
    assert result.target == "Home lights"
    assert result.observed == {
        "zone": "everything",
        "lights": 3,
        "on": 3,
        "isOn": True,
    }
    assert [request.url.path for request in requests] == ["/api/lights/on", "/api/state"]
    await provider.close()


def test_indoor_shortcut_verification_requires_every_light_for_on() -> None:
    partial_zone = {
        "id": "everything",
        "isOn": True,
        "entities": [
            {**STATE["entities"][0], "state": "on"},
            *STATE["entities"][1:],
        ],
    }

    verified, observed = NovaProvider._verify_lighting_shortcut(
        {"zones": [partial_zone]}, "indoors", "on"
    )

    assert not verified
    assert observed["on"] == 1
    assert observed["lights"] == 3


def test_missing_zone_cannot_verify_an_off_shortcut() -> None:
    verified, observed = NovaProvider._verify_lighting_shortcut(
        {"zones": []}, "outside", "off"
    )

    assert not verified
    assert observed == {"zone": "outside", "available": False}


@pytest.mark.asyncio
async def test_registry_rejects_unallowlisted_tool_arguments() -> None:
    provider = NovaProvider(NovaDashboardClient("http://nova.test"))
    registry = CapabilityRegistry(allowlist={"nova"})
    registry.register(provider)

    with pytest.raises(ValueError, match="schema"):
        registry.validate_action(
            PlannedAction(
                id="a1",
                order=0,
                call=CapabilityToolCall(
                    provider="nova",
                    tool="nova.control",
                    arguments={"target": "lounge light", "action": "delete_everything"},
                ),
            )
        )
    await registry.close()


@pytest.mark.asyncio
async def test_registry_canonicalizes_provider_scoped_short_tool_name() -> None:
    provider = NovaProvider(NovaDashboardClient("http://nova.test"))
    registry = CapabilityRegistry(allowlist={"nova"})
    registry.register(provider)
    action = PlannedAction(
        id="a1",
        order=0,
        call=CapabilityToolCall(
            provider="nova",
            tool="control",
            arguments={"target": "lounge light", "action": "turn_on"},
        ),
    )

    canonical = registry.validate_action(action)

    assert canonical.call.tool == "nova.control"
    await registry.close()


@pytest.mark.asyncio
async def test_provider_maps_and_verifies_entity_action() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/api/state":
            return httpx.Response(200, json=STATE)
        if request.url.path == "/api/entity":
            body = json.loads(request.content)
            assert body == {
                "entityId": "light.lounge_light",
                "domain": "light",
                "service": "turn_on",
                "data": {},
            }
            changed = {
                **STATE,
                "entities": [{**STATE["entities"][0], "state": "on"}, *STATE["entities"][1:]],
            }
            return httpx.Response(200, json=changed)
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    client = NovaDashboardClient("http://nova.test", transport=httpx.MockTransport(handler))
    provider = NovaProvider(client)
    await provider.refresh()
    result = await provider.execute(
        PlannedAction(
            id="a1",
            order=0,
            call=CapabilityToolCall(
                provider="nova",
                tool="nova.control",
                arguments={"target": "lounge light", "action": "turn_on"},
            ),
        )
    )
    assert result.ok
    assert result.observed and result.observed["state"] == "on"
    assert [request.url.path for request in requests] == ["/api/state", "/api/entity"]
    await provider.close()


@pytest.mark.asyncio
async def test_light_waits_for_home_assistant_to_publish_success() -> None:
    polls = 0
    off = STATE

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal polls
        if request.url.path == "/api/entity":
            return httpx.Response(200, json=off)
        if request.url.path == "/api/state":
            polls += 1
            changed = {
                **off,
                "entities": [
                    {**off["entities"][0], "state": "on"},
                    *off["entities"][1:],
                ],
            }
            return httpx.Response(200, json=changed if polls >= 3 else off)
        raise AssertionError(request.url.path)

    provider = NovaProvider(
        NovaDashboardClient("http://nova.test", transport=httpx.MockTransport(handler)),
        climate_verify_timeout_seconds=1,
        climate_verify_interval_seconds=0,
    )
    result = await provider.execute(
        PlannedAction(
            id="light-on",
            order=0,
            call=CapabilityToolCall(
                provider="nova",
                tool="nova.control",
                arguments={"target": "lounge light", "action": "turn_on"},
            ),
        )
    )

    assert result.ok
    assert result.observed and result.observed["state"] == "on"
    # Initial alias/state load plus two post-command verification refreshes.
    assert polls == 3
    await provider.close()


@pytest.mark.asyncio
async def test_ralph_loop_stops_at_iteration_cap_without_resending_command() -> None:
    state_reads = 0
    mutations = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal mutations, state_reads
        if request.url.path == "/api/state":
            state_reads += 1
            return httpx.Response(200, json=STATE)
        if request.url.path == "/api/entity":
            mutations += 1
            return httpx.Response(200, json=STATE)
        raise AssertionError(request.url.path)

    provider = NovaProvider(
        NovaDashboardClient("http://nova.test", transport=httpx.MockTransport(handler)),
        climate_verify_timeout_seconds=1,
        climate_verify_interval_seconds=0,
        verification_max_iterations=2,
    )
    result = await provider.execute(
        PlannedAction(
            id="light-on",
            order=0,
            call=CapabilityToolCall(
                provider="nova",
                tool="nova.control",
                arguments={"target": "lounge light", "action": "turn_on"},
            ),
        )
    )

    assert not result.ok
    assert result.code == "unverified"
    assert state_reads == 3  # one initial read plus two bounded loop reads
    assert mutations == 1
    await provider.close()


@pytest.mark.asyncio
async def test_disabled_ralph_loop_uses_only_immediate_dashboard_snapshot() -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return httpx.Response(200, json=STATE)

    provider = NovaProvider(
        NovaDashboardClient("http://nova.test", transport=httpx.MockTransport(handler)),
        verification_loop_enabled=False,
    )
    result = await provider.execute(
        PlannedAction(
            id="light-on",
            order=0,
            call=CapabilityToolCall(
                provider="nova",
                tool="nova.control",
                arguments={"target": "lounge light", "action": "turn_on"},
            ),
        )
    )

    assert not result.ok
    assert paths == ["/api/state", "/api/entity"]
    await provider.close()


@pytest.mark.asyncio
async def test_panel_heater_waits_for_published_power_state_and_returns_canonical_result() -> None:
    polls = 0
    off = {
        "zones": [{"id": "bedroom", "name": "Bedroom"}, {"id": "climate", "name": "Climate"}],
        "entities": [
            {
                "entity_id": "climate.panel_heater_2",
                "name": "Panel Heater",
                "area_id": "climate",
                "domain": "climate",
                "state": "off",
                "attributes": {"temperature": 22, "current_temperature": 21},
            }
        ],
        "preferences": {},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal polls
        if request.url.path == "/api/entity":
            assert json.loads(request.content) == {
                "entityId": "climate.panel_heater_2",
                "domain": "climate",
                "service": "turn_on",
                "data": {},
            }
            return httpx.Response(200, json=off)
        if request.url.path == "/api/state":
            polls += 1
            state = off if polls < 3 else {
                **off,
                "entities": [{**off["entities"][0], "state": "heat"}],
            }
            return httpx.Response(200, json=state)
        raise AssertionError(request.url.path)

    provider = NovaProvider(
        NovaDashboardClient("http://nova.test", transport=httpx.MockTransport(handler)),
        climate_verify_timeout_seconds=1,
        climate_verify_interval_seconds=0,
    )
    result = await provider.execute(
        PlannedAction(
            id="heater-on",
            order=0,
            call=CapabilityToolCall(
                provider="nova",
                tool="nova.control",
                arguments={"target": "bedroom heater", "action": "turn_on"},
            ),
        )
    )

    assert result.ok
    assert result.observed == {
        "name": "Panel Heater",
        "room": "bedroom",
        "power": "on",
        "targetTemperatureC": 22.0,
        "roomTemperatureC": 21.0,
        "supportedActions": ["turn_on", "turn_off", "set_temperature"],
    }
    assert polls == 3
    await provider.close()


@pytest.mark.asyncio
async def test_panel_heater_temperature_is_not_claimed_when_setpoint_never_changes() -> None:
    state = {
        "zones": [{"id": "bedroom", "name": "Bedroom"}, {"id": "climate", "name": "Climate"}],
        "entities": [
            {
                "entity_id": "climate.panel_heater_2",
                "name": "Panel Heater",
                "area_id": "climate",
                "domain": "climate",
                "state": "heat",
                "attributes": {"temperature": 20, "current_temperature": 19},
            }
        ],
        "preferences": {},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=state)

    provider = NovaProvider(
        NovaDashboardClient("http://nova.test", transport=httpx.MockTransport(handler)),
        climate_verify_timeout_seconds=0,
        climate_verify_interval_seconds=0,
    )
    result = await provider.execute(
        PlannedAction(
            id="heater-temperature",
            order=0,
            call=CapabilityToolCall(
                provider="nova",
                tool="nova.control",
                arguments={
                    "target": "bedroom heater",
                    "action": "set_temperature",
                    "value": 24,
                },
            ),
        )
    )

    assert not result.ok
    assert result.code == "unverified"
    assert result.observed and result.observed["targetTemperatureC"] == 20.0
    await provider.close()


@pytest.mark.asyncio
async def test_dashboard_http_error_preserves_safe_json_detail() -> None:
    client = NovaDashboardClient(
        "http://nova.test",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(503, json={"detail": "MCP token missing"})
        ),
    )

    with pytest.raises(RuntimeError, match=r"HTTP 503\): MCP token missing"):
        await client.mcp_call("nova.dashboard.health")
    await client.close()


@pytest.mark.asyncio
async def test_health_does_not_force_a_full_state_refetch_each_poll() -> None:
    state_calls = 0
    version_calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal state_calls, version_calls
        if request.url.path == "/api/state":
            state_calls += 1
            return httpx.Response(200, json=STATE)
        if request.url.path == "/api/version":
            version_calls += 1
            return httpx.Response(200, json={"buildId": "build-xyz"})
        raise AssertionError(f"unexpected path {request.url.path}")

    client = NovaDashboardClient(
        "http://nova.test", transport=httpx.MockTransport(handler)
    )
    provider = NovaProvider(client, alias_refresh_seconds=30)

    first = await provider.health()
    second = await provider.health()

    assert first["ok"] is True
    assert second["ok"] is True
    assert first["dashboardBuildId"] == "build-xyz"
    assert first["stateGeneratedAt"] == STATE["generatedAt"]
    # /health is polled every few seconds by the status strip: it must probe
    # reachability (version) without re-pulling the entire HA snapshot each
    # time. With the old force=True this fetched /api/state on every call.
    assert version_calls == 2
    assert state_calls == 1  # only the initial cold-cache fetch
    await client.close()
