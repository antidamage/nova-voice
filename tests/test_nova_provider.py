from __future__ import annotations

import json

import httpx
import pytest

from nova_voice.capabilities.registry import CapabilityRegistry
from nova_voice.domain import CapabilityToolCall, PlannedAction
from nova_voice.providers.nova.client import NovaDashboardClient
from nova_voice.providers.nova.provider import AliasIndex, NovaProvider

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


def test_llm_control_contract_exposes_only_idempotent_state_changes() -> None:
    provider = NovaProvider(object())
    manifest = provider.manifest()
    control = next(tool for tool in manifest.tools if tool["function"]["name"] == "nova.control")

    actions = control["function"]["parameters"]["properties"]["action"]["enum"]
    assert "toggle" not in actions
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
