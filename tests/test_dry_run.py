from __future__ import annotations

import httpx
import pytest
from conftest import interpretation

from nova_voice.config import Settings
from nova_voice.domain import (
    CapabilityToolCall,
    Decision,
    PlannedAction,
    SpeechAct,
    Utterance,
)
from nova_voice.dry_run import begin_dry_run, current_dry_run, end_dry_run
from nova_voice.policy import ExecutionPolicy
from nova_voice.providers.nova.client import NovaDashboardClient
from nova_voice.providers.nova.provider import NovaProvider

STATE = {
    "zones": [
        {
            "id": "kitchen",
            "name": "Kitchen",
            "isOn": False,
            "entities": [
                {
                    "entity_id": "light.kitchen",
                    "name": "Kitchen light",
                    "domain": "light",
                    "state": "off",
                }
            ],
        }
    ],
    "entities": [
        {
            "entity_id": "light.kitchen",
            "name": "Kitchen light",
            "domain": "light",
            "state": "off",
            "area_id": "kitchen",
        }
    ],
}


def _client(seen: list[httpx.Request]) -> NovaDashboardClient:
    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/api/state":
            return httpx.Response(200, json=STATE)
        return httpx.Response(200, json={"ok": True})

    return NovaDashboardClient(
        "http://nova.test", transport=httpx.MockTransport(handler)
    )


def _action(tool: str, arguments: dict) -> PlannedAction:
    return PlannedAction(
        id="a1",
        order=0,
        call=CapabilityToolCall(provider="nova", tool=tool, arguments=arguments),
    )


@pytest.fixture
def dry_run():
    token = begin_dry_run()
    try:
        yield current_dry_run()
    finally:
        end_dry_run(token)


async def test_entity_mutation_is_recorded_and_never_sent(dry_run) -> None:
    seen: list[httpx.Request] = []
    provider = NovaProvider(_client(seen))

    result = await provider.execute(
        _action("nova.control", {"target": "Kitchen light", "action": "turn_on"})
    )

    assert result.code == "dry_run"
    assert result.ok is True
    # The exact body a real run would have POSTed, so a test can assert on what
    # would have happened rather than on what was said about it.
    assert result.requested == {
        "entityId": "light.kitchen",
        "domain": "light",
        "service": "turn_on",
        "data": {},
    }
    # Pre-action state: the "before" side of a diff that was never applied.
    assert result.observed is not None and result.observed["state"] == "off"
    assert [request.url.path for request in seen] == ["/api/state"]
    assert [(item.method, item.path) for item in dry_run.requests] == [
        ("POST", "/api/entity")
    ]


async def test_lighting_shortcut_is_withheld(dry_run) -> None:
    seen: list[httpx.Request] = []
    provider = NovaProvider(_client(seen))
    await provider.refresh(force=True)

    result = await provider.execute(
        _action("nova.lighting_shortcut", {"scope": "indoors", "action": "on"})
    )

    assert result.code == "dry_run"
    assert result.requested == {"scope": "indoors", "action": "on"}
    # A GET that mutates is still a mutation; the guard is not method-based.
    assert dry_run.requests[0].path == "/api/lights/on"
    assert "/api/lights/on" not in [request.url.path for request in seen]


async def test_mode_is_withheld(dry_run) -> None:
    seen: list[httpx.Request] = []
    provider = NovaProvider(_client(seen))

    result = await provider.execute(
        _action("nova.mode", {"mode": "house_party", "action": "on"})
    )

    assert result.code == "dry_run"
    assert result.requested == {"mode": "house-party", "enabled": True}
    assert dry_run.requests[0].path == "/api/modes"


async def test_reads_still_reach_the_dashboard(dry_run) -> None:
    seen: list[httpx.Request] = []
    provider = NovaProvider(_client(seen))

    # A dry run needs live state to resolve targets and report the before-state,
    # so reads deliberately pass through the guard untouched.
    await provider.refresh(force=True)

    assert [request.url.path for request in seen] == ["/api/state"]
    assert dry_run.requests == []


def test_dry_run_turn_passes_the_shadow_mode_short_circuit() -> None:
    settings = Settings(shadow_mode=True)
    policy = ExecutionPolicy(settings)
    plan = interpretation(
        speech_act=SpeechAct.DIRECTIVE,
        decision=Decision.EXECUTE,
        actions=[_action("nova.control", {"target": "Kitchen light", "action": "turn_on"})],
    )

    shadowed = policy.evaluate(
        Utterance.text("turn on the kitchen light", wake_detected=True),
        plan,
        session_active=False,
    )
    assert (shadowed.execute, shadowed.shadowed) == (False, True)

    # A dry run is strictly more informative than shadow mode and just as safe,
    # so it must reach the provider rather than stopping before any request
    # body is built.
    dry = policy.evaluate(
        Utterance.text("turn on the kitchen light", wake_detected=True, dry_run=True),
        plan,
        session_active=False,
    )
    assert (dry.execute, dry.shadowed, dry.dry_run) == (True, False, True)


def test_verified_targets_waive_a_low_confidence_score() -> None:
    settings = Settings(shadow_mode=False)
    policy = ExecutionPolicy(settings)
    plan = interpretation(
        speech_act=SpeechAct.DIRECTIVE,
        decision=Decision.EXECUTE,
        confidence=0.1,
        actions=[_action("nova.control", {"target": "Kitchen light", "action": "turn_on"})],
    )
    utterance = Utterance.text("turn on the kitchen light", wake_detected=True)

    assert policy.evaluate(utterance, plan, session_active=False).execute is False

    outcome = policy.evaluate(utterance, plan, session_active=False, targets_verified=True)
    assert outcome.execute is True
    assert "waived low confidence" in outcome.reason


def test_verified_targets_do_not_waive_the_addressing_gate() -> None:
    settings = Settings(shadow_mode=False)
    policy = ExecutionPolicy(settings)
    plan = interpretation(
        speech_act=SpeechAct.DIRECTIVE,
        decision=Decision.EXECUTE,
        addressed=0.05,
        actions=[_action("nova.control", {"target": "Kitchen light", "action": "turn_on"})],
    )

    outcome = policy.evaluate(
        Utterance.text("turn on the kitchen light"),
        plan,
        session_active=False,
        targets_verified=True,
    )
    assert outcome.execute is False
    assert "addressing probability" in outcome.reason
