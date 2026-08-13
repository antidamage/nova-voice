from __future__ import annotations

import httpx
import pytest

from nova_voice.domain import CapabilityToolCall, PlannedAction
from nova_voice.providers.nova.client import NovaDashboardClient
from nova_voice.providers.nova.provider import NovaProvider, resolve_color


def _provider(handler) -> NovaProvider:
    return NovaProvider(
        NovaDashboardClient("http://nova.test", transport=httpx.MockTransport(handler))
    )


def _mode_action(mode: str, action: str) -> PlannedAction:
    return PlannedAction(
        id="m1",
        order=0,
        call=CapabilityToolCall(
            provider="nova", tool="nova.mode", arguments={"mode": mode, "action": action}
        ),
    )


async def test_house_party_on_is_verified_from_the_reply() -> None:
    sent: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        import json

        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"mode": "house-party", "enabled": True})

    result = await _provider(handler).execute(_mode_action("house_party", "on"))

    assert sent == [{"mode": "house-party", "enabled": True}]
    assert (result.ok, result.code) == (True, "ok")


async def test_a_mode_the_dashboard_did_not_apply_is_unverified() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"mode": "house-party", "enabled": False})

    result = await _provider(handler).execute(_mode_action("house_party", "on"))

    # There is no device to catch up here, so a mismatch is a real failure
    # rather than a slow one — it must not be reported as success.
    assert (result.ok, result.code) == (False, "unverified")


def test_named_colors_resolve_without_the_model_inventing_a_triple() -> None:
    assert resolve_color("blue") == [40, 90, 255]
    assert resolve_color("Warm  White") == [255, 190, 120]
    assert resolve_color([10, 20, 30]) == [10, 20, 30]


@pytest.mark.parametrize("value", ["puce", [1, 2], [300, 0, 0], 5])
def test_an_unresolvable_colour_is_an_error_not_a_guess(value) -> None:
    # Lighting the room the wrong colour is worse than saying so.
    with pytest.raises(ValueError):
        resolve_color(value)


def test_mode_states_are_read_from_the_state_snapshot() -> None:
    state = {"preferences": {"phonoscope": {"houseParty": {"enabled": True}}}}
    assert NovaProvider._mode_states(state) == {"house_party": True}
    assert NovaProvider._mode_states({}) == {"house_party": False}
