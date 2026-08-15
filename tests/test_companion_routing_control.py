"""The runtime rollback switches, and why they exist at the router.

Two reasons, and the second is the one that prompted this.

Rolling back used to mean editing the environment file and restarting the whole
voice stack — on a live household, to switch off a feature that is already
misbehaving.

And routing could not be *measured*. Comparing companion against local by
disconnecting the phone is not a control: the phone reconnects on its own,
silently, and everything after that is a mixture with no way to tell which
samples were which. I drew a conclusion from exactly that contaminated data
before building this.
"""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from nova_voice.api import create_app
from nova_voice.companion.router import CompanionWorkloadRouter
from nova_voice.companion.session import CompanionSessionManager
from nova_voice.config import Settings

pytestmark = pytest.mark.asyncio


def _client(router: CompanionWorkloadRouter | None, sessions=None):
    service = SimpleNamespace(companion_router=router, companion_sessions=sessions)
    app = create_app(Settings(), service=service)
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://voice.test"
    )


def _router(**kwargs) -> CompanionWorkloadRouter:
    return CompanionWorkloadRouter(CompanionSessionManager(), **kwargs)


async def test_force_local_takes_effect_without_a_restart() -> None:
    router = _router(enabled=True)

    async with _client(router) as client:
        response = await client.post("/v1/companion/routing", json={"forceLocal": True})

    assert response.status_code == 200
    assert response.json()["applied"] == {"forceLocal": True}
    assert router.force_local is True
    # The reason is the operator-facing part: a switch that changed nothing
    # visible would be indistinguishable from one that did not work.
    assert router.eligibility("classify_icon").reason == "force-local is on"


async def test_the_companion_can_be_switched_off_entirely() -> None:
    router = _router(enabled=True)

    async with _client(router) as client:
        await client.post("/v1/companion/routing", json={"enabled": False})

    assert router.enabled is False
    assert router.eligibility("classify_icon").reason == "companion feature is off"


async def test_status_reports_what_the_router_is_doing_not_what_the_file_says() -> None:
    """A status endpoint that disagrees with behaviour is worse than none.

    Settings default both switches off; this router was constructed on, and an
    override then turns force-local on. Neither value comes from configuration.
    """

    router = _router(enabled=True)
    sessions = router.sessions

    async with _client(router, sessions) as client:
        await client.post("/v1/companion/routing", json={"forceLocal": True})
        status = (await client.get("/v1/companion/status")).json()

    assert status["enabled"] is True
    assert status["forceLocal"] is True


async def test_a_non_boolean_is_refused_rather_than_coerced() -> None:
    """"false" is truthy, and a rollback switch that silently reads a string as
    "on" would fail in the direction that matters."""

    router = _router(enabled=True)

    async with _client(router) as client:
        response = await client.post("/v1/companion/routing", json={"forceLocal": "false"})

    assert response.status_code == 400
    assert router.force_local is False


async def test_a_workload_route_can_be_switched_at_runtime() -> None:
    """Hot-path passes ship local, so measuring them needs a switch.

    Without one, the only way to try the companion on a spoken turn is a
    redeploy — which is a bad way to run an experiment on a live household, and
    a worse way to end one that is going badly.
    """

    router = _router(enabled=True)
    assert router.route("render_response").mode == "local"

    async with _client(router) as client:
        response = await client.post(
            "/v1/companion/routing", json={"routes": {"render_response": "companion_preferred"}}
        )

    assert response.status_code == 200
    assert router.route("render_response").mode == "companion_preferred"


async def test_an_unknown_route_mode_is_refused() -> None:
    """A typo must not silently leave the route as it was.

    ``companion_prefered`` looks like it worked and changes nothing, which is
    the sort of thing that gets read as "the phone never answers".
    """

    router = _router(enabled=True)

    async with _client(router) as client:
        response = await client.post(
            "/v1/companion/routing", json={"routes": {"render_response": "companion_prefered"}}
        )

    assert response.status_code == 400
    assert router.route("render_response").mode == "local"


async def test_an_unknown_workload_is_refused() -> None:
    router = _router(enabled=True)

    async with _client(router) as client:
        response = await client.post(
            "/v1/companion/routing", json={"routes": {"transcribe": "local"}}
        )

    assert response.status_code == 400


async def test_an_empty_request_is_refused() -> None:
    router = _router(enabled=True)

    async with _client(router) as client:
        response = await client.post("/v1/companion/routing", json={})

    assert response.status_code == 400


async def test_the_control_is_unavailable_when_routing_is_not_configured() -> None:
    async with _client(None) as client:
        response = await client.post("/v1/companion/routing", json={"forceLocal": True})

    assert response.status_code == 503
