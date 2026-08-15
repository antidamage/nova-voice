"""The companion status surface (NPT-106).

Two properties matter: it explains *why* a workload is or is not being offered,
and it leaks nothing. A status endpoint is exactly the thing that ends up
proxied to a browser and written to logs, and a companion may be carrying
calendar and health context.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import httpx
import pytest
from conftest import issue_certificate
from cryptography.hazmat.primitives import serialization

from nova_voice.api import create_app
from nova_voice.companion.auth import AuthenticatedIdentity
from nova_voice.companion.protocol import (
    PROTOCOL_VERSION,
    CompanionHello,
    CompanionTelemetry,
    ModelAvailability,
)
from nova_voice.companion.router import CompanionWorkloadRouter
from nova_voice.companion.session import CompanionSessionManager


@pytest.fixture
def ca_path(tmp_path):
    _, certificate = issue_certificate("nova-household-ca", ca=True)
    path = tmp_path / "ca.crt"
    path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    return path


def _app(ca_path, sessions, router, **overrides):
    from nova_voice.config import Settings

    settings = Settings(companion_enabled=True, tls_ca_path=ca_path, **overrides)
    service = SimpleNamespace(companion_sessions=sessions, companion_router=router)
    return create_app(settings, service=service)


async def _get(app, path="/v1/companion/status") -> dict:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="https://voice.test") as client:
        response = await client.get(path)
        assert response.status_code == 200
        return response.json()


def _register(sessions, *, locality="home_lan"):
    return sessions.register(
        identity=AuthenticatedIdentity(
            identity="companion-1",
            roles=("companion",),
            not_after=datetime.now(UTC) + timedelta(days=1),
            fingerprint="abc123",
        ),
        hello=CompanionHello(
            protocolVersion=PROTOCOL_VERSION,
            displayName="Companion",
            roles=["companion"],
            appVersion="1.0.0",
            osVersion="26.0",
            workloads=["classify_icon", "interpret"],
            personalTools=["calendar.events"],
            telemetry=CompanionTelemetry(
                battery=0.9,
                charging=True,
                models=ModelAvailability(hotAvailable=True, hotContextTokens=4096),
            ),
        ),
        locality=locality,
        send=_noop_send,
    )


async def _noop_send(_payload: dict) -> None:
    return None


async def test_status_is_complete_when_nothing_is_connected():
    """"No companion" is the normal state, not an error."""

    sessions = CompanionSessionManager()
    router = CompanionWorkloadRouter(sessions, enabled=True)
    payload = await _get(_app(None, sessions, router))

    assert payload["connected"] is False
    assert payload["enabled"] is True
    assert "routes" in payload


async def test_status_explains_why_a_workload_is_not_offered(ca_path):
    sessions = CompanionSessionManager()
    router = CompanionWorkloadRouter(sessions, enabled=True)
    payload = await _get(_app(ca_path, sessions, router))

    assert payload["routes"]["classify_icon"]["eligibility"] == "no companion session"


async def test_status_reports_a_connected_companion(ca_path):
    sessions = CompanionSessionManager()
    router = CompanionWorkloadRouter(sessions, enabled=True)
    _register(sessions)
    payload = await _get(_app(ca_path, sessions, router))

    assert payload["connected"] is True
    assert payload["identity"] == "companion-1"
    assert payload["locality"] == "home_lan"
    assert payload["tier"] == "full"
    assert payload["workloads"] == ["classify_icon", "interpret"]
    assert payload["routes"]["classify_icon"]["eligibility"] == "eligible"


async def test_status_shows_a_tailnet_session_blocked_from_home_lan_routes(ca_path):
    sessions = CompanionSessionManager()
    router = CompanionWorkloadRouter(sessions, enabled=True)
    _register(sessions, locality="tailnet")
    payload = await _get(_app(ca_path, sessions, router))

    assert payload["locality"] == "tailnet"
    assert "route needs home_lan" in payload["routes"]["classify_icon"]["eligibility"]


async def test_force_local_is_visible_as_the_reason(ca_path):
    sessions = CompanionSessionManager()
    router = CompanionWorkloadRouter(sessions, enabled=True, force_local=True)
    _register(sessions)
    payload = await _get(_app(ca_path, sessions, router, companion_force_local=True))

    assert payload["forceLocal"] is True
    # Asked of a workload that would otherwise be offered: `interpret` ships
    # routed local, so it would report "route is local" whether or not the
    # global switch were pulled, and prove nothing about it.
    assert payload["routes"]["classify_icon"]["eligibility"] == "force-local is on"


async def test_status_carries_no_secret_or_personal_material(ca_path):
    """Certificates, prompts and payloads must not reach this surface."""

    sessions = CompanionSessionManager()
    router = CompanionWorkloadRouter(sessions, enabled=True)
    _register(sessions)
    body = json.dumps(await _get(_app(ca_path, sessions, router))).lower()

    for forbidden in (
        "begin certificate",
        "private key",
        "fingerprint",
        "abc123",
        "signature",
        "nonce",
        "transcript",
        "payload",
    ):
        assert forbidden not in body, f"status leaked {forbidden!r}"


async def test_status_does_not_name_the_deployment(ca_path):
    sessions = CompanionSessionManager()
    router = CompanionWorkloadRouter(sessions, enabled=True)
    payload = await _get(_app(ca_path, sessions, router))

    assert "neptunium" not in json.dumps(payload).lower()
