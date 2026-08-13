"""The companion channel end to end, driven by the headless reference peer.

This is what makes the whole companion layer testable without Xcode or a
physical phone: the reference peer speaks the real protocol over a real
WebSocket into the real endpoint, so authentication, registration, supersede,
disconnect and locality are exercised rather than asserted about.

The peer is driven synchronously here on purpose. Starlette's ``TestClient``
runs the application in its own portal thread, and each ``send_text`` /
``receive_text`` blocks this thread while the server side advances — so a
coroutine whose awaits bottom out in those blocking calls can be run with
``asyncio.run`` without the two loops fighting.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from conftest import issue_certificate
from cryptography.hazmat.primitives import serialization
from fastapi.testclient import TestClient

from nova_voice.api import create_app
from nova_voice.companion.reference import ReferenceCompanion
from nova_voice.companion.session import CompanionSessionManager
from nova_voice.companion.tiers import CompanionTier
from nova_voice.config import Settings


@pytest.fixture
def household_ca(tmp_path):
    key, certificate = issue_certificate("nova-household-ca", ca=True)
    path = tmp_path / "ca.crt"
    path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    return key, certificate, path


@pytest.fixture
def sessions():
    return CompanionSessionManager()


def _app(ca_path, sessions, **overrides):
    settings = Settings(companion_enabled=True, tls_ca_path=ca_path, **overrides)
    service = SimpleNamespace(companion_sessions=sessions, companion_router=None)
    return create_app(settings, service=service)


def _identity(household_ca, name="companion-1"):
    ca_key, ca_certificate, _ = household_ca
    key, certificate = issue_certificate(
        name, issuer_key=ca_key, issuer_name=ca_certificate.subject
    )
    return key, certificate.public_bytes(serialization.Encoding.PEM).decode()


def _peer(websocket, key, certificate_pem, *, announced_id="companion-1", **overrides):
    async def send(text: str) -> None:
        websocket.send_text(text)

    async def receive() -> str:
        return websocket.receive_text()

    return ReferenceCompanion(
        announced_id=announced_id,
        private_key=key,
        certificate_pem=certificate_pem,
        send=send,
        receive=receive,
        **overrides,
    )


def test_an_authenticated_peer_registers_a_session(household_ca, sessions):
    _, _, ca_path = household_ca
    key, certificate_pem = _identity(household_ca)

    with TestClient(_app(ca_path, sessions)).websocket_connect("/v1/companion") as websocket:
        peer = _peer(websocket, key, certificate_pem)
        ack = asyncio.run(peer.authenticate())

        assert ack["type"] == "hello_ack"
        assert ack["authenticatedId"] == "companion-1"

        snapshot = sessions.snapshot()
        assert snapshot.connected is True
        assert snapshot.identity == "companion-1"
        assert "companion" in snapshot.roles
        # The hello's telemetry establishes the tier immediately; a freshly
        # connected healthy device must not sit out a dwell window.
        assert snapshot.tier is CompanionTier.FULL


def test_a_certificate_may_not_announce_another_identity(household_ca, sessions):
    """A cert issued for indium must not register as the companion."""

    _, _, ca_path = household_ca
    key, certificate_pem = _identity(household_ca, name="indium")

    with TestClient(_app(ca_path, sessions)).websocket_connect("/v1/companion") as websocket:
        peer = _peer(websocket, key, certificate_pem, announced_id="companion-1")
        with pytest.raises(Exception):
            asyncio.run(peer.authenticate())

    assert sessions.snapshot().connected is False


def test_a_foreign_certificate_is_refused(household_ca, sessions, tmp_path):
    _, _, ca_path = household_ca
    other_ca_key, other_ca = issue_certificate("other-ca", ca=True)
    key, certificate = issue_certificate(
        "companion-1", issuer_key=other_ca_key, issuer_name=other_ca.subject
    )
    pem = certificate.public_bytes(serialization.Encoding.PEM).decode()

    with TestClient(_app(ca_path, sessions)).websocket_connect("/v1/companion") as websocket:
        peer = _peer(websocket, key, pem)
        with pytest.raises(Exception):
            asyncio.run(peer.authenticate())

    assert sessions.snapshot().connected is False


def test_a_newer_session_supersedes_the_older_one(household_ca, sessions):
    """A phone that just changed network is likelier live than a stale socket."""

    _, _, ca_path = household_ca
    key, certificate_pem = _identity(household_ca)
    client = TestClient(_app(ca_path, sessions))

    with client.websocket_connect("/v1/companion") as first:
        asyncio.run(_peer(first, key, certificate_pem).authenticate())
        first_session = sessions.snapshot().session_id

        with client.websocket_connect("/v1/companion") as second:
            asyncio.run(_peer(second, key, certificate_pem).authenticate())
            second_session = sessions.snapshot().session_id

            assert second_session != first_session
            assert sessions.snapshot().connected is True


def test_disconnect_leaves_no_stale_session(household_ca, sessions):
    _, _, ca_path = household_ca
    key, certificate_pem = _identity(household_ca)

    with TestClient(_app(ca_path, sessions)).websocket_connect("/v1/companion") as websocket:
        asyncio.run(_peer(websocket, key, certificate_pem).authenticate())
        assert sessions.snapshot().connected is True

    assert sessions.snapshot().connected is False
    assert sessions.snapshot().identity is None


def test_a_malformed_frame_closes_the_socket(household_ca, sessions):
    """Garbage never reaches a model or a provider; the edge rejects it."""

    _, _, ca_path = household_ca
    key, certificate_pem = _identity(household_ca)

    with TestClient(_app(ca_path, sessions)).websocket_connect("/v1/companion") as websocket:
        asyncio.run(_peer(websocket, key, certificate_pem).authenticate())
        websocket.send_text(json.dumps({"type": "definitely_not_a_message"}))
        with pytest.raises(Exception):
            websocket.receive_text()

    assert sessions.snapshot().connected is False


def test_the_channel_is_closed_when_the_feature_is_off(household_ca, sessions):
    """companion_enabled=false must make the socket refuse to open at all."""

    _, _, ca_path = household_ca
    settings = Settings(companion_enabled=False, tls_ca_path=ca_path)
    service = SimpleNamespace(companion_sessions=sessions, companion_router=None)
    client = TestClient(create_app(settings, service=service))

    with pytest.raises(Exception):
        with client.websocket_connect("/v1/companion"):
            pass

    assert sessions.snapshot().connected is False


def test_a_device_claiming_to_be_home_does_not_become_home(household_ca, sessions):
    """Locality is the server's judgement of the peer address, not a claim.

    The peer here reports a home network in its telemetry. With no home subnet
    configured, the session must still be classified away — otherwise a
    compromised or buggy client could unlock the home-LAN-only routes just by
    saying so.
    """

    _, _, ca_path = household_ca
    key, certificate_pem = _identity(household_ca)

    with TestClient(_app(ca_path, sessions)).websocket_connect("/v1/companion") as websocket:
        peer = _peer(websocket, key, certificate_pem)
        peer.telemetry = peer.telemetry.model_copy(update={"reported_network": "home-wifi"})
        ack = asyncio.run(peer.authenticate())

    assert ack["locality"] != "home_lan"
