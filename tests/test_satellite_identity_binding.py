"""A household certificate must not be able to impersonate another satellite.

The mTLS listener only proves a peer holds *a* certificate the household CA
signed. Until this binding, any of them could announce any ``satelliteId`` —
which mattered little while every holder was a fixed machine, and matters a
great deal once one of them is a phone that leaves the house.

The exchange is opt-in by the client so the deployed fleet keeps working
through the rollout; these tests pin both sides of that switch.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from conftest import issue_certificate
from cryptography.hazmat.primitives import serialization
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from nova_voice.api import create_app
from nova_voice.companion.auth import challenge_material
from nova_voice.companion.protocol import PROTOCOL_VERSION, AuthResponse, serialize
from nova_voice.companion.reference import sign_challenge
from nova_voice.config import Settings


@pytest.fixture
def household_ca(tmp_path):
    key, certificate = issue_certificate("nova-household-ca", ca=True)
    path = tmp_path / "ca.crt"
    path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    return key, certificate, path


def _app(ca_path, **overrides):
    settings = Settings(tls_ca_path=ca_path, **overrides)
    service = SimpleNamespace(
        companion_sessions=None, companion_router=None, voice_settings=None
    )
    # The satellite socket refuses outright without an audio runtime; identity
    # binding happens before any of it is touched.
    audio = SimpleNamespace(set_monitor_sink=lambda _sink: None)
    return create_app(settings, service=service, audio_runtime=audio)


def _hello(satellite_id="nocturnium", *, identity_proof=False, client="linux-native"):
    return {
        "protocolVersion": 1,
        "satelliteId": satellite_id,
        "displayName": "Test Satellite",
        "roomId": "lounge",
        "client": client,
        "supervisor": "none" if client == "browser" else "systemd",
        "capturePolicy": "always",
        "capabilities": {
            "microphone": True,
            "speaker": True,
            "identityProof": identity_proof,
        },
    }


def _issue_for(household_ca, name):
    ca_key, ca_certificate, _ = household_ca
    key, certificate = issue_certificate(
        name, issuer_key=ca_key, issuer_name=ca_certificate.subject
    )
    return key, certificate.public_bytes(serialization.Encoding.PEM).decode()


def _sign(key, pem, *, announced_id, nonce):
    return serialize(
        AuthResponse(
            protocolVersion=PROTOCOL_VERSION,
            announcedId=announced_id,
            roles=["satellite"],
            certificateChain=[pem],
            signature=sign_challenge(
                key,
                challenge_material(
                    nonce=nonce,
                    protocol_version=PROTOCOL_VERSION,
                    announced_id=announced_id,
                    roles=["satellite"],
                ),
            ),
        )
    )


def test_a_legacy_satellite_is_admitted_while_the_switch_is_on(household_ca):
    """The deployed fleet keeps working through the rollout."""

    _, _, ca_path = household_ca
    app = _app(ca_path, companion_allow_unbound_satellites=True)

    with TestClient(app).websocket_connect("/v1/satellites") as websocket:
        websocket.send_text(json.dumps(_hello()))
        ack = json.loads(websocket.receive_text())

    assert ack["type"] == "hello"
    assert ack["satelliteId"] == "nocturnium"


def test_a_legacy_satellite_is_refused_once_the_switch_is_off(household_ca):
    """Ending the migration window is a configuration change, not a code one."""

    _, _, ca_path = household_ca
    app = _app(ca_path, companion_allow_unbound_satellites=False)

    with pytest.raises((WebSocketDisconnect, RuntimeError)):
        with TestClient(app).websocket_connect("/v1/satellites") as websocket:
            websocket.send_text(json.dumps(_hello()))
            websocket.receive_text()


def test_a_satellite_proving_its_own_identity_is_admitted(household_ca):
    _, _, ca_path = household_ca
    key, pem = _issue_for(household_ca, "nocturnium")
    app = _app(ca_path, companion_allow_unbound_satellites=False)

    with TestClient(app).websocket_connect("/v1/satellites") as websocket:
        websocket.send_text(json.dumps(_hello(identity_proof=True)))
        challenge = json.loads(websocket.receive_text())
        assert challenge["type"] == "auth_challenge"
        websocket.send_text(
            json.dumps(_sign(key, pem, announced_id="nocturnium", nonce=challenge["nonce"]))
        )
        ack = json.loads(websocket.receive_text())

    assert ack["type"] == "hello"


def test_a_certificate_cannot_impersonate_another_satellite(household_ca):
    """The whole point: indium's certificate may not announce nocturnium."""

    _, _, ca_path = household_ca
    key, pem = _issue_for(household_ca, "indium")
    app = _app(ca_path, companion_allow_unbound_satellites=False)

    with pytest.raises((WebSocketDisconnect, RuntimeError)):
        with TestClient(app).websocket_connect("/v1/satellites") as websocket:
            websocket.send_text(json.dumps(_hello("nocturnium", identity_proof=True)))
            challenge = json.loads(websocket.receive_text())
            # Signed honestly, for the identity this certificate really is.
            websocket.send_text(
                json.dumps(_sign(key, pem, announced_id="indium", nonce=challenge["nonce"]))
            )
            websocket.receive_text()


def test_signing_the_announced_name_with_the_wrong_certificate_fails(household_ca):
    """Claiming the right name with the wrong key is the obvious attack."""

    _, _, ca_path = household_ca
    key, pem = _issue_for(household_ca, "indium")
    app = _app(ca_path, companion_allow_unbound_satellites=False)

    with pytest.raises((WebSocketDisconnect, RuntimeError)):
        with TestClient(app).websocket_connect("/v1/satellites") as websocket:
            websocket.send_text(json.dumps(_hello("nocturnium", identity_proof=True)))
            challenge = json.loads(websocket.receive_text())
            websocket.send_text(
                json.dumps(
                    _sign(key, pem, announced_id="nocturnium", nonce=challenge["nonce"])
                )
            )
            websocket.receive_text()


def test_a_browser_satellite_is_exempt(household_ca):
    """A browser cannot hold a certificate; the dashboard relays it instead."""

    _, _, ca_path = household_ca
    app = _app(ca_path, companion_allow_unbound_satellites=False)

    with TestClient(app).websocket_connect("/v1/satellites") as websocket:
        websocket.send_text(json.dumps(_hello("web-ipad", client="browser")))
        ack = json.loads(websocket.receive_text())

    assert ack["type"] == "hello"


def test_a_satellite_that_promises_proof_and_sends_none_is_refused(household_ca):
    _, _, ca_path = household_ca
    app = _app(ca_path, companion_allow_unbound_satellites=True)

    with pytest.raises((WebSocketDisconnect, RuntimeError)):
        with TestClient(app).websocket_connect("/v1/satellites") as websocket:
            websocket.send_text(json.dumps(_hello(identity_proof=True)))
            json.loads(websocket.receive_text())
            websocket.send_text(json.dumps({"type": "heartbeat", "sentAt": "2026-08-13T00:00:00Z"}))
            websocket.receive_text()

