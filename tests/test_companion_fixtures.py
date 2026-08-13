"""The frozen protocol fixtures are the contract; these tests hold them to it.

The fixtures are static committed files, not generated at test time, which is
what makes these tests meaningful: an incompatible change to a model breaks a
fixture rather than silently rewriting it. The same files are decoded by the
Swift package (NPT-905), so both sides agree without a shared code generator.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nova_voice.companion.protocol import (
    MAX_FRAME_BYTES,
    ClientMessage,
    ServerMessage,
    parse_client_message,
    serialize,
)
from nova_voice.companion.sensitivity import MESSAGE_SENSITIVITY

FIXTURES = Path(__file__).resolve().parent.parent / "docs" / "companion-protocol"
# Cross-language signing vectors, not a protocol message. It lives alongside
# the message fixtures because it is part of the same frozen contract, but it
# has no `type` and must not be validated as a frame.
CHALLENGE_VECTORS = FIXTURES / "challenge-material.json"
VALID = sorted(path for path in FIXTURES.glob("*.json") if path != CHALLENGE_VECTORS)
INVALID = sorted((FIXTURES / "invalid").glob("*.json"))


def _message_types(union) -> set[str]:
    types = set()
    for member in union.__args__[0].__args__:
        types.update(member.model_fields["type"].annotation.__args__)
    return types


CLIENT_TYPES = _message_types(ClientMessage)
SERVER_TYPES = _message_types(ServerMessage)


def test_fixtures_exist():
    assert VALID, "no protocol fixtures found; run ops/generate_companion_fixtures.py"
    assert INVALID


def test_every_message_type_has_a_fixture():
    """NPT-006: every message, not just the interesting ones."""

    documented = {path.stem for path in VALID}
    assert CLIENT_TYPES | SERVER_TYPES == documented


@pytest.mark.parametrize("path", VALID, ids=lambda path: path.stem)
def test_fixture_declares_its_own_type(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["type"] == path.stem


@pytest.mark.parametrize("path", VALID, ids=lambda path: path.stem)
def test_client_fixtures_round_trip_through_the_strict_models(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["type"] not in CLIENT_TYPES:
        pytest.skip("server-to-client message")
    message = parse_client_message(path.read_text(encoding="utf-8"))
    # Re-serialising must reproduce the file exactly, or the fixture has
    # drifted from the implementation.
    assert serialize(message) == payload


@pytest.mark.parametrize("path", VALID, ids=lambda path: path.stem)
def test_fixture_is_within_the_frame_cap(path):
    assert len(path.read_bytes()) <= MAX_FRAME_BYTES


@pytest.mark.parametrize("path", VALID, ids=lambda path: path.stem)
def test_fixture_type_is_classified_for_redaction(path):
    assert path.stem in MESSAGE_SENSITIVITY


@pytest.mark.parametrize("path", INVALID, ids=lambda path: path.stem)
def test_invalid_fixtures_are_rejected(path):
    """Strictness is part of the contract, so its failures are fixtures too."""

    with pytest.raises(ValueError):
        parse_client_message(path.read_text(encoding="utf-8"))


def test_oversized_frame_is_rejected_before_parsing():
    payload = json.dumps(
        {"type": "heartbeat", "sentAt": "2026-08-13T09:00:00Z", "pad": "x" * MAX_FRAME_BYTES}
    )
    with pytest.raises(ValueError, match="exceeds"):
        parse_client_message(payload)


def test_no_real_deployment_identity_leaks_into_fixtures():
    """Tracked configuration carries placeholders only.

    The fixtures are documentation a new deployment reads, so a real hostname,
    tailnet or SSID appearing here would be both a leak and misleading.
    """

    forbidden = ("neptunium", "iridium", "nocturnium", "indium", "tuatara-dory", "192.168.")
    for path in [*VALID, *INVALID]:
        content = path.read_text(encoding="utf-8").lower()
        for token in forbidden:
            assert token not in content, f"{path.name} contains {token!r}"


def test_timestamps_are_utc_with_an_explicit_offset():
    for path in VALID:
        for line in path.read_text(encoding="utf-8").splitlines():
            if "At\":" in line or "Deadline\":" in line:
                assert "Z" in line or "+00:00" in line, f"{path.name}: {line.strip()}"


def test_challenge_vectors_match_the_current_implementation():
    """The Swift client signs these exact bytes (NovaCompanionKit).

    Regenerating the fixtures is a deliberate act; if `challenge_material`
    changes without it, the committed vectors stop matching and this fails —
    rather than the handshake failing later with a signature that simply never
    verifies.
    """

    from nova_voice.companion.auth import challenge_material

    vectors = json.loads(CHALLENGE_VECTORS.read_text(encoding="utf-8"))
    assert vectors, "no challenge vectors committed"
    for vector in vectors:
        produced = challenge_material(
            nonce=vector["nonce"],
            protocol_version=vector["protocolVersion"],
            announced_id=vector["announcedId"],
            roles=list(vector["roles"]),
        ).decode("utf-8")
        assert produced == vector["material"]


def test_challenge_vectors_cover_normalisation():
    """Role order, role case and surrounding whitespace must not matter."""

    vectors = json.loads(CHALLENGE_VECTORS.read_text(encoding="utf-8"))
    assert any(
        role != role.lower() for vector in vectors for role in vector["roles"]
    ), "no vector exercises role case normalisation"
    assert any(
        vector["roles"] != sorted(role.lower() for role in vector["roles"])
        for vector in vectors
    ), "no vector exercises role ordering"
    assert any(
        vector["announcedId"] != vector["announcedId"].strip() for vector in vectors
    ), "no vector exercises whitespace trimming"
