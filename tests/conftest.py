from __future__ import annotations

from datetime import UTC, datetime

import pytest

from nova_voice.domain import (
    ActiveGoal,
    Decision,
    Emotion,
    EmotionLabel,
    GoalStatus,
    Interpretation,
    ResponsePlan,
    SpeechAct,
    Utterance,
)


@pytest.fixture
def utterance() -> Utterance:
    now = datetime.now(UTC)
    return Utterance(
        id="utterance-1",
        satellite_id="test",
        room_id="lounge",
        started_at=now,
        ended_at=now,
        transcript="Turn the lounge light on",
        wake_detected=False,
    )


def interpretation(
    *,
    speech_act: SpeechAct = SpeechAct.DIRECTIVE,
    decision: Decision = Decision.IGNORE,
    addressed: float = 0.99,
    confidence: float = 0.95,
    actions: list | None = None,
) -> Interpretation:
    return Interpretation(
        emotion=Emotion(label=EmotionLabel.NEUTRAL, confidence=0.8, intensity=0.2),
        speech_act=speech_act,
        addressed_probability=addressed,
        decision=decision,
        confidence=confidence,
        active_goal=ActiveGoal(summary="test", status=GoalStatus.NEW),
        actions=actions or [],
        response_plan=ResponsePlan(),
    )


def issue_certificate(
    common_name: str,
    *,
    issuer_key=None,
    issuer_name=None,
    not_before=None,
    not_after=None,
    client_auth: bool = True,
    ca: bool = False,
):
    """Mint a household-style client identity for companion auth tests.

    Mirrors what ``ops/issue-satellite-identity.sh`` produces — an EC P-256 key,
    ``clientAuth`` extended key usage, and the identity in both the common name
    and a DNS subject alternative name — so tests exercise the same shape the
    real issuance path emits.
    """

    from datetime import UTC, datetime, timedelta

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])
    now = datetime.now(UTC)
    builder = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer_name or subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(not_before or now - timedelta(days=1))
        .not_valid_after(not_after or now + timedelta(days=365))
        .add_extension(x509.BasicConstraints(ca=ca, path_length=None), critical=True)
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName(common_name)]), critical=False
        )
    )
    if client_auth:
        builder = builder.add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CLIENT_AUTH]), critical=False
        )
    return key, builder.sign(issuer_key or key, hashes.SHA256())
