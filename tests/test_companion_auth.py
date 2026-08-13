"""Identity binding on the companion channel.

The property under test is the one that did not exist before: holding a
household certificate is not enough — the connection must prove it holds the
key for the identity it announces.
"""

from __future__ import annotations

import base64
from datetime import UTC, datetime, timedelta

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from nova_voice.companion.auth import (
    AuthenticationError,
    CompanionAuthenticator,
    challenge_material,
)
from nova_voice.companion.reference import sign_challenge


def _issue(
    common_name: str,
    *,
    issuer_key=None,
    issuer_name: x509.Name | None = None,
    not_before: datetime | None = None,
    not_after: datetime | None = None,
    client_auth: bool = True,
    ca: bool = False,
):
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
    )
    if client_auth:
        builder = builder.add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CLIENT_AUTH]), critical=False
        )
    builder = builder.add_extension(
        x509.SubjectAlternativeName([x509.DNSName(common_name)]), critical=False
    )
    certificate = builder.sign(issuer_key or key, hashes.SHA256())
    return key, certificate


@pytest.fixture
def household_ca():
    return _issue("nova-household-ca", ca=True)


@pytest.fixture
def authenticator(household_ca):
    _, ca_certificate = household_ca
    return CompanionAuthenticator(
        ca_certificate.public_bytes(serialization.Encoding.PEM)
    )


def _authenticate(authenticator, key, certificate, *, announced_id, roles=("companion",)):
    nonce = authenticator.issue_nonce()
    material = challenge_material(
        nonce=nonce, protocol_version=1, announced_id=announced_id, roles=list(roles)
    )
    return authenticator.authenticate(
        nonce=nonce,
        protocol_version=1,
        announced_id=announced_id,
        roles=list(roles),
        certificate_chain=[certificate.public_bytes(serialization.Encoding.PEM).decode()],
        signature=sign_challenge(key, material),
    )


def test_valid_identity_authenticates(authenticator, household_ca):
    ca_key, ca_certificate = household_ca
    key, certificate = _issue(
        "companion-1", issuer_key=ca_key, issuer_name=ca_certificate.subject
    )
    identity = _authenticate(authenticator, key, certificate, announced_id="companion-1")
    assert identity.identity == "companion-1"
    assert identity.roles == ("companion",)


def test_certificate_may_not_announce_another_identity(authenticator, household_ca):
    """A cert issued for indium must not be able to claim to be companion-1."""

    ca_key, ca_certificate = household_ca
    key, certificate = _issue("indium", issuer_key=ca_key, issuer_name=ca_certificate.subject)
    with pytest.raises(AuthenticationError, match="may not announce"):
        _authenticate(authenticator, key, certificate, announced_id="companion-1")


def test_foreign_ca_is_rejected(authenticator):
    other_ca_key, other_ca = _issue("other-ca", ca=True)
    key, certificate = _issue(
        "companion-1", issuer_key=other_ca_key, issuer_name=other_ca.subject
    )
    with pytest.raises(AuthenticationError):
        _authenticate(authenticator, key, certificate, announced_id="companion-1")


def test_expired_certificate_is_rejected(authenticator, household_ca):
    ca_key, ca_certificate = household_ca
    past = datetime.now(UTC) - timedelta(days=2)
    key, certificate = _issue(
        "companion-1",
        issuer_key=ca_key,
        issuer_name=ca_certificate.subject,
        not_before=past - timedelta(days=1),
        not_after=past,
    )
    with pytest.raises(AuthenticationError, match="expired"):
        _authenticate(authenticator, key, certificate, announced_id="companion-1")


def test_certificate_without_client_auth_is_rejected(authenticator, household_ca):
    ca_key, ca_certificate = household_ca
    key, certificate = _issue(
        "companion-1",
        issuer_key=ca_key,
        issuer_name=ca_certificate.subject,
        client_auth=False,
    )
    with pytest.raises(AuthenticationError, match="extended key usage"):
        _authenticate(authenticator, key, certificate, announced_id="companion-1")


def test_signature_over_other_material_is_rejected(authenticator, household_ca):
    ca_key, ca_certificate = household_ca
    key, certificate = _issue(
        "companion-1", issuer_key=ca_key, issuer_name=ca_certificate.subject
    )
    nonce = authenticator.issue_nonce()
    # Signed with a different role list than the one announced.
    wrong = challenge_material(
        nonce=nonce, protocol_version=1, announced_id="companion-1", roles=["satellite"]
    )
    with pytest.raises(AuthenticationError, match="signature"):
        authenticator.authenticate(
            nonce=nonce,
            protocol_version=1,
            announced_id="companion-1",
            roles=["companion"],
            certificate_chain=[
                certificate.public_bytes(serialization.Encoding.PEM).decode()
            ],
            signature=sign_challenge(key, wrong),
        )


def test_nonce_cannot_be_replayed(authenticator, household_ca):
    ca_key, ca_certificate = household_ca
    key, certificate = _issue(
        "companion-1", issuer_key=ca_key, issuer_name=ca_certificate.subject
    )
    nonce = authenticator.issue_nonce()
    material = challenge_material(
        nonce=nonce, protocol_version=1, announced_id="companion-1", roles=["companion"]
    )
    chain = [certificate.public_bytes(serialization.Encoding.PEM).decode()]
    signature = sign_challenge(key, material)
    kwargs = dict(
        nonce=nonce,
        protocol_version=1,
        announced_id="companion-1",
        roles=["companion"],
        certificate_chain=chain,
        signature=signature,
    )
    assert authenticator.authenticate(**kwargs).identity == "companion-1"
    with pytest.raises(AuthenticationError, match="nonce"):
        authenticator.authenticate(**kwargs)


def test_unknown_nonce_is_rejected(authenticator, household_ca):
    ca_key, ca_certificate = household_ca
    key, certificate = _issue(
        "companion-1", issuer_key=ca_key, issuer_name=ca_certificate.subject
    )
    material = challenge_material(
        nonce="never-issued", protocol_version=1, announced_id="companion-1", roles=["companion"]
    )
    with pytest.raises(AuthenticationError, match="nonce"):
        authenticator.authenticate(
            nonce="never-issued",
            protocol_version=1,
            announced_id="companion-1",
            roles=["companion"],
            certificate_chain=[
                certificate.public_bytes(serialization.Encoding.PEM).decode()
            ],
            signature=sign_challenge(key, material),
        )


def test_identity_allowlist_restricts_further(household_ca):
    ca_key, ca_certificate = household_ca
    authenticator = CompanionAuthenticator(
        ca_certificate.public_bytes(serialization.Encoding.PEM),
        allowed_identities=frozenset({"companion-2"}),
    )
    key, certificate = _issue(
        "companion-1", issuer_key=ca_key, issuer_name=ca_certificate.subject
    )
    with pytest.raises(AuthenticationError, match="not permitted"):
        _authenticate(authenticator, key, certificate, announced_id="companion-1")


def test_malformed_signature_is_rejected(authenticator, household_ca):
    ca_key, ca_certificate = household_ca
    _, certificate = _issue(
        "companion-1", issuer_key=ca_key, issuer_name=ca_certificate.subject
    )
    nonce = authenticator.issue_nonce()
    with pytest.raises(AuthenticationError):
        authenticator.authenticate(
            nonce=nonce,
            protocol_version=1,
            announced_id="companion-1",
            roles=["companion"],
            certificate_chain=[
                certificate.public_bytes(serialization.Encoding.PEM).decode()
            ],
            signature=base64.b64encode(b"not a signature").decode(),
        )
