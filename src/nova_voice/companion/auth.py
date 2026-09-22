"""Bind an announced client identity to the certificate it actually holds.

Until now the mTLS listener proved only that a peer held *some* certificate
signed by the household CA; nothing checked that the id in its hello was the id
that certificate was issued for. Any household cert could claim to be any
device. That was survivable while every holder was a fixed machine on the home
LAN. A companion is the first mobile holder and the first to present a
household identity from outside the LAN, so the binding is enforced.

The binding is done at the application layer rather than from the TLS peer
certificate, because uvicorn does not publish the peer certificate into the
ASGI scope and there is no supported way to reach it from the application. So
the client presents its chain and signs a server-issued nonce:

1. the server sends a fresh single-use nonce and the protocol range it accepts;
2. the client returns its certificate chain plus a signature over the nonce,
   protocol version, announced id and announced roles — all four together, so
   none can be swapped after signing;
3. the server validates the chain to the household CA, checks validity dates
   and client-auth extended key usage, verifies the signature against the leaf
   public key, and maps SAN/CN to the announced id;
4. only then may the connection register an identity.

Nonces are single-use and short-lived, so a captured signature cannot be
replayed against a later connection.

This runs *in addition to* transport TLS, not instead of it. The listener still
requires a client certificate; this proves the connection's owner is the device
whose name it is using.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import secrets
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, padding, rsa
from cryptography.hazmat.primitives.hashes import SHA256
from cryptography.x509.oid import ExtendedKeyUsageOID

logger = logging.getLogger(__name__)

NONCE_BYTES = 32
DEFAULT_NONCE_TTL_SECONDS = 30.0


class AuthenticationError(Exception):
    """The peer may not register the identity it announced."""


@dataclass(frozen=True)
class AuthenticatedIdentity:
    identity: str
    roles: tuple[str, ...]
    not_after: datetime
    # Fingerprint of the leaf, so a revoked or rotated certificate is
    # distinguishable in audit records without storing the certificate itself.
    fingerprint: str
    # The verified leaf, kept because approvals are signed *later* on an
    # already-authenticated session and there is nothing else to check them
    # against. Optional so the many places that construct an identity for a
    # test do not have to mint a certificate they never use.
    certificate: x509.Certificate | None = None


def challenge_material(
    *, nonce: str, protocol_version: int, announced_id: str, roles: list[str]
) -> bytes:
    """The exact bytes both sides sign, in a canonical form.

    Roles are sorted so the two sides cannot disagree about ordering, and every
    field is length-delimited by the separator so no two different inputs can
    produce the same signed string.
    """

    joined = "|".join(
        [
            "nova-companion-auth-v1",
            nonce,
            str(protocol_version),
            announced_id.strip().lower(),
            ",".join(sorted(role.strip().lower() for role in roles)),
        ]
    )
    return joined.encode("utf-8")


def approval_material(*, approval_id: str, approved: bool, nonce: str) -> bytes:
    """The bytes an approval decision is signed over.

    Built like ``challenge_material`` and for the same reasons: canonical,
    length-delimited by the separator, and prefixed with its own domain string
    so a signature collected for one purpose can never be presented as the
    other.

    The nonce is what makes a *decision* single-use rather than merely
    authentic. Without it, "yes" to one approval is a valid "yes" forever, and
    anyone who captured it could replay it against a later approval carrying
    the same id — which matters because the phone is asked for the same shapes
    of permission repeatedly.
    """

    joined = "|".join(
        [
            "nova-companion-approval-v1",
            approval_id,
            "approved" if approved else "denied",
            nonce,
        ]
    )
    return joined.encode("utf-8")


def verify_approval(
    identity: AuthenticatedIdentity,
    *,
    approval_id: str,
    approved: bool,
    nonce: str,
    signature: str,
) -> None:
    """Check a decision really came from the device that was asked.

    Raises ``AuthenticationError`` rather than returning a boolean, so a
    caller that forgets to check the result cannot silently execute a mutation
    nobody approved.
    """

    if identity.certificate is None:
        raise AuthenticationError("this session has no certificate to check an approval against")
    try:
        raw = base64.b64decode(signature, validate=True)
    except (ValueError, TypeError) as error:
        raise AuthenticationError("approval signature was not valid base64") from error
    _verify_signature(
        identity.certificate,
        raw,
        approval_material(approval_id=approval_id, approved=approved, nonce=nonce),
    )


@dataclass
class NonceIssuer:
    """Issues single-use nonces and retires them on first use or expiry."""

    ttl_seconds: float = DEFAULT_NONCE_TTL_SECONDS
    _outstanding: dict[str, float] = field(default_factory=dict)

    def issue(self, *, now: float | None = None) -> str:
        current = time.monotonic() if now is None else now
        self._expire(current)
        nonce = secrets.token_urlsafe(NONCE_BYTES)
        self._outstanding[nonce] = current + self.ttl_seconds
        return nonce

    def consume(self, nonce: str, *, now: float | None = None) -> bool:
        """Spend a nonce. False when it is unknown, expired or already used."""

        current = time.monotonic() if now is None else now
        self._expire(current)
        expiry = self._outstanding.pop(nonce, None)
        return expiry is not None and expiry >= current

    def _expire(self, now: float) -> None:
        for nonce, expiry in list(self._outstanding.items()):
            if expiry < now:
                del self._outstanding[nonce]


def _identity_names(certificate: x509.Certificate) -> set[str]:
    """Subject common name plus every DNS subject alternative name."""

    names: set[str] = set()
    for attribute in certificate.subject.get_attributes_for_oid(x509.oid.NameOID.COMMON_NAME):
        value = attribute.value
        if isinstance(value, bytes):
            value = value.decode("utf-8", "replace")
        if value:
            names.add(value.strip().lower())
    try:
        san = certificate.extensions.get_extension_for_class(x509.SubjectAlternativeName)
    except x509.ExtensionNotFound:
        return names
    for dns_name in san.value.get_values_for_type(x509.DNSName):
        if dns_name:
            names.add(dns_name.strip().lower())
    return names


def _verify_signature(certificate: x509.Certificate, signature: bytes, material: bytes) -> None:
    public_key = certificate.public_key()
    try:
        if isinstance(public_key, ec.EllipticCurvePublicKey):
            public_key.verify(signature, material, ec.ECDSA(SHA256()))
        elif isinstance(public_key, rsa.RSAPublicKey):
            public_key.verify(signature, material, padding.PKCS1v15(), SHA256())
        elif isinstance(public_key, ed25519.Ed25519PublicKey):
            public_key.verify(signature, material)
        else:
            raise AuthenticationError("unsupported client key type")
    except InvalidSignature as error:
        raise AuthenticationError("challenge signature did not verify") from error


def _verify_chain(
    chain: list[x509.Certificate], ca_certificates: list[x509.Certificate], now: datetime
) -> None:
    """Validate leaf-to-CA issuance and validity windows.

    This deliberately walks the supplied chain rather than delegating to a
    verifier: the household CA is a single directly-issuing root, so the
    property being checked is narrow and worth being explicit about.
    """

    for certificate in chain:
        if certificate.not_valid_before_utc > now:
            raise AuthenticationError("client certificate is not yet valid")
        if certificate.not_valid_after_utc < now:
            raise AuthenticationError("client certificate has expired")

    # Each certificate must be signed by the next one up, ending at the CA.
    for index, certificate in enumerate(chain):
        issuers = (
            [chain[index + 1]]
            if index + 1 < len(chain)
            else [ca for ca in ca_certificates if ca.subject == certificate.issuer]
        )
        if not issuers:
            raise AuthenticationError("client certificate was not issued by the household CA")
        for issuer in issuers:
            try:
                issuer.public_key().verify(
                    certificate.signature,
                    certificate.tbs_certificate_bytes,
                    *_signature_arguments(issuer, certificate),
                )
                break
            except (InvalidSignature, AuthenticationError, TypeError, ValueError):
                continue
        else:
            raise AuthenticationError("client certificate chain does not verify")


def _signature_arguments(issuer: x509.Certificate, certificate: x509.Certificate):
    public_key = issuer.public_key()
    algorithm = certificate.signature_hash_algorithm
    if isinstance(public_key, ec.EllipticCurvePublicKey):
        return (ec.ECDSA(algorithm),)
    if isinstance(public_key, rsa.RSAPublicKey):
        return (padding.PKCS1v15(), algorithm)
    if isinstance(public_key, ed25519.Ed25519PublicKey):
        return ()
    raise AuthenticationError("unsupported issuer key type")


def _require_client_auth(certificate: x509.Certificate) -> None:
    try:
        usage = certificate.extensions.get_extension_for_class(x509.ExtendedKeyUsage)
    except x509.ExtensionNotFound:
        # An identity without an explicit purpose is not accepted as a client:
        # the issuing script always sets clientAuth.
        raise AuthenticationError("client certificate has no extended key usage") from None
    if ExtendedKeyUsageOID.CLIENT_AUTH not in usage.value:
        raise AuthenticationError("client certificate is not valid for client authentication")


class CompanionAuthenticator:
    """Validate one ``auth_response`` against an issued challenge."""

    def __init__(
        self,
        ca_pem: bytes | str | None,
        *,
        nonce_ttl_seconds: float = DEFAULT_NONCE_TTL_SECONDS,
        allowed_identities: frozenset[str] = frozenset(),
    ) -> None:
        self._ca_certificates: list[x509.Certificate] = []
        if ca_pem:
            data = ca_pem.encode() if isinstance(ca_pem, str) else ca_pem
            self._ca_certificates = x509.load_pem_x509_certificates(data)
        self.nonces = NonceIssuer(ttl_seconds=nonce_ttl_seconds)
        # Empty means "any identity the household CA issued whose certificate
        # matches the id it announces". Naming identities is an additional
        # restriction, not the mechanism that makes the socket safe — and
        # keeping it configurable is what keeps deployment names out of git.
        self._allowed = frozenset(name.strip().lower() for name in allowed_identities)

    @property
    def configured(self) -> bool:
        return bool(self._ca_certificates)

    def issue_nonce(self) -> str:
        return self.nonces.issue()

    def authenticate(
        self,
        *,
        nonce: str,
        protocol_version: int,
        announced_id: str,
        roles: list[str],
        certificate_chain: list[str],
        signature: str,
        now: datetime | None = None,
    ) -> AuthenticatedIdentity:
        if not self._ca_certificates:
            raise AuthenticationError("no household CA is configured for companion auth")
        if not self.nonces.consume(nonce):
            # Covers unknown, expired and replayed nonces alike.
            raise AuthenticationError("challenge nonce is unknown, expired or already used")

        identity = announced_id.strip().lower()
        if self._allowed and identity not in self._allowed:
            raise AuthenticationError(f"identity {identity!r} is not permitted to connect")

        try:
            chain = [
                x509.load_pem_x509_certificate(pem.encode())
                for pem in certificate_chain
            ]
        except ValueError as error:
            raise AuthenticationError("certificate chain could not be parsed") from error

        moment = (now or datetime.now(UTC)).astimezone(UTC)
        _verify_chain(chain, self._ca_certificates, moment)
        leaf = chain[0]
        _require_client_auth(leaf)

        try:
            raw_signature = base64.b64decode(signature, validate=True)
        except (ValueError, TypeError) as error:
            raise AuthenticationError("signature was not valid base64") from error
        _verify_signature(
            leaf,
            raw_signature,
            challenge_material(
                nonce=nonce,
                protocol_version=protocol_version,
                announced_id=announced_id,
                roles=roles,
            ),
        )

        names = _identity_names(leaf)
        if identity not in names:
            raise AuthenticationError(
                f"certificate identity {sorted(names) or ['unnamed']} may not announce {identity!r}"
            )

        fingerprint = hashlib.sha256(
            leaf.public_bytes(encoding=serialization.Encoding.DER)
        ).hexdigest()
        return AuthenticatedIdentity(
            identity=identity,
            roles=tuple(sorted(role.strip().lower() for role in roles)),
            not_after=leaf.not_valid_after_utc,
            fingerprint=fingerprint,
            certificate=leaf,
        )
