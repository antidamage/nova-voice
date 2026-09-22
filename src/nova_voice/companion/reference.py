"""A headless companion peer for tests and bring-up.

Every server-side behaviour worth testing — authentication, offer/accept,
rejection, tool callbacks, malformed results, slow results, disconnects — is
exercised through this rather than through an iPhone. That keeps the whole
routing and session layer testable under ``pytest`` alone, with no Xcode and no
physical device, which is the only way this stays maintainable.

It is a test peer, not a client library: it is deliberately eager to
misbehave on request.
"""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, padding, rsa

from nova_voice.companion.auth import challenge_material
from nova_voice.companion.protocol import (
    PROTOCOL_VERSION,
    AuthResponse,
    CompanionHello,
    CompanionTelemetry,
    Heartbeat,
    JobAccept,
    JobFailed,
    JobOffer,
    JobProgress,
    JobReject,
    JobResult,
    ModelAvailability,
    RejectReason,
    ToolCall,
    serialize,
)

# Given one offer, decide what this peer does with it. Returning a dict
# completes the job with that result; returning a RejectReason declines it;
# returning None fails the attempt.
JobHandler = Callable[[JobOffer], Awaitable[dict | RejectReason | None]]


def sign_challenge(private_key, material: bytes) -> str:
    if isinstance(private_key, ec.EllipticCurvePrivateKey):
        signature = private_key.sign(material, ec.ECDSA(hashes.SHA256()))
    elif isinstance(private_key, rsa.RSAPrivateKey):
        signature = private_key.sign(material, padding.PKCS1v15(), hashes.SHA256())
    elif isinstance(private_key, ed25519.Ed25519PrivateKey):
        signature = private_key.sign(material)
    else:
        raise TypeError("unsupported private key type")
    return base64.b64encode(signature).decode()


@dataclass
class ReferenceCompanion:
    """Drives one companion session over an already-connected transport.

    ``send``/``receive`` are supplied by the caller so the same peer works over
    a real WebSocket, a Starlette test-client socket, or a pair of in-memory
    queues.
    """

    announced_id: str
    private_key: Any
    certificate_pem: str
    send: Callable[[str], Awaitable[None]]
    receive: Callable[[], Awaitable[str]]
    roles: tuple[str, ...] = ("companion",)
    workloads: tuple[str, ...] = ("interpret", "classify_icon")
    display_name: str = "Reference Companion"
    telemetry: CompanionTelemetry = field(
        default_factory=lambda: CompanionTelemetry(
            battery=1.0,
            charging=True,
            models=ModelAvailability(hotAvailable=True, hotContextTokens=4096),
        )
    )
    job_handler: JobHandler | None = None
    # Delay before answering an offer, for acceptance-timeout tests.
    accept_delay_seconds: float = 0.0
    # Tool calls this peer will make before returning its result.
    tool_calls: tuple[tuple[str, str, dict], ...] = ()
    tool_results: list = field(default_factory=list)
    accepted: list[str] = field(default_factory=list)

    async def authenticate(self) -> dict:
        """Answer the challenge and complete the hello exchange."""

        challenge = _decode(await self.receive())
        material = challenge_material(
            nonce=challenge["nonce"],
            protocol_version=PROTOCOL_VERSION,
            announced_id=self.announced_id,
            roles=list(self.roles),
        )
        await self._send(
            AuthResponse(
                protocolVersion=PROTOCOL_VERSION,
                announcedId=self.announced_id,
                roles=list(self.roles),
                certificateChain=[self.certificate_pem],
                signature=sign_challenge(self.private_key, material),
            )
        )
        await self._send(
            CompanionHello(
                protocolVersion=PROTOCOL_VERSION,
                displayName=self.display_name,
                roles=list(self.roles),
                appVersion="reference",
                osVersion="reference",
                workloads=list(self.workloads),
                telemetry=self.telemetry,
            )
        )
        return _decode(await self.receive())

    async def run(self, *, until: int = 1) -> None:
        """Service offers until ``until`` jobs have reached a terminal state."""

        finished = 0
        while finished < until:
            message = _decode(await self.receive())
            if message.get("type") != "job_offer":
                continue
            offer = JobOffer.model_validate(message)
            if await self._handle_offer(offer):
                finished += 1

    async def heartbeat(self) -> None:
        await self._send(Heartbeat(sentAt=datetime.now(UTC)))

    async def _handle_offer(self, offer: JobOffer) -> bool:
        envelope = offer.envelope
        if self.accept_delay_seconds:
            await asyncio.sleep(self.accept_delay_seconds)
        outcome = (
            await self.job_handler(offer)
            if self.job_handler is not None
            else {"echo": offer.payload}
        )
        if isinstance(outcome, str):
            await self._send(
                JobReject(
                    jobId=envelope.job_id, attemptId=envelope.attempt_id, reason=outcome
                )
            )
            return True

        await self._send(
            JobAccept(jobId=envelope.job_id, attemptId=envelope.attempt_id)
        )
        self.accepted.append(envelope.attempt_id)

        for index, (provider, tool, arguments) in enumerate(self.tool_calls):
            await self._send(
                ToolCall(
                    jobId=envelope.job_id,
                    attemptId=envelope.attempt_id,
                    callId=f"{envelope.attempt_id}-{index}",
                    provider=provider,
                    tool=tool,
                    arguments=arguments,
                )
            )
            self.tool_results.append(_decode(await self.receive()))

        await self._send(
            JobProgress(
                jobId=envelope.job_id,
                attemptId=envelope.attempt_id,
                sequence=0,
                stage="reasoning",
            )
        )
        if outcome is None:
            await self._send(
                JobFailed(
                    jobId=envelope.job_id,
                    attemptId=envelope.attempt_id,
                    reason="model_error",
                )
            )
            return True
        await self._send(
            JobResult(
                jobId=envelope.job_id, attemptId=envelope.attempt_id, result=outcome
            )
        )
        return True

    async def _send(self, message) -> None:
        await self.send(json.dumps(serialize(message)))


def _decode(raw: str) -> dict:
    return json.loads(raw)


def certificate_pem(certificate) -> str:
    return certificate.public_bytes(serialization.Encoding.PEM).decode()
