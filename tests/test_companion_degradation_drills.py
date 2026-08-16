"""What the companion path does when things go wrong, case by case.

NPT-706 and NPT-707. The roadmap asks for a drill per failure mode with a
documented expected state, so this file is that table in executable form. Each
test names the failure, and asserts the two things that must hold across all of
them: **the work still gets done exactly once**, and **the failure does not
become the user's problem** — no stalled turn, no duplicate execution, no
policy bypassed.

Everything runs against the real session manager, the real router and the real
reference peer over a pair of in-memory queues. Only the WebSocket is stood in
for, so a drill that passes here is a statement about the shipped code path.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime, timedelta

import pytest
from conftest import issue_certificate

from nova_voice.companion.auth import AuthenticatedIdentity
from nova_voice.companion.dispatch import dispatch_companion_message
from nova_voice.companion.protocol import (
    PROTOCOL_VERSION,
    CompanionHello,
    CompanionTelemetry,
    ModelAvailability,
    parse_client_message,
)
from nova_voice.companion.reference import ReferenceCompanion
from nova_voice.companion.router import CompanionWorkloadRouter
from nova_voice.companion.session import CompanionSessionManager
from nova_voice.companion.tiers import CompanionTier, TierThresholds


class _Wire:
    def __init__(self) -> None:
        self.to_peer: asyncio.Queue[str] = asyncio.Queue()
        self.to_server: asyncio.Queue[str] = asyncio.Queue()

    async def server_send(self, payload: dict) -> None:
        await self.to_peer.put(json.dumps(payload))

    async def peer_send(self, text: str) -> None:
        await self.to_server.put(text)

    async def peer_receive(self) -> str:
        return await self.to_peer.get()


def _telemetry(**overrides) -> CompanionTelemetry:
    base = {
        "battery": 1.0,
        "charging": True,
        "models": ModelAvailability(hotAvailable=True, hotContextTokens=4096),
    }
    return CompanionTelemetry(**{**base, **overrides})


def _hello(*workloads: str, telemetry: CompanionTelemetry | None = None) -> CompanionHello:
    return CompanionHello(
        protocolVersion=PROTOCOL_VERSION,
        displayName="Reference Companion",
        roles=["companion"],
        appVersion="test",
        osVersion="test",
        workloads=list(workloads),
        telemetry=telemetry or _telemetry(),
    )


def _identity() -> AuthenticatedIdentity:
    return AuthenticatedIdentity(
        identity="companion-1",
        roles=("companion",),
        not_after=datetime.now(UTC) + timedelta(days=1),
        fingerprint="test",
    )


async def _connect(sessions, wire, *workloads, telemetry=None):
    session = sessions.register(
        identity=_identity(),
        hello=_hello(*workloads, telemetry=telemetry),
        locality="home_lan",
        send=wire.server_send,
    )

    async def pump() -> None:
        while True:
            raw = await wire.to_server.get()
            dispatch_companion_message(sessions, session, parse_client_message(raw))

    return session, asyncio.create_task(pump())


def _peer(wire, key, pem, **overrides) -> ReferenceCompanion:
    return ReferenceCompanion(
        announced_id="companion-1",
        private_key=key,
        certificate_pem=pem,
        send=wire.peer_send,
        receive=wire.peer_receive,
        **overrides,
    )


@pytest.fixture
def credentials():
    from cryptography.hazmat.primitives import serialization

    key, certificate = issue_certificate("companion-1")
    return key, certificate.public_bytes(serialization.Encoding.PEM).decode()


async def _local():
    return {"icon": "local-fallback"}


# -- drills: the device declines or cannot take it -----------------------------


async def test_drill_low_power_rejection_falls_back_without_counting_as_failure(
    credentials,
):
    """Expected: local result, counted as a rejection, breaker untouched.

    A phone declining on battery grounds is the system working as designed. If
    that counted against the failure budget, an evening of sensible refusals
    would trip the breaker and disable a healthy device.
    """

    key, pem = credentials
    wire = _Wire()
    sessions = CompanionSessionManager()
    router = CompanionWorkloadRouter(sessions, enabled=True)
    _, pump = await _connect(sessions, wire, "classify_icon")

    peer = _peer(
        wire,
        key,
        pem,
        workloads=("classify_icon",),
        job_handler=lambda offer: _resolved("battery"),
    )
    peer_task = asyncio.create_task(peer.run(until=1))

    result = await router.run("classify_icon", {"name": "x"}, _local)
    await peer_task
    pump.cancel()

    assert result.source == "local"
    counters = router.counters()["classify_icon"]
    assert counters["rejected"] == 1
    assert counters["failed"] == 0
    assert counters["paused"] is False


async def test_drill_thermal_critical_makes_the_device_ineligible_before_any_offer():
    """Expected: no offer at all, and a reason that names the gate."""

    sessions = CompanionSessionManager()
    router = CompanionWorkloadRouter(sessions, enabled=True)
    wire = _Wire()
    _, pump = await _connect(
        sessions, wire, "classify_icon", telemetry=_telemetry(thermalState="critical")
    )

    decision = router.eligibility("classify_icon")
    result = await router.run("classify_icon", {"name": "x"}, _local)
    pump.cancel()

    assert decision.eligible is False
    assert "below" in decision.reason
    assert result.source == "local"
    # Never offered, so the phone was never woken to be told no.
    assert router.counters().get("classify_icon", {}).get("offered", 0) == 0


async def test_drill_stale_telemetry_disables_offers_rather_than_assuming_health():
    """Expected: silence stops being read as health, on one shared clock.

    Both assertions use `time.monotonic`, which is what the tier tracker and
    the router both read. Mixing in the event loop's clock here would compare
    two unrelated origins and prove nothing.
    """

    sessions = CompanionSessionManager(
        thresholds=TierThresholds(stale_after_seconds=1.0)
    )
    router = CompanionWorkloadRouter(sessions, enabled=True)
    wire = _Wire()
    _, pump = await _connect(sessions, wire, "classify_icon")

    now = time.monotonic()
    assert router.eligibility("classify_icon", now=now).eligible is True
    assert sessions.snapshot(now=now).tier is CompanionTier.FULL

    stale = router.eligibility("classify_icon", now=now + 600)
    pump.cancel()

    assert stale.eligible is False
    assert "stale" in stale.reason


# -- drills: the device goes away ----------------------------------------------


async def test_drill_disconnect_before_acceptance_falls_back_once(credentials):
    """Expected: exactly one local run, no hanging future."""

    key, pem = credentials
    wire = _Wire()
    sessions = CompanionSessionManager(accept_timeout_seconds=0.1)
    router = CompanionWorkloadRouter(sessions, enabled=True)
    session, pump = await _connect(sessions, wire, "classify_icon")

    runs: list[int] = []

    async def local():
        runs.append(1)
        return {"icon": "local-fallback"}

    # Nobody is answering the offer.
    result = await router.run("classify_icon", {"name": "x"}, local)
    pump.cancel()

    assert result.source == "local"
    assert runs == [1]
    assert session.attempts == {}


async def test_drill_disconnect_after_acceptance_resolves_rather_than_hanging(
    credentials,
):
    """Expected: the accepted attempt terminates, and local runs exactly once.

    This is the dangerous one. Between acceptance and disconnect nothing local
    is running by design — that is the no-eager-hedge rule — so if the attempt
    did not resolve, the turn would wait for its whole deadline with no work
    happening anywhere.
    """

    key, pem = credentials
    wire = _Wire()
    sessions = CompanionSessionManager()
    router = CompanionWorkloadRouter(sessions, enabled=True)
    session, pump = await _connect(sessions, wire, "classify_icon")

    runs: list[int] = []

    async def local():
        runs.append(1)
        return {"icon": "local-fallback"}

    async def accept_then_vanish(offer):
        # Accept, then the socket dies before a result is sent.
        asyncio.get_running_loop().call_soon(lambda: sessions.release(session))
        return None

    peer = _peer(
        wire, key, pem, workloads=("classify_icon",), job_handler=accept_then_vanish
    )
    peer_task = asyncio.create_task(peer.run(until=1))

    result = await router.run("classify_icon", {"name": "x"}, local)
    peer_task.cancel()
    pump.cancel()

    assert result.source == "local"
    assert runs == [1]


# -- drills: the device answers badly ------------------------------------------


async def test_drill_an_invalid_result_falls_back_and_counts_as_a_failure(credentials):
    """Expected: local result, failure counted, so repeats can trip the breaker."""

    key, pem = credentials
    wire = _Wire()
    sessions = CompanionSessionManager()
    router = CompanionWorkloadRouter(sessions, enabled=True)
    _, pump = await _connect(sessions, wire, "classify_icon")

    peer = _peer(
        wire,
        key,
        pem,
        workloads=("classify_icon",),
        job_handler=lambda offer: _resolved({"nonsense": True}),
    )
    peer_task = asyncio.create_task(peer.run(until=1))

    def parse(payload):
        if "icon" not in payload:
            raise ValueError("not an icon choice")
        return payload

    result = await router.run("classify_icon", {"name": "x"}, _local, parse=parse)
    await peer_task
    pump.cancel()

    assert result.source == "local"
    assert router.counters()["classify_icon"]["failed"] == 1


async def test_drill_a_failing_device_is_paused_rather_than_slowing_every_turn(
    credentials,
):
    """NPT-706. Expected: after the budget, offers stop until the pause elapses.

    Without this, a phone that fails every job adds its acceptance timeout to
    every single voice turn — the companion being *present* would make the
    assistant slower than not having one at all.
    """

    key, pem = credentials
    wire = _Wire()
    sessions = CompanionSessionManager()
    router = CompanionWorkloadRouter(
        sessions, enabled=True, failure_budget=2, breaker_pause_seconds=300
    )
    _, pump = await _connect(sessions, wire, "classify_icon")

    def parse(payload):
        raise ValueError("always invalid")

    for _ in range(2):
        peer = _peer(
            wire,
            key,
            pem,
            workloads=("classify_icon",),
            job_handler=lambda offer: _resolved({"nonsense": True}),
        )
        task = asyncio.create_task(peer.run(until=1))
        await router.run("classify_icon", {"name": "x"}, _local, parse=parse)
        await task

    decision = router.eligibility("classify_icon")
    offered_before = router.counters()["classify_icon"]["offered"]
    result = await router.run("classify_icon", {"name": "x"}, _local, parse=parse)
    pump.cancel()

    assert decision.eligible is False
    assert "circuit breaker" in decision.reason
    assert result.source == "local"
    # The point of the breaker: no further round trip was spent on the device.
    assert router.counters()["classify_icon"]["offered"] == offered_before


async def test_drill_the_breaker_recovers_after_its_pause(credentials):
    """A breaker that never releases is just a broken feature."""

    key, pem = credentials
    wire = _Wire()
    sessions = CompanionSessionManager()
    router = CompanionWorkloadRouter(
        sessions, enabled=True, failure_budget=1, breaker_pause_seconds=0.05
    )
    _, pump = await _connect(sessions, wire, "classify_icon")

    def parse(payload):
        raise ValueError("always invalid")

    peer = _peer(
        wire,
        key,
        pem,
        workloads=("classify_icon",),
        job_handler=lambda offer: _resolved({"nonsense": True}),
    )
    task = asyncio.create_task(peer.run(until=1))
    await router.run("classify_icon", {"name": "x"}, _local, parse=parse)
    await task

    assert router.eligibility("classify_icon").eligible is False
    await asyncio.sleep(0.1)
    assert router.eligibility("classify_icon").eligible is True
    pump.cancel()


# -- drills: the operator intervenes -------------------------------------------


async def test_drill_force_local_stops_offers_without_disconnecting_the_device():
    """Expected: rollback switch works instantly and the session survives.

    `force_local` has to be usable during an incident, which means it must not
    require the owner to close the app or the device to reconnect afterwards.
    """

    sessions = CompanionSessionManager()
    router = CompanionWorkloadRouter(sessions, enabled=True)
    wire = _Wire()
    _, pump = await _connect(sessions, wire, "classify_icon")

    assert router.eligibility("classify_icon").eligible is True
    router.set_force_local(True)
    assert router.eligibility("classify_icon").eligible is False
    assert sessions.snapshot().connected is True

    router.set_force_local(False)
    assert router.eligibility("classify_icon").eligible is True
    pump.cancel()


async def test_drill_disabling_the_feature_restores_the_previous_behaviour_exactly():
    """The rollback invariant: `companion_enabled=false` means the old path."""

    sessions = CompanionSessionManager()
    router = CompanionWorkloadRouter(sessions, enabled=False)
    wire = _Wire()
    _, pump = await _connect(sessions, wire, "classify_icon")

    result = await router.run("classify_icon", {"name": "x"}, _local)
    pump.cancel()

    assert result.source == "local"
    assert result.reason == "companion feature is off"
    assert router.counters()["classify_icon"]["offered"] == 0


async def test_drill_a_companion_only_workload_reports_unavailable_rather_than_guessing():
    """Expected: an explicit no-result, not silently different semantics."""

    sessions = CompanionSessionManager()
    router = CompanionWorkloadRouter(sessions, enabled=True)
    router.override("classify_icon", mode="companion_only")

    result = await router.run("classify_icon", {"name": "x"}, _local)

    assert result.source == "none"
    assert result.value is None


async def _resolved(value):
    return value
