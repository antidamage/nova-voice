"""What a reconnecting device says it is still holding.

The other half of NPT-403. `reconcile()` was complete and tested, and nothing
on the wire carried the device's claim — so the function could not have run in
production, and jobs leased to a device that had restarted would have waited
out their leases instead of being released.

The load-bearing case is the *empty* claim. A device that crashed mid-job
reconnects holding nothing, and that emptiness is the signal.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from nova_voice.companion.auth import AuthenticatedIdentity
from nova_voice.companion.protocol import (
    PROTOCOL_VERSION,
    CompanionHello,
    CompanionTelemetry,
    ModelAvailability,
    parse_client_message,
)
from nova_voice.companion.session import CompanionSessionManager


async def _noop_send(_payload: dict) -> None:
    return None


def _identity() -> AuthenticatedIdentity:
    return AuthenticatedIdentity(
        identity="companion-1",
        roles=("companion",),
        not_after=datetime.now(UTC) + timedelta(days=1),
        fingerprint="test",
    )


def _hello(active_jobs: list[str] | None = None) -> CompanionHello:
    return CompanionHello(
        protocolVersion=PROTOCOL_VERSION,
        displayName="Companion",
        roles=["companion"],
        appVersion="1.0",
        osVersion="26.6",
        workloads=["classify_icon"],
        telemetry=CompanionTelemetry(
            battery=1.0,
            charging=True,
            models=ModelAvailability(hotAvailable=True, hotContextTokens=4096),
        ),
        **({"activeJobs": active_jobs} if active_jobs is not None else {}),
    )


async def _settle(rounds: int = 40) -> None:
    for _ in range(rounds):
        await asyncio.sleep(0)


# -- the wire ------------------------------------------------------------------


def test_a_hello_without_the_field_parses_as_holding_nothing():
    """Absence and emptiness must mean the same thing.

    An older client that never sends the field has, in fact, restarted and
    holds nothing. Treating absence as "unknown" instead would leave its jobs
    leased to a device that will never finish them — the precise failure this
    field exists to fix.
    """

    raw = (
        '{"type":"hello","protocolVersion":1,"displayName":"Companion",'
        '"roles":["companion"],"appVersion":"1.0","osVersion":"26.6"}'
    )
    message = parse_client_message(raw)

    assert message.active_jobs == []


def test_a_claim_is_carried_onto_the_snapshot():
    sessions = CompanionSessionManager()
    sessions.register(
        identity=_identity(),
        hello=_hello(["job-1", "job-2"]),
        locality="home_lan",
        send=_noop_send,
    )

    assert sessions.snapshot().claimed_jobs == ("job-1", "job-2")


# -- reconciliation on register ------------------------------------------------


async def test_registering_reconciles_what_the_device_claims():
    seen: list[tuple[str, tuple[str, ...]]] = []

    async def reconcile(session_id: str, claimed: tuple[str, ...]) -> tuple[str, ...]:
        seen.append((session_id, claimed))
        return ()

    sessions = CompanionSessionManager()
    sessions.bind_reconciler(reconcile)
    session = sessions.register(
        identity=_identity(), hello=_hello(["job-1"]), locality="home_lan", send=_noop_send
    )
    await _settle()

    assert seen == [(session.session_id, ("job-1",))]


async def test_a_restarted_device_reconciles_with_an_empty_claim():
    # The case the whole field exists for: nothing held, so anything the
    # server thinks it leased to this device is releasable now rather than
    # when the lease happens to run out.
    seen: list[tuple[str, ...]] = []

    async def reconcile(session_id: str, claimed: tuple[str, ...]) -> tuple[str, ...]:
        seen.append(claimed)
        return ()

    sessions = CompanionSessionManager()
    sessions.bind_reconciler(reconcile)
    sessions.register(
        identity=_identity(), hello=_hello([]), locality="home_lan", send=_noop_send
    )
    await _settle()

    assert seen == [()]


async def test_a_job_the_device_claims_but_we_do_not_own_is_cancelled_on_the_device():
    """Otherwise it finishes work whose result we will refuse — and if the job
    was reassigned, the same job runs twice."""

    sent: list[dict] = []

    async def send(payload: dict) -> None:
        sent.append(payload)

    async def reconcile(session_id: str, claimed: tuple[str, ...]) -> tuple[str, ...]:
        return ("ghost-job",)

    sessions = CompanionSessionManager()
    sessions.bind_reconciler(reconcile)
    sessions.register(
        identity=_identity(), hello=_hello(["ghost-job"]), locality="home_lan", send=send
    )
    await _settle()

    cancels = [frame for frame in sent if frame.get("type") == "job_cancel"]
    assert [frame["jobId"] for frame in cancels] == ["ghost-job"]
    assert cancels[0]["reason"] == "superseded"


async def test_registration_does_not_wait_for_reconciliation():
    """A device waiting on its hello_ack is a device not yet answering anything.

    So the store read happens after registration returns, not inside it.
    """

    started = asyncio.Event()
    release = asyncio.Event()

    async def reconcile(session_id: str, claimed: tuple[str, ...]) -> tuple[str, ...]:
        started.set()
        await release.wait()
        return ()

    sessions = CompanionSessionManager()
    sessions.bind_reconciler(reconcile)
    session = sessions.register(
        identity=_identity(), hello=_hello(["job-1"]), locality="home_lan", send=_noop_send
    )

    # Registration has already returned a usable session.
    assert sessions.is_current(session)
    await asyncio.wait_for(started.wait(), timeout=1)
    release.set()
    await _settle()


async def test_a_failing_reconciliation_does_not_stop_the_device_connecting():
    # Cleanup failing is not a reason to refuse a working phone.
    async def reconcile(session_id: str, claimed: tuple[str, ...]) -> tuple[str, ...]:
        raise RuntimeError("the store is unavailable")

    sessions = CompanionSessionManager()
    sessions.bind_reconciler(reconcile)
    session = sessions.register(
        identity=_identity(), hello=_hello(["job-1"]), locality="home_lan", send=_noop_send
    )
    await _settle()

    assert sessions.is_current(session)
    assert sessions.snapshot().connected is True


async def test_a_deployment_with_no_reconciler_registers_normally():
    """A deployment with no durable store has nothing to reconcile, and must
    not fail to register a device over its absence."""

    sessions = CompanionSessionManager()
    session = sessions.register(
        identity=_identity(), hello=_hello(["job-1"]), locality="home_lan", send=_noop_send
    )
    await _settle()

    assert sessions.is_current(session)


# -- server-driven liveness ----------------------------------------------------


async def test_the_server_pings_so_the_heartbeat_field_means_something():
    """Without this it grew without bound from the moment of connect.

    A status field that always looks alarming and never means anything is
    worse than no field: it is the one an operator learns to ignore, and then
    ignores on the day it matters.
    """

    sent: list[dict] = []

    async def send(payload: dict) -> None:
        sent.append(payload)

    sessions = CompanionSessionManager()
    sessions.register(
        identity=_identity(), hello=_hello(), locality="home_lan", send=send
    )
    loop = asyncio.create_task(sessions.ping_loop(interval_seconds=0.01))
    await asyncio.sleep(0.05)
    loop.cancel()

    assert any(frame.get("type") == "ping" for frame in sent)


async def test_pinging_with_nobody_connected_is_harmless():
    sessions = CompanionSessionManager()
    loop = asyncio.create_task(sessions.ping_loop(interval_seconds=0.01))
    await asyncio.sleep(0.05)
    loop.cancel()

    assert sessions.snapshot().connected is False


async def test_a_failed_ping_does_not_tear_the_session_down():
    # The receive loop owns teardown. A ping failing is how we learn the socket
    # is gone, not a reason to act on it from here.
    async def send(_payload: dict) -> None:
        raise RuntimeError("socket closed")

    sessions = CompanionSessionManager()
    session = sessions.register(
        identity=_identity(), hello=_hello(), locality="home_lan", send=send
    )
    loop = asyncio.create_task(sessions.ping_loop(interval_seconds=0.01))
    await asyncio.sleep(0.05)
    loop.cancel()

    assert sessions.is_current(session)
