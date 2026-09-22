"""Walking out of the house, and back into it.

NPT-704. The transition is where the locality contract earns its keep. Leaving
must withdraw the reasoning-replacement routes *immediately* — a phone on
cellular answering a household turn is the thing the home-LAN requirement
exists to prevent — while leaving the away-permitted work alone. Coming back
must restore them without anything manual.

The awkward part is that neither direction is announced. There is no "I am
leaving" frame: the socket closes and a new one opens from a different address,
sometimes seconds later, sometimes with a job still in flight.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from nova_voice.companion.auth import AuthenticatedIdentity
from nova_voice.companion.jobs import TERMINAL
from nova_voice.companion.ledger import CompanionJobLedger
from nova_voice.companion.protocol import (
    PROTOCOL_VERSION,
    CompanionHello,
    CompanionTelemetry,
    ModelAvailability,
)
from nova_voice.companion.router import CompanionWorkloadRouter
from nova_voice.companion.session import CompanionSessionManager
from nova_voice.durable.models import CompanionJobRecord
from nova_voice.durable.store import DurableAgentStore


async def _noop_send(_payload: dict) -> None:
    return None


def _identity() -> AuthenticatedIdentity:
    return AuthenticatedIdentity(
        identity="companion-1",
        roles=("companion",),
        not_after=datetime.now(UTC) + timedelta(days=1),
        fingerprint="test",
    )


def _hello() -> CompanionHello:
    return CompanionHello(
        protocolVersion=PROTOCOL_VERSION,
        displayName="Companion",
        roles=["companion"],
        appVersion="1.0",
        osVersion="26.6",
        workloads=["classify_icon", "interpret"],
        personalTools=["companion.calendar.list"],
        telemetry=CompanionTelemetry(
            battery=1.0,
            charging=True,
            models=ModelAvailability(hotAvailable=True, hotContextTokens=4096),
        ),
    )


def _connect(sessions: CompanionSessionManager, locality: str):
    return sessions.register(
        identity=_identity(), hello=_hello(), locality=locality, send=_noop_send
    )


def _router(sessions: CompanionSessionManager) -> CompanionWorkloadRouter:
    router = CompanionWorkloadRouter(sessions, enabled=True)
    router.override("classify_icon", mode="companion_preferred", locality="home_lan")
    return router


# -- leaving -------------------------------------------------------------------


def test_a_tailnet_session_cannot_run_a_home_lan_route():
    sessions = CompanionSessionManager()
    router = _router(sessions)
    _connect(sessions, "tailnet")

    decision = router.eligibility("classify_icon")

    assert decision.eligible is False
    assert "tailnet" in decision.reason
    assert "home_lan" in decision.reason


def test_an_unrecognised_network_is_refused_like_the_open_internet():
    # A subnet nobody configured is not a weaker home; it is somewhere else.
    sessions = CompanionSessionManager()
    router = _router(sessions)
    _connect(sessions, "other")

    assert router.eligibility("classify_icon").eligible is False


def test_reconnecting_from_the_tailnet_withdraws_the_home_routes_at_once():
    """No grace period. The old session's locality does not linger.

    A phone that has just handed over to cellular is not "still nearly home"
    for a minute — it is out, and a route that waits before believing that is a
    route that answers a household turn from a car.
    """

    sessions = CompanionSessionManager()
    router = _router(sessions)
    _connect(sessions, "home_lan")
    assert router.eligibility("classify_icon").eligible is True

    # Same device, new socket, new address. Supersedes the old session.
    _connect(sessions, "tailnet")

    assert router.eligibility("classify_icon").eligible is False


def test_an_away_permitted_route_survives_the_same_transition():
    """Leaving withdraws reasoning replacement, not everything.

    Personal tools are a narrower permission that the roadmap allows over an
    authenticated tailnet path, so a transition that disabled them too would be
    over-applying the rule.
    """

    sessions = CompanionSessionManager()
    router = CompanionWorkloadRouter(sessions, enabled=True)
    router.override("classify_icon", mode="companion_preferred", locality="tailnet_ok")
    _connect(sessions, "tailnet")

    assert router.eligibility("classify_icon").eligible is True


# -- coming back ---------------------------------------------------------------


def test_returning_home_restores_the_routes_with_nothing_manual():
    sessions = CompanionSessionManager()
    router = _router(sessions)
    _connect(sessions, "tailnet")
    assert router.eligibility("classify_icon").eligible is False

    _connect(sessions, "home_lan")

    assert router.eligibility("classify_icon").eligible is True


def test_the_route_configuration_is_untouched_by_the_transition():
    # Locality gates eligibility; it must not rewrite what the operator chose,
    # or a walk to the shops would silently reset the dashboard.
    sessions = CompanionSessionManager()
    router = _router(sessions)
    _connect(sessions, "home_lan")
    _connect(sessions, "tailnet")

    assert router.route("classify_icon").mode == "companion_preferred"
    assert router.route("classify_icon").locality == "home_lan"


# -- jobs caught in the transition ---------------------------------------------


async def test_a_job_in_flight_when_the_device_leaves_is_resolved_not_stranded():
    """The socket closes with no warning and a job still accepted.

    Nothing local is running by design — that is the no-eager-hedge rule — so a
    teardown that failed to resolve the attempt would leave the caller waiting
    for its whole deadline with no work happening anywhere. That is the worst
    available outcome: worse than falling back immediately, and worse than
    never having offered it.
    """

    sent: list[dict] = []

    async def send(payload: dict) -> None:
        sent.append(payload)

    sessions = CompanionSessionManager()
    session = sessions.register(
        identity=_identity(), hello=_hello(), locality="home_lan", send=send
    )

    offer = asyncio.create_task(
        sessions.offer(
            workload="classify_icon",
            payload={},
            idempotency_key="key-1",
            input_revision="rev-1",
            result_schema="icon_choice",
            trace_id="trace-1",
            complete_deadline_seconds=30.0,
        )
    )
    for _ in range(50):
        await asyncio.sleep(0)
        if sent:
            break
    envelope = sent[0]["envelope"]
    sessions.handle_accept(session, envelope["jobId"], envelope["attemptId"])
    result = await offer
    assert result is not None
    _, attempt, outcome = result
    assert outcome.accepted is True

    # The phone walks out of range.
    sessions.release(session)

    settled = await asyncio.wait_for(attempt.result, timeout=1)
    assert settled.ok is False
    assert settled.failure == "disconnected"


async def test_a_durable_job_leased_to_the_old_session_is_reclaimed(tmp_path):
    """The device reconnects from elsewhere and no longer holds its job.

    Reclaimed rather than left to expire: the lease may have minutes on it, and
    the device has just told us — by not claiming it — that nothing is working
    on it.
    """

    store = DurableAgentStore(tmp_path / "durable.sqlite3")
    await store.initialize()
    ledger = CompanionJobLedger(store)

    from nova_voice.companion import jobs

    await ledger.create(
        CompanionJobRecord(
            id="job-1",
            workload="interpret",
            input_revision="rev-1",
            idempotency_key="key-1",
            trace_id="trace-1",
        )
    )
    await ledger.apply(
        "job-1",
        lambda job: jobs.offer(
            job, attempt_id="a1", session_id="home-session", lease_seconds=600
        ),
    )
    await ledger.apply("job-1", lambda job: jobs.accept(job, attempt_id="a1"))
    await ledger.apply("job-1", lambda job: jobs.start(job, attempt_id="a1"))

    reclaimed, unknown = await ledger.reconcile(
        session_id="home-session", device_job_ids=[]
    )

    assert reclaimed == ("job-1",)
    assert unknown == ()
    record = await ledger.get("job-1")
    assert record is not None
    assert record.lease_owner is None
    assert record.status not in TERMINAL, "a reclaimed job must be re-offerable"


async def test_a_job_the_returning_device_still_holds_is_left_alone(tmp_path):
    # Wi-Fi flapped, the app never stopped working. Reclaiming would duplicate
    # the work it is still doing.
    store = DurableAgentStore(tmp_path / "durable.sqlite3")
    await store.initialize()
    ledger = CompanionJobLedger(store)

    from nova_voice.companion import jobs

    await ledger.create(
        CompanionJobRecord(
            id="job-1",
            workload="interpret",
            input_revision="rev-1",
            idempotency_key="key-1",
            trace_id="trace-1",
        )
    )
    await ledger.apply(
        "job-1",
        lambda job: jobs.offer(
            job, attempt_id="a1", session_id="session-1", lease_seconds=600
        ),
    )
    await ledger.apply("job-1", lambda job: jobs.accept(job, attempt_id="a1"))
    await ledger.apply("job-1", lambda job: jobs.start(job, attempt_id="a1"))

    reclaimed, _ = await ledger.reconcile(
        session_id="session-1", device_job_ids=["job-1"]
    )

    assert reclaimed == ()
    record = await ledger.get("job-1")
    assert record is not None and record.lease_owner == "session-1"


def test_a_superseded_session_leaves_nothing_registered():
    """One device, one session. The old one must not linger as a second peer."""

    sessions = CompanionSessionManager()
    first = _connect(sessions, "home_lan")
    second = _connect(sessions, "tailnet")

    assert sessions.is_current(second) is True
    assert sessions.is_current(first) is False
    assert sessions.snapshot().locality == "tailnet"
