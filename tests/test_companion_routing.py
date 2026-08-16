"""Routing, locality, tiers and the strict protocol edge.

The load-bearing test here is the no-eager-hedge one: once the companion
accepts an attempt, Iridium's local path must not run. Everything else in the
offload design is pointless if both paths execute, because the ``--parallel 1``
slot stays occupied either way.
"""

from __future__ import annotations

import pytest

from nova_voice.companion.locality import LocalityClassifier
from nova_voice.companion.protocol import (
    CompanionTelemetry,
    ModelAvailability,
    parse_client_message,
)
from nova_voice.companion.router import CompanionWorkloadRouter
from nova_voice.companion.session import JobOutcome, OfferOutcome
from nova_voice.companion.tiers import (
    CompanionTier,
    TierThresholds,
    TierTracker,
)

# -- locality -----------------------------------------------------------------


def test_locality_is_derived_from_the_peer_address():
    classifier = LocalityClassifier.from_settings(["192.168.8.0/24"])
    assert classifier.classify("192.168.8.30") == "home_lan"
    assert classifier.classify("100.72.234.9") == "tailnet"
    assert classifier.classify("203.0.113.5") == "other"
    assert classifier.classify(None) == "other"


def test_unconfigured_home_subnets_never_qualify_as_home():
    """The safe failure direction: nothing is home until it is configured."""

    classifier = LocalityClassifier.from_settings([])
    assert classifier.classify("192.168.8.30") == "other"


def test_malformed_subnets_are_ignored_not_fatal():
    classifier = LocalityClassifier.from_settings(["not-a-subnet", "192.168.8.0/24"])
    assert classifier.classify("192.168.8.30") == "home_lan"


# -- tiers --------------------------------------------------------------------


def _telemetry(**changes) -> CompanionTelemetry:
    values = {
        "battery": 1.0,
        "charging": False,
        "models": ModelAvailability(hotAvailable=True),
    }
    values.update(changes)
    return CompanionTelemetry(**values)


def test_degradation_is_immediate_but_recovery_must_dwell():
    tracker = TierTracker(thresholds=TierThresholds(dwell_seconds=60))
    tracker.update(_telemetry(battery=0.9), now=0)
    assert tracker.tier is CompanionTier.FULL

    # Falling is instant: a nearly-flat phone stops being asked at once.
    tracker.update(_telemetry(battery=0.3), now=1)
    assert tracker.tier is CompanionTier.REDUCED

    # Climbing back needs the improvement to hold.
    tracker.update(_telemetry(battery=0.9), now=2)
    assert tracker.tier is CompanionTier.REDUCED
    tracker.update(_telemetry(battery=0.9), now=30)
    assert tracker.tier is CompanionTier.REDUCED
    tracker.update(_telemetry(battery=0.9), now=70)
    assert tracker.tier is CompanionTier.FULL


def test_hysteresis_stops_a_boundary_reading_flapping():
    thresholds = TierThresholds(dwell_seconds=0, hysteresis=0.05)
    tracker = TierTracker(thresholds=thresholds)
    tracker.update(_telemetry(battery=0.3), now=0)
    assert tracker.tier is CompanionTier.REDUCED

    tracker.update(_telemetry(battery=0.2), now=1)
    assert tracker.tier is CompanionTier.ADVISORY
    # 0.25 is the boundary; without headroom it must not climb back.
    tracker.update(_telemetry(battery=0.26), now=2)
    assert tracker.tier is CompanionTier.ADVISORY
    tracker.update(_telemetry(battery=0.31), now=3)
    assert tracker.tier is CompanionTier.REDUCED


def test_critical_thermal_and_missing_model_force_off():
    tracker = TierTracker(thresholds=TierThresholds(dwell_seconds=0))
    tracker.update(_telemetry(battery=1.0, charging=True), now=0)
    assert tracker.tier is CompanionTier.FULL
    tracker.update(_telemetry(battery=1.0, charging=True, thermalState="critical"), now=1)
    assert tracker.tier is CompanionTier.OFF
    tracker.update(
        _telemetry(battery=1.0, charging=True, models=ModelAvailability(hotAvailable=False)),
        now=2,
    )
    assert tracker.tier is CompanionTier.OFF


def test_stale_telemetry_is_not_evidence_of_health():
    tracker = TierTracker(thresholds=TierThresholds(stale_after_seconds=180))
    tracker.update(_telemetry(battery=1.0, charging=True), now=0)
    assert tracker.current(now=10).tier is CompanionTier.FULL
    assert tracker.current(now=500).tier is CompanionTier.OFF


# -- protocol strictness ------------------------------------------------------


def test_unknown_message_type_is_rejected():
    with pytest.raises(ValueError):
        parse_client_message('{"type":"take_over_the_house"}')


def test_unknown_field_is_rejected():
    with pytest.raises(ValueError):
        parse_client_message(
            '{"type":"heartbeat","sentAt":"2026-08-13T00:00:00Z","extra":1}'
        )


def test_oversized_frame_is_rejected_before_parsing():
    payload = '{"type":"heartbeat","sentAt":"2026-08-13T00:00:00Z","pad":"' + "x" * 400_000 + '"}'
    with pytest.raises(ValueError, match="exceeds"):
        parse_client_message(payload)


def test_valid_heartbeat_parses():
    message = parse_client_message('{"type":"heartbeat","sentAt":"2026-08-13T00:00:00Z"}')
    assert message.type == "heartbeat"


# -- routing ------------------------------------------------------------------


class _FakeAttempt:
    def __init__(self) -> None:
        self.job_id = "job"
        self.attempt_id = "attempt"


class _FakeSnapshot:
    def __init__(self, **changes) -> None:
        self.connected = True
        self.roles = ("companion",)
        self.workloads = frozenset({"interpret", "classify_icon"})
        self.schema_versions = (1,)
        self.tier = CompanionTier.FULL
        # Carried into the ineligibility message, because "tier off is below
        # reduced" does not say whether the battery is flat, the device is hot,
        # or its telemetry stopped arriving.
        self.tier_reason = "charging"
        self.locality = "home_lan"
        self.telemetry_age_seconds = 1.0
        self.__dict__.update(changes)


class _FakeSessions:
    """Stands in for the session manager, recording what it was asked to do."""

    def __init__(self, *, accept=True, outcome=None, snapshot=None) -> None:
        self._accept = accept
        self._outcome = outcome or JobOutcome(ok=True, result={"value": "companion"})
        self._snapshot = snapshot or _FakeSnapshot()
        self.offers = 0

    def snapshot(self, *, now=None):
        return self._snapshot

    async def offer(self, **kwargs):
        self.offers += 1
        attempt = _FakeAttempt()
        return (
            object(),
            attempt,
            OfferOutcome(
                accepted=self._accept,
                attempt_id="attempt",
                reason=None if self._accept else "battery",
            ),
        )

    async def await_result(self, session, attempt, *, timeout_seconds):
        return self._outcome


async def test_accepted_attempt_never_runs_the_local_path():
    """The no-eager-hedge rule: acceptance means Iridium's slot stays free."""

    local_calls = 0

    async def local():
        nonlocal local_calls
        local_calls += 1
        return {"value": "local"}

    sessions = _FakeSessions(accept=True)
    router = CompanionWorkloadRouter(sessions, enabled=True)
    router.override("interpret", mode="companion_preferred")
    result = await router.run("interpret", {"transcript": "hello"}, local)

    assert result.source == "companion"
    assert result.value == {"value": "companion"}
    assert local_calls == 0


async def test_rejection_falls_back_locally_exactly_once():
    local_calls = 0

    async def local():
        nonlocal local_calls
        local_calls += 1
        return {"value": "local"}

    sessions = _FakeSessions(accept=False)
    router = CompanionWorkloadRouter(sessions, enabled=True)
    router.override("interpret", mode="companion_preferred")
    result = await router.run("interpret", {"transcript": "hello"}, local)

    assert result.source == "local"
    assert local_calls == 1
    assert "declined" in result.reason


async def test_failure_after_acceptance_falls_back_once():
    local_calls = 0

    async def local():
        nonlocal local_calls
        local_calls += 1
        return {"value": "local"}

    sessions = _FakeSessions(
        accept=True, outcome=JobOutcome(ok=False, failure="timeout", detail="deadline")
    )
    router = CompanionWorkloadRouter(sessions, enabled=True)
    router.override("interpret", mode="companion_preferred")
    result = await router.run("interpret", {"transcript": "hello"}, local)

    assert result.source == "local"
    assert local_calls == 1


async def test_invalid_companion_result_falls_back():
    async def local():
        return {"value": "local"}

    def parse(payload):
        raise ValueError("not the expected shape")

    sessions = _FakeSessions(accept=True)
    router = CompanionWorkloadRouter(sessions, enabled=True)
    router.override("interpret", mode="companion_preferred")
    result = await router.run("interpret", {}, local, parse=parse)

    assert result.source == "local"
    assert "invalid result" in result.reason


async def test_companion_only_returns_unavailable_rather_than_running_locally():
    async def local():
        raise AssertionError("companion_only must not run the local path")

    sessions = _FakeSessions(accept=False)
    router = CompanionWorkloadRouter(sessions, enabled=True)
    router.override("interpret", mode="companion_preferred")
    router.override("interpret", mode="companion_only")
    result = await router.run("interpret", {}, local)

    assert result.source == "none"


async def test_feature_switch_keeps_everything_local():
    async def local():
        return {"value": "local"}

    sessions = _FakeSessions()
    router = CompanionWorkloadRouter(sessions, enabled=False)
    router.override("interpret", mode="companion_preferred")
    result = await router.run("interpret", {}, local)

    assert result.source == "local"
    assert sessions.offers == 0
    assert result.reason == "companion feature is off"


async def test_force_local_stops_offers_without_disabling_the_feature():
    async def local():
        return {"value": "local"}

    sessions = _FakeSessions()
    router = CompanionWorkloadRouter(sessions, enabled=True, force_local=True)
    router.override("interpret", mode="companion_preferred")
    result = await router.run("interpret", {}, local)

    assert result.source == "local"
    assert sessions.offers == 0


async def test_tailnet_cannot_unlock_a_home_lan_route():
    async def local():
        return {"value": "local"}

    sessions = _FakeSessions(snapshot=_FakeSnapshot(locality="tailnet"))
    router = CompanionWorkloadRouter(sessions, enabled=True)
    router.override("interpret", mode="companion_preferred")
    result = await router.run("interpret", {}, local)

    assert result.source == "local"
    assert sessions.offers == 0
    assert "tailnet" in result.reason


async def test_tier_below_the_route_minimum_stays_local():
    async def local():
        return {"value": "local"}

    sessions = _FakeSessions(snapshot=_FakeSnapshot(tier=CompanionTier.ADVISORY))
    router = CompanionWorkloadRouter(sessions, enabled=True)
    router.override("interpret", mode="companion_preferred")
    result = await router.run("interpret", {}, local)

    assert result.source == "local"
    assert sessions.offers == 0


async def test_unadvertised_workload_is_not_offered():
    async def local():
        return {"value": "local"}

    sessions = _FakeSessions(snapshot=_FakeSnapshot(workloads=frozenset({"classify_icon"})))
    router = CompanionWorkloadRouter(sessions, enabled=True)
    router.override("interpret", mode="companion_preferred")
    result = await router.run("interpret", {}, local)

    assert result.source == "local"
    assert sessions.offers == 0


async def test_repeated_failures_pause_the_workload():
    """A failing phone must not add its timeout to every single turn."""

    async def local():
        return {"value": "local"}

    sessions = _FakeSessions(
        accept=True, outcome=JobOutcome(ok=False, failure="failed", detail="model error")
    )
    router = CompanionWorkloadRouter(sessions, enabled=True, failure_budget=2)
    router.override("interpret", mode="companion_preferred")

    await router.run("interpret", {}, local)
    await router.run("interpret", {}, local)
    offers_before = sessions.offers

    result = await router.run("interpret", {}, local)
    assert sessions.offers == offers_before
    assert "circuit breaker" in result.reason


async def test_disabled_route_runs_nothing():
    async def local():
        raise AssertionError("a disabled workload must not run")

    sessions = _FakeSessions()
    router = CompanionWorkloadRouter(sessions, enabled=True)
    router.override("interpret", mode="companion_preferred")
    router.override("classify_icon", mode="disabled")
    result = await router.run("classify_icon", {}, local)

    assert result.source == "none"


async def test_route_overrides_are_per_workload():
    sessions = _FakeSessions()
    router = CompanionWorkloadRouter(sessions, enabled=True)
    router.override("interpret", mode="companion_preferred")
    router.override("classify_icon", mode="local")

    assert router.route("classify_icon").mode == "local"
    assert router.route("confirm_objective").mode == "companion_preferred"


async def test_spoken_turn_workloads_ship_local():
    """The hot-path defaults, which are a measured decision and not caution.

    On device, replies came back at a latency comparable to Iridium's but in a
    flat assistant voice with Nova's personality gone, answering the previous
    question rather than the current one. ``render_response`` *is* the
    assistant's voice, so that is a regression no latency parity pays for —
    and ``interpret`` plans the actions, which is worse again.

    The capability stays built and switchable at runtime. It is off until the
    output is good, not until the plumbing works.
    """

    router = CompanionWorkloadRouter(_FakeSessions(), enabled=True)

    assert router.route("interpret").mode == "local"
    assert router.route("render_response").mode == "local"
    # The passes outside the spoken turn are not affected: those measured well.
    assert router.route("classify_icon").mode == "companion_preferred"
    assert router.route("confirm_objective").mode == "companion_preferred"
    assert router.route("extract_self_profile_update").mode == "companion_preferred"
