"""Which route was faster, and which was better.

NPT-310. Two separate questions, and the second is the one that has actually
mattered: `render_response` matched the voice server for speed on the phone and
lost Nova's persona entirely. A timing comparison alone would have said ship it.

So this covers both — per-arm latency that a fallback cannot distort, and a
comparison mode that keeps both *answers* so a person can judge them.
"""

from __future__ import annotations

import asyncio

import pytest

from nova_voice.companion.router import CompanionWorkloadRouter, RouteTimings


class _Snapshot:
    connected = True
    roles = ("companion",)
    workloads = frozenset({"classify_icon", "render_response"})
    schema_versions = (1,)
    locality = "home_lan"
    telemetry_age_seconds = 1.0
    tier_reason = "charging"

    def __init__(self, tier) -> None:
        self.tier = tier


class _Outcome:
    def __init__(self, accepted=True, reason=None) -> None:
        self.accepted = accepted
        self.attempt_id = "a1"
        self.reason = reason
        self.retry_after_seconds = None


class _JobOutcome:
    def __init__(self, ok=True, result=None, failure=None, detail=None) -> None:
        self.ok = ok
        self.result = result
        self.failure = failure
        self.detail = detail


class _Sessions:
    """A session manager that answers on the companion after a set delay."""

    def __init__(self, *, tier, result=None, delay=0.0, accept=True, ok=True) -> None:
        from nova_voice.companion.tiers import CompanionTier

        self._snapshot = _Snapshot(tier or CompanionTier.FULL)
        self._result = result
        self._delay = delay
        self._accept = accept
        self._ok = ok

    def snapshot(self, *, now=None):
        return self._snapshot

    async def offer(self, **kwargs):
        return object(), object(), _Outcome(accepted=self._accept, reason="battery")

    async def await_result(self, session, attempt, *, timeout_seconds):
        await asyncio.sleep(self._delay)
        if not self._ok:
            return _JobOutcome(ok=False, failure="failed", detail="model error")
        return _JobOutcome(ok=True, result=self._result)


def _router(sessions) -> CompanionWorkloadRouter:
    router = CompanionWorkloadRouter(sessions, enabled=True)
    router.override("classify_icon", mode="companion_preferred")
    return router


# -- the window ----------------------------------------------------------------


def test_the_sample_window_is_bounded():
    """Unbounded percentiles are a leak in a process that runs for weeks."""

    timings = RouteTimings(window=5)
    for value in range(50):
        timings.add("local", float(value))

    assert len(timings.local) == 5
    # And it keeps the *recent* ones, so an improvement shows up rather than
    # being averaged away by a week of history.
    assert timings.local == [45.0, 46.0, 47.0, 48.0, 49.0]


def test_percentiles_are_absent_rather_than_zero_when_nothing_ran():
    # Zero would read as "instant", which is the opposite of "no data".
    summary = RouteTimings().summary()

    assert summary["companion"]["n"] == 0
    assert summary["companion"]["p50"] is None
    assert summary["local"]["p95"] is None


def test_percentiles_describe_the_samples():
    timings = RouteTimings()
    for value in [10, 20, 30, 40, 100]:
        timings.add("companion", float(value))

    summary = timings.summary()["companion"]
    assert summary["n"] == 5
    assert summary["p50"] == 30.0
    assert summary["p95"] == 100.0


# -- per-arm timing --------------------------------------------------------------


async def test_a_companion_answer_is_timed_against_the_companion_arm():
    sessions = _Sessions(tier=None, result={"icon": "pill"}, delay=0.02)
    router = _router(sessions)

    async def local():
        return {"icon": "local"}

    result = await router.run("classify_icon", {}, local)

    assert result.source == "companion"
    summary = router.timings()["classify_icon"]
    assert summary["companion"]["n"] == 1
    assert summary["local"]["n"] == 0


async def test_a_fallback_does_not_charge_the_companion_attempt_to_local():
    """The distortion this whole split exists to prevent.

    A fallback's total contains the failed attempt plus the local run. Charging
    that to the local arm would make the voice server look slower every time
    the phone let it down — the opposite of the truth.
    """

    sessions = _Sessions(tier=None, ok=False, delay=0.05)
    router = _router(sessions)

    async def local():
        await asyncio.sleep(0.01)
        return {"icon": "local"}

    result = await router.run("classify_icon", {}, local)

    assert result.source == "local"
    summary = router.timings()["classify_icon"]
    # The local arm holds only the local work.
    assert summary["local"]["n"] == 1
    assert summary["local"]["p50"] < 40
    # And the wasted attempt is recorded as its own number, because "what does
    # a failed attempt cost me" is the question that decides the route.
    assert summary["fallbackOverhead"]["n"] == 1
    assert summary["fallbackOverhead"]["p50"] >= 40


async def test_a_pass_that_never_goes_to_the_phone_still_records_local():
    sessions = _Sessions(tier=None)
    router = CompanionWorkloadRouter(sessions, enabled=True)
    router.override("classify_icon", mode="local")

    async def local():
        return {"icon": "local"}

    await router.run("classify_icon", {}, local)

    summary = router.timings()["classify_icon"]
    assert summary["local"]["n"] == 1
    # Never offered, so no overhead was paid and none is invented.
    assert summary["fallbackOverhead"]["n"] == 0


# -- "both" means both ---------------------------------------------------------


def test_no_pass_runs_on_both_stacks_by_default():
    """Running both gives up the entire benefit of offloading.

    It occupies llama.cpp's slot *and* the phone, so it must never be
    something a deployment ends up in by accident.
    """

    router = _router(_Sessions(tier=None))
    assert router.comparing() == frozenset()


def test_comparing_is_derived_from_the_route_rather_than_a_second_flag():
    # The earlier design held this as a separate runtime switch, which meant
    # the dropdown could read "Both" while nothing was being compared — and it
    # silently reset on every deploy.
    router = _router(_Sessions(tier=None))
    router.override("classify_icon", mode="both")

    assert router.comparing() == frozenset({"classify_icon"})


async def test_both_runs_both_arms_and_keeps_both_answers():
    sessions = _Sessions(tier=None, result={"text": "the phone's words"}, delay=0.01)
    router = _router(sessions)
    router.override("classify_icon", mode="both")

    ran: list[str] = []

    async def local():
        ran.append("local")
        return {"text": "the voice server's words"}

    result = await router.run("classify_icon", {}, local)

    # The local arm really ran — this is the whole difference from
    # `companion_preferred`, which only runs it when the phone fails.
    assert ran == ["local"]
    entry = router.comparisons()[-1]
    assert entry["companion"]["text"] == "the phone's words"
    assert entry["local"]["text"] == "the voice server's words"
    assert entry["companion"]["elapsedMs"] is not None
    assert entry["local"]["elapsedMs"] is not None
    assert result.source in {"companion", "local"}
    assert entry["spoken"] == result.source


async def test_both_reports_a_timing_for_each_arm():
    sessions = _Sessions(tier=None, result={"text": "phone"}, delay=0.01)
    router = _router(sessions)
    router.override("classify_icon", mode="both")

    async def local():
        return {"text": "server"}

    await router.run("classify_icon", {}, local)

    summary = router.timings()["classify_icon"]
    assert summary["companion"]["n"] == 1
    assert summary["local"]["n"] == 1


async def test_the_faster_arm_supplies_the_answer():
    # "The first back can have the speech": the local arm here is immediate
    # and the phone is slow, so the reply must be the local one.
    sessions = _Sessions(tier=None, result={"text": "phone"}, delay=0.2)
    router = _router(sessions)
    router.override("classify_icon", mode="both")

    async def local():
        return {"text": "server"}

    result = await router.run("classify_icon", {}, local)

    assert result.source == "local"
    assert result.value == {"text": "server"}
    # And the slower arm is still recorded, because the turn waits for it.
    assert router.comparisons()[-1]["companion"]["text"] == "phone"


async def test_both_still_answers_when_the_phone_fails():
    sessions = _Sessions(tier=None, ok=False)
    router = _router(sessions)
    router.override("classify_icon", mode="both")

    async def local():
        return {"text": "server"}

    result = await router.run("classify_icon", {}, local)

    assert result.source == "local"
    assert result.value == {"text": "server"}


async def test_a_real_local_failure_is_not_swallowed_by_the_race():
    """A declining phone is ordinary; a raising local arm is a fault."""

    sessions = _Sessions(tier=None, ok=False)
    router = _router(sessions)
    router.override("classify_icon", mode="both")

    async def local():
        raise RuntimeError("the local model is down")

    with pytest.raises(RuntimeError, match="local model is down"):
        await router.run("classify_icon", {}, local)


async def test_both_falls_back_to_local_alone_when_the_phone_is_ineligible():
    # Nothing to compare against, so it behaves like an ordinary local run
    # rather than refusing to answer.
    from nova_voice.companion.tiers import CompanionTier

    sessions = _Sessions(tier=CompanionTier.OFF)
    router = _router(sessions)
    router.override("classify_icon", mode="both")

    async def local():
        return {"text": "server"}

    result = await router.run("classify_icon", {}, local)

    assert result.source == "local"
    assert router.comparisons() == []


async def test_the_comparison_log_is_bounded():
    # It holds model output. An unbounded list of it is both a leak and a
    # growing pile of household text nobody asked to keep.
    sessions = _Sessions(tier=None, result={"text": "phone"})
    router = _router(sessions)
    router.override("classify_icon", mode="both")

    async def local():
        return {"text": "server"}

    for _ in range(30):
        await router.run("classify_icon", {}, local)

    assert len(router.comparisons()) == 20


async def test_a_pass_that_was_never_offered_reports_no_wasted_time():
    """"Never eligible" cost nothing and must not be reported as waste.

    Every pass routed `local` still travels through the fallback path. Timing
    its sub-millisecond bookkeeping as "wasted on a failed try" would put a
    number in front of an operator that describes nothing at all — which is
    exactly what the first version of this did.
    """

    from nova_voice.companion.tiers import CompanionTier

    sessions = _Sessions(tier=CompanionTier.OFF)
    router = _router(sessions)

    async def local():
        return {"icon": "local"}

    await router.run("classify_icon", {}, local)

    summary = router.timings()["classify_icon"]
    assert summary["local"]["n"] == 1
    assert summary["fallbackOverhead"]["n"] == 0


async def test_a_pass_that_was_offered_and_failed_does_report_wasted_time():
    sessions = _Sessions(tier=None, ok=False, delay=0.03)
    router = _router(sessions)

    async def local():
        return {"icon": "local"}

    await router.run("classify_icon", {}, local)

    assert router.timings()["classify_icon"]["fallbackOverhead"]["n"] == 1
