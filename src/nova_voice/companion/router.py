"""Decide, per workload, whether the companion or Iridium does the work.

The routing is granular rather than a single on/off switch because the
workloads differ enormously in risk. Moving ``classify_icon`` to a phone costs
nothing if the phone is slow or wrong. Moving ``interpret`` there puts a spoken
turn behind a device that might be in a pocket.

**No eager hedge.** This is the rule the whole design turns on. Once the
companion *accepts* an attempt, Iridium does not start the same LLM workload
locally. Running both would keep the ``--parallel 1`` slot occupied and defeat
the entire purpose of offloading. The local path starts only on a terminal
outcome for that attempt — rejection, acceptance timeout, disconnect, invalid
result, cancellation or the workload deadline — and then it runs exactly once.

Before acceptance the cost of falling back is a fast round trip, which is why
the acceptance deadline is short and separate from the completion deadline.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any, Literal

from nova_voice.companion.protocol import CompanionWorkload, Locality, Sensitivity
from nova_voice.companion.session import CompanionSessionManager
from nova_voice.companion.tiers import CompanionTier, tier_rank
from nova_voice.companion.workloads import WORKLOADS

logger = logging.getLogger(__name__)

RouteMode = Literal[
    "local",
    "companion_preferred",
    "companion_only",
    "companion_fallback",
    # Run the pass on **both** stacks, speak whichever answers first, and keep
    # both answers and both timings.
    #
    # This deliberately breaks the no-eager-hedge rule the rest of the design
    # turns on: it occupies llama.cpp's single slot *and* the phone, so it
    # gives up the entire latency benefit of offloading. That is the trade it
    # exists to make — you cannot compare two models by running one of them,
    # and "which is faster" and "which is better" are different questions that
    # both need answering before a pass is moved for real.
    "both",
    "disabled",
]
RouteLocality = Literal["home_lan", "tailnet_ok"]


@dataclass(frozen=True)
class WorkloadRoute:
    mode: RouteMode = "local"
    locality: RouteLocality = "home_lan"
    min_tier: CompanionTier = CompanionTier.REDUCED
    # Completion budget for one attempt on the companion.
    deadline_seconds: float = 20.0
    sensitivity: Sensitivity = "ordinary"


def _default_routes() -> dict[CompanionWorkload, WorkloadRoute]:
    """One route per registered workload, derived from its spec.

    Deriving rather than restating means a workload can never appear in the
    route table without a handler and a result schema, and a hot-path deadline
    can never drift away from the one the spec chose to leave room for
    fallback plus TTS.

    Every route defaults to home-LAN only. A tailnet peer is not proof of being
    home, and the roadmap's rollout sequence turns routes on deliberately —
    ``companion_enabled`` is false by default, so none of this is consulted
    until it is switched on.
    """

    return {
        name: WorkloadRoute(
            # Hot-path passes default to **local**, and that is a measured
            # decision rather than caution. Apple's on-device model rendered
            # replies at a latency comparable to Iridium's (~1.9-2.2s) but
            # dropped Nova's persona entirely — flat "I'm just an AI, I don't
            # have feelings" answers where the local model speaks in character
            # — and answered the previous question rather than the current one.
            # ``render_response`` *is* the assistant's voice, so that is a
            # regression no latency parity pays for.
            #
            # The capability is built and switchable; it is off until the
            # output is good, not until the plumbing works.
            # See docs/evidence/companion-offload-live-20260815.md.
            mode="local" if entry.hot_path else "companion_preferred",
            locality="home_lan",
            # Hot-path work needs a healthy device; background work tolerates a
            # phone that is merely not in trouble.
            min_tier=CompanionTier.FULL if entry.hot_path else CompanionTier.REDUCED,
            deadline_seconds=entry.default_timeout_seconds,
            sensitivity=entry.sensitivity,
        )
        for name, entry in WORKLOADS.items()
    }


DEFAULT_ROUTES: dict[CompanionWorkload, WorkloadRoute] = _default_routes()


class _ArmUnavailable(RuntimeError):
    """The companion arm could not answer. Ordinary flow, not a fault."""


@dataclass(frozen=True)
class Eligibility:
    """Why a workload may or may not be offered right now."""

    eligible: bool
    reason: str
    route: WorkloadRoute


@dataclass
class WorkloadCounters:
    offered: int = 0
    accepted: int = 0
    rejected: int = 0
    completed: int = 0
    failed: int = 0
    fell_back: int = 0
    # Consecutive failures, for the circuit breaker.
    consecutive_failures: int = 0
    paused_until: float = 0.0


@dataclass
class RouteTimings:
    """How long a pass takes, kept separately for each place it can run.

    The two arms have to be timed *separately* or the comparison lies. A
    fallback's total elapsed time contains a failed companion attempt plus the
    local run, so charging that to "local" would make the voice server look
    slower every time the phone let it down — the opposite of the truth.

    Bounded: a fixed window of recent samples per arm. Percentiles over the
    whole history would take days to notice an improvement, and an unbounded
    list is a leak in a process that runs for weeks.
    """

    window: int = 60
    companion: list[float] = field(default_factory=list)
    local: list[float] = field(default_factory=list)
    # What a failed attempt cost the caller: total minus the local run. This
    # is the price of trying the phone and being let down, and it is the number
    # that decides whether `companion_preferred` is worth it at all.
    fallback_overhead: list[float] = field(default_factory=list)

    def add(self, arm: str, value: float) -> None:
        samples = getattr(self, arm, None)
        if samples is None:
            return
        samples.append(value)
        if len(samples) > self.window:
            del samples[: len(samples) - self.window]

    @staticmethod
    def _percentile(samples: list[float], fraction: float) -> float | None:
        if not samples:
            return None
        ordered = sorted(samples)
        # Nearest-rank. With windows this small an interpolating percentile
        # implies a precision the sample size does not support.
        index = min(len(ordered) - 1, int(fraction * len(ordered)))
        return round(ordered[index], 1)

    def summary(self) -> dict:
        return {
            arm: {
                "n": len(samples),
                "p50": self._percentile(samples, 0.5),
                "p95": self._percentile(samples, 0.95),
            }
            for arm, samples in (
                ("companion", self.companion),
                ("local", self.local),
                ("fallbackOverhead", self.fallback_overhead),
            )
        }


@dataclass
class RouteComparison:
    """One turn answered by both, kept so quality can be judged and not guessed.

    Speed was never the hard part. `render_response` matched the voice server
    for latency on the phone and lost Nova's persona entirely — a comparison of
    timings alone would have said ship it.
    """

    workload: CompanionWorkload
    at: str
    companion_text: str | None
    local_text: str | None
    companion_ms: float | None
    local_ms: float | None
    spoken: Literal["companion", "local", "neither"]

    def as_dict(self) -> dict:
        return {
            "workload": self.workload,
            "at": self.at,
            "spoken": self.spoken,
            "companion": {"text": self.companion_text, "elapsedMs": self.companion_ms},
            "local": {"text": self.local_text, "elapsedMs": self.local_ms},
        }


@dataclass
class RouteResult:
    """What actually happened, for measurement and for the dashboard."""

    workload: CompanionWorkload
    source: Literal["companion", "local", "none"]
    value: Any = None
    reason: str = ""
    elapsed_ms: float = 0.0


class CompanionWorkloadRouter:
    def __init__(
        self,
        sessions: CompanionSessionManager,
        *,
        routes: dict[CompanionWorkload, WorkloadRoute] | None = None,
        enabled: bool = False,
        force_local: bool = False,
        failure_budget: int = 3,
        breaker_pause_seconds: float = 120.0,
    ) -> None:
        self.sessions = sessions
        self._routes: dict[CompanionWorkload, WorkloadRoute] = dict(DEFAULT_ROUTES)
        if routes:
            self._routes.update(routes)
        self._enabled = enabled
        self._force_local = force_local
        self._failure_budget = failure_budget
        self._breaker_pause = breaker_pause_seconds
        self._counters: dict[CompanionWorkload, WorkloadCounters] = {}
        self._timings: dict[CompanionWorkload, RouteTimings] = {}
        self._comparisons: list[RouteComparison] = []
        self._comparison_window = 20

    # -- configuration --------------------------------------------------------

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled

    def set_force_local(self, force_local: bool) -> None:
        """Global override: keep every reasoning workload on Iridium."""

        self._force_local = force_local

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def force_local(self) -> bool:
        """What the router is *actually* doing, which the configuration may not.

        Both switches can be moved at runtime for a rollback, so reporting the
        configured value would mean a status endpoint that disagrees with the
        behaviour it is describing — during exactly the incident someone is
        using it to understand.
        """

        return self._force_local

    def route(self, workload: CompanionWorkload) -> WorkloadRoute:
        return self._routes.get(workload, WorkloadRoute())

    def override(self, workload: CompanionWorkload, **changes) -> None:
        self._routes[workload] = replace(self.route(workload), **changes)

    def routes(self) -> dict[CompanionWorkload, WorkloadRoute]:
        return dict(self._routes)

    def comparing(self) -> frozenset[CompanionWorkload]:
        """Passes currently running on both stacks.

        Derived from the route table rather than held as a second flag. The
        earlier runtime-only switch was a separate piece of state saying the
        same thing, which meant the dropdown could read "Both" while nothing
        was being compared — and, worse, it silently reset on every deploy.
        A route mode is persisted with the rest of the dashboard's settings, so
        it survives a restart the way an operator expects a setting to.
        """

        return frozenset(
            workload for workload, route in self._routes.items() if route.mode == "both"
        )

    def timings(self) -> dict[str, dict]:
        return {workload: timing.summary() for workload, timing in self._timings.items()}

    def comparisons(self) -> list[dict]:
        return [comparison.as_dict() for comparison in self._comparisons]

    def pop_comparison(self, workload: CompanionWorkload) -> dict | None:
        """Claim the most recent comparison for a pass, for this turn's transcript.

        Matched by workload and recency rather than by turn id. Two turns
        overlapping could in principle swap their entries — which is a display
        ordering wart in an evaluation mode, not a correctness problem, and not
        worth threading a trace id through three layers to avoid.
        """

        for index in range(len(self._comparisons) - 1, -1, -1):
            if self._comparisons[index].workload == workload:
                return self._comparisons[index].as_dict()
        return None

    def _record_comparison(self, comparison: RouteComparison) -> None:
        self._comparisons.append(comparison)
        if len(self._comparisons) > self._comparison_window:
            del self._comparisons[: len(self._comparisons) - self._comparison_window]

    def _timing(self, workload: CompanionWorkload) -> RouteTimings:
        return self._timings.setdefault(workload, RouteTimings())

    def counters(self) -> dict[str, dict]:
        return {
            workload: {
                "offered": counter.offered,
                "accepted": counter.accepted,
                "rejected": counter.rejected,
                "completed": counter.completed,
                "failed": counter.failed,
                "fellBack": counter.fell_back,
                "paused": counter.paused_until > time.monotonic(),
            }
            for workload, counter in self._counters.items()
        }

    # -- eligibility ----------------------------------------------------------

    def eligibility(self, workload: CompanionWorkload, *, now: float | None = None) -> Eligibility:
        moment = time.monotonic() if now is None else now
        route = self.route(workload)

        def no(reason: str) -> Eligibility:
            return Eligibility(False, reason, route)

        if route.mode == "disabled":
            return no("workload is disabled")
        if route.mode == "local":
            return no("route is local")
        if not self._enabled:
            return no("companion feature is off")
        if self._force_local:
            return no("force-local is on")
        counter = self._counters.get(workload)
        if counter is not None and counter.paused_until > moment:
            return no("workload is paused by the failure circuit breaker")

        snapshot = self.sessions.snapshot(now=moment)
        if not snapshot.connected:
            return no("no companion session")
        if "companion" not in snapshot.roles:
            return no("session does not hold the companion role")
        if workload not in snapshot.workloads:
            return no(f"companion does not advertise {workload}")
        if 1 not in snapshot.schema_versions:
            return no("companion does not support the current schema version")
        if snapshot.telemetry_age_seconds is None:
            return no("no telemetry received")
        if tier_rank(snapshot.tier) < tier_rank(route.min_tier):
            # The tier alone does not explain itself. "tier off is below
            # reduced" is true and useless: off because the battery is flat,
            # because the device is hot, and because its telemetry stopped
            # arriving are three different problems with three different
            # fixes, and the operator reading this is trying to tell them
            # apart. Carrying the tier's own reason through is the difference
            # between a status line and a diagnosis.
            return no(
                f"tier {snapshot.tier.value} ({snapshot.tier_reason}) "
                f"is below {route.min_tier.value}"
            )
        if not self._locality_allows(route.locality, snapshot.locality):
            return no(f"connected via {snapshot.locality}, route needs {route.locality}")
        return Eligibility(True, "eligible", route)

    @staticmethod
    def _locality_allows(required: RouteLocality, actual: Locality) -> bool:
        if required == "home_lan":
            return actual == "home_lan"
        return actual in ("home_lan", "tailnet")

    # -- dispatch -------------------------------------------------------------

    async def run(
        self,
        workload: CompanionWorkload,
        payload: dict,
        local: Callable[[], Awaitable[Any]] | None,
        *,
        parse: Callable[[dict], Any] | None = None,
        idempotency_key: str | None = None,
        trace_id: str | None = None,
        allowed_tools: frozenset[str] | None = None,
    ) -> RouteResult:
        """Run one workload, honouring its route and the no-eager-hedge rule.

        ``local`` is a *factory*, not a coroutine, precisely so it is not
        started until it is actually needed. Handing in an already-created
        coroutine would make the eager hedge impossible to avoid.
        """

        started = time.perf_counter()
        route = self.route(workload)
        decision = self.eligibility(workload)
        counter = self._counters.setdefault(workload, WorkloadCounters())

        def elapsed() -> float:
            return round((time.perf_counter() - started) * 1000, 1)

        timing = self._timing(workload)
        # Whether the phone was actually asked. Distinguishes "fell back after
        # trying" from "was never eligible in the first place", which cost
        # nothing and must not be reported as waste.
        attempted = False

        if route.mode == "both" and local is not None and decision.eligible:
            return await self._run_both(
                workload,
                payload,
                local,
                parse=parse,
                idempotency_key=idempotency_key,
                trace_id=trace_id,
                allowed_tools=allowed_tools,
                route=route,
                counter=counter,
                started=started,
            )

        async def run_local() -> tuple[Any, float]:
            """The local arm, timed on its own.

            Separate from the turn's total on purpose: a fallback's total
            contains the failed companion attempt, and charging that to the
            local arm would make the voice server look slower precisely when
            the phone had let it down.
            """

            local_started = time.perf_counter()
            value = await local()  # type: ignore[misc]
            local_ms = round((time.perf_counter() - local_started) * 1000, 1)
            timing.add("local", local_ms)
            _record_turn_route(workload, "local", local_ms)
            return value, local_ms

        async def fall_back(reason: str) -> RouteResult:
            if route.mode == "disabled":
                # An optional workload that is switched off does not run at
                # all — not on the companion, and not locally either.
                return RouteResult(workload, "none", None, reason, elapsed())
            if route.mode == "companion_only":
                # Explicitly unavailable rather than silently different
                # semantics: the caller asked for the companion specifically.
                return RouteResult(workload, "none", None, reason, elapsed())
            if local is None:
                return RouteResult(workload, "none", None, reason, elapsed())
            counter.fell_back += 1
            total_before_local = elapsed()
            value, local_ms = await run_local()
            if attempted:
                # Everything spent before the local run began: the offer, the
                # wait, the timeout. This is what trying the phone cost when it
                # did not work out, and it is the number that decides whether
                # `companion_preferred` earns its place.
                #
                # Gated on an offer having actually been made. A pass routed
                # `local` still passes through here, and recording its
                # sub-millisecond bookkeeping as "wasted on a failed try" would
                # put a number in front of an operator that describes nothing.
                timing.add("fallback_overhead", total_before_local)
            return RouteResult(workload, "local", value, reason, elapsed())

        if not decision.eligible:
            return await fall_back(decision.reason)

        input_revision = _revision(payload)
        offered = await self.sessions.offer(
            workload=workload,
            payload=payload,
            idempotency_key=idempotency_key or input_revision,
            input_revision=input_revision,
            result_schema=WORKLOADS[workload].result_schema,
            trace_id=trace_id or uuid.uuid4().hex,
            complete_deadline_seconds=route.deadline_seconds,
            sensitivity=route.sensitivity,
            allowed_tools=allowed_tools,
            # A callback must not be able to outlive the turn that asked for
            # it. Both ceilings are derived from the workload's own deadline so
            # a hot-path pass cannot spend its whole budget waiting on tools
            # and leave nothing for the answer or for local fallback.
            callback_deadline_seconds=max(1.0, route.deadline_seconds / 3),
            callback_budget_seconds=max(1.0, route.deadline_seconds * 0.75),
        )
        if offered is None:
            return await fall_back("companion session disappeared before the offer")
        session, attempt, outcome = offered
        counter.offered += 1
        attempted = True

        if not outcome.accepted:
            # A rejection is ordinary flow. It does not count against the
            # failure budget, because a phone declining on battery grounds is
            # the system working as intended.
            counter.rejected += 1
            return await fall_back(f"companion declined: {outcome.reason}")

        counter.accepted += 1
        # From here the companion owns the attempt and nothing local runs.
        # Running both is the `both` route mode's job, handled above, so this
        # path keeps the no-eager-hedge rule without qualification.
        companion_started = time.perf_counter()
        result = await self.sessions.await_result(
            session, attempt, timeout_seconds=route.deadline_seconds
        )
        companion_ms = round((time.perf_counter() - companion_started) * 1000, 1)
        if result.ok and result.result is not None:
            value = result.result
            if parse is not None:
                try:
                    value = parse(result.result)
                except (ValueError, TypeError, KeyError):
                    self._record_failure(counter)
                    return await fall_back("companion returned an invalid result")
                if value is None:
                    self._record_failure(counter)
                    return await fall_back("companion returned an unusable result")
            counter.completed += 1
            counter.consecutive_failures = 0
            timing.add("companion", companion_ms)
            _record_turn_route(workload, "companion", companion_ms)
            return RouteResult(
                workload, "companion", value, "companion completed", elapsed()
            )

        self._record_failure(counter)
        return await fall_back(f"companion {result.failure}: {result.detail or ''}".strip())

    async def _run_both(
        self,
        workload: CompanionWorkload,
        payload: dict,
        local: Callable[[], Awaitable[Any]],
        *,
        parse: Callable[[dict], Any] | None,
        idempotency_key: str | None,
        trace_id: str | None,
        allowed_tools: frozenset[str] | None,
        route: WorkloadRoute,
        counter: WorkloadCounters,
        started: float,
    ) -> RouteResult:
        """Run the pass on both stacks, answer with the first, keep both.

        **Both arms are awaited before the turn completes.** The first to
        succeed supplies the answer, so the reply is the faster model's — but
        the turn does not finish until the other lands, because a comparison
        you can only half see is not a comparison. In an evaluation mode that
        has already given up the free slot, spending the difference in
        wall-clock to get complete, trustworthy data is the right trade; a
        transcript that sometimes showed one arm and sometimes two would be
        actively misleading for the thing this exists to answer.
        """

        timing = self._timing(workload)

        def elapsed() -> float:
            return round((time.perf_counter() - started) * 1000, 1)

        async def local_arm() -> tuple[Any, float]:
            begin = time.perf_counter()
            value = await local()
            ms = round((time.perf_counter() - begin) * 1000, 1)
            timing.add("local", ms)
            _record_turn_route(workload, "local", ms)
            return value, ms

        async def companion_arm() -> tuple[Any, float]:
            begin = time.perf_counter()
            offered = await self.sessions.offer(
                workload=workload,
                payload=payload,
                idempotency_key=idempotency_key or _revision(payload),
                input_revision=_revision(payload),
                result_schema=WORKLOADS[workload].result_schema,
                trace_id=trace_id or uuid.uuid4().hex,
                complete_deadline_seconds=route.deadline_seconds,
                sensitivity=route.sensitivity,
                allowed_tools=allowed_tools,
                callback_deadline_seconds=max(1.0, route.deadline_seconds / 3),
                callback_budget_seconds=max(1.0, route.deadline_seconds * 0.75),
            )
            if offered is None:
                raise _ArmUnavailable("companion session disappeared before the offer")
            session, attempt, outcome = offered
            counter.offered += 1
            if not outcome.accepted:
                counter.rejected += 1
                raise _ArmUnavailable(f"companion declined: {outcome.reason}")
            counter.accepted += 1
            result = await self.sessions.await_result(
                session, attempt, timeout_seconds=route.deadline_seconds
            )
            if not (result.ok and result.result is not None):
                self._record_failure(counter)
                raise _ArmUnavailable(f"companion {result.failure}: {result.detail or ''}".strip())
            value = result.result
            if parse is not None:
                try:
                    value = parse(result.result)
                except (ValueError, TypeError, KeyError):
                    self._record_failure(counter)
                    raise _ArmUnavailable("companion returned an invalid result") from None
                if value is None:
                    self._record_failure(counter)
                    raise _ArmUnavailable("companion returned an unusable result")
            counter.completed += 1
            counter.consecutive_failures = 0
            ms = round((time.perf_counter() - begin) * 1000, 1)
            timing.add("companion", ms)
            _record_turn_route(workload, "companion", ms)
            return value, ms

        arms: dict[asyncio.Task, str] = {
            asyncio.create_task(local_arm()): "local",
            asyncio.create_task(companion_arm()): "companion",
        }
        outcomes: dict[str, tuple[Any, float]] = {}
        winner: str | None = None

        pending = set(arms)
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                arm = arms[task]
                try:
                    outcomes[arm] = task.result()
                except _ArmUnavailable as unavailable:
                    # The phone declining or failing is ordinary. The local arm
                    # raising is not, and is left to propagate below.
                    logger.info("companion arm unavailable for %s: %s", workload, unavailable)
                    continue
                if winner is None:
                    winner = arm

        if winner is None:
            # Both arms failed. The local one raising is a real fault, so let
            # it surface the way it would on any other route rather than being
            # flattened into "no companion".
            for task, arm in arms.items():
                if arm == "local" and task.exception() is not None:
                    raise task.exception()  # type: ignore[misc]
            return RouteResult(workload, "none", None, "both arms failed", elapsed())

        self._record_comparison(
            RouteComparison(
                workload=workload,
                at=datetime.now(UTC).isoformat(),
                companion_text=_readable(outcomes.get("companion", (None, 0))[0]),
                local_text=_readable(outcomes.get("local", (None, 0))[0]),
                companion_ms=outcomes.get("companion", (None, None))[1],
                local_ms=outcomes.get("local", (None, None))[1],
                spoken=winner,  # type: ignore[arg-type]
            )
        )
        value, _ = outcomes[winner]
        if winner == "local":
            counter.fell_back += 1
        return RouteResult(
            workload, winner, value, f"both ran; {winner} answered first", elapsed()
        )



    def _record_failure(self, counter: WorkloadCounters) -> None:
        counter.failed += 1
        counter.consecutive_failures += 1
        if counter.consecutive_failures >= self._failure_budget:
            # A failing phone must not add its timeout to every single turn.
            counter.paused_until = time.monotonic() + self._breaker_pause
            counter.consecutive_failures = 0
            logger.warning(
                "companion workload paused after repeated failures for %.0fs",
                self._breaker_pause,
            )


# Where each pass of the *current turn* actually ran.
#
# A ContextVar rather than an argument threaded through the interpreter,
# because the router is three layers below the thing that wants the answer and
# the passes are called from different places. It is per-task and inherited by
# child tasks, which is exactly the shape of a turn.
#
# Unset outside a turn, and appending to nothing is a no-op — so a background
# pass that runs outside any turn is simply not attributed to one.
_TURN_ROUTES: ContextVar[list[dict] | None] = ContextVar("nova_turn_routes", default=None)


@contextmanager
def turn_route_log() -> Iterator[list[dict]]:
    """Collect where every pass ran, for the duration of one turn."""

    entries: list[dict] = []
    token = _TURN_ROUTES.set(entries)
    try:
        yield entries
    finally:
        _TURN_ROUTES.reset(token)


def _record_turn_route(workload: str, source: str, elapsed_ms: float) -> None:
    entries = _TURN_ROUTES.get()
    if entries is None:
        return
    # One entry per arm that actually executed. A turn where both ran shows up
    # as two entries for the same pass, which is the point: "was this processed
    # twice" should be readable rather than inferred.
    entries.append({"pass": workload, "source": source, "ms": round(elapsed_ms, 1)})


def _readable(value: Any) -> str | None:
    """The answer as a person would read it, for a side-by-side comparison.

    Every routable pass returns something different — a rendered string, a
    parsed `Interpretation`, an icon choice — and the point of the comparison
    is that a human can look at the two and say which is better. So this
    reaches for the human-facing part first and falls back to a bounded repr
    rather than dumping a model.
    """

    if value is None:
        return None
    for attribute in ("text", "icon", "summary"):
        held = getattr(value, attribute, None)
        if isinstance(held, str) and held.strip():
            return held[:500]
    if isinstance(value, str):
        return value[:500]
    if isinstance(value, dict):
        for key in ("text", "icon", "summary"):
            held = value.get(key)
            if isinstance(held, str) and held.strip():
                return held[:500]
    return repr(value)[:500]


def _revision(payload: dict) -> str:
    """Stable hash of a request, used as input revision and idempotency key."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:32]
