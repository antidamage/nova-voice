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

import hashlib
import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Any, Literal

from nova_voice.companion.protocol import CompanionWorkload, Locality, Sensitivity
from nova_voice.companion.session import CompanionSessionManager
from nova_voice.companion.tiers import CompanionTier, tier_rank
from nova_voice.companion.workloads import WORKLOADS

logger = logging.getLogger(__name__)

RouteMode = Literal[
    "local", "companion_preferred", "companion_only", "companion_fallback", "disabled"
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
            return no(f"tier {snapshot.tier.value} is below {route.min_tier.value}")
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
            value = await local()
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
        )
        if offered is None:
            return await fall_back("companion session disappeared before the offer")
        session, attempt, outcome = offered
        counter.offered += 1

        if not outcome.accepted:
            # A rejection is ordinary flow. It does not count against the
            # failure budget, because a phone declining on battery grounds is
            # the system working as intended.
            counter.rejected += 1
            return await fall_back(f"companion declined: {outcome.reason}")

        counter.accepted += 1
        # From here the companion owns the attempt. Nothing local runs.
        result = await self.sessions.await_result(
            session, attempt, timeout_seconds=route.deadline_seconds
        )
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
            return RouteResult(workload, "companion", value, "companion completed", elapsed())

        self._record_failure(counter)
        return await fall_back(f"companion {result.failure}: {result.detail or ''}".strip())

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


def _revision(payload: dict) -> str:
    """Stable hash of a request, used as input revision and idempotency key."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:32]
