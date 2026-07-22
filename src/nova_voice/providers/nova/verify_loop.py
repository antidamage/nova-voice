"""Bounded, LLM-confirmed device-verification loop ("the wiggum loop").

This module is deliberately independent of ``NovaProvider``/``service.py``: it
takes a small set of tasks and callables and knows nothing about the
dashboard client, aliases, or the interpretation pipeline. That keeps it
directly testable with fake pollers/confirmers and reusable by any capability
provider that needs to wait for a mutation to become observable.

Design, in one paragraph: a mutation has already been issued once by the
caller (this module never re-issues a command — devices here are assumed
merely slow to report, not lossy). ``run`` re-reads state on a bounded
schedule, applying a cheap deterministic check on every read. When that check
is not yet satisfied, a throttled JSON-only LLM pass may be asked whether the
observed (possibly still-settling) state actually satisfies the turn's
objective; its verdict is authoritative. If the loop runs long, a one-shot,
turn-scoped "thinking" callback lets a human monitoring the transcript know
verification is still in flight. Group (multi-item) results accumulate into
one ``GroupVerificationResult``.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import logging
import math
import time
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from nova_voice.domain import VerificationVerdict

logger = logging.getLogger(__name__)

PollFn = Callable[[], Awaitable[dict[str, Any]]]
ThinkingFn = Callable[[], Awaitable[None]]
LlmConfirmFn = Callable[
    [list["VerificationTask"], dict[str, Any], list["VerificationItemResult"]],
    Awaitable[VerificationVerdict | None],
]


@dataclass(frozen=True)
class VerificationTask:
    """One device's task + objective to confirm.

    ``verify`` is the existing deterministic state check (e.g. brightness
    within tolerance); ``objective`` is a short human-readable description of
    what "done" means, handed to the LLM confirmation pass so it can reason
    about the observed state without seeing raw provider internals.
    """

    action_id: str
    label: str
    objective: str
    verify: Callable[[dict[str, Any]], tuple[bool, dict[str, Any] | None]]


@dataclass
class VerificationItemResult:
    action_id: str
    target: str
    confirmed: bool = False
    observed: dict[str, Any] | None = None
    reason: str = "pending"
    attempts: int = 0


@dataclass(frozen=True)
class GroupVerificationResult:
    items: list[VerificationItemResult]
    all_confirmed: bool
    iterations: int
    elapsed_ms: float
    final_state: dict[str, Any]

    def item(self, action_id: str) -> VerificationItemResult | None:
        return next((entry for entry in self.items if entry.action_id == action_id), None)


@dataclass(frozen=True)
class VerificationLoopConfig:
    enabled: bool = True
    max_iterations: int = 20
    sleep_seconds: float = 0.5
    failure_seconds: float = 8.0
    thinking_threshold_seconds: float = 2.5
    llm_verify_enabled: bool = True
    llm_verify_min_interval_seconds: float = 1.5
    # Hard cutoff for a single LLM confirmation call. Without this, a slow or
    # hanging LLM backend is bounded only by the interpreter's own HTTP client
    # timeout (tens of seconds), silently defeating failure_seconds. The final,
    # end-of-loop confirmation call always gets this full budget; in-loop calls
    # get whatever is smaller (this budget, or the time left before deadline).
    llm_confirm_timeout_seconds: float = 3.0


@dataclass
class TurnVerificationContext:
    """Shared, per-turn state so multiple sequential device loops in the same
    turn behave as one: only one "*Thinking*" marker and one shared LLM-call
    throttle clock, no matter how many items are verified in sequence.
    """

    thinking: ThinkingFn | None = None
    llm_confirm: LlmConfirmFn | None = None
    thinking_emitted: bool = False
    llm_last_call_monotonic: float = field(default=-math.inf)


_current_turn_context: contextvars.ContextVar[TurnVerificationContext | None] = (
    contextvars.ContextVar("nova_verification_turn_context", default=None)
)


@contextlib.contextmanager
def turn_scope(
    *,
    thinking: ThinkingFn | None = None,
    llm_confirm: LlmConfirmFn | None = None,
) -> Iterator[TurnVerificationContext]:
    """Scope one voice turn's shared verification state via a contextvar.

    Concurrent turns in different rooms are separate asyncio tasks, so this
    never leaks between them the way an instance attribute on the (singleton)
    provider would.
    """

    context = TurnVerificationContext(thinking=thinking, llm_confirm=llm_confirm)
    token = _current_turn_context.set(context)
    try:
        yield context
    finally:
        _current_turn_context.reset(token)


def current_turn_context() -> TurnVerificationContext | None:
    return _current_turn_context.get()


async def run(
    tasks: list[VerificationTask],
    initial_state: dict[str, Any],
    *,
    poll: PollFn,
    config: VerificationLoopConfig,
    context: TurnVerificationContext | None = None,
) -> GroupVerificationResult:
    """Bounded wiggum loop: repeat reads, never the mutation that preceded them.

    The immediate (possibly stale) state is checked first. If any task is not
    yet satisfied, the authoritative state is re-read on a bounded schedule
    until every task confirms, the iteration budget is exhausted, or the
    wall-clock deadline wins. Transient poll errors are tolerated inside those
    bounds.
    """

    ctx = context if context is not None else TurnVerificationContext()
    started = time.monotonic()
    items: dict[str, VerificationItemResult] = {
        task.action_id: VerificationItemResult(action_id=task.action_id, target=task.label)
        for task in tasks
    }

    def pending_tasks() -> list[VerificationTask]:
        return [task for task in tasks if not items[task.action_id].confirmed]

    def all_confirmed() -> bool:
        return all(item.confirmed for item in items.values())

    def deterministic_pass(state: dict[str, Any]) -> None:
        for task in pending_tasks():
            item = items[task.action_id]
            item.attempts += 1
            ok, observed = task.verify(state)
            item.observed = observed
            if ok:
                item.confirmed = True
                item.reason = "deterministic state match"

    async def llm_pass(state: dict[str, Any], *, force: bool = False) -> None:
        if not config.llm_verify_enabled or ctx.llm_confirm is None:
            return
        remaining = pending_tasks()
        if not remaining:
            return
        now = time.monotonic()
        since_last_call = now - ctx.llm_last_call_monotonic
        if not force and since_last_call < config.llm_verify_min_interval_seconds:
            return
        ctx.llm_last_call_monotonic = now
        # Hard cutoff: bound this one call so a slow/hanging LLM backend can
        # never make the loop run past failure_seconds by more than this fixed
        # budget, regardless of the interpreter's own (much larger) client
        # timeout. An in-loop call is further capped by whatever time remains
        # before the loop's own deadline; the final forced call always gets
        # the full budget since the deadline has already passed by then.
        call_budget = config.llm_confirm_timeout_seconds
        if not force:
            time_left = deadline - time.monotonic()
            call_budget = min(call_budget, max(time_left, 0.05))
        try:
            verdict = await asyncio.wait_for(
                ctx.llm_confirm(remaining, state, list(items.values())), timeout=call_budget
            )
        except TimeoutError:
            logger.warning("LLM verification confirm timed out after %.1fs", call_budget)
            return
        except Exception:
            logger.warning("LLM verification confirm failed", exc_info=True)
            return
        if verdict is None:
            return
        by_target = {entry.target: entry for entry in verdict.items}
        for task in remaining:
            entry = by_target.get(task.label)
            if entry is None:
                continue
            item = items[task.action_id]
            item.reason = entry.reason
            if entry.confirmed:
                item.confirmed = True

    async def maybe_think() -> None:
        if ctx.thinking is None or ctx.thinking_emitted:
            return
        if (time.monotonic() - started) < config.thinking_threshold_seconds:
            return
        ctx.thinking_emitted = True
        try:
            await ctx.thinking()
        except Exception:
            logger.warning("thinking marker emission failed", exc_info=True)

    def finish(iterations: int) -> GroupVerificationResult:
        return GroupVerificationResult(
            items=list(items.values()),
            all_confirmed=all_confirmed(),
            iterations=iterations,
            elapsed_ms=round((time.monotonic() - started) * 1000, 3),
            final_state=state,
        )

    state = initial_state
    deterministic_pass(state)
    if all_confirmed() or not config.enabled:
        return finish(0)

    deadline = started + config.failure_seconds
    iteration = 0
    while iteration < config.max_iterations:
        iteration += 1
        remaining_time = deadline - time.monotonic()
        if remaining_time <= 0:
            break
        if config.sleep_seconds:
            # Preserve time for the actual poll even when someone sets a pause
            # longer than the whole deadline.
            await asyncio.sleep(min(config.sleep_seconds, remaining_time / 2))
        remaining_time = deadline - time.monotonic()
        if remaining_time <= 0:
            break
        try:
            state = await asyncio.wait_for(poll(), timeout=remaining_time)
        except TimeoutError:
            break
        except Exception:
            logger.warning("verification poll failed; retrying", exc_info=True)
            await maybe_think()
            continue
        deterministic_pass(state)
        if not all_confirmed():
            await llm_pass(state)
        await maybe_think()
        if all_confirmed():
            break

    if not all_confirmed():
        await llm_pass(state, force=True)

    return finish(iteration)
