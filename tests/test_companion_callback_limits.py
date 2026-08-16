"""What a companion may ask Iridium to do, and what it may spend doing it.

NPT-308 and NPT-309. The tool callback loop is the one place the phone reaches
back into the house, so every bound on it is tested from the outside: the
catalogue it was given, the number of calls, how long one may take, how long
they may take in total, how many may run at once, and how many bytes may cross
in each direction. Cancellation is tested here too, because an outstanding
callback is exactly the thing a cancelled turn leaves behind.

These drive the real session manager rather than the reference peer: the gates
live in the manager, and asserting on them directly says which gate fired
instead of only that something was refused.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta

import pytest

from nova_voice.companion.auth import AuthenticatedIdentity
from nova_voice.companion.protocol import (
    PROTOCOL_VERSION,
    CompanionHello,
    CompanionTelemetry,
    ModelAvailability,
    ToolCall,
    ToolResultMessage,
)
from nova_voice.companion.session import CompanionSessionManager


class _Wire:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send(self, payload: dict) -> None:
        self.sent.append(payload)

    def of_type(self, kind: str) -> list[dict]:
        return [message for message in self.sent if message.get("type") == kind]


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
        displayName="Reference Companion",
        roles=["companion"],
        appVersion="test",
        osVersion="test",
        workloads=["classify_icon"],
        telemetry=CompanionTelemetry(
            battery=1.0,
            charging=True,
            models=ModelAvailability(hotAvailable=True, hotContextTokens=4096),
        ),
    )


async def _accepted(sessions: CompanionSessionManager, wire: _Wire, **offer_kwargs):
    """Get an accepted attempt without going through a socket.

    The offer blocks until the companion answers, so acceptance is delivered
    from this side once the offer frame has been written.
    """

    session = sessions.register(
        identity=_identity(), hello=_hello(), locality="home_lan", send=wire.send
    )
    task = asyncio.create_task(
        sessions.offer(
            workload="classify_icon",
            payload={},
            idempotency_key="key",
            input_revision="revision",
            result_schema="icon_choice",
            trace_id="trace",
            complete_deadline_seconds=30.0,
            **offer_kwargs,
        )
    )
    for _ in range(50):
        await asyncio.sleep(0)
        if wire.of_type("job_offer"):
            break
    offer = wire.of_type("job_offer")[0]
    sessions.handle_accept(session, offer["envelope"]["jobId"], offer["envelope"]["attemptId"])
    outcome = await task
    assert outcome is not None
    return session, outcome[1], offer


def _call(attempt, *, provider="nova", tool="light_set", arguments=None, call_id="c1") -> ToolCall:
    return ToolCall(
        jobId=attempt.job_id,
        attemptId=attempt.attempt_id,
        callId=call_id,
        provider=provider,
        tool=tool,
        arguments=arguments or {},
    )


async def _settle(rounds: int = 60) -> None:
    for _ in range(rounds):
        await asyncio.sleep(0)


def _ok(call: ToolCall, observed=None) -> ToolResultMessage:
    return ToolResultMessage(
        jobId=call.job_id,
        attemptId=call.attempt_id,
        callId=call.call_id,
        ok=True,
        code="ok",
        message="done",
        observed=observed,
    )


@pytest.fixture
def executed() -> list[str]:
    return []


# -- the catalogue ------------------------------------------------------------


async def test_the_offer_names_the_only_tools_the_phone_may_call():
    """The phone is told the catalogue rather than left to guess it."""

    async def execute(call):
        return _ok(call)

    sessions = CompanionSessionManager(execute_tool_call=execute)
    wire = _Wire()
    _, _, offer = await _accepted(
        sessions, wire, allowed_tools=frozenset({"nova.light_set", "nova.climate_set"})
    )

    assert offer["toolCatalogue"] == ["nova.climate_set", "nova.light_set"]
    assert offer["callbackBudget"] > 0


async def test_a_tool_outside_the_catalogue_never_reaches_the_registry(executed):
    """An unadvertised tool is refused by name, before anything runs."""

    async def execute(call):
        executed.append(call.tool)
        return _ok(call)

    sessions = CompanionSessionManager(execute_tool_call=execute)
    wire = _Wire()
    _, attempt, _ = await _accepted(sessions, wire, allowed_tools=frozenset({"nova.light_set"}))

    sessions.dispatch_tool_call(_session(sessions), _call(attempt, tool="unlock_door"))
    await _settle()

    result = wire.of_type("tool_result")[-1]
    assert result["ok"] is False
    assert result["code"] == "blocked"
    assert result["message"] == "tool was not offered with this job"
    # The gate is the point: execution was never attempted.
    assert executed == []


async def test_an_empty_catalogue_means_no_callbacks_at_all(executed):
    """Most workloads are one typed question. They get no tool loop."""

    async def execute(call):
        executed.append(call.tool)
        return _ok(call)

    sessions = CompanionSessionManager(execute_tool_call=execute)
    wire = _Wire()
    _, attempt, offer = await _accepted(sessions, wire)

    assert offer["toolCatalogue"] == []
    assert offer["callbackBudget"] == 0

    sessions.dispatch_tool_call(_session(sessions), _call(attempt))
    await _settle()

    assert wire.of_type("tool_result")[-1]["code"] == "blocked"
    assert executed == []


async def test_a_bare_catalogue_name_matches_a_qualified_call(executed):
    """Catalogues name some tools bare; the two forms must not disagree."""

    async def execute(call):
        executed.append(call.tool)
        return _ok(call)

    sessions = CompanionSessionManager(execute_tool_call=execute)
    wire = _Wire()
    _, attempt, _ = await _accepted(sessions, wire, allowed_tools=frozenset({"light_set"}))

    sessions.dispatch_tool_call(
        _session(sessions), _call(attempt, provider="nova", tool="light_set")
    )
    await _settle()

    assert wire.of_type("tool_result")[-1]["ok"] is True
    assert executed == ["light_set"]


# -- the resource bounds ------------------------------------------------------


async def test_a_slow_tool_is_cut_off_at_the_callback_deadline():
    """One hanging provider must not hold a voice turn open."""

    async def execute(call):
        await asyncio.sleep(5)
        return _ok(call)

    sessions = CompanionSessionManager(execute_tool_call=execute)
    wire = _Wire()
    _, attempt, _ = await _accepted(
        sessions,
        wire,
        allowed_tools=frozenset({"nova.light_set"}),
        callback_deadline_seconds=0.05,
    )

    sessions.dispatch_tool_call(_session(sessions), _call(attempt))
    await asyncio.sleep(0.2)

    result = wire.of_type("tool_result")[-1]
    assert result["ok"] is False
    assert result["code"] == "timeout"


async def test_the_cumulative_budget_stops_a_chain_of_slow_calls(executed):
    """Count alone is not a bound: twelve slow calls are twelve times the wait."""

    async def execute(call):
        executed.append(call.call_id)
        await asyncio.sleep(0.2)
        return _ok(call)

    sessions = CompanionSessionManager(execute_tool_call=execute)
    wire = _Wire()
    # One call is allowed to take four times the whole budget, so the budget is
    # unambiguously spent afterwards. Numbers close enough to be decided by
    # clock granularity would make this a coin flip on Windows, where
    # `monotonic()` ticks about every 16ms.
    _, attempt, _ = await _accepted(
        sessions,
        wire,
        allowed_tools=frozenset({"nova.light_set"}),
        callback_deadline_seconds=1.0,
        callback_budget_seconds=0.05,
    )

    sessions.dispatch_tool_call(_session(sessions), _call(attempt, call_id="first"))
    await asyncio.sleep(0.4)
    sessions.dispatch_tool_call(_session(sessions), _call(attempt, call_id="second"))
    await _settle()

    assert executed == ["first"]
    refusal = wire.of_type("tool_result")[-1]
    assert refusal["code"] == "blocked"
    assert refusal["message"] == "callback time budget exhausted"


async def test_too_many_callbacks_at_once_are_refused():
    """Concurrency is capped so one job cannot fan out across the house."""

    gate = asyncio.Event()

    async def execute(call):
        await gate.wait()
        return _ok(call)

    sessions = CompanionSessionManager(execute_tool_call=execute)
    wire = _Wire()
    _, attempt, _ = await _accepted(
        sessions,
        wire,
        allowed_tools=frozenset({"nova.light_set"}),
        max_concurrent_callbacks=1,
    )

    sessions.dispatch_tool_call(_session(sessions), _call(attempt, call_id="first"))
    await _settle()
    sessions.dispatch_tool_call(_session(sessions), _call(attempt, call_id="second"))
    await _settle()

    refusal = wire.of_type("tool_result")[-1]
    assert refusal["callId"] == "second"
    assert refusal["message"] == "too many callbacks in flight"

    gate.set()
    await _settle()


async def test_oversized_arguments_never_reach_the_registry(executed):
    """A payload bomb is refused at the size check, not parsed downstream."""

    async def execute(call):
        executed.append(call.call_id)
        return _ok(call)

    sessions = CompanionSessionManager(execute_tool_call=execute, callback_argument_bytes=256)
    wire = _Wire()
    _, attempt, _ = await _accepted(sessions, wire, allowed_tools=frozenset({"nova.light_set"}))

    sessions.dispatch_tool_call(
        _session(sessions), _call(attempt, arguments={"blob": "x" * 4000})
    )
    await _settle()

    result = wire.of_type("tool_result")[-1]
    assert result["ok"] is False
    assert result["code"] == "invalid"
    assert executed == []


async def test_an_oversized_result_is_truncated_rather_than_dropped():
    """The action succeeded. Losing that would invite the phone to retry it."""

    async def execute(call):
        return _ok(call, observed={"rows": ["y" * 100 for _ in range(200)]})

    sessions = CompanionSessionManager(execute_tool_call=execute, callback_result_bytes=512)
    wire = _Wire()
    _, attempt, _ = await _accepted(sessions, wire, allowed_tools=frozenset({"nova.light_set"}))

    sessions.dispatch_tool_call(_session(sessions), _call(attempt))
    await _settle()

    result = wire.of_type("tool_result")[-1]
    assert result["ok"] is True
    assert result["observed"] == {
        "truncated": True,
        "reason": "result exceeded the callback size limit",
    }
    assert len(json.dumps(result)) < 2000


# -- NPT-309: cancellation and turn ownership ---------------------------------


async def test_cancelling_a_turn_cancels_its_outstanding_callbacks():
    """A superseded turn leaves no tool work running and no future pending."""

    started = asyncio.Event()
    finished: list[str] = []

    async def execute(call):
        started.set()
        await asyncio.sleep(5)
        finished.append(call.call_id)
        return _ok(call)

    sessions = CompanionSessionManager(execute_tool_call=execute)
    wire = _Wire()
    session, attempt, _ = await _accepted(
        sessions, wire, allowed_tools=frozenset({"nova.light_set"})
    )

    sessions.dispatch_tool_call(session, _call(attempt))
    await started.wait()
    assert attempt.callbacks_in_flight == 1

    await sessions.cancel(session, attempt, "superseded")
    await _settle()

    assert finished == []
    assert attempt.tool_tasks == set()
    # A cancelled callback is not reported as a failure: the phone is being
    # told to stop the whole attempt on the same socket.
    assert wire.of_type("tool_result") == []
    assert wire.of_type("job_cancel")[-1]["reason"] == "superseded"
    assert attempt.result.result().failure == "cancelled"


async def test_a_callback_arriving_after_cancellation_is_refused(executed):
    """The frame in flight when the turn died must not execute."""

    async def execute(call):
        executed.append(call.call_id)
        return _ok(call)

    sessions = CompanionSessionManager(execute_tool_call=execute)
    wire = _Wire()
    session, attempt, _ = await _accepted(
        sessions, wire, allowed_tools=frozenset({"nova.light_set"})
    )

    await sessions.cancel(session, attempt, "user_cancelled")
    sessions.dispatch_tool_call(session, _call(attempt))
    await _settle()

    assert executed == []
    assert wire.of_type("tool_result")[-1]["code"] == "invalid"


async def test_a_disconnect_cancels_callbacks_as_well_as_jobs():
    """Losing the phone must not leave its tool work running against the house."""

    started = asyncio.Event()
    finished: list[str] = []

    async def execute(call):
        started.set()
        await asyncio.sleep(5)
        finished.append(call.call_id)
        return _ok(call)

    sessions = CompanionSessionManager(execute_tool_call=execute)
    wire = _Wire()
    session, attempt, _ = await _accepted(
        sessions, wire, allowed_tools=frozenset({"nova.light_set"})
    )

    sessions.dispatch_tool_call(session, _call(attempt))
    await started.wait()

    sessions.release(session)
    await _settle()

    assert finished == []
    assert attempt.result.result().failure == "disconnected"


def _session(sessions: CompanionSessionManager):
    current = sessions.current
    assert current is not None
    return current
