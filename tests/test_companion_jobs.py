"""Offering work to a companion and getting it back, end to end.

The socket tests cover the transport; these cover what the transport is *for*.
Both sides run in one event loop over a pair of queues, so the real session
manager, the real router, the real dispatcher and the real reference peer all
take part — the only thing standing in is the WebSocket itself.

The load-bearing assertion is the no-eager-hedge one. Once the peer accepts,
Iridium's local path must not run: executing both would keep the ``--parallel
1`` slot occupied and defeat the entire point of offloading.
"""

from __future__ import annotations

import asyncio
import json
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
    ToolResultMessage,
    parse_client_message,
)
from nova_voice.companion.reference import ReferenceCompanion
from nova_voice.companion.router import CompanionWorkloadRouter
from nova_voice.companion.session import CompanionSessionManager


class _Wire:
    """Two queues standing in for the socket, so both sides share one loop."""

    def __init__(self) -> None:
        self.to_peer: asyncio.Queue[str] = asyncio.Queue()
        self.to_server: asyncio.Queue[str] = asyncio.Queue()

    async def server_send(self, payload: dict) -> None:
        await self.to_peer.put(json.dumps(payload))

    async def peer_send(self, text: str) -> None:
        await self.to_server.put(text)

    async def peer_receive(self) -> str:
        return await self.to_peer.get()


def _hello(*workloads: str) -> CompanionHello:
    return CompanionHello(
        protocolVersion=PROTOCOL_VERSION,
        displayName="Reference Companion",
        roles=["companion"],
        appVersion="test",
        osVersion="test",
        workloads=list(workloads),
        telemetry=CompanionTelemetry(
            battery=1.0,
            charging=True,
            models=ModelAvailability(hotAvailable=True, hotContextTokens=4096),
        ),
    )


def _identity() -> AuthenticatedIdentity:
    return AuthenticatedIdentity(
        identity="companion-1",
        roles=("companion",),
        not_after=datetime.now(UTC) + timedelta(days=1),
        fingerprint="test",
    )


async def _connect(sessions, wire, *workloads, execute_tool_call=None):
    if execute_tool_call is not None:
        sessions.bind_tool_executor(execute_tool_call)
    session = sessions.register(
        identity=_identity(),
        hello=_hello(*workloads),
        locality="home_lan",
        send=wire.server_send,
    )

    async def pump() -> None:
        while True:
            raw = await wire.to_server.get()
            dispatch_companion_message(sessions, session, parse_client_message(raw))

    return session, asyncio.create_task(pump())


def _peer(wire, key, certificate_pem, **overrides) -> ReferenceCompanion:
    return ReferenceCompanion(
        announced_id="companion-1",
        private_key=key,
        certificate_pem=certificate_pem,
        send=wire.peer_send,
        receive=wire.peer_receive,
        **overrides,
    )


@pytest.fixture
def credentials():
    key, certificate = issue_certificate("companion-1")
    from cryptography.hazmat.primitives import serialization

    return key, certificate.public_bytes(serialization.Encoding.PEM).decode()


async def test_an_accepted_job_keeps_the_local_path_unused(credentials):
    """The no-eager-hedge contract, proven against a real peer."""

    key, pem = credentials
    wire = _Wire()
    sessions = CompanionSessionManager()
    router = CompanionWorkloadRouter(sessions, enabled=True)
    _, pump = await _connect(sessions, wire, "classify_icon")

    local_calls = 0

    async def local():
        nonlocal local_calls
        local_calls += 1
        return {"icon": "local-fallback"}

    async def handler(offer):
        return {"icon": "pill"}

    peer = _peer(wire, key, pem, workloads=("classify_icon",), job_handler=handler)
    peer_task = asyncio.create_task(peer.run(until=1))

    result = await router.run("classify_icon", {"name": "Estrogen"}, local)
    await peer_task
    pump.cancel()

    assert result.source == "companion"
    assert result.value == {"icon": "pill"}
    assert local_calls == 0


async def test_a_rejected_job_falls_back_locally_exactly_once(credentials):
    """Declining is ordinary flow, not a failure."""

    key, pem = credentials
    wire = _Wire()
    sessions = CompanionSessionManager()
    router = CompanionWorkloadRouter(sessions, enabled=True)
    _, pump = await _connect(sessions, wire, "classify_icon")

    local_calls = 0

    async def local():
        nonlocal local_calls
        local_calls += 1
        return {"icon": "local-fallback"}

    async def handler(offer):
        return "battery"

    peer = _peer(wire, key, pem, workloads=("classify_icon",), job_handler=handler)
    peer_task = asyncio.create_task(peer.run(until=1))

    result = await router.run("classify_icon", {"name": "Estrogen"}, local)
    await peer_task
    pump.cancel()

    assert result.source == "local"
    assert result.value == {"icon": "local-fallback"}
    assert local_calls == 1


async def test_a_failed_attempt_falls_back_locally(credentials):
    key, pem = credentials
    wire = _Wire()
    sessions = CompanionSessionManager()
    router = CompanionWorkloadRouter(sessions, enabled=True)
    _, pump = await _connect(sessions, wire, "classify_icon")

    async def local():
        return {"icon": "local-fallback"}

    async def handler(offer):
        return None  # accepted, then failed

    peer = _peer(wire, key, pem, workloads=("classify_icon",), job_handler=handler)
    peer_task = asyncio.create_task(peer.run(until=1))

    result = await router.run("classify_icon", {"name": "Estrogen"}, local)
    await peer_task
    pump.cancel()

    assert result.source == "local"


async def test_a_companion_tool_call_runs_through_iridium(credentials):
    """The companion plans; Iridium executes. It never touches a provider."""

    key, pem = credentials
    wire = _Wire()
    sessions = CompanionSessionManager()
    router = CompanionWorkloadRouter(sessions, enabled=True)

    executed: list[tuple[str, str]] = []

    async def execute_tool_call(call) -> ToolResultMessage:
        executed.append((call.provider, call.tool))
        return ToolResultMessage(
            jobId=call.job_id,
            attemptId=call.attempt_id,
            callId=call.call_id,
            ok=True,
            code="ok",
            message="done",
            observed={"isOn": True},
        )

    _, pump = await _connect(
        sessions, wire, "classify_icon", execute_tool_call=execute_tool_call
    )

    async def local():
        return {"icon": "local-fallback"}

    async def handler(offer):
        return {"icon": "pill"}

    peer = _peer(
        wire,
        key,
        pem,
        workloads=("classify_icon",),
        job_handler=handler,
        tool_calls=(("nova", "nova.lighting_shortcut", {"state": "on"}),),
    )
    peer_task = asyncio.create_task(peer.run(until=1))

    result = await router.run(
        "classify_icon",
        {"name": "Estrogen"},
        local,
        allowed_tools=frozenset({"nova.lighting_shortcut"}),
    )
    await peer_task
    pump.cancel()

    assert executed == [("nova", "nova.lighting_shortcut")]
    assert result.source == "companion"
    # The peer saw the result of the action Iridium ran on its behalf.
    assert peer.tool_results[0]["ok"] is True


async def test_the_callback_budget_is_enforced(credentials):
    """A runaway job cannot keep asking Iridium to do things."""

    key, pem = credentials
    wire = _Wire()
    sessions = CompanionSessionManager(callback_cap=1)
    router = CompanionWorkloadRouter(sessions, enabled=True)

    executed: list[str] = []

    async def execute_tool_call(call) -> ToolResultMessage:
        executed.append(call.call_id)
        return ToolResultMessage(
            jobId=call.job_id,
            attemptId=call.attempt_id,
            callId=call.call_id,
            ok=True,
            code="ok",
            message="done",
        )

    _, pump = await _connect(
        sessions, wire, "classify_icon", execute_tool_call=execute_tool_call
    )

    async def local():
        return {"icon": "local-fallback"}

    peer = _peer(
        wire,
        key,
        pem,
        workloads=("classify_icon",),
        job_handler=lambda offer: _resolved({"icon": "pill"}),
        tool_calls=(
            ("nova", "nova.lighting_shortcut", {}),
            ("nova", "nova.lighting_shortcut", {}),
        ),
    )
    peer_task = asyncio.create_task(peer.run(until=1))

    await router.run(
        "classify_icon",
        {"name": "Estrogen"},
        local,
        allowed_tools=frozenset({"nova.lighting_shortcut"}),
    )
    await peer_task
    pump.cancel()

    assert len(executed) == 1
    refusal = peer.tool_results[1]
    assert refusal["ok"] is False
    assert refusal["code"] == "blocked"


async def test_a_disconnect_mid_job_resolves_rather_than_hanging(credentials):
    """Nothing may wait forever on a phone that vanished."""

    key, pem = credentials
    wire = _Wire()
    sessions = CompanionSessionManager()
    router = CompanionWorkloadRouter(sessions, enabled=True)
    session, pump = await _connect(sessions, wire, "classify_icon")

    async def local():
        return {"icon": "local-fallback"}

    async def handler(offer):
        # Accept, then go quiet: the session is released underneath the job.
        sessions.release(session)
        return {"icon": "never-delivered"}

    peer = _peer(wire, key, pem, workloads=("classify_icon",), job_handler=handler)
    peer_task = asyncio.create_task(peer.run(until=1))

    result = await asyncio.wait_for(
        router.run("classify_icon", {"name": "Estrogen"}, local), timeout=5
    )
    peer_task.cancel()
    pump.cancel()

    assert result.source == "local"


def _resolved(value):
    future: asyncio.Future = asyncio.get_event_loop().create_future()
    future.set_result(value)
    return future
