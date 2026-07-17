from __future__ import annotations

import asyncio

import pytest

from nova_voice.domain import CapabilityToolCall, PlannedAction, ToolResult
from nova_voice.service import NovaVoiceService


class _FakeProvider:
    def __init__(self, *, fail: set[str] | None = None) -> None:
        self.fail = fail or set()
        self.running = 0
        self.max_running = 0
        self.calls: list[str] = []

    async def execute(self, action: PlannedAction) -> ToolResult:
        self.calls.append(action.id)
        self.running += 1
        self.max_running = max(self.max_running, self.running)
        try:
            await asyncio.sleep(0.01)
            if action.id in self.fail:
                raise RuntimeError("simulated provider failure")
            return ToolResult(action_id=action.id, ok=True, code="ok", message="done")
        finally:
            self.running -= 1


class _FakeRegistry:
    def __init__(
        self,
        provider: _FakeProvider,
        *,
        parallel_safe: bool,
        invalid: set[str] | None = None,
        idempotent: bool = False,
        risk: str = "low",
        requires_confirmation: bool = False,
        include_policy: bool = True,
    ) -> None:
        self.provider_instance = provider
        self.parallel_safe = parallel_safe
        self.invalid = invalid or set()
        self.idempotent = idempotent
        self.risk = risk
        self.requires_confirmation = requires_confirmation
        self.include_policy = include_policy

    def validate_action(self, action: PlannedAction) -> PlannedAction:
        if action.id in self.invalid:
            raise ValueError("invalid")
        return action

    def policy_for(self, _provider: str, _tool: str):
        if not self.include_policy:
            return None
        return type(
            "Policy",
            (),
            {
                "risk": self.risk,
                "parallel_safe": self.parallel_safe,
                "requires_confirmation": self.requires_confirmation,
                "idempotent": self.idempotent,
            },
        )()

    def provider(self, _provider: str) -> _FakeProvider:
        return self.provider_instance


def _action(action_id: str, order: int, *, depends_on: list[str] | None = None) -> PlannedAction:
    return PlannedAction(
        id=action_id,
        order=order,
        depends_on=depends_on or [],
        call=CapabilityToolCall(
            provider="fixture",
            tool="fixture.query",
            arguments={"target": action_id},
        ),
    )


@pytest.mark.asyncio
async def test_independent_parallel_safe_actions_share_execution_wave() -> None:
    provider = _FakeProvider()
    service = object.__new__(NovaVoiceService)
    service.registry = _FakeRegistry(provider, parallel_safe=True)

    results = await service._execute_plan([_action("a1", 0), _action("a2", 1)])

    assert [result.action_id for result in results] == ["a1", "a2"]
    assert all(result.ok for result in results)
    assert provider.max_running == 2


@pytest.mark.asyncio
async def test_dependency_waits_for_predecessor_and_provider_failure_is_local() -> None:
    provider = _FakeProvider(fail={"a1"})
    service = object.__new__(NovaVoiceService)
    service.registry = _FakeRegistry(provider, parallel_safe=False)

    results = await service._execute_plan([_action("a1", 0), _action("a2", 1, depends_on=["a1"])])

    assert [result.action_id for result in results] == ["a1", "a2"]
    assert results[0].code == "backend_error"
    assert results[1].code == "blocked"
    assert provider.calls == ["a1"]


@pytest.mark.asyncio
async def test_invalid_mixed_plan_does_not_partially_execute() -> None:
    provider = _FakeProvider()
    service = object.__new__(NovaVoiceService)
    service.registry = _FakeRegistry(provider, parallel_safe=True, invalid={"a2"})

    results = await service._execute_plan([_action("a1", 0), _action("a2", 1)])

    assert [result.code for result in results] == ["blocked", "invalid"]
    assert provider.calls == []


@pytest.mark.asyncio
async def test_duplicate_idempotent_action_is_coalesced() -> None:
    provider = _FakeProvider()
    service = object.__new__(NovaVoiceService)
    service.registry = _FakeRegistry(provider, parallel_safe=True, idempotent=True)
    duplicate = _action("a2", 1)
    duplicate.call.arguments["target"] = "a1"

    results = await service._execute_plan([_action("a1", 0), duplicate])

    assert [result.action_id for result in results] == ["a1", "a2"]
    assert results[1].message == "Duplicate idempotent action coalesced"
    assert provider.calls == ["a1"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs",
    [{"risk": "confirmation"}, {"requires_confirmation": True}, {"include_policy": False}],
)
async def test_unapproved_or_unclassified_capability_never_executes(kwargs: dict) -> None:
    provider = _FakeProvider()
    service = object.__new__(NovaVoiceService)
    service.registry = _FakeRegistry(provider, parallel_safe=True, **kwargs)

    results = await service._execute_plan([_action("a1", 0)])

    assert results[0].code == "blocked"
    assert provider.calls == []
