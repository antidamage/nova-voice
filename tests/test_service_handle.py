from __future__ import annotations

import asyncio
from typing import Any

import pytest
from conftest import interpretation

from nova_voice.config import Settings
from nova_voice.domain import (
    CapabilityToolCall,
    Decision,
    GoalStatus,
    PlannedAction,
    SpeechAct,
    ToolResult,
)
from nova_voice.interpretation.base import Interpreter
from nova_voice.providers.nova.client import NovaDashboardError
from nova_voice.service import NovaVoiceService


class _Interpreter(Interpreter):
    def __init__(self, value, *, rendered: str | None = None) -> None:
        self.value = value
        self.rendered = rendered
        self.contexts: list[dict[str, Any]] = []
        self.conversations = []
        self.render_calls: list[dict[str, Any]] = []

    async def interpret(
        self,
        utterance,
        *,
        active_goal,
        relevant_state,
        tools,
        conversation=None,
    ):
        self.contexts.append(relevant_state)
        self.conversations.append(conversation)
        return self.value.model_copy(deep=True)

    async def render_response(
        self,
        utterance,
        interpretation,
        results,
        *,
        persona,
        environment=None,
        conversation=None,
    ):
        self.render_calls.append(
            {
                "utterance": utterance,
                "interpretation": interpretation,
                "results": results,
                "persona": persona,
                "environment": environment,
                "conversation": conversation,
            }
        )
        return self.rendered


class _Provider:
    def __init__(
        self,
        *,
        unavailable: bool = False,
        delay: float = 0,
        weather: dict[str, Any] | None = None,
    ) -> None:
        self.unavailable = unavailable
        self.delay = delay
        self.weather = weather
        self.executed: list[str] = []

    async def refresh(self, *, force: bool = False) -> dict:
        if self.unavailable:
            raise NovaDashboardError("unavailable")
        return {}

    async def prompt_context(self, room: str) -> dict:
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.unavailable:
            raise NovaDashboardError("unavailable")
        return {
            "room": room,
            "zones": ["Lounge"],
            "nearbyTargets": [],
            "weather": self.weather,
        }

    async def health(self) -> dict:
        return {"ok": not self.unavailable}

    async def execute(self, action: PlannedAction) -> ToolResult:
        self.executed.append(action.id)
        return ToolResult(
            action_id=action.id,
            ok=True,
            code="ok",
            target=str(action.call.arguments.get("target", "lights")),
            observed={"isOn": True},
            message="Dashboard action completed and was verified",
        )


class _Registry:
    def __init__(self, provider: _Provider) -> None:
        self.provider_instance = provider

    def tool_catalog(self) -> list[dict]:
        return []

    def validate_action(self, action: PlannedAction) -> PlannedAction:
        return action

    def policy_for(self, _provider: str, _tool: str):
        return type(
            "Policy",
            (),
            {
                "risk": "low",
                "parallel_safe": False,
                "requires_confirmation": False,
                "idempotent": False,
            },
        )()

    def provider(self, _provider: str) -> _Provider:
        return self.provider_instance

    async def close(self) -> None:
        return None


class _Store:
    def __init__(self) -> None:
        self.saved: list[tuple] = []

    async def initialize(self) -> None:
        return None

    async def delete_expired(self) -> int:
        return 0

    async def add(self, utterance, value) -> None:
        self.saved.append((utterance, value))

    def stop(self) -> None:
        return None

    async def count(self) -> int:
        return len(self.saved)


class _Persona:
    response_prompt = "fixture persona"

    def render(self, _utterance, decision, _results, *, shadowed: bool):
        return None if shadowed else ("Reply" if decision == Decision.REPLY else None)

    def tone_instruction(self, _emotion) -> str:
        return "Natural conversational delivery."


def _action(*, provider: str = "fixture", tool: str = "fixture.control") -> PlannedAction:
    return PlannedAction(
        id="a1",
        order=0,
        call=CapabilityToolCall(
            provider=provider,
            tool=tool,
            arguments={"target": "lounge light"},
        ),
    )


def _service(settings: Settings, interpreter: _Interpreter, provider: _Provider, store: _Store):
    return NovaVoiceService(
        settings,
        interpreter,
        _Registry(provider),
        provider,
        store,
        _Persona(),
    )


@pytest.mark.asyncio
async def test_dashboard_outage_keeps_interpretation_and_retention_available(utterance) -> None:
    value = interpretation(decision=Decision.REPLY)
    interpreter = _Interpreter(value)
    provider = _Provider(unavailable=True)
    store = _Store()
    service = _service(Settings(), interpreter, provider, store)
    spoken = utterance.model_copy(update={"wake_detected": True})

    result = await service.handle(spoken)

    assert result.response_text == "Reply"
    packet = interpreter.contexts[0]
    assert packet == {
        "room": "lounge",
        "zones": [],
        "nearbyTargets": [],
    }
    assert len(interpreter.contexts) == 1
    assert store.saved == [(spoken, value)]
    assert set(result.timings_ms) == {
        "providerContext",
        "interpretation",
        "retention",
        "policy",
        "execution",
        "session",
        "response",
        "total",
    }
    assert all(value >= 0 for value in result.timings_ms.values())


@pytest.mark.asyncio
async def test_unaddressed_ambient_speech_is_classified_but_not_retained(utterance) -> None:
    value = interpretation(decision=Decision.IGNORE, addressed=0.1, confidence=0.1)
    interpreter = _Interpreter(value)
    provider = _Provider()
    store = _Store()
    service = _service(Settings(), interpreter, provider, store)

    result = await service.handle(utterance)

    assert not result.executed
    assert store.saved == []


@pytest.mark.asyncio
async def test_passively_executed_dashboard_command_is_retained(utterance) -> None:
    # A confident directive without a wake word can still pass the
    # passive-execution policy; that genuine dashboard command earns
    # retention even though it was never wake-worded.
    value = interpretation(decision=Decision.EXECUTE, actions=[_action()], addressed=0.99)
    interpreter = _Interpreter(value)
    provider = _Provider()
    store = _Store()
    service = _service(
        Settings(shadow_mode=False, passive_execution_enabled=True), interpreter, provider, store
    )

    result = await service.handle(utterance)

    assert result.executed
    assert store.saved == [(utterance, result.interpretation)]


@pytest.mark.asyncio
async def test_slow_dashboard_context_does_not_block_the_voice_turn(utterance) -> None:
    value = interpretation(decision=Decision.REPLY)
    interpreter = _Interpreter(value)
    provider = _Provider(delay=0.1)
    store = _Store()
    service = _service(
        Settings(provider_context_timeout_seconds=0.01), interpreter, provider, store
    )

    result = await service.handle(utterance)

    assert result.response_text == "Reply"
    packet = interpreter.contexts[0]
    assert packet == {
        "room": "lounge",
        "zones": [],
        "nearbyTargets": [],
    }
    assert result.timings_ms["providerContext"] < 50


@pytest.mark.asyncio
async def test_addressed_question_cannot_be_ignored_as_a_non_tool_turn(utterance) -> None:
    value = interpretation(
        speech_act=SpeechAct.QUESTION,
        decision=Decision.IGNORE,
        addressed=0.9,
    )
    interpreter = _Interpreter(value)
    provider = _Provider()
    store = _Store()
    service = _service(Settings(), interpreter, provider, store)
    spoken = utterance.model_copy(
        update={"transcript": "Hello Nova, how are you feeling?", "wake_detected": True}
    )

    result = await service.handle(spoken)

    assert result.interpretation.decision == Decision.REPLY
    assert result.response_text == "Reply"


@pytest.mark.asyncio
async def test_addressed_non_tool_request_with_empty_execute_plan_becomes_reply(utterance) -> None:
    value = interpretation(
        speech_act=SpeechAct.DIRECTIVE,
        decision=Decision.EXECUTE,
        addressed=0.9,
    )
    interpreter = _Interpreter(value)
    provider = _Provider()
    store = _Store()
    service = _service(Settings(), interpreter, provider, store)
    spoken = utterance.model_copy(
        update={"transcript": "Tell me a joke", "wake_detected": True}
    )

    result = await service.handle(spoken)

    assert result.interpretation.decision == Decision.REPLY
    assert result.response_text == "Reply"
    assert not result.executed


@pytest.mark.asyncio
async def test_self_intention_cannot_execute_even_when_model_requests_action(utterance) -> None:
    value = interpretation(decision=Decision.EXECUTE, actions=[_action()])
    interpreter = _Interpreter(value)
    provider = _Provider()
    store = _Store()
    service = _service(
        Settings(shadow_mode=False, passive_execution_enabled=True), interpreter, provider, store
    )
    spoken = utterance.model_copy(update={"transcript": "I gotta turn the lounge light on"})

    result = await service.handle(spoken)

    assert not result.executed
    assert not result.shadowed
    assert result.interpretation.actions == []
    assert provider.executed == []


@pytest.mark.asyncio
async def test_shadow_mode_records_valid_plan_without_mutating_provider(utterance) -> None:
    value = interpretation(decision=Decision.EXECUTE, actions=[_action()])
    interpreter = _Interpreter(value)
    provider = _Provider()
    store = _Store()
    service = _service(Settings(), interpreter, provider, store)

    result = await service.handle(utterance)

    assert result.shadowed
    assert not result.executed
    assert [item.code for item in result.results] == ["shadowed"]
    assert provider.executed == []


@pytest.mark.asyncio
async def test_conversation_start_snapshots_environment_and_persona_once(utterance) -> None:
    value = interpretation(decision=Decision.REPLY)
    interpreter = _Interpreter(value, rendered="Of course.")
    provider = _Provider(weather={"condition": "rainy", "temperatureC": 11})
    store = _Store()
    service = _service(Settings(), interpreter, provider, store)
    first = utterance.model_copy(
        update={
            "transcript": "Bandit, how are you?",
            "wake_detected": True,
            "conversation_active": True,
        }
    )
    second = utterance.model_copy(
        update={
            "id": "utterance-2",
            "transcript": "What did I just ask?",
            "wake_detected": False,
            "conversation_active": True,
        }
    )

    await service.handle(first)
    provider.weather = {"condition": "sunny", "temperatureC": 20}
    await service.handle(second)

    first_snapshot, second_snapshot = interpreter.conversations
    assert first_snapshot.initial_environment is not None
    assert set(first_snapshot.initial_environment["now"]) == {"iso", "date", "time"}
    assert first_snapshot.initial_environment["now"]["time"].endswith(("am", "pm"))
    assert first_snapshot.initial_environment["weather"] == {
        "condition": "rainy",
        "temperatureC": 11,
    }
    assert first_snapshot.persona_prompt == "fixture persona"
    assert second_snapshot.initial_environment == first_snapshot.initial_environment
    assert [(message.role, message.content) for message in second_snapshot.messages] == [
        ("user", "Bandit, how are you?"),
        ("assistant", "Of course."),
    ]
    assert "now" not in interpreter.contexts[0]
    assert "now" not in interpreter.contexts[1]


@pytest.mark.asyncio
async def test_social_turn_model_status_cannot_close_the_conversation(utterance) -> None:
    value = interpretation(speech_act=SpeechAct.SOCIAL, decision=Decision.REPLY)
    value.active_goal.status = GoalStatus.ABANDONED
    interpreter = _Interpreter(value, rendered="I'm here.")
    service = _service(Settings(), interpreter, _Provider(), _Store())
    spoken = utterance.model_copy(
        update={
            "transcript": "Hey Bandit, are you there?",
            "wake_detected": True,
            "conversation_active": True,
        }
    )

    await service.handle(spoken)

    assert service.conversations.active(spoken.room_id)


@pytest.mark.asyncio
async def test_verified_dashboard_command_always_gets_persona_aware_confirmation(
    utterance,
) -> None:
    action = _action(provider="nova", tool="nova.control")
    value = interpretation(decision=Decision.EXECUTE, actions=[action])
    interpreter = _Interpreter(value, rendered="Done — the lounge light is on.")
    provider = _Provider()
    store = _Store()
    service = _service(
        Settings(shadow_mode=False, passive_execution_enabled=True),
        interpreter,
        provider,
        store,
    )
    spoken = utterance.model_copy(
        update={"transcript": "Bandit, turn the lounge light on", "wake_detected": True}
    )

    result = await service.handle(spoken)

    assert result.executed
    assert result.response_text == "Done — the lounge light is on."
    assert provider.executed == ["a1"]
    assert len(interpreter.render_calls) == 1
    render_call = interpreter.render_calls[0]
    assert render_call["persona"] == "fixture persona"
    assert render_call["environment"] is None
    assert render_call["results"][0].observed == {"isOn": True}
