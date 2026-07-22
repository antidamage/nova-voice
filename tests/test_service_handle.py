from __future__ import annotations

import asyncio
from typing import Any

import pytest
from conftest import interpretation
from pydantic import ValidationError

from nova_voice.agent_settings import AgentSettings
from nova_voice.config import Settings
from nova_voice.domain import (
    CapabilityToolCall,
    Decision,
    GoalStatus,
    PlannedAction,
    SelfProfileUpdate,
    SpeakerIdentity,
    SpeechAct,
    ToolResult,
    TurnStage,
    TurnTerminalStatus,
)
from nova_voice.interpretation.base import Interpreter
from nova_voice.providers.nova.client import NovaDashboardError
from nova_voice.service import (
    NovaVoiceService,
    command_word_count,
    explicit_self_profile_update,
)


class _Interpreter(Interpreter):
    def __init__(
        self,
        value,
        *,
        rendered: str | None = None,
        profile_update: SelfProfileUpdate | None = None,
    ) -> None:
        self.value = value
        self.rendered = rendered
        self.profile_update = profile_update
        self.contexts: list[dict[str, Any]] = []
        self.conversations = []
        self.active_goals = []
        self.render_calls: list[dict[str, Any]] = []
        self.profile_calls = []

    async def extract_self_profile_update(self, utterance):
        self.profile_calls.append(utterance)
        return self.profile_update

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
        self.active_goals.append(active_goal)
        return self.value.model_copy(deep=True)

    async def render_response(
        self,
        utterance,
        interpretation,
        results,
        *,
        persona,
        environment=None,
        relevant_state=None,
        conversation=None,
        temperature=None,
        command_max_words=None,
    ):
        self.render_calls.append(
            {
                "utterance": utterance,
                "interpretation": interpretation,
                "results": results,
                "persona": persona,
                "environment": environment,
                "relevant_state": relevant_state,
                "conversation": conversation,
                "temperature": temperature,
                "command_max_words": command_max_words,
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
        fail: bool = False,
        invalid_device_action_ids: set[str] | None = None,
    ) -> None:
        self.unavailable = unavailable
        self.delay = delay
        self.weather = weather
        self.fail = fail
        self.invalid_ids = invalid_device_action_ids or set()
        self.executed: list[str] = []
        self.verification_config: dict[str, object] | None = None

    def configure_verification_loop(self, **values: object) -> None:
        self.verification_config = values

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

    async def invalid_device_action_ids(self, _actions) -> set[str]:
        return self.invalid_ids

    async def execute(self, action: PlannedAction) -> ToolResult:
        self.executed.append(action.id)
        if self.fail:
            return ToolResult(
                action_id=action.id,
                ok=False,
                code="backend_error",
                target=str(action.call.arguments.get("target", "lights")),
                observed=None,
                message="Dashboard rejected the command",
            )
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


class _SpeakerProfiles:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def apply_disclosure(self, identity, update, transcript):
        self.calls.append((identity, update, transcript))
        return SpeakerIdentity(
            status="recognized",
            template_id=identity.template_id,
            person_id="person-a",
            display_name=update.name or "Addie",
            pronouns=update.pronouns,
            confidence=identity.confidence,
        )


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


def test_command_word_count_removes_one_leading_wake_phrase() -> None:
    assert command_word_count("Hey Nova, turn the lights on", ["Nova", "Hey Nova"]) == 4
    assert command_word_count("Turn the lights on", ["Nova", "Hey Nova"]) == 4


def test_explicit_profile_fallback_extracts_clear_addressed_identity_claims() -> None:
    combined = explicit_self_profile_update(
        "Nova, actually, my name is Addie and I use she/her pronouns",
        ["Nova"],
    )
    introduction = explicit_self_profile_update("Nova, this is Adeline speaking", ["Nova"])
    conversational = explicit_self_profile_update(
        "By the way, my name is Adeline and my pronouns are she her", ["Nova"]
    )

    assert combined is not None
    assert combined.name == "Addie"
    assert combined.pronouns == "she/her"
    assert introduction is not None
    assert introduction.name == "Adeline"
    assert conversational is not None
    assert conversational.name == "Adeline"


@pytest.mark.parametrize(
    "transcript",
    [
        "She said my name is Addie",
        "Nova, she said my name is Addie",
        "Nova, repeat the phrase my name is Addie",
    ],
)
def test_explicit_profile_fallback_rejects_third_party_or_quoted_claims(
    transcript: str,
) -> None:
    assert explicit_self_profile_update(transcript, ["Nova"]) is None


@pytest.mark.asyncio
async def test_short_execute_from_unrecognized_speaker_is_ignored(utterance) -> None:
    value = interpretation(decision=Decision.EXECUTE, actions=[_action()], addressed=1.0)
    interpreter = _Interpreter(value)
    provider = _Provider()
    store = _Store()
    service = NovaVoiceService(
        Settings(shadow_mode=False, passive_execution_enabled=True),
        interpreter,
        _Registry(provider),
        provider,
        store,
        _Persona(),
        speaker_profiles=_SpeakerProfiles(),
    )
    spoken = utterance.model_copy(update={"transcript": "Nova, lights on", "wake_detected": True})

    result = await service.handle(spoken)

    assert not result.executed
    assert result.interpretation.decision == Decision.IGNORE
    assert result.interpretation.actions == []
    assert provider.executed == []


@pytest.mark.asyncio
async def test_short_execute_from_recognized_speaker_remains_executable(utterance) -> None:
    value = interpretation(decision=Decision.EXECUTE, actions=[_action()], addressed=1.0)
    interpreter = _Interpreter(value)
    provider = _Provider()
    service = NovaVoiceService(
        Settings(shadow_mode=False, passive_execution_enabled=True),
        interpreter,
        _Registry(provider),
        provider,
        _Store(),
        _Persona(),
        speaker_profiles=_SpeakerProfiles(),
    )
    spoken = utterance.model_copy(
        update={
            "transcript": "Nova, lights on",
            "wake_detected": True,
            "speaker": SpeakerIdentity(
                status="recognized",
                template_id="voice-a",
                person_id="person-a",
                display_name="Addie",
                confidence=0.91,
            ),
        }
    )

    result = await service.handle(spoken)

    assert result.executed
    assert provider.executed == ["a1"]


@pytest.mark.asyncio
async def test_longer_execute_from_unrecognized_speaker_uses_normal_policy(utterance) -> None:
    value = interpretation(decision=Decision.EXECUTE, actions=[_action()], addressed=1.0)
    interpreter = _Interpreter(value)
    provider = _Provider()
    service = NovaVoiceService(
        Settings(shadow_mode=False, passive_execution_enabled=True),
        interpreter,
        _Registry(provider),
        provider,
        _Store(),
        _Persona(),
        speaker_profiles=_SpeakerProfiles(),
    )
    spoken = utterance.model_copy(
        update={"transcript": "Nova, please turn the lounge lights on", "wake_detected": True}
    )

    result = await service.handle(spoken)

    assert result.executed
    assert provider.executed == ["a1"]


def test_agent_settings_apply_live_to_provider_without_changing_voice() -> None:
    provider = _Provider()
    service = _service(
        Settings(), _Interpreter(interpretation(decision=Decision.REPLY)), provider, _Store()
    )

    service.apply_agent_settings(
        AgentSettings(
            ralphLoopEnabled=False,
            ralphLoopMaxIterations=6,
            ralphLoopSleepMs=300,
            ralphLoopFailureSeconds=5,
        )
    )

    assert provider.verification_config == {
        "enabled": False,
        "max_iterations": 6,
        "sleep_seconds": 0.3,
        "failure_seconds": 5,
        "thinking_threshold_seconds": 2.5,
        "llm_verify_enabled": True,
        "llm_verify_min_interval_seconds": 1.5,
        "llm_confirm_timeout_seconds": 3.0,
    }
    assert service.voice_settings is None


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
async def test_current_addressed_self_disclosure_updates_current_speaker(utterance) -> None:
    profile_update = SelfProfileUpdate(
        name="Addie",
        pronouns="she/her",
        evidence="my name is Addie and I use she/her",
    )
    value = interpretation(
        decision=Decision.REPLY,
        speech_act=SpeechAct.SELF_INTENTION,
    )
    interpreter = _Interpreter(
        value,
        rendered="Nice to meet you.",
        profile_update=profile_update,
    )
    provider = _Provider()
    store = _Store()
    profiles = _SpeakerProfiles()
    service = NovaVoiceService(
        Settings(),
        interpreter,
        _Registry(provider),
        provider,
        store,
        _Persona(),
        speaker_profiles=profiles,
    )
    spoken = utterance.model_copy(
        update={
            "wake_detected": True,
            "transcript": "Nova, my name is Addie and I use she/her",
            "speaker": SpeakerIdentity(
                status="provisional", template_id="voice-a", confidence=1.0
            ),
        }
    )

    result = await service.handle(spoken)

    assert len(profiles.calls) == 1
    assert interpreter.profile_calls == [spoken]
    assert store.saved[0][0].speaker.display_name == "Addie"
    assert interpreter.render_calls[0]["utterance"].speaker.person_id == "person-a"
    assert interpreter.render_calls[0]["interpretation"].self_profile_update == (
        profile_update
    )
    assert result.speaker is not None
    assert result.speaker.display_name == "Addie"
    assert result.speaker.pronouns == "she/her"


@pytest.mark.asyncio
async def test_dedicated_identity_result_replaces_general_interpreter_value(
    utterance,
) -> None:
    stale = SelfProfileUpdate(name="Wrong", evidence="my name is Wrong")
    extracted = SelfProfileUpdate(
        name="Adeline",
        pronouns="she/her",
        evidence="my name is Adeline and my pronouns are she her",
    )
    value = interpretation(
        decision=Decision.REPLY,
        speech_act=SpeechAct.SELF_INTENTION,
    ).model_copy(update={"self_profile_update": stale})
    interpreter = _Interpreter(value, profile_update=extracted)
    profiles = _SpeakerProfiles()
    provider = _Provider()
    store = _Store()
    service = NovaVoiceService(
        Settings(),
        interpreter,
        _Registry(provider),
        provider,
        store,
        _Persona(),
        speaker_profiles=profiles,
    )
    spoken = utterance.model_copy(
        update={
            "conversation_active": True,
            "transcript": (
                "By the way, my name is Adeline and my pronouns are she her"
            ),
            "speaker": SpeakerIdentity(
                status="provisional", template_id="voice-a", confidence=0.83
            ),
        }
    )

    result = await service.handle(spoken)

    assert profiles.calls == [(spoken.speaker, extracted, spoken.transcript)]
    assert result.interpretation.self_profile_update == extracted


@pytest.mark.asyncio
async def test_clear_self_disclosure_updates_profile_when_model_omits_the_field(
    utterance,
) -> None:
    interpreter = _Interpreter(interpretation(decision=Decision.REPLY))
    interpreter.agent_name = "Nova"
    provider = _Provider()
    store = _Store()
    profiles = _SpeakerProfiles()
    service = NovaVoiceService(
        Settings(),
        interpreter,
        _Registry(provider),
        provider,
        store,
        _Persona(),
        speaker_profiles=profiles,
    )
    spoken = utterance.model_copy(
        update={
            "wake_detected": True,
            "transcript": "Nova, my name is Addie and I use she/her pronouns",
            "speaker": SpeakerIdentity(
                status="provisional", template_id="voice-a", confidence=1.0
            ),
        }
    )

    result = await service.handle(spoken)

    assert len(profiles.calls) == 1
    update = profiles.calls[0][1]
    assert update.name == "Addie"
    assert update.pronouns == "she/her"
    assert result.interpretation.self_profile_update == update


@pytest.mark.asyncio
async def test_quoted_profile_claim_never_updates_speaker(utterance) -> None:
    value = interpretation(
        decision=Decision.IGNORE,
        speech_act=SpeechAct.QUOTED_OR_MEDIA,
    ).model_copy(
        update={
            "self_profile_update": SelfProfileUpdate(
                name="Addie", evidence="my name is Addie"
            )
        }
    )
    profiles = _SpeakerProfiles()
    provider = _Provider()
    service = NovaVoiceService(
        Settings(),
        _Interpreter(value),
        _Registry(provider),
        provider,
        _Store(),
        _Persona(),
        speaker_profiles=profiles,
    )
    spoken = utterance.model_copy(
        update={"wake_detected": True, "transcript": "She said my name is Addie"}
    )

    await service.handle(spoken)

    assert profiles.calls == []


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
async def test_nonexistent_device_plan_is_not_a_passive_dashboard_command(utterance) -> None:
    value = interpretation(decision=Decision.EXECUTE, actions=[_action()], addressed=0.99)
    interpreter = _Interpreter(value)
    provider = _Provider(invalid_device_action_ids={"a1"})
    store = _Store()
    service = _service(
        Settings(shadow_mode=False, passive_execution_enabled=True), interpreter, provider, store
    )

    result = await service.handle(utterance)

    assert not result.executed
    assert not result.shadowed
    assert result.interpretation.decision == Decision.IGNORE
    assert result.interpretation.actions == []
    assert provider.executed == []
    assert store.saved == []


@pytest.mark.asyncio
async def test_early_execute_plan_is_not_vetoed_by_lower_address_score(utterance) -> None:
    # Regression for a live passive turn where interpretation correctly returned
    # a directive plus dashboard actions at confidence 0.9, but the independent
    # 0.8 address score prevented execution until a conversation was opened.
    value = interpretation(
        decision=Decision.EXECUTE,
        actions=[_action()],
        addressed=0.8,
        confidence=0.9,
    )
    interpreter = _Interpreter(value)
    provider = _Provider()
    store = _Store()
    service = _service(
        Settings(shadow_mode=False, passive_execution_enabled=True), interpreter, provider, store
    )

    result = await service.handle(utterance)

    assert result.executed
    assert result.policy_reason == "allowed"
    assert result.interpretation.addressed_probability == 1.0
    assert provider.executed == ["a1"]
    assert store.saved == [(utterance, result.interpretation)]


@pytest.mark.asyncio
async def test_failed_dashboard_command_opens_conversation_and_forces_zero_temperature(
    utterance,
) -> None:
    value = interpretation(
        decision=Decision.EXECUTE,
        actions=[_action(provider="nova", tool="nova.control")],
        addressed=0.99,
    )
    interpreter = _Interpreter(value, rendered="I could not do that")
    interpreter.render_temperature = 0.8  # configured non-zero wording
    provider = _Provider(fail=True)
    store = _Store()
    service = _service(
        Settings(shadow_mode=False, passive_execution_enabled=True), interpreter, provider, store
    )

    result = await service.handle(utterance)

    assert any(not outcome.ok for outcome in result.results)
    # A conversation is opened so the user can respond without a wake word,
    assert service.conversations.active(utterance.room_id)
    # and the failed-command reply is rendered at forced-zero temperature.
    assert interpreter.render_calls
    assert interpreter.render_calls[-1]["temperature"] == 0.0


@pytest.mark.asyncio
async def test_zero_temperature_lock_clears_when_the_conversation_ends(utterance) -> None:
    value = interpretation(
        decision=Decision.EXECUTE,
        actions=[_action(provider="nova", tool="nova.control")],
        addressed=0.99,
    )
    interpreter = _Interpreter(value, rendered="nope")
    provider = _Provider(fail=True)
    service = _service(
        Settings(shadow_mode=False, passive_execution_enabled=True), interpreter, provider, _Store()
    )

    await service.handle(utterance)
    assert service._scope_key(utterance.room_id) in service._zero_render_temperature_scopes

    # An explicit dismissal ends the conversation and lifts the temperature lock.
    goodbye = utterance.model_copy(
        update={"transcript": "goodbye", "conversation_active": True}
    )
    await service.handle(goodbye)
    assert service._scope_key(goodbye.room_id) not in service._zero_render_temperature_scopes
    assert not service.conversations.active(goodbye.room_id)


@pytest.mark.asyncio
async def test_command_reply_length_is_rolled_and_zero_is_silent(utterance, monkeypatch) -> None:
    from nova_voice import service as service_module
    from nova_voice.voice_settings import VoiceSettings

    value = interpretation(
        decision=Decision.EXECUTE,
        actions=[_action(provider="nova", tool="nova.control")],
        addressed=0.99,
    )

    # A rolled length of zero is a silent acknowledgement: the command runs but
    # nothing is spoken and the renderer is not consulted.
    interpreter = _Interpreter(value, rendered="Done, dude")
    service = _service(
        Settings(shadow_mode=False, passive_execution_enabled=True),
        interpreter,
        _Provider(),
        _Store(),
    )
    service.voice_settings = VoiceSettings(command_reply_max_words=5)
    monkeypatch.setattr(service_module.random, "randint", lambda _low, _high: 0)
    result = await service.handle(utterance.model_copy(update={"wake_detected": True}))
    assert result.executed
    assert result.response_text is None
    assert interpreter.render_calls == []

    # A non-zero roll speaks, and the budget is passed to the renderer.
    interpreter2 = _Interpreter(value, rendered="On it, all done")
    service2 = _service(
        Settings(shadow_mode=False, passive_execution_enabled=True),
        interpreter2,
        _Provider(),
        _Store(),
    )
    service2.voice_settings = VoiceSettings(command_reply_max_words=5)
    monkeypatch.setattr(service_module.random, "randint", lambda _low, _high: 4)
    result2 = await service2.handle(utterance.model_copy(update={"wake_detected": True}))
    assert result2.response_text == "On it, all done"
    assert interpreter2.render_calls[-1]["command_max_words"] == 4


@pytest.mark.asyncio
async def test_new_conversation_does_not_inherit_a_stale_goal(utterance) -> None:
    # Turn 1: a failed command leaves an in-progress goal on the session.
    interpreter = _Interpreter(
        interpretation(
            decision=Decision.EXECUTE,
            actions=[_action(provider="nova", tool="nova.control")],
            addressed=0.99,
        ),
        rendered="nope",
    )
    service = _service(
        Settings(shadow_mode=False, passive_execution_enabled=True),
        interpreter,
        _Provider(fail=True),
        _Store(),
    )
    await service.handle(utterance.model_copy(update={"wake_detected": True}))
    assert service.sessions.active_goal(utterance.room_id) is not None

    # The conversation window is gone (ended by the user or timed out) while the
    # goal still lingered on its own clock — the classic stale-context leak.
    service.conversations.end(utterance.room_id)
    assert service.sessions.active_goal(utterance.room_id) is not None

    # A brand-new wake opens a clean conversation: interpret sees no old goal.
    interpreter.value = interpretation(decision=Decision.REPLY)
    await service.handle(
        utterance.model_copy(update={"wake_detected": True, "transcript": "hello"})
    )
    assert interpreter.active_goals[-1] is None


@pytest.mark.asyncio
async def test_media_speech_in_a_conversation_is_not_remembered(utterance) -> None:
    interpreter = _Interpreter(interpretation(decision=Decision.REPLY), rendered="Hi")
    service = _service(Settings(), interpreter, _Provider(), _Store())

    # A real wake turn is recorded (user + assistant).
    await service.handle(
        utterance.model_copy(update={"wake_detected": True, "transcript": "hello"})
    )
    first = service.conversations.snapshot(utterance.room_id)
    assert first is not None and len(first.messages) == 2

    # Television dialogue during the open conversation must not be recorded.
    interpreter.value = interpretation(
        speech_act=SpeechAct.QUOTED_OR_MEDIA, decision=Decision.REPLY, addressed=1.0
    )
    await service.handle(
        utterance.model_copy(
            update={"conversation_active": True, "transcript": "buy now, five easy payments"}
        )
    )
    second = service.conversations.snapshot(utterance.room_id)
    assert second is not None
    assert len(second.messages) == 2
    assert all("five easy payments" not in message.content for message in second.messages)


@pytest.mark.asyncio
async def test_successful_dashboard_results_are_retained_as_observations(utterance) -> None:
    value = interpretation(
        decision=Decision.EXECUTE,
        actions=[_action(provider="nova", tool="nova.control")],
    )
    interpreter = _Interpreter(value, rendered="Done")
    provider = _Provider()
    service = _service(Settings(shadow_mode=False), interpreter, provider, _Store())
    spoken = utterance.model_copy(update={"wake_detected": True})

    await service.handle(spoken)

    snapshot = service.conversations.snapshot(spoken.room_id)
    assert snapshot is not None
    assert any("isOn" in entry for entry in snapshot.observations)


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
async def test_foreground_turn_emits_complete_immutable_trace(utterance) -> None:
    value = interpretation(
        decision=Decision.EXECUTE,
        actions=[_action(provider="nova", tool="nova.control")],
        addressed=0.99,
    )
    interpreter = _Interpreter(value, rendered="Done")
    provider = _Provider()
    service = _service(
        Settings(shadow_mode=False, passive_execution_enabled=True),
        interpreter,
        provider,
        _Store(),
    )
    spoken = utterance.model_copy(update={"wake_detected": True})

    result = await service.handle(spoken)

    trace = result.turn_trace
    assert trace is not None
    assert trace.utterance_id == spoken.id
    assert trace.input_revision.startswith("sha256:")
    assert trace.context_revision is not None
    assert [stage.stage for stage in trace.stages] == list(TurnStage)
    assert trace.terminal_status == TurnTerminalStatus.COMPLETED
    assert trace.policy is not None and trace.policy.execute
    assert trace.tool_journal[0].status == "completed"
    assert trace.verification[0].ok
    assert trace.verification[0].observed_revision is not None
    assert trace.response_revisions
    assert trace.total_ms >= 0
    assert spoken.transcript not in trace.model_dump_json()
    with pytest.raises(ValidationError):
        trace.terminal_status = TurnTerminalStatus.FAILED


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
