from __future__ import annotations

from typing import Any

import pytest
from conftest import interpretation

from nova_voice.audio.conversation import ConversationMessage, ConversationSnapshot
from nova_voice.capabilities.base import CapabilityManifest, ToolPolicy
from nova_voice.capabilities.registry import CapabilityRegistry
from nova_voice.config import Settings
from nova_voice.domain import Decision, SpeechAct, ToolResult
from nova_voice.interpretation.base import Interpreter
from nova_voice.providers.web.provider import WEB_TOOLS
from nova_voice.service import (
    NovaVoiceService,
    knowledge_fallback_query,
    knowledge_web_fallback_relevant,
    response_has_knowledge_failure,
)
from nova_voice.voice_settings import VoiceSettings


class _Interpreter(Interpreter):
    def __init__(self, value, *responses: str | None) -> None:
        self.value = value
        self.responses = list(responses)
        self.render_calls: list[tuple] = []

    async def interpret(
        self,
        _utterance,
        *,
        active_goal,
        relevant_state,
        tools,
        conversation=None,
    ):
        return self.value.model_copy(deep=True)

    async def render_response(
        self,
        utterance,
        value,
        results,
        *,
        persona,
        environment=None,
        relevant_state=None,
        conversation=None,
        temperature=None,
        command_max_words=None,
    ) -> str | None:
        self.render_calls.append((utterance, value, results))
        return self.responses.pop(0) if self.responses else None


class _NovaProvider:
    agent_name = "Nova"

    async def prompt_context(self, room: str) -> dict[str, Any]:
        return {"room": room, "zones": [], "nearbyTargets": []}

    async def invalid_device_action_ids(self, _actions) -> set[str]:
        return set()


class _WebProvider:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.actions = []

    def manifest(self) -> CapabilityManifest:
        return CapabilityManifest(
            id="web",
            version="test",
            contract_version="test",
            execution_class="iridium_local",
            tools=WEB_TOOLS,
            skill_files=[],
            tool_policies={
                "web.ask": ToolPolicy(
                    risk="low",
                    reversible=True,
                    idempotent=True,
                    parallel_safe=True,
                )
            },
        )

    async def execute(self, action) -> ToolResult:
        self.actions.append(action)
        if self.fail:
            return ToolResult(
                action_id=action.id,
                ok=False,
                code="backend_error",
                target="web",
                requested=action.call.arguments,
                message="Web lookup failed",
            )
        return ToolResult(
            action_id=action.id,
            ok=True,
            code="ok",
            target="web",
            requested=action.call.arguments,
            observed={"answer": "Wellington is New Zealand's capital."},
            message="Web answer retrieved",
        )


class _Store:
    async def add(self, _utterance, _interpretation) -> None:
        return None


class _Persona:
    response_prompt = "fixture persona"

    def render(self, _utterance, decision, results, *, shadowed: bool):
        if shadowed:
            return None
        if results and not all(result.ok for result in results):
            return results[0].message
        return "Reply" if decision == Decision.REPLY else "Done."

    def tone_instruction(self, _emotion) -> str:
        return "Natural conversational delivery."


def _service(value, *responses: str | None, web_fail: bool = False):
    interpreter = _Interpreter(value, *responses)
    nova = _NovaProvider()
    web = _WebProvider(fail=web_fail)
    registry = CapabilityRegistry()
    registry.register(web)
    service = NovaVoiceService(
        Settings(shadow_mode=False),
        interpreter,
        registry,
        nova,
        _Store(),
        _Persona(),
        web_provider=web,
    )
    service.voice_settings = VoiceSettings(web_access_enabled=True)
    return service, interpreter, web


@pytest.mark.parametrize(
    "response",
    [
        "I don't know.",
        "I'm not sure.",
        "I’m not certain.",
        "I can't do that.",
        "I don't have that information.",
        "I don't have enough context.",
        "I lack the answer.",
        "I can't provide that information.",
    ],
)
def test_detects_common_knowledge_failure_responses(response: str) -> None:
    assert response_has_knowledge_failure(response)


@pytest.mark.parametrize(
    ("transcript", "speech_act"),
    [
        ("How warm is it in here?", SpeechAct.QUESTION),
        ("Can you turn on the lounge light?", SpeechAct.DIRECTIVE),
        ("Tell me a joke", SpeechAct.DIRECTIVE),
        ("What do you remember about me?", SpeechAct.QUESTION),
        ("Can you send an email?", SpeechAct.QUESTION),
    ],
)
def test_rejects_non_knowledge_failure_categories(utterance, transcript, speech_act) -> None:
    spoken = utterance.model_copy(
        update={"transcript": transcript, "wake_detected": True}
    )
    value = interpretation(speech_act=speech_act, decision=Decision.REPLY)
    assert not knowledge_web_fallback_relevant(spoken, value, "I can't do that.")


def test_rejects_unaddressed_knowledge_failure(utterance) -> None:
    spoken = utterance.model_copy(
        update={"transcript": "Who discovered penicillin?", "wake_detected": False}
    )
    value = interpretation(speech_act=SpeechAct.QUESTION, decision=Decision.REPLY)
    assert not knowledge_web_fallback_relevant(spoken, value, "I don't know.")


def test_follow_up_query_includes_recent_user_context() -> None:
    conversation = ConversationSnapshot(
        id="c1",
        room_id="lounge",
        initial_environment=None,
        personality="",
        persona_prompt="",
        messages=(
            ConversationMessage("user", "Who is Jacinda Ardern?"),
            ConversationMessage("assistant", "She is a former prime minister."),
        ),
    )
    assert knowledge_fallback_query("Who is her husband?", conversation) == (
        "Who is Jacinda Ardern. Follow-up: Who is her husband?"
    )


@pytest.mark.asyncio
async def test_addressed_knowledge_failure_runs_one_web_lookup_and_rerenders(utterance) -> None:
    value = interpretation(speech_act=SpeechAct.QUESTION, decision=Decision.REPLY)
    service, interpreter, web = _service(
        value,
        "I don't know.",
        "Wellington is New Zealand's capital.",
    )
    spoken = utterance.model_copy(
        update={
            "transcript": "What is the capital of New Zealand?",
            "wake_detected": True,
        }
    )

    result = await service.handle(spoken)

    assert result.executed
    assert result.response_text == "Wellington is New Zealand's capital."
    assert len(web.actions) == 1
    assert web.actions[0].call.arguments == {
        "query": "What is the capital of New Zealand?"
    }
    assert len(interpreter.render_calls) == 2
    assert result.interpretation.actions[0].call.tool == "web.ask"
    assert result.timings_ms["knowledgeFallback"] >= 0


@pytest.mark.asyncio
async def test_failed_fallback_search_terminates_after_one_attempt(utterance) -> None:
    value = interpretation(speech_act=SpeechAct.QUESTION, decision=Decision.REPLY)
    service, interpreter, web = _service(
        value,
        "I don't know.",
        "I couldn't search the web just now.",
        web_fail=True,
    )
    spoken = utterance.model_copy(
        update={"transcript": "Who discovered penicillin?", "wake_detected": True}
    )

    result = await service.handle(spoken)

    assert result.executed
    assert result.response_text == "I couldn't search the web just now."
    assert len(web.actions) == 1
    assert len(interpreter.render_calls) == 2


@pytest.mark.asyncio
async def test_disabled_web_access_leaves_the_honest_failure_without_lookup(utterance) -> None:
    value = interpretation(speech_act=SpeechAct.QUESTION, decision=Decision.REPLY)
    service, interpreter, web = _service(value, "I don't know.")
    service.voice_settings = VoiceSettings(web_access_enabled=False)
    spoken = utterance.model_copy(
        update={"transcript": "Who discovered penicillin?", "wake_detected": True}
    )

    result = await service.handle(spoken)

    assert not result.executed
    assert result.response_text == "I don't know."
    assert web.actions == []
    assert len(interpreter.render_calls) == 1
