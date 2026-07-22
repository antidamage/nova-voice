from __future__ import annotations

import pytest
from conftest import interpretation

from nova_voice.config import Settings
from nova_voice.domain import CapabilityToolCall, Decision, PlannedAction, SpeechAct
from nova_voice.interpretation.llama_cpp import web_lookup_is_relevant
from nova_voice.policy import ExecutionPolicy


def _web_action() -> PlannedAction:
    return PlannedAction(
        id="w1",
        order=0,
        call=CapabilityToolCall(provider="web", tool="web.ask", arguments={"query": "who won"}),
    )


def _nova_action() -> PlannedAction:
    return PlannedAction(
        id="n1",
        order=0,
        call=CapabilityToolCall(
            provider="nova",
            tool="nova.control",
            arguments={"target": "lights", "action": "turn_on"},
        ),
    )


# -- Explicit-cue gate -------------------------------------------------------

def test_web_lookup_relevant_for_explicit_cues() -> None:
    assert web_lookup_is_relevant("look that up for me")
    assert web_lookup_is_relevant("search the web for the score")
    assert web_lookup_is_relevant("google the opening hours")
    assert web_lookup_is_relevant("what's the latest on the election")
    assert web_lookup_is_relevant("find out when the shop closes")


def test_web_lookup_not_relevant_for_control_or_chat() -> None:
    assert not web_lookup_is_relevant("turn on the lounge lights")
    assert not web_lookup_is_relevant("set the aircon to twenty one")
    assert not web_lookup_is_relevant("tell me a joke")
    assert not web_lookup_is_relevant("how warm is it in here")


# -- Policy carve-out for read-only web lookups ------------------------------

def _addressed(utterance):
    return utterance.model_copy(update={"wake_detected": True, "transcript": "who won the rugby"})


def test_web_question_executes_when_addressed(utterance) -> None:
    settings = Settings(shadow_mode=False, passive_execution_enabled=False)
    outcome = ExecutionPolicy(settings).evaluate(
        _addressed(utterance),
        interpretation(
            speech_act=SpeechAct.QUESTION, decision=Decision.EXECUTE, actions=[_web_action()]
        ),
        session_active=False,
    )
    assert outcome.execute
    assert not outcome.shadowed


def test_web_question_blocked_when_not_addressed(utterance) -> None:
    # Even with passive execution enabled for household control, a web lookup
    # never fires from unaddressed ambient speech.
    settings = Settings(shadow_mode=False, passive_execution_enabled=True)
    passive = utterance.model_copy(
        update={"wake_detected": False, "transcript": "who won the rugby"}
    )
    outcome = ExecutionPolicy(settings).evaluate(
        passive,
        interpretation(
            speech_act=SpeechAct.QUESTION, decision=Decision.EXECUTE, actions=[_web_action()]
        ),
        session_active=False,
    )
    assert not outcome.execute


@pytest.mark.parametrize(
    "speech_act",
    [SpeechAct.THIRD_PARTY, SpeechAct.QUOTED_OR_MEDIA, SpeechAct.OBSERVATION],
)
def test_web_non_executable_speech_acts_still_blocked(utterance, speech_act) -> None:
    settings = Settings(shadow_mode=False, passive_execution_enabled=True)
    outcome = ExecutionPolicy(settings).evaluate(
        _addressed(utterance),
        interpretation(speech_act=speech_act, decision=Decision.EXECUTE, actions=[_web_action()]),
        session_active=False,
    )
    assert not outcome.execute


def test_mixed_web_and_household_question_uses_strict_gate(utterance) -> None:
    # A plan mixing a web lookup with a household mutation is not "web-only", so
    # the strict directive/desired-state gate applies: a QUESTION is rejected.
    settings = Settings(shadow_mode=False, passive_execution_enabled=False)
    outcome = ExecutionPolicy(settings).evaluate(
        _addressed(utterance),
        interpretation(
            speech_act=SpeechAct.QUESTION,
            decision=Decision.EXECUTE,
            actions=[_web_action(), _nova_action()],
        ),
        session_active=False,
    )
    assert not outcome.execute


def test_web_directive_executes_normally(utterance) -> None:
    settings = Settings(shadow_mode=False, passive_execution_enabled=False)
    outcome = ExecutionPolicy(settings).evaluate(
        _addressed(utterance),
        interpretation(
            speech_act=SpeechAct.DIRECTIVE, decision=Decision.EXECUTE, actions=[_web_action()]
        ),
        session_active=False,
    )
    assert outcome.execute


def test_web_shadowed_under_shadow_mode(utterance) -> None:
    settings = Settings(shadow_mode=True, passive_execution_enabled=False)
    outcome = ExecutionPolicy(settings).evaluate(
        _addressed(utterance),
        interpretation(
            speech_act=SpeechAct.QUESTION, decision=Decision.EXECUTE, actions=[_web_action()]
        ),
        session_active=False,
    )
    assert not outcome.execute
    assert outcome.shadowed
