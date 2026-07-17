from __future__ import annotations

import pytest
from conftest import interpretation
from pydantic import ValidationError

from nova_voice.config import Settings
from nova_voice.domain import (
    CapabilityToolCall,
    Decision,
    GoalStatus,
    PlannedAction,
    SpeechAct,
)
from nova_voice.interpretation.speech_cues import (
    enforce_decision_consistency,
    enforce_speech_cues,
    has_speech_interrupt,
)
from nova_voice.policy import ExecutionPolicy


def action(action_id: str = "a1", *, depends_on: list[str] | None = None) -> PlannedAction:
    return PlannedAction(
        id=action_id,
        order=0,
        depends_on=depends_on or [],
        call=CapabilityToolCall(
            provider="nova",
            tool="nova.control",
            arguments={"target": "lounge light", "action": "turn_on"},
        ),
    )


def test_non_execute_cannot_contain_actions() -> None:
    with pytest.raises(ValidationError):
        interpretation(decision=Decision.IGNORE, actions=[action()])


def test_dependency_must_refer_to_earlier_action() -> None:
    with pytest.raises(ValidationError):
        interpretation(decision=Decision.EXECUTE, actions=[action(depends_on=["missing"])])


@pytest.mark.parametrize(
    "speech_act",
    [
        SpeechAct.SELF_INTENTION,
        SpeechAct.OBSERVATION,
        SpeechAct.THIRD_PARTY,
        SpeechAct.QUOTED_OR_MEDIA,
    ],
)
def test_negative_speech_acts_never_execute(utterance, speech_act: SpeechAct) -> None:
    settings = Settings(shadow_mode=False, passive_execution_enabled=True)
    outcome = ExecutionPolicy(settings).evaluate(
        utterance,
        interpretation(speech_act=speech_act, decision=Decision.EXECUTE, actions=[action()]),
        session_active=False,
    )
    assert not outcome.execute
    assert not outcome.shadowed


def test_shadow_mode_records_but_never_executes(utterance) -> None:
    settings = Settings(shadow_mode=True, passive_execution_enabled=False)
    outcome = ExecutionPolicy(settings).evaluate(
        utterance,
        interpretation(decision=Decision.EXECUTE, actions=[action()]),
        session_active=False,
    )
    assert not outcome.execute
    assert outcome.shadowed


def test_passive_threshold_is_higher(utterance) -> None:
    settings = Settings(
        shadow_mode=False,
        passive_execution_enabled=True,
        passive_addressed_threshold=0.92,
        active_addressed_threshold=0.55,
    )
    value = interpretation(decision=Decision.EXECUTE, addressed=0.8, actions=[action()])
    assert not ExecutionPolicy(settings).evaluate(utterance, value, session_active=False).execute
    assert ExecutionPolicy(settings).evaluate(utterance, value, session_active=True).execute


def test_low_confidence_interpretation_never_executes_passively(utterance) -> None:
    settings = Settings(shadow_mode=False, passive_execution_enabled=True)
    value = interpretation(
        decision=Decision.EXECUTE,
        confidence=0.7,
        addressed=0.99,
        actions=[action()],
    )

    outcome = ExecutionPolicy(settings).evaluate(utterance, value, session_active=False)

    assert not outcome.execute
    assert "confidence" in outcome.reason


@pytest.mark.parametrize(
    "transcript",
    [
        "I gotta turn the air con on",
        "I need to go and turn the air con on",
        "We are going to switch the lounge lights off",
        "I plan to set the heater temperature",
    ],
)
def test_explicit_self_agency_never_executes(utterance, transcript: str) -> None:
    settings = Settings(shadow_mode=False, passive_execution_enabled=True)
    spoken = utterance.model_copy(update={"transcript": transcript})
    value = interpretation(decision=Decision.EXECUTE, actions=[action()])

    outcome = ExecutionPolicy(settings).evaluate(spoken, value, session_active=True)

    assert not outcome.execute
    assert "speaker" in outcome.reason


@pytest.mark.parametrize(
    "transcript",
    [
        "I want the air con to be on",
        "I need you to turn the air con on",
        "I want Nova to switch the lounge lights off",
    ],
)
def test_desired_state_or_addressee_is_not_self_agency(utterance, transcript: str) -> None:
    settings = Settings(shadow_mode=False, passive_execution_enabled=True)
    spoken = utterance.model_copy(update={"transcript": transcript})
    value = interpretation(decision=Decision.EXECUTE, actions=[action()])

    assert ExecutionPolicy(settings).evaluate(spoken, value, session_active=True).execute


def test_self_agency_normalizes_model_output(utterance) -> None:
    value = interpretation(decision=Decision.EXECUTE, actions=[action()])

    normalized = enforce_speech_cues("I gotta turn the air con on", value)

    assert normalized.speech_act == SpeechAct.SELF_INTENTION
    assert normalized.decision == Decision.IGNORE
    assert normalized.actions == []


def test_explicit_negated_command_cannot_execute(utterance) -> None:
    value = interpretation(decision=Decision.EXECUTE, actions=[action()])

    normalized = enforce_speech_cues("don't turn the air con on", value)

    assert normalized.decision == Decision.IGNORE
    assert normalized.actions == []


@pytest.mark.parametrize(
    "transcript",
    ["shut up", "Be quiet!", "please stop talking", "Okay Bandit, shut up."],
)
def test_direct_speech_interrupt_phrases_are_recognized(transcript: str) -> None:
    assert has_speech_interrupt(transcript)


def test_speech_interrupt_uses_the_configured_wake_word_list() -> None:
    assert has_speech_interrupt("Beamoh, stop talking", ["beemo", "beamoh"])
    assert not has_speech_interrupt("Bandit, stop talking", ["beemo", "beamoh"])


@pytest.mark.parametrize(
    "transcript",
    [
        "He told me to shut up",
        "The television said be quiet",
        "Should I tell them to stop talking?",
        "Shut up and turn the lights off",
    ],
)
def test_indirect_or_compound_interrupt_phrases_do_not_cancel_speech(transcript: str) -> None:
    assert not has_speech_interrupt(transcript)


@pytest.mark.parametrize("speech_act", [SpeechAct.THIRD_PARTY, SpeechAct.QUOTED_OR_MEDIA])
def test_open_conversation_treats_all_user_speech_as_addressed_without_executing(
    utterance,
    speech_act: SpeechAct,
) -> None:
    spoken = utterance.model_copy(
        update={"transcript": "She said to turn it off", "conversation_active": True}
    )
    value = interpretation(
        speech_act=speech_act,
        decision=Decision.IGNORE,
        addressed=0.1,
    )

    normalized = enforce_decision_consistency(spoken, value, addressed_threshold=0.55)

    assert normalized.addressed_probability == 1
    assert normalized.decision == Decision.REPLY
    assert normalized.actions == []


def test_never_mind_ends_an_open_conversation_without_a_followup_question(utterance) -> None:
    spoken = utterance.model_copy(
        update={"transcript": "never mind", "conversation_active": True}
    )
    value = enforce_speech_cues(
        spoken.transcript,
        interpretation(
            speech_act=SpeechAct.UNCLEAR,
            decision=Decision.CLARIFY,
            addressed=1,
        ),
    )

    normalized = enforce_decision_consistency(spoken, value, addressed_threshold=0.55)

    assert normalized.active_goal.status.value == "abandoned"
    assert normalized.decision == Decision.IGNORE
    assert normalized.actions == []


def test_model_cannot_abandon_an_addressed_social_conversation(utterance) -> None:
    spoken = utterance.model_copy(
        update={
            "transcript": "Hey Bandit, are you there?",
            "wake_detected": True,
            "conversation_active": True,
        }
    )
    value = interpretation(
        speech_act=SpeechAct.SOCIAL,
        decision=Decision.IGNORE,
        addressed=1,
    )
    value.active_goal.status = GoalStatus.ABANDONED

    normalized = enforce_decision_consistency(spoken, value, addressed_threshold=0.55)

    assert normalized.decision == Decision.REPLY


@pytest.mark.parametrize("transcript", ["that's all", "that will be all", "we're done"])
def test_natural_conversation_end_phrases_are_abandonment(transcript: str) -> None:
    normalized = enforce_speech_cues(
        transcript,
        interpretation(speech_act=SpeechAct.SOCIAL, decision=Decision.REPLY),
    )

    assert normalized.active_goal.status == GoalStatus.ABANDONED
    assert normalized.decision == Decision.IGNORE
