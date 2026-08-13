"""The reply pass must know about the household when the turn was about it.

The defect these cover: the interpretation pass always receives
``relevantState``, but the reply pass received it only when a regex matched the
raw transcript. A turn could therefore be planned correctly against every zone
and then *spoken* by a model that had been told nothing about any lights — the
assistant denying it knew about lights it had just been asked to change.
"""

from __future__ import annotations

import pytest

from nova_voice.audio.runtime import _format_timings, _slowest_stage
from nova_voice.domain import (
    ActiveGoal,
    CapabilityToolCall,
    Decision,
    Emotion,
    Interpretation,
    PlannedAction,
    SpeechAct,
)
from nova_voice.interpretation.llama_cpp import (
    household_state_is_relevant,
    turn_concerns_household,
)


def _interpretation(*providers: str) -> Interpretation:
    return Interpretation(
        emotion=Emotion(confidence=0.9, intensity=0.1),
        speech_act=SpeechAct.DIRECTIVE,
        addressed_probability=0.95,
        decision=Decision.EXECUTE if providers else Decision.REPLY,
        confidence=0.9,
        active_goal=ActiveGoal(),
        actions=[
            PlannedAction(
                id=f"a{index}",
                order=index,
                call=CapabilityToolCall(
                    provider=provider, tool=f"{provider}.do", arguments={}
                ),
            )
            for index, provider in enumerate(providers)
        ],
    )


# -- the regression itself ----------------------------------------------------


@pytest.mark.parametrize(
    "transcript",
    [
        "make it brighter in here",
        "a bit dimmer please",
        "too dark",
        "go red",
        "warmer colour",
    ],
)
def test_a_planned_household_action_carries_state_whatever_the_words(transcript):
    """The interpretation settles it, not the vocabulary."""

    assert turn_concerns_household(transcript, _interpretation("nova")) is True


def test_planned_action_wins_even_when_the_transcript_says_nothing_device_like():
    """The exact shape of the reported bug: correct plan, mute reply pass."""

    assert household_state_is_relevant("do that thing again") is False
    assert turn_concerns_household("do that thing again", _interpretation("nova")) is True


def test_non_household_providers_do_not_pull_in_household_state():
    """A web search must not drag every zone into the reply prompt."""

    assert turn_concerns_household("who won the cricket", _interpretation("web")) is False


def test_digital_twin_counts_as_household():
    assert turn_concerns_household("how is everything", _interpretation("household_digital_twin"))


def test_falls_back_to_the_transcript_when_nothing_was_planned():
    """Questions plan no actions, so the lexical signal still has to work."""

    assert turn_concerns_household("are the lights on", _interpretation()) is True
    assert turn_concerns_household("tell me a joke", _interpretation()) is False


def test_no_interpretation_available_still_works():
    """The interpret pass calls this before any interpretation exists."""

    assert turn_concerns_household("turn the lights off", None) is True
    assert turn_concerns_household("tell me a joke", None) is False


# -- vocabulary the regex used to miss ---------------------------------------


@pytest.mark.parametrize(
    "transcript",
    [
        "make it brighter",
        "brighter please",
        "can you dim it",
        "dimmer in here",
        "it is too dark",
        "make it darker",
        "turn the brightness down",
        "put the lighting on",
    ],
)
def test_brightness_language_is_recognised(transcript):
    assert household_state_is_relevant(transcript) is True


@pytest.mark.parametrize(
    "transcript",
    # No device verb either, so the regex has nothing at all to go on — which
    # is precisely why the interpretation has to be the deciding signal.
    ["change the colour to blue", "make it blue", "go red", "run the sunset scene"],
)
def test_colour_commands_ride_the_interpretation_not_the_regex(transcript):
    """Colour words are too ambiguous to gate on lexically.

    "What's your favourite colour" is chit-chat, and dumping every zone into
    that turn is exactly what this gate exists to prevent. A real colour
    command plans a household action, so the interpretation carries it.
    """

    assert household_state_is_relevant(transcript) is False
    assert turn_concerns_household(transcript, _interpretation("nova")) is True


def test_asking_about_a_favourite_colour_is_not_a_household_turn():
    assert household_state_is_relevant("what's your favourite colour") is False
    assert turn_concerns_household("what's your favourite colour", _interpretation()) is False


@pytest.mark.parametrize(
    "transcript",
    ["tell me a joke", "what is the capital of France", "how do I poach an egg"],
)
def test_unrelated_turns_still_get_no_household_dump(transcript):
    """The gate exists to stop context being dumped into every turn."""

    assert household_state_is_relevant(transcript) is False
    assert turn_concerns_household(transcript, _interpretation()) is False


# -- stage timing logging -----------------------------------------------------


def test_timings_are_ordered_by_cost_not_by_name():
    formatted = _format_timings({"stt": 120.0, "tts": 940.0, "denoise": 12.0})
    assert formatted.startswith("tts=940.0")
    assert formatted.split() == ["tts=940.0", "stt=120.0", "denoise=12.0"]


def test_trivial_timings_are_omitted():
    """Near-zero phases are noise that pushes the interesting ones off the end."""

    assert "prefetchHit" not in _format_timings({"stt": 90.0, "prefetchHit": 0.0})


def test_missing_timings_do_not_break_the_line():
    assert _format_timings({}) == ""
    assert _slowest_stage({}) == "none"


def test_slowest_stage_ignores_aggregates():
    """`service` contains the phases below it, so naming it says nothing."""

    slowest = _slowest_stage(
        {"stt": 100.0, "service": 5000.0},
        {"interpretation": 812.4, "execution": 40.0, "total": 5200.0},
    )
    assert slowest == "interpretation:812.4"


def test_slowest_stage_survives_a_non_numeric_entry():
    assert _slowest_stage({"stt": 100.0, "note": None}) == "stt:100.0"
