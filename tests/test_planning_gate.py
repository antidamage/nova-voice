from __future__ import annotations

import pytest

from nova_voice.domain import ActiveGoal, GoalStatus
from nova_voice.interpretation.planning_gate import planning_offered

STATE = {
    "roomDevices": [
        {"name": "Neon lights", "room": "lounge"},
        {"name": "TV light", "room": "lounge"},
    ],
    "climateControls": [{"name": "Panel Heater", "room": "bedroom"}],
    "indoorRooms": ["bedroom", "conservatory"],
    "modes": {"house_party": True},
}


def offered(transcript: str, **kwargs) -> bool:
    kwargs.setdefault("household_state", STATE)
    kwargs.setdefault("active_goal", None)
    return planning_offered(transcript, **kwargs)


@pytest.mark.parametrize(
    "transcript",
    [
        "are you there?",
        "hey, testing. are you there?",
        "tell me a joke",
        "tell me another joke please",
        "how are you doing",
        "how's it going?",
        "thanks mate",
        "thank you!",
        "who are you",
        "what's your name?",
        "good morning",
    ],
)
def test_recognised_small_talk_is_planned_as_a_reply(transcript: str) -> None:
    assert not offered(transcript)


def test_the_wake_word_and_name_are_not_part_of_the_clause() -> None:
    assert not offered("Nova, are you there?", address_words=["beemo", "Nova"])
    assert not offered("hey beemo, how are you", address_words=["beemo", "Nova"])


@pytest.mark.parametrize(
    "transcript",
    [
        # The phrasings a closed action vocabulary dropped: none is small talk.
        "it's too hot in here",
        "I'm cold",
        "it's freezing",
        "hotter",
        "stuffy in here",
        "louder",
        "quieter",
        "too loud",
        "next",
        "previous",
        "I'm leaving",
        "I'm home",
        "I'm back",
        "good night",
        "movie time",
        "is anyone home",
        "who's home",
        "no",
        "both",
        "all of them",
        # Ordinary commands and requests.
        "turn on the lounge lights",
        "what's the temperature in the lounge?",
        "remind me to call mum",
        "tell Sam dinner is ready",
        "what do you think about cats",
        "yes",
        "21",
    ],
)
def test_anything_but_recognised_small_talk_keeps_planning(transcript: str) -> None:
    assert offered(transcript)


@pytest.mark.parametrize(
    "transcript",
    [
        "hey, turn the lights on",
        "thanks, now the heater",
        "hello, house party",
        "good morning, neon please",
        "tell me a joke in five words",
    ],
)
def test_a_chat_clause_beside_a_request_keeps_planning(transcript: str) -> None:
    assert offered(transcript)


@pytest.mark.parametrize(
    "goal",
    [
        ActiveGoal(status=GoalStatus.NEEDS_CLARIFICATION),
        ActiveGoal(status=GoalStatus.IN_PROGRESS),
        ActiveGoal(pending=["a1"]),
    ],
)
def test_an_open_goal_keeps_planning(goal: ActiveGoal) -> None:
    assert offered("thanks", household_state={}, active_goal=goal)


def test_a_settled_goal_does_not_force_planning() -> None:
    goal = ActiveGoal(status=GoalStatus.SATISFIED)

    assert not offered("cheers", household_state={}, active_goal=goal)


def test_an_explicit_lookup_keeps_planning() -> None:
    assert offered("who are you", web_lookup_requested=True)
