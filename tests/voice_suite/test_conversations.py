"""Multi-turn behaviour: what stays open, what closes, and what still gets done.

Two things are being pinned here. First, that a conversation ends — the
original complaint was one that carried on for hours without a wake word.
Second, that ending it never costs a command: an instruction inside a
conversation is still an instruction, including when it arrives in the middle
of a chatty sentence with two others.
"""

from __future__ import annotations

import asyncio

import pytest

from .assertions import MUTATION_PATHS
from .runner import TurnRequest

pytestmark = pytest.mark.voice_suite

LOUNGE = {"satellite_id": "indium", "room_id": "lounge"}


def _paths(outcome) -> list[str]:
    return [str(item.get("path") or "") for item in outcome.requests]


async def test_follow_up_needs_no_wake_word(harness):
    await harness.say(TurnRequest(phrase="Turn on the lights", wake_detected=True, **LOUNGE))

    follow_up = await harness.say(
        TurnRequest(phrase="Turn off the lights", wake_detected=False, **LOUNGE)
    )

    assert not follow_up.dropped
    assert follow_up.decision == "execute"
    assert "/api/lights/off" in _paths(follow_up)


async def test_ambient_speech_does_not_hold_a_conversation_open(harness, voice_host):
    """The hours-later bug, as a test.

    Background talk used to refresh the idle window, so any noisy room kept a
    conversation alive indefinitely. These turns are addressed to nobody and
    must let the window die on schedule.
    """

    await harness.say(TurnRequest(phrase="Turn on the lights", wake_detected=True, **LOUNGE))

    # Chatter, at a cadence that would have refreshed the old rule forever.
    for _ in range(3):
        await asyncio.sleep(20)
        await harness.say(
            TurnRequest(
                phrase="anyway I told him it was probably fine either way",
                wake_detected=False,
                speaker_name=None,
                **LOUNGE,
            )
        )

    await asyncio.sleep(45)
    stranded = await harness.say(
        TurnRequest(phrase="Turn off the lights", wake_detected=False, **LOUNGE)
    )

    # The window has closed, so an unwaked command is no longer addressed.
    assert not any(path in MUTATION_PATHS or "lights" in path for path in _paths(stranded))


async def test_abandonment_ends_the_conversation_immediately(harness):
    await harness.say(TurnRequest(phrase="Turn on the lights", wake_detected=True, **LOUNGE))
    await harness.say(TurnRequest(phrase="never mind, that's all", wake_detected=False, **LOUNGE))

    stranded = await harness.say(
        TurnRequest(phrase="Turn off the lights", wake_detected=False, **LOUNGE)
    )

    assert "/api/lights/off" not in _paths(stranded)


async def test_a_chained_command_keeps_every_clause(harness):
    outcome = await harness.say(
        TurnRequest(
            phrase=(
                "Tell me the weather and then turn on the kitchen lights, "
                "and turn off the bedroom heater"
            ),
            wake_detected=True,
            **LOUNGE,
        )
    )

    assert outcome.decision == "execute"
    # Both device clauses survived — the failure this catches is answering the
    # conversational part and quietly dropping the rest.
    targets = " ".join(outcome.targets).casefold()
    assert "kitchen" in targets, outcome.results
    assert "heater" in targets, outcome.results
    assert outcome.response_text, "the weather clause should still have been answered"


async def test_a_command_inside_a_conversation_is_still_acted_on(harness):
    await harness.say(TurnRequest(phrase="Tell me the weather", wake_detected=True, **LOUNGE))

    outcome = await harness.say(
        TurnRequest(
            phrase="thanks, that's useful — turn on the kitchen lights",
            wake_detected=False,
            **LOUNGE,
        )
    )

    assert outcome.decision == "execute"
    assert "kitchen" in " ".join(outcome.targets).casefold(), outcome.results


@pytest.mark.slow
async def test_the_conversation_ceiling_eventually_requires_the_wake_word(harness):
    """Continuous engaged speech must still hit the absolute ceiling.

    Genuinely slow — it asserts a five-minute lifetime in real time — so it
    carries its own marker and is deselected by default with `-m "not slow"`.
    Keeping the wall-clock honest is the point: a fake clock would test the
    tracker, which the unit tests already do, rather than the deployed host.
    """

    await harness.say(TurnRequest(phrase="Turn on the lights", wake_detected=True, **LOUNGE))

    # Engaged, addressed follow-ups at half the idle window: under the idle
    # rule alone this conversation would never close.
    deadline = 330
    elapsed = 0
    while elapsed < deadline:
        await asyncio.sleep(30)
        elapsed += 30
        await harness.say(
            TurnRequest(phrase="Tell me the time", wake_detected=False, **LOUNGE)
        )

    stranded = await harness.say(
        TurnRequest(phrase="Turn off the lights", wake_detected=False, **LOUNGE)
    )

    assert "/api/lights/off" not in _paths(stranded)
