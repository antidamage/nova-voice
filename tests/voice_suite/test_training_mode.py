"""Training mode against a live host, in both states.

The suite's speaker override is what makes this runnable at all: the clips are
synthesized speech the household has never heard, so without an override every
case would be an unrecognized voice regardless of the switch.

Switching the household's live setting is a real change to a running house, so
these tests restore whatever they found — including when they fail.
"""

from __future__ import annotations

import pytest

from .runner import TurnRequest

pytestmark = pytest.mark.voice_suite

LOUNGE = {"satellite_id": "indium", "room_id": "lounge"}
COMMAND = "Please turn on the lounge lights for me now"


@pytest.fixture
async def training(harness):
    """Set voice training for the duration of a case, then put it back.

    This changes a setting on a running house, so the restore is in a finally:
    a failing test must not leave the household unable to talk to its own
    assistant.
    """

    async def set_mode(enabled: bool) -> None:
        await harness.set_voice_setting("voiceTrainingEnabled", enabled)

    original = await harness.voice_setting("voiceTrainingEnabled")
    try:
        yield set_mode
    finally:
        if isinstance(original, bool):
            await set_mode(original)


async def test_an_unrecognized_voice_is_ignored_when_training_is_off(harness, training):
    await training(False)

    outcome = await harness.say(
        TurnRequest(phrase=COMMAND, wake_detected=True, speaker_name=None, **LOUNGE)
    )

    # Dropped at the wake word, or ignored at the plan — either is the rule
    # working; what must not happen is a household request.
    assert outcome.dropped or outcome.decision == "ignore", outcome.payload
    assert not outcome.requests


async def test_an_unrecognized_voice_may_command_when_training_is_on(harness, training):
    await training(True)

    outcome = await harness.say(
        TurnRequest(phrase=COMMAND, wake_detected=True, speaker_name=None, **LOUNGE)
    )

    assert outcome.decision == "execute", outcome.payload
    assert outcome.requests


async def test_the_test_voice_override_works_in_either_mode(harness, training):
    for enabled in (True, False):
        await training(enabled)
        await harness.end_conversations()

        outcome = await harness.say(
            TurnRequest(phrase=COMMAND, wake_detected=True, speaker_name="Test Speaker", **LOUNGE)
        )

        assert outcome.decision == "execute", (enabled, outcome.payload)
