from __future__ import annotations

from conftest import interpretation

from nova_voice.audio.conversation import ConversationTracker
from nova_voice.domain import Decision, SpeechAct
from nova_voice.service import turn_extends_conversation


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _tracker(clock: _Clock, *, idle: float = 60.0, maximum: float | None = 300.0):
    return ConversationTracker(idle_seconds=idle, max_seconds=maximum, monotonic=clock)


def test_wake_word_always_extends() -> None:
    assert turn_extends_conversation(
        wake_detected=True,
        interpretation=interpretation(
            speech_act=SpeechAct.UNCLEAR, decision=Decision.IGNORE, addressed=0.0
        ),
        addressed_threshold=0.5,
    )


def test_unintelligible_speech_does_not_extend() -> None:
    # The hours-later bug: "unclear" was not an ambient speech act, so any
    # noise the model could not parse kept refreshing the idle window.
    assert not turn_extends_conversation(
        wake_detected=False,
        interpretation=interpretation(speech_act=SpeechAct.UNCLEAR, decision=Decision.REPLY),
        addressed_threshold=0.5,
    )


def test_ignored_turns_do_not_extend() -> None:
    assert not turn_extends_conversation(
        wake_detected=False,
        interpretation=interpretation(speech_act=SpeechAct.OBSERVATION, decision=Decision.IGNORE),
        addressed_threshold=0.5,
    )


def test_unaddressed_turns_do_not_extend() -> None:
    assert not turn_extends_conversation(
        wake_detected=False,
        interpretation=interpretation(
            speech_act=SpeechAct.OBSERVATION, decision=Decision.REPLY, addressed=0.2
        ),
        addressed_threshold=0.5,
    )


def test_ambient_speech_acts_do_not_extend() -> None:
    for act in (SpeechAct.THIRD_PARTY, SpeechAct.QUOTED_OR_MEDIA, SpeechAct.SELF_INTENTION):
        assert not turn_extends_conversation(
            wake_detected=False,
            interpretation=interpretation(speech_act=act, decision=Decision.REPLY),
            addressed_threshold=0.5,
        )


def test_an_addressed_understood_turn_extends() -> None:
    assert turn_extends_conversation(
        wake_detected=False,
        interpretation=interpretation(speech_act=SpeechAct.QUESTION, decision=Decision.REPLY),
        addressed_threshold=0.5,
    )


def test_the_ceiling_closes_a_continuously_refreshed_conversation() -> None:
    clock = _Clock()
    tracker = _tracker(clock, idle=60.0, maximum=300.0)
    tracker.start("lounge")

    # Engaged speech every 30 seconds keeps the idle window alive forever; only
    # the absolute ceiling can end this.
    for _ in range(9):
        clock.advance(30.0)
        tracker.refresh("lounge")
        assert tracker.active("lounge")

    clock.advance(30.0)
    tracker.refresh("lounge")

    assert not tracker.active("lounge")


def test_refresh_cannot_resurrect_a_conversation_past_the_ceiling() -> None:
    clock = _Clock()
    tracker = _tracker(clock, idle=600.0, maximum=300.0)
    tracker.start("lounge")

    clock.advance(400.0)
    tracker.refresh("lounge")

    assert not tracker.active("lounge")


def test_refresh_still_tolerates_a_slow_in_flight_turn() -> None:
    # Refresh deliberately does not apply the idle clock: a turn that took
    # longer than the window to render and play must not expire before the
    # user has had their follow-up window.
    clock = _Clock()
    tracker = _tracker(clock, idle=60.0, maximum=300.0)
    tracker.start("lounge")

    clock.advance(90.0)
    tracker.refresh("lounge")

    assert tracker.active("lounge")


def test_no_ceiling_configured_keeps_the_old_behaviour() -> None:
    clock = _Clock()
    tracker = _tracker(clock, idle=60.0, maximum=None)
    tracker.start("lounge")

    for _ in range(100):
        clock.advance(30.0)
        tracker.refresh("lounge")

    assert tracker.active("lounge")


def test_set_max_seconds_applies_to_an_open_conversation() -> None:
    clock = _Clock()
    tracker = _tracker(clock, idle=60.0, maximum=None)
    tracker.start("lounge")
    clock.advance(30.0)

    tracker.set_max_seconds(10.0)

    assert not tracker.active("lounge")


def test_the_clock_can_be_refreshed_without_thawing_the_snapshot() -> None:
    clock = _Clock()
    tracker = _tracker(clock)
    tracker.start("lounge")
    tracker.initialize_prompt(
        "lounge",
        environment={"now": "9:00am", "weather": {"condition": "clear"}},
        personality="",
        persona_prompt="",
    )

    tracker.set_environment_value("lounge", "now", "9:20am")

    snapshot = tracker.snapshot("lounge")
    assert snapshot is not None
    assert snapshot.initial_environment == {
        "now": "9:20am",
        "weather": {"condition": "clear"},
    }
