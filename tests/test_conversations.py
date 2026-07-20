from __future__ import annotations

from nova_voice.audio.conversation import ConversationTracker


class FakeClock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def test_record_observations_dedups_and_bounds() -> None:
    conversations = ConversationTracker(idle_seconds=20)
    conversations.start("lounge")

    conversations.record_observations("lounge", ["lights: on", "lights: on", "  ", "aircon: 22C"])
    snapshot = conversations.snapshot("lounge")
    assert snapshot is not None
    # Consecutive duplicates and blanks are dropped.
    assert snapshot.observations == ("lights: on", "aircon: 22C")

    conversations.record_observations("lounge", [f"item {n}" for n in range(10)], limit=4)
    newest = conversations.snapshot("lounge")
    assert newest is not None
    # Newest entries win once the bound is exceeded.
    assert newest.observations == ("item 6", "item 7", "item 8", "item 9")


def test_message_history_is_bounded_to_recent_turns() -> None:
    conversations = ConversationTracker(idle_seconds=60)
    conversations.start("lounge")
    for n in range(20):
        conversations.append_turn("lounge", f"u{n}", f"a{n}")
    snapshot = conversations.snapshot("lounge")
    assert snapshot is not None
    assert len(snapshot.messages) == ConversationTracker.MESSAGE_HISTORY_LIMIT
    # Newest turns are kept; the oldest have aged out of context.
    assert snapshot.messages[-1].content == "a19"
    assert all(message.content != "u0" for message in snapshot.messages)


def test_record_observations_ignores_unknown_rooms() -> None:
    conversations = ConversationTracker(idle_seconds=20)
    conversations.record_observations("lounge", ["ignored"])
    assert conversations.snapshot("lounge") is None


def test_conversation_expires_after_twenty_seconds_of_inactivity() -> None:
    clock = FakeClock()
    conversations = ConversationTracker(idle_seconds=20, monotonic=clock)

    assert conversations.start("lounge")
    clock.advance(19.999)
    assert conversations.active("lounge")
    clock.advance(0.001)
    assert not conversations.active("lounge")
    assert conversations.snapshot("lounge") is None


def test_refresh_restarts_the_idle_window_and_end_clears_context() -> None:
    clock = FakeClock()
    conversations = ConversationTracker(idle_seconds=20, monotonic=clock)
    conversations.start("lounge")
    clock.advance(15)
    conversations.refresh("lounge")
    clock.advance(15)

    assert conversations.active("lounge")

    conversations.end("lounge")

    assert not conversations.active("lounge")


def test_speaker_template_affinity_lasts_only_for_the_live_conversation() -> None:
    clock = FakeClock()
    conversations = ConversationTracker(idle_seconds=20, monotonic=clock)
    conversations.start("lounge")

    conversations.bind_speaker_template("lounge", "voice-a")
    assert conversations.speaker_template("lounge") == "voice-a"

    clock.advance(20)
    assert conversations.speaker_template("lounge") is None


def test_recognized_speaker_name_and_pronouns_are_retained_per_turn() -> None:
    conversations = ConversationTracker()
    conversations.start("lounge")

    conversations.append_turn(
        "lounge",
        "What is on?",
        "The lounge light.",
        speaker_name="Adeline",
        speaker_pronouns="she/her",
    )

    snapshot = conversations.snapshot("lounge")
    assert snapshot is not None
    assert snapshot.messages[0].speaker_name == "Adeline"
    assert snapshot.messages[0].speaker_pronouns == "she/her"

    conversations.start("lounge")
    assert conversations.speaker_template("lounge") is None


def test_conversation_prompt_and_history_are_isolated_by_room() -> None:
    clock = FakeClock()
    conversations = ConversationTracker(idle_seconds=20, monotonic=clock)
    conversations.start("lounge")
    conversations.start("office")

    first = conversations.initialize_prompt(
        "lounge",
        environment={"now": {"time": "8:14 am"}, "weather": {"condition": "rainy"}},
        personality="Bright and bubbly",
        persona_prompt="Be cheerful.",
    )
    conversations.initialize_prompt(
        "lounge",
        environment={"now": {"time": "9:00 pm"}, "weather": None},
        personality="This must not replace the initial personality",
        persona_prompt="This must not replace the initial persona",
    )
    conversations.append_turn("lounge", "How are you?", "Quite well.")

    lounge = conversations.snapshot("lounge")
    office = conversations.snapshot("office")

    assert first is not None
    assert lounge is not None
    assert lounge.initial_environment == first.initial_environment
    assert lounge.personality == "Bright and bubbly"
    assert lounge.persona_prompt == "Be cheerful."
    assert [(message.role, message.content) for message in lounge.messages] == [
        ("user", "How are you?"),
        ("assistant", "Quite well."),
    ]
    assert office is not None
    assert office.initial_environment is None
    assert office.messages == ()


def test_idle_window_is_live_tunable() -> None:
    clock = FakeClock()
    conversations = ConversationTracker(idle_seconds=20, monotonic=clock)
    conversations.start("lounge")

    conversations.set_idle_seconds(60)
    clock.advance(45)

    assert conversations.active("lounge")
    clock.advance(16)
    assert not conversations.active("lounge")


def test_household_key_shares_one_conversation_across_rooms() -> None:
    clock = FakeClock()
    conversations = ConversationTracker(
        idle_seconds=20, monotonic=clock, key_fn=lambda room_id: "household"
    )
    conversations.start("lounge")
    clock.advance(15)
    # A follow-up elected on the other room's satellite refreshes the same window.
    conversations.refresh("office")
    clock.advance(15)

    assert conversations.active("lounge")
    assert conversations.active("office")

    conversations.end("office")

    assert not conversations.active("lounge")
