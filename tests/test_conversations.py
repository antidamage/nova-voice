from __future__ import annotations

from nova_voice.audio.conversation import ConversationTracker


class FakeClock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


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
