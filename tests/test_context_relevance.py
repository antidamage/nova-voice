from __future__ import annotations

from nova_voice.interpretation.llama_cpp import household_state_is_relevant


def test_household_state_relevant_for_device_and_status_turns() -> None:
    assert household_state_is_relevant("is the aircon still on")
    assert household_state_is_relevant("what's the temperature set to")
    assert household_state_is_relevant("turn off the lounge lights")
    assert household_state_is_relevant("what's on right now")
    assert household_state_is_relevant("how warm is it in here")


def test_household_state_not_relevant_for_unrelated_turns() -> None:
    assert not household_state_is_relevant("tell me a joke")
    assert not household_state_is_relevant("what's your favourite colour")
    assert not household_state_is_relevant("say hello to everyone")
