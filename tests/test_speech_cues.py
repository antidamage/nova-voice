from __future__ import annotations

from nova_voice.interpretation.speech_cues import has_abandonment


def test_bare_abandonment_phrases_end_the_conversation() -> None:
    assert has_abandonment("Never mind.")
    assert has_abandonment("that's all")
    assert has_abandonment("That will be all")
    assert has_abandonment("that'll be all")
    assert has_abandonment("forget it")
    assert has_abandonment("stop that")
    assert has_abandonment("we're done")


def test_courtesy_and_wake_wrapping_still_matches() -> None:
    assert has_abandonment("Okay, never mind, thanks")
    assert has_abandonment("Hey Bandit, that's all", wake_words=("bandit",))
    assert has_abandonment("cancel that please")
    assert has_abandonment("that's all for now")
    assert has_abandonment("that's all, thank you")


def test_sentences_merely_containing_a_cue_do_not_match() -> None:
    assert not has_abandonment("that's all the lights are for")
    assert not has_abandonment("stop that music in the lounge")
    assert not has_abandonment("forget it ever happened and tell me a story")
    assert not has_abandonment("never mind the weather, turn on the heater")


def test_unrelated_requests_do_not_match() -> None:
    assert not has_abandonment("turn off the lounge lights")
    assert not has_abandonment("tell me a joke")
