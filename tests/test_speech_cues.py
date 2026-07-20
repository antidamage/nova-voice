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


def test_dismissals_end_the_conversation() -> None:
    # Explicit dismissals the user expects to end a conversation, including when
    # the wake word opened it in the same utterance.
    assert has_abandonment("be quiet")
    assert has_abandonment("Be quiet.")
    assert has_abandonment("shut up")
    assert has_abandonment("stop talking")
    assert has_abandonment("goodbye")
    assert has_abandonment("Good bye!")
    assert has_abandonment("bye")
    assert has_abandonment("goodnight")
    assert has_abandonment("Hey Bandit, be quiet", wake_words=("bandit",))
    assert has_abandonment("beemo, goodbye please", wake_words=("beemo",))


def test_dismissal_words_inside_a_request_do_not_match() -> None:
    # Whole-utterance matching: a dismissal word buried in a real request must
    # not end the conversation.
    assert not has_abandonment("be quiet is what you should tell the lounge speaker")
    assert not has_abandonment("say goodbye to the lights and turn them off")
    assert not has_abandonment("shut up shop and lock the door")


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
