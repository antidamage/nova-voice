from __future__ import annotations

import pytest

from nova_voice.affectations import apply_affectations, drop_person_pronouns
from nova_voice.voice_settings import VoiceAffectations, VoiceSettings


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # A first-person subjective contraction swaps to its bare verb.
        ("I'm checking the weather.", "Am checking the weather."),
        # Bare "I"/"we" is dropped; second-person and objective forms stay.
        ("I will check the weather for you.", "Will check the weather for you."),
        ("We've turned on your lights.", "Have turned on your lights."),
        # Only the FIRST subjective pronoun in a sentence goes; later ones stay.
        ("I think I can do it.", "Think I can do it."),
        # A period starts a fresh sentence, so each drops its own first one.
        ("I am here. I am ready.", "Am here. Am ready."),
        # Curly apostrophes come straight from the language model.
        ("I’ll get right on it.", "Will get right on it."),
        # Second- and third-person forms are untouched.
        ("You should close the window.", "You should close the window."),
        ("She said they would bring theirs.", "She said they would bring theirs."),
        # Objective/possessive first-person forms are untouched.
        ("It's on me, I promise!", "It's on me, promise!"),
        # Plural subjective works the same way, sentence by sentence.
        ("We are all set. We can go now.", "Are all set. Can go now."),
        # Words merely containing the pronoun letters are untouched.
        ("We were wet.", "Were wet."),
    ],
)
def test_drop_person_pronouns(text: str, expected: str) -> None:
    assert drop_person_pronouns(text) == expected


def test_drop_person_pronouns_never_returns_an_empty_reply() -> None:
    assert drop_person_pronouns("I.") == "I."
    assert drop_person_pronouns("We.") == "We."


def test_apply_affectations_respects_the_flag() -> None:
    quirky = VoiceAffectations(pronoun_drop=True)
    plain = VoiceAffectations()
    assert apply_affectations("I'm here.", quirky) == "Am here."
    assert apply_affectations("I'm here.", plain) == "I'm here."


def test_voice_settings_parse_dashboard_affectations() -> None:
    settings = VoiceSettings.model_validate({"affectations": {"pronounDrop": True}})
    assert settings.affectations.pronoun_drop is True
    assert (
        settings.model_dump(mode="json", by_alias=True)["affectations"]["pronounDrop"]
        is True
    )


def test_voice_settings_affectations_default_off_and_survive_junk() -> None:
    assert VoiceSettings.model_validate({}).affectations.pronoun_drop is False
    junk = VoiceSettings.model_validate({"affectations": "nonsense"})
    assert junk.affectations.pronoun_drop is False
