from __future__ import annotations

from nova_voice.audio.runtime import WakePhraseMatcher


def test_beemo_is_recognised_with_common_greeting_prefixes() -> None:
    matcher = WakePhraseMatcher(
        ["beemo", "bimo", "bemo", "beamo", "bmo"],
        ("hey", "yo", "ok"),
    )
    assert matcher.matches("beemo turn on the lounge lights")
    assert matcher.matches("yo beemo what is the weather")
    assert matcher.matches("okay bimo tell me a joke")
    assert matcher.matches("bee mo, stop talking")


def test_beemo_is_only_accepted_early_and_not_as_an_article_noun() -> None:
    matcher = WakePhraseMatcher("beemo")
    assert not matcher.matches("I watched beemo on television")
    assert not matcher.matches("the beemo character spoke")


def test_only_explicitly_configured_mistranscriptions_are_accepted() -> None:
    matcher = WakePhraseMatcher(["beemo", "beamoh"])

    assert matcher.matches("beamoh turn the lights off")
    assert not matcher.matches("bimo turn the lights off")
