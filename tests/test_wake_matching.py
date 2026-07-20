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


def test_fused_greeting_wake_renderings_are_recognised() -> None:
    matcher = WakePhraseMatcher(["nova"], ("hey", "ok"))

    assert matcher.matches("hanova turn on the lights")
    assert matcher.matches("heynova what's the weather")
    # Ordinary words that merely end with the wake word are not addressing.
    assert not matcher.matches("casanova was on television")
    assert not matcher.matches("supernova detected in the news")


def test_rewrite_replaces_wake_word_with_agent_name() -> None:
    matcher = WakePhraseMatcher(["nova"], ("hey", "ok"))

    assert (
        matcher.rewrite("nova, turn on the lounge lights", "Nova")
        == "Nova, turn on the lounge lights"
    )
    assert (
        matcher.rewrite("hey nova what's the weather", "Nova")
        == "hey Nova what's the weather"
    )
    assert matcher.rewrite("no va please stop", "Nova") == "Nova please stop"


def test_rewrite_restores_a_greeting_for_fused_renderings() -> None:
    matcher = WakePhraseMatcher(["nova", "bandit"], ("hey", "ok"))

    assert (
        matcher.rewrite("Hanova turn on the lights", "Nova")
        == "Hey Nova turn on the lights"
    )
    # A fragment that is itself an accepted greeting is kept as spoken.
    assert (
        matcher.rewrite("okbandit lights off please", "Nova")
        == "ok Nova lights off please"
    )
    # An explicit greeting is never doubled by the fused fragment.
    assert (
        matcher.rewrite("hey hanova lights off", "Nova") == "hey Nova lights off"
    )


def test_rewrite_leaves_unaddressed_transcripts_alone() -> None:
    matcher = WakePhraseMatcher(["bandit"], ("hey", "ok"))

    assert (
        matcher.rewrite("the bandit stole the show", "Nova")
        == "the bandit stole the show"
    )
    assert matcher.rewrite("turn on the lights", "Nova") == "turn on the lights"
    assert matcher.rewrite("bandit lights on", "") == "bandit lights on"
