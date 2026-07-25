from __future__ import annotations

import time

import numpy as np

from nova_voice.audio.echo import PlaybackEchoGuard


def test_echo_reference_is_the_recent_nova_playback_output() -> None:
    guard = PlaybackEchoGuard()
    # The guard receives the exact PCM chunk after runtime DSP, before it is
    # handed to the room sink. Health must expose that a live room reference
    # exists without exposing the audio itself.
    guard.note_playback("office", b"\x01\x00" * 2_400, 24_000)

    status = guard.health()
    assert status["enabled"] is True
    assert status["scope"] == "room"
    assert status["references"] == 1
    assert status["activeReferences"] == 1


def test_echo_reference_is_shared_by_every_satellite_in_the_room() -> None:
    guard = PlaybackEchoGuard()
    rate = 16_000
    sample = np.arange(rate * 2)
    pcm16 = (
        np.sin(2 * np.pi * 3 * sample / rate)
        * np.sin(2 * np.pi * 220 * sample / rate)
        * 12_000
    ).astype(np.int16).tobytes()
    guard.note_playback("office", pcm16, rate)
    guard.note_response_text("office", "Why did the toaster go to therapy?")

    # Indium and Nocturnium both pass their shared room id to these checks, so
    # either microphone sees the same reference. A genuinely separate room is
    # isolated and cannot suppress unrelated household speech.
    assert guard.echo_score("office", pcm16, rate) > 0.99
    assert guard.echo_score("lounge", pcm16, rate) == 0
    assert guard.transcript_matches_response("office", "why did the toaster go to therapy")
    assert not guard.transcript_matches_response("lounge", "why did the toaster go to therapy")


# Real reply/echo pairs observed on this household's lounge satellite. The echo is
# re-recognised speech, so it comes back mangled; literal token overlap scored
# these 0.545-0.786 against its own 0.6 bar, leaking the low ones.
_GARBLED_ECHOES = (
    (
        "I am here, ready to roll. no drama just chilling",
        "You're good and you're ready to roll no drama just chillin.",
    ),
    (
        "I would love to spin a tale, but I do not actually tell stories like a human.",
        "I'd love to spit a tail at, but I don't actually tell stories like a human.",
    ),
    (
        "I am going to just wait for my morning coffee to drip, Addie. "
        "What can I help you with?",
        "I'm gonna just win for my morning coffee to drip Ellie. What can I help you?",
    ),
    (
        "I am good, just hanging out. What is on your mind?",
        "But I'm good just you're hanging out what's on your mind.",
    ),
    # Only the tail of a long reply reaches the microphone.
    ("I am here, ready to roll. no drama just chilling", "no drama just chillin"),
)


def test_garbled_echo_is_recognised_despite_asr_mangling() -> None:
    for spoken, heard in _GARBLED_ECHOES:
        guard = PlaybackEchoGuard()
        guard.note_response_text("lounge", spoken)
        verdict = guard.transcript_echo_verdict("lounge", heard)
        assert verdict.matched, (heard, verdict)
        assert verdict.signal in {"coverage", "run", "ratio"}


# The failure that matters more than a leak: going deaf to real requests.
_GENUINE_SPEECH = (
    ("I am good, just hanging out. What is on your mind?", "turn on the lounge lights"),
    # Shares the assistant's words, but is a person answering it.
    ("I am good, just hanging out. What is on your mind?", "yeah I'm good thanks"),
    ("I am good, just hanging out. What is on your mind?", "okay"),
    (
        "I would love to spin a tale, but I do not actually tell stories like a human.",
        "tell me a story about a toaster",
    ),
    ("I am here, ready to roll. no drama just chilling", "play some music in the kitchen"),
    ("The lounge is warm, Adeline.", "set the heater to twenty degrees"),
)


def test_genuine_household_speech_is_never_intercepted() -> None:
    for spoken, heard in _GENUINE_SPEECH:
        guard = PlaybackEchoGuard()
        guard.note_response_text("lounge", spoken)
        verdict = guard.transcript_echo_verdict("lounge", heard)
        assert not verdict.matched, (heard, verdict)


def test_phonetic_keying_collapses_the_common_asr_substitutions() -> None:
    from nova_voice.audio.echo import _phonetic_key

    # Dropped "-g" and homophone vowels must not defeat the comparison.
    assert _phonetic_key("chilling") == _phonetic_key("chillin")
    assert _phonetic_key("tale") == _phonetic_key("tail")
    # Genuinely different words must stay distinct.
    assert _phonetic_key("spin") != _phonetic_key("spit")


def test_transcript_thresholds_are_reported_for_tuning() -> None:
    status = PlaybackEchoGuard().health()
    assert status["transcriptCoverageThreshold"] == 0.6
    assert status["transcriptRunThreshold"] == 4
    assert status["transcriptRatioThreshold"] == 0.55
    assert status["transcriptMinContentWords"] == 3


def test_interception_expires_with_its_window() -> None:
    guard = PlaybackEchoGuard(transcript_window_seconds=0.05)
    guard.note_response_text("lounge", "no drama just chilling")
    assert guard.transcript_matches_response("lounge", "no drama just chillin")
    # A reply the assistant made long ago must not suppress speech forever, or a
    # user echoing its phrasing much later would be silenced.
    time.sleep(0.12)
    assert not guard.transcript_matches_response("lounge", "no drama just chillin")
