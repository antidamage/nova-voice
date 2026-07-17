from __future__ import annotations

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
