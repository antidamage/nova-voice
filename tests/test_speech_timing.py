from nova_voice.audio.speech_timing import (
    DEFAULT_CHARS_PER_SECOND,
    clamp_chars_per_second,
    consonant_onsets_ms,
    estimate_speech_duration_ms,
)


def test_estimate_scales_with_text_length() -> None:
    short = estimate_speech_duration_ms("Hi there.", DEFAULT_CHARS_PER_SECOND)
    long = estimate_speech_duration_ms(
        "The lounge lights are now set to candlelight, as requested.",
        DEFAULT_CHARS_PER_SECOND,
    )
    assert long > short > 0


def test_onsets_are_sorted_within_duration_and_spaced() -> None:
    text = "The lounge lights are now set to candlelight, as requested."
    duration_ms = estimate_speech_duration_ms(text, DEFAULT_CHARS_PER_SECOND)
    onsets = consonant_onsets_ms(text, duration_ms)
    assert onsets, "real sentences must produce pulses"
    assert onsets == sorted(onsets)
    assert onsets[0] >= 0
    assert onsets[-1] <= duration_ms
    gaps = [b - a for a, b in zip(onsets, onsets[1:], strict=False)]
    assert all(gap >= 90 for gap in gaps)


def test_cluster_starts_pulse_once() -> None:
    # "st" and "tr" are single attacks: one onset per cluster, not per letter.
    onsets = consonant_onsets_ms("street", 1000)
    assert len(onsets) == 2  # "str" cluster + trailing "t"


def test_degenerate_inputs_produce_no_onsets() -> None:
    assert consonant_onsets_ms("", 1000) == []
    assert consonant_onsets_ms("aeiou", 0) == []
    assert consonant_onsets_ms("...", 1000) == []


def test_rate_clamp_rejects_nonsense() -> None:
    assert clamp_chars_per_second(0.0) == DEFAULT_CHARS_PER_SECOND
    assert clamp_chars_per_second(float("nan")) == DEFAULT_CHARS_PER_SECOND
    assert clamp_chars_per_second(1.0) == 8.0
    assert clamp_chars_per_second(500.0) == 24.0
