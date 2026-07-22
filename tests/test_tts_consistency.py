import numpy as np

from nova_voice.audio.consistency import acoustic_consistency_metrics, consistency_drift


def test_acoustic_consistency_metrics_measure_duration_rate_and_pitch() -> None:
    sample_rate = 24_000
    seconds = 1.0
    timeline = np.arange(int(sample_rate * seconds)) / sample_rate
    samples = (np.sin(2 * np.pi * 200 * timeline) * 0.25 * 32_767).astype("<i2")

    metrics = acoustic_consistency_metrics(
        samples.tobytes(),
        sample_rate,
        text_characters=20,
    )

    assert metrics["durationMs"] == 1_000
    assert metrics["charactersPerSecond"] == 20
    assert 195 <= metrics["medianPitchHz"] <= 205
    assert -16 <= metrics["rmsDb"] <= -14


def test_acoustic_consistency_metrics_tolerate_empty_audio() -> None:
    metrics = acoustic_consistency_metrics(b"", 24_000, text_characters=10)

    assert metrics["durationMs"] == 0
    assert metrics["medianPitchHz"] is None


def test_identical_text_drift_flags_large_pitch_or_rate_changes() -> None:
    stable = consistency_drift(
        {"medianPitchHz": 200.0, "charactersPerSecond": 12.0},
        {"medianPitchHz": 210.0, "charactersPerSecond": 13.0},
    )
    unstable = consistency_drift(
        {"medianPitchHz": 200.0, "charactersPerSecond": 12.0},
        {"medianPitchHz": 145.0, "charactersPerSecond": 8.0},
    )

    assert not stable["consistencyAlert"]
    assert unstable["consistencyAlert"]
