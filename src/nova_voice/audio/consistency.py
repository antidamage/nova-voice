from __future__ import annotations

import numpy as np

from nova_voice.audio.pcm import pcm16_to_float32


def acoustic_consistency_metrics(
    pcm16: bytes,
    sample_rate: int,
    *,
    text_characters: int,
) -> dict[str, float | None]:
    """Return compact non-content diagnostics for one synthesized response."""

    samples = pcm16_to_float32(pcm16)
    if sample_rate <= 0 or samples.size == 0:
        return {
            "durationMs": 0.0,
            "charactersPerSecond": None,
            "medianPitchHz": None,
            "rmsDb": None,
        }
    duration_seconds = samples.size / sample_rate
    rms = float(np.sqrt(np.mean(np.square(samples)) + 1e-12))
    pitches: list[float] = []
    frame_size = max(64, sample_rate * 40 // 1000)
    step = max(frame_size, sample_rate // 10)
    min_lag = max(1, sample_rate // 350)
    max_lag = min(frame_size - 2, sample_rate // 70)
    for start in range(0, max(1, samples.size - frame_size + 1), step):
        frame = samples[start : start + frame_size]
        if frame.size < frame_size:
            break
        frame = frame - float(np.mean(frame))
        frame_rms = float(np.sqrt(np.mean(np.square(frame)) + 1e-12))
        if frame_rms < 0.01:
            continue
        correlation = np.correlate(frame, frame, mode="full")[frame_size - 1 :]
        zero_lag = float(correlation[0])
        if zero_lag <= 0 or max_lag <= min_lag:
            continue
        region = correlation[min_lag : max_lag + 1]
        offset = int(np.argmax(region))
        peak = float(region[offset]) / zero_lag
        if peak < 0.25:
            continue
        pitches.append(sample_rate / (min_lag + offset))
    return {
        "durationMs": round(duration_seconds * 1000, 1),
        "charactersPerSecond": (
            round(text_characters / duration_seconds, 2) if duration_seconds else None
        ),
        "medianPitchHz": round(float(np.median(pitches)), 1) if pitches else None,
        "rmsDb": round(20 * float(np.log10(max(rms, 1e-9))), 1),
    }


def consistency_drift(
    previous: dict[str, float | None],
    current: dict[str, float | None],
) -> dict[str, float | bool | None]:
    def percent(key: str) -> float | None:
        before = previous.get(key)
        after = current.get(key)
        if before is None or after is None or before <= 0:
            return None
        return round(abs(after - before) / before * 100, 1)

    pitch = percent("medianPitchHz")
    rate = percent("charactersPerSecond")
    return {
        "pitchDriftPercent": pitch,
        "rateDriftPercent": rate,
        "consistencyAlert": bool(
            (pitch is not None and pitch > 20) or (rate is not None and rate > 25)
        ),
    }
