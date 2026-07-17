"""Small, deterministic prosody features used as LLM interpretation evidence.

This module intentionally does not classify emotion.  It only derives bounded
features from the already framed PCM segment; lexical/contextual evidence and
the model remain responsible for the label.
"""

from __future__ import annotations

import numpy as np

from nova_voice.audio.pcm import pcm16_to_float32, rms_db
from nova_voice.domain import AcousticFeatures

SAMPLE_RATE = 16_000
_FRAME_SAMPLES = 480  # 30 ms


def _pitch_values(samples: np.ndarray) -> np.ndarray:
    values: list[float] = []
    for offset in range(0, samples.size - _FRAME_SAMPLES + 1, _FRAME_SAMPLES):
        frame = samples[offset : offset + _FRAME_SAMPLES]
        frame = frame - frame.mean()
        energy = float(np.sqrt(np.mean(frame * frame)))
        if energy < 0.01:
            continue
        correlation = np.correlate(frame, frame, mode="full")[_FRAME_SAMPLES - 1 :]
        # Human speech fundamentals are approximately 80-400 Hz.
        low, high = SAMPLE_RATE // 400, SAMPLE_RATE // 80
        high = min(high, correlation.size - 1)
        if high <= low:
            continue
        lag = low + int(np.argmax(correlation[low : high + 1]))
        if correlation[lag] / max(correlation[0], 1e-9) >= 0.30:
            values.append(SAMPLE_RATE / lag)
    return np.asarray(values, dtype=np.float32)


def extract_acoustic_features(pcm16: bytes) -> AcousticFeatures:
    samples = pcm16_to_float32(pcm16)
    if samples.size == 0:
        return AcousticFeatures()
    frame_count = max(1, samples.size // _FRAME_SAMPLES)
    frames = samples[: frame_count * _FRAME_SAMPLES]
    if frames.size >= _FRAME_SAMPLES:
        frames = frames.reshape(-1, _FRAME_SAMPLES)
        envelope = np.sqrt(np.mean(frames * frames, axis=1) + 1e-9)
        frame_db = 20 * np.log10(np.maximum(envelope, 1e-6))
        pause_ratio = float(np.mean(frame_db < -45.0))
        # Count sustained energy onsets as a conservative speech-rate proxy.
        active = frame_db > max(-45.0, float(frame_db.max()) - 18.0)
        onsets = int(np.count_nonzero(active[1:] & ~active[:-1]))
        syllables_per_second = float(onsets / max(samples.size / SAMPLE_RATE, 0.1))
    else:
        pause_ratio = 0.0
        syllables_per_second = None

    pitches = _pitch_values(samples)
    if pitches.size:
        reference = 150.0
        relative = 12 * np.log2(pitches / reference)
        pitch_median_relative = float(np.median(relative))
        pitch_range = float(np.ptp(relative))
    else:
        pitch_median_relative = None
        pitch_range = None
    return AcousticFeatures(
        duration_ms=round(samples.size * 1000 / SAMPLE_RATE),
        rms_db=rms_db(samples),
        peak_db=float(20 * np.log10(max(float(np.max(np.abs(samples))), 1e-6))),
        pitch_median_relative=pitch_median_relative,
        pitch_range=pitch_range,
        syllables_per_second=syllables_per_second,
        pause_ratio=pause_ratio,
    )
