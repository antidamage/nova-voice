from __future__ import annotations

import math

import numpy as np

SAMPLE_RATE = 16_000
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2
FRAME_MS = 20
SAMPLES_PER_FRAME = SAMPLE_RATE * FRAME_MS // 1000
BYTES_PER_FRAME = SAMPLES_PER_FRAME * SAMPLE_WIDTH_BYTES


def pcm16_to_float32(payload: bytes) -> np.ndarray:
    if len(payload) % 2:
        raise ValueError("PCM16 payload length must be even")
    return np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0


def float32_to_pcm16(samples: np.ndarray) -> bytes:
    clipped = np.clip(samples, -1, 1)
    return (clipped * 32767).astype("<i2").tobytes()


def scale_pcm16(payload: bytes, percent: float) -> bytes:
    """Apply a linear volume gain (percent of the original level) to PCM16."""

    if percent == 100:
        return payload
    return float32_to_pcm16(pcm16_to_float32(payload) * (percent / 100))


def rms_db(samples: np.ndarray) -> float:
    if samples.size == 0:
        return -120
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    return max(-120, 20 * math.log10(max(rms, 1e-6)))


def peak_db(samples: np.ndarray) -> float:
    if samples.size == 0:
        return -120
    peak = float(np.max(np.abs(samples)))
    return max(-120, 20 * math.log10(max(peak, 1e-6)))
