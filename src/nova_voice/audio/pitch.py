"""DSP pitch shifting for synthesized response audio.

Qwen3-TTS ignores pitch instructions outright — measured on the deployed
0.6B CustomVoice checkpoint, a ±20 percent instructed shift moves the median
F0 by under 2 percent, both in numeric and qualitative phrasings.  The
dashboard's pitch setting is therefore applied as signal processing on the
synthesized PCM instead of as prompt text.

Chunks are shifted with a short left-context carried over from the previous
chunk so successive STFT windows line up and chunk boundaries don't click.
Processing is CPU-bound (librosa phase vocoder); callers on an event loop
must run :meth:`process` in a worker thread.
"""

from __future__ import annotations

import math

import numpy as np

# One STFT window of preceding audio is enough to stabilise the vocoder at
# the chunk seam; the shifted context itself is discarded.
_CONTEXT_SAMPLES = 2048


def pitch_percent_to_semitones(percent: int) -> float:
    """Map the dashboard's ±20 percent pitch offset to vocoder semitones."""

    return 12 * math.log2(1 + percent / 100)


class StreamingPitchShifter:
    """Shift 16-bit mono PCM chunks by a fixed ratio, seam-safely."""

    def __init__(self, percent: int, sample_rate: int) -> None:
        self.semitones = pitch_percent_to_semitones(percent)
        self.sample_rate = sample_rate
        self._context = np.zeros(0, dtype=np.float32)

    def process(self, pcm16: bytes) -> bytes:
        import librosa

        samples = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        if samples.size == 0:
            return pcm16
        context_length = self._context.size
        joined = np.concatenate((self._context, samples))
        shifted = librosa.effects.pitch_shift(
            y=joined, sr=self.sample_rate, n_steps=self.semitones
        )
        self._context = samples[-_CONTEXT_SAMPLES:].copy()
        result = shifted[context_length : context_length + samples.size]
        return (np.clip(result, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
