"""Offline stutter/dropout metric for a recorded voice response.

Scores a WAV recording (e.g. a microphone capture of the satellite playing a
spoken reply) for playback glitches: short silent gaps inside continuous
speech, which is exactly what an underrun or an engine rebuild sounds like.

Usage:
    python stutter_metric.py recording.wav

Output (JSON on stdout):
    {
      "speech_span_ms": ...,      # first..last speech activity
      "dropouts": ...,            # count of 40-600 ms silent gaps inside speech
      "dropout_total_ms": ...,
      "longest_gap_ms": ...,
      "stutter_score": ...,       # dropouts per 10 s of speech (0 = clean)
      "ok": true/false            # true when stutter_score < 1.0
    }

No external dependencies beyond numpy; reads 16-bit PCM WAV of any rate.
Playwright (or any harness) can shell out to this and assert on the JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np

FRAME_MS = 10
# Gaps shorter than this are normal articulation; longer than the maximum is a
# deliberate pause between sentences rather than a glitch.
MIN_GAP_MS = 40
MAX_GAP_MS = 600
ACTIVITY_MARGIN_DB = 27.0


def load_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())
    if width != 2:
        raise SystemExit(f"expected 16-bit PCM WAV, got sample width {width}")
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, rate


def analyze(samples: np.ndarray, rate: int) -> dict:
    frame = max(1, rate * FRAME_MS // 1000)
    usable = (samples.size // frame) * frame
    if usable == 0:
        return {"error": "recording too short", "ok": False}
    blocks = samples[:usable].reshape(-1, frame)
    rms = np.sqrt(np.mean(np.square(blocks), axis=1))
    db = 20 * np.log10(np.maximum(rms, 1e-7))

    # Reference level: the loud portion of the recording (the spoken reply).
    loud = float(np.percentile(db, 95))
    if loud < -55.0:
        # Nothing but noise floor — the microphone heard no speech at all.
        return {"error": "no speech detected", "ok": False, "peak_db": round(loud, 1)}
    threshold = loud - ACTIVITY_MARGIN_DB
    active = db > threshold
    active_indices = np.flatnonzero(active)
    if active_indices.size < 5:
        return {"error": "no speech detected", "ok": False, "peak_db": loud}

    start, end = int(active_indices[0]), int(active_indices[-1])
    span = active[start : end + 1]
    span_ms = span.size * FRAME_MS

    dropouts = 0
    dropout_total_ms = 0
    longest_gap_ms = 0
    gap = 0
    for value in span:
        if not value:
            gap += 1
            continue
        gap_ms = gap * FRAME_MS
        if MIN_GAP_MS <= gap_ms <= MAX_GAP_MS:
            dropouts += 1
            dropout_total_ms += gap_ms
        if gap_ms > longest_gap_ms:
            longest_gap_ms = gap_ms
        gap = 0

    stutter_score = dropouts / max(span_ms / 10_000, 0.1)
    return {
        "speech_span_ms": int(span_ms),
        "dropouts": int(dropouts),
        "dropout_total_ms": int(dropout_total_ms),
        "longest_gap_ms": int(longest_gap_ms),
        "stutter_score": round(float(stutter_score), 2),
        "peak_db": round(loud, 1),
        "ok": bool(stutter_score < 1.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", type=Path)
    arguments = parser.parse_args()
    samples, rate = load_wav_mono(arguments.wav)
    print(json.dumps(analyze(samples, rate)))


if __name__ == "__main__":
    sys.exit(main())
