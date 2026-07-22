from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from nova_voice.audio.pcm import pcm16_to_float32

_TRAILING_WORD = re.compile(r"[a-z']+", re.IGNORECASE)
_UNFINISHED_WORDS = frozenset(
    {
        "a", "an", "and", "as", "at", "because", "but", "by", "for", "from",
        "if", "in", "into", "my", "of", "on", "or", "so", "than", "that", "the",
        "then", "to", "unless", "until", "when", "where", "while", "with", "your",
    }
)


class EndpointDecision(StrEnum):
    COMPLETE = "complete"
    WAIT = "wait"
    CONTINUE = "continue"


@dataclass(frozen=True)
class EndpointResult:
    decision: EndpointDecision
    completion_probability: float
    additional_wait_ms: int


class SemanticEndpointDetector:
    """Audio-native cadence detector evaluated after base VAD silence."""

    def __init__(
        self,
        *,
        wait_threshold: float = 0.65,
        continue_threshold: float = 0.35,
        intermediate_wait_ms: int = 300,
        max_pause_ms: int = 3_500,
        sample_rate: int = 16_000,
    ) -> None:
        if not 0 <= continue_threshold <= wait_threshold <= 1:
            raise ValueError("endpoint thresholds must satisfy 0 <= continue <= wait <= 1")
        self.wait_threshold = wait_threshold
        self.continue_threshold = continue_threshold
        self.intermediate_wait_ms = max(0, intermediate_wait_ms)
        self.max_pause_ms = max(self.intermediate_wait_ms, max_pause_ms)
        self.sample_rate = sample_rate

    def decide(
        self,
        pcm16: bytes,
        *,
        trailing_silence_ms: int,
        interim_transcript: str | None = None,
    ) -> EndpointResult:
        samples = pcm16_to_float32(pcm16)
        silence_samples = min(
            len(samples),
            max(0, trailing_silence_ms) * self.sample_rate // 1000,
        )
        voiced = samples[:-silence_samples] if silence_samples else samples
        # The final 360 ms carries the completion cadence. Divide it into three
        # equal windows and compare energy: falling cadence is completion-like;
        # rising cadence usually means a clause is continuing after a pause.
        tail = voiced[-max(1, self.sample_rate * 360 // 1000) :]
        if len(tail) < 3:
            probability = 0.5
        else:
            windows = np.array_split(tail, 3)
            energies = [float(np.sqrt(np.mean(np.square(window)) + 1e-12)) for window in windows]
            scale = max(max(energies), 1e-6)
            slope = (energies[-1] - energies[0]) / scale
            duration_ms = len(voiced) * 1000 / self.sample_rate
            duration_bonus = min(0.12, max(0.0, (duration_ms - 500) / 10_000))
            probability = min(1.0, max(0.0, 0.5 - (0.45 * slope) + duration_bonus))

        transcript = (interim_transcript or "").strip()
        if transcript:
            words = _TRAILING_WORD.findall(transcript.casefold())
            unfinished = bool(
                transcript.endswith((",", "-", "—"))
                or (words and words[-1] in _UNFINISHED_WORDS)
            )
            if unfinished:
                probability = min(probability, self.continue_threshold - 0.01)
            elif transcript.endswith((".", "?", "!")):
                probability = max(probability, self.wait_threshold)

        if probability >= self.wait_threshold:
            return EndpointResult(EndpointDecision.COMPLETE, probability, 0)
        if probability <= self.continue_threshold:
            return EndpointResult(
                EndpointDecision.CONTINUE,
                probability,
                self.max_pause_ms,
            )
        return EndpointResult(
            EndpointDecision.WAIT,
            probability,
            self.intermediate_wait_ms,
        )
