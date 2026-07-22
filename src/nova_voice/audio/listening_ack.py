from __future__ import annotations

import math
import struct
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class ListeningAcknowledgement:
    sound_id: str
    pcm16: bytes
    sample_rate: int
    implies_completion: bool = False


class ListeningAckController:
    """A bounded, deterministic non-verbal cue for an extended user pause."""

    def __init__(self, *, cooldown_seconds: float = 8.0, sample_rate: int = 16_000) -> None:
        self.cooldown_seconds = max(0.0, cooldown_seconds)
        self.sample_rate = sample_rate
        self._last_played: dict[str, float] = {}

    def choose(self, room_id: str, *, addressed: bool) -> ListeningAcknowledgement | None:
        now = time.monotonic()
        if not addressed or now - self._last_played.get(room_id, -1e9) < self.cooldown_seconds:
            return None
        self._last_played[room_id] = now
        duration_s = 0.07
        frames = int(self.sample_rate * duration_s)
        # A quiet two-tone earcon is recognisable but cannot be mistaken for a
        # word, an action confirmation, or a conversational claim.
        pcm = bytearray()
        for index in range(frames):
            position = index / frames
            envelope = math.sin(math.pi * position) ** 2
            frequency = 660 if position < 0.5 else 780
            phase = 2 * math.pi * frequency * index / self.sample_rate
            sample = int(2_000 * envelope * math.sin(phase))
            pcm.extend(struct.pack("<h", sample))
        return ListeningAcknowledgement(
            "listening-soft-double-tick-v1",
            bytes(pcm),
            self.sample_rate,
        )
