from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import numpy as np

from nova_voice.audio.pcm import SAMPLES_PER_FRAME, pcm16_to_float32
from nova_voice.audio.segmenter import SpeechSegment


def energy_envelope(segment: SpeechSegment) -> np.ndarray:
    samples = pcm16_to_float32(segment.pcm16)
    frame_count = samples.size // SAMPLES_PER_FRAME
    if frame_count < 4:
        return np.empty(0, dtype=np.float32)
    framed = samples[: frame_count * SAMPLES_PER_FRAME].reshape(frame_count, SAMPLES_PER_FRAME)
    envelope = np.sqrt(np.mean(np.square(framed), axis=1) + 1e-9)
    deviation = float(envelope.std())
    if deviation < 1e-6:
        return np.empty(0, dtype=np.float32)
    return ((envelope - envelope.mean()) / deviation).astype(np.float32)


def envelopes_match(left: np.ndarray, right: np.ndarray, threshold: float = 0.62) -> bool:
    if left.size < 4 or right.size < 4 or abs(left.size - right.size) > 30:
        return False
    best = -1.0
    for offset in range(-10, 11):
        if offset < 0:
            a, b = left[-offset:], right[: left.size + offset]
        elif offset > 0:
            a, b = left[: right.size - offset], right[offset:]
        else:
            size = min(left.size, right.size)
            a, b = left[:size], right[:size]
        size = min(a.size, b.size)
        if size < 4:
            continue
        correlation = float(np.mean(a[:size] * b[:size]))
        best = max(best, correlation)
    return best >= threshold


@dataclass(frozen=True)
class SegmentCandidate:
    satellite_id: str
    room_id: str
    segment: SpeechSegment
    wake_detected: bool
    envelope: np.ndarray
    ended_monotonic: float = 0.0

    @property
    def quality(self) -> float:
        acoustic = self.segment.acoustic
        signal = acoustic.snr_db if acoustic.snr_db is not None else acoustic.rms_db
        return float(signal) + (20.0 if self.wake_detected else 0.0)

    @property
    def duration_seconds(self) -> float:
        return max(0.0, float(self.segment.acoustic.duration_ms) / 1000.0)


def capture_intervals_overlap(left: SegmentCandidate, right: SegmentCandidate) -> bool:
    """True when two candidates captured substantially the same span of time.

    Different microphones close the same utterance with different VAD
    boundaries, which defeats strict envelope matching.  Two segments whose
    capture intervals overlap by at least half of the shorter segment are the
    same utterance heard twice.
    """

    left_start = left.ended_monotonic - left.duration_seconds
    right_start = right.ended_monotonic - right.duration_seconds
    overlap = min(left.ended_monotonic, right.ended_monotonic) - max(left_start, right_start)
    shorter = min(left.duration_seconds, right.duration_seconds)
    if shorter <= 0:
        return False
    return overlap >= shorter * 0.5


@dataclass
class ElectionGroup:
    created_at: float
    scope_id: str
    candidates: list[SegmentCandidate]
    winner: asyncio.Future[str]
    final_satellite_id: str | None = None
    finalize_task: asyncio.Task | None = field(default=None, repr=False)


class SegmentElection:
    """Short debounce that elects one source for the same heard utterance."""

    def __init__(self, election_window_seconds: float = 0.25, history_seconds: float = 2.0) -> None:
        self.election_window_seconds = election_window_seconds
        self.history_seconds = history_seconds
        self._groups: list[ElectionGroup] = []
        self._lock = asyncio.Lock()

    async def elect(
        self,
        satellite_id: str,
        segment: SpeechSegment,
        *,
        wake_detected: bool,
        room_id: str = "",
        scope_id: str | None = None,
    ) -> bool:
        now = time.monotonic()
        candidate = SegmentCandidate(
            satellite_id=satellite_id,
            room_id=room_id,
            segment=segment,
            wake_detected=wake_detected,
            envelope=energy_envelope(segment),
            ended_monotonic=now,
        )
        scope = scope_id if scope_id is not None else room_id
        async with self._lock:
            self._groups = [
                group for group in self._groups if now - group.created_at <= self.history_seconds
            ]
            # Envelope correlation confirms the same utterance when both VADs
            # cut similar segments; the capture-interval overlap catches the
            # same utterance when they did not (different boundaries, lengths).
            group = next(
                (
                    existing
                    for existing in self._groups
                    if existing.scope_id == scope
                    and (
                        envelopes_match(existing.candidates[0].envelope, candidate.envelope)
                        or capture_intervals_overlap(existing.candidates[0], candidate)
                    )
                ),
                None,
            )
            if group is None:
                loop = asyncio.get_running_loop()
                group = ElectionGroup(now, scope, [candidate], loop.create_future())
                group.finalize_task = asyncio.create_task(self._finalize(group))
                self._groups.append(group)
            elif group.final_satellite_id is not None:
                return group.final_satellite_id == satellite_id
            elif all(item.satellite_id != satellite_id for item in group.candidates):
                group.candidates.append(candidate)

        winner = await asyncio.shield(group.winner)
        return winner == satellite_id

    async def _finalize(self, group: ElectionGroup) -> None:
        await asyncio.sleep(self.election_window_seconds)
        winner = max(group.candidates, key=lambda candidate: candidate.quality)
        group.final_satellite_id = winner.satellite_id
        if not group.winner.done():
            group.winner.set_result(winner.satellite_id)
