from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from nova_voice.audio.pcm import pcm16_to_float32

# Envelope resolution.  50 blocks/second (20 ms) tracks speech energy closely
# enough to recognise the assistant's own voice while keeping the correlation
# search trivially cheap.
_BLOCKS_PER_SECOND = 50


def _energy_envelope(pcm16: bytes, sample_rate: int) -> np.ndarray:
    samples = pcm16_to_float32(pcm16)
    block = max(1, sample_rate // _BLOCKS_PER_SECOND)
    usable = (samples.size // block) * block
    if usable == 0:
        return np.empty(0, dtype=np.float32)
    blocks = samples[:usable].reshape(-1, block)
    return np.sqrt(np.mean(np.square(blocks), axis=1))


def _normalized_peak_correlation(reference: np.ndarray, segment: np.ndarray) -> float:
    """Peak Pearson correlation of ``segment`` against every lag of ``reference``."""

    if segment.size < _BLOCKS_PER_SECOND // 2 or reference.size < segment.size:
        return 0.0
    segment = segment - segment.mean()
    segment_norm = float(np.linalg.norm(segment))
    if segment_norm == 0:
        return 0.0
    best = 0.0
    # The envelopes are short (tens of seconds at 50 Hz), so a direct sliding
    # window stays well under a millisecond and avoids FFT edge artefacts.
    for offset in range(reference.size - segment.size + 1):
        window = reference[offset : offset + segment.size]
        window = window - window.mean()
        window_norm = float(np.linalg.norm(window))
        if window_norm == 0:
            continue
        value = float(np.dot(window, segment)) / (window_norm * segment_norm)
        if value > best:
            best = value
    return best


@dataclass
class _Reference:
    envelope: list[float] = field(default_factory=list)
    last_chunk_monotonic: float = 0.0
    playback_ends_monotonic: float = 0.0
    response_texts: list[tuple[float, str]] = field(default_factory=list)


class PlaybackEchoGuard:
    """Recognise the assistant's own speech coming back through a room.

    The server knows exactly what audio it streamed into every room.  This
    guard keeps a short energy-envelope history of that playback per room and
    matches candidate microphone segments from any satellite in the room
    against it.  An utterance that is acoustically Nova's own voice is dropped
    before STT/interpretation even when one satellite hears another satellite's
    speaker or playback-active tagging misses a reverberant tail.
    """

    def __init__(
        self,
        *,
        history_seconds: float = 30.0,
        correlation_threshold: float = 0.55,
        transcript_window_seconds: float = 25.0,
    ) -> None:
        self._history_blocks = int(history_seconds * _BLOCKS_PER_SECOND)
        self.correlation_threshold = correlation_threshold
        self.transcript_window = transcript_window_seconds
        self._references: dict[str, _Reference] = {}

    def _reference(self, room_id: str) -> _Reference:
        reference = self._references.get(room_id)
        if reference is None:
            reference = _Reference()
            self._references[room_id] = reference
        return reference

    def note_playback(self, room_id: str, pcm16: bytes, sample_rate: int) -> None:
        reference = self._reference(room_id)
        envelope = _energy_envelope(pcm16, sample_rate)
        reference.envelope.extend(envelope.tolist())
        if len(reference.envelope) > self._history_blocks:
            del reference.envelope[: len(reference.envelope) - self._history_blocks]
        now = time.monotonic()
        reference.last_chunk_monotonic = now
        # Playback on the satellite cannot finish earlier than the audio that
        # has been sent so far takes to play out; arrival is usually faster
        # than realtime, so track the projected acoustic end.
        chunk_seconds = len(pcm16) / 2 / sample_rate
        reference.playback_ends_monotonic = (
            max(reference.playback_ends_monotonic, now) + chunk_seconds
        )

    def note_response_text(self, room_id: str, text: str) -> None:
        reference = self._reference(room_id)
        now = time.monotonic()
        reference.response_texts.append((now, text))
        reference.response_texts = [
            (stamp, value)
            for stamp, value in reference.response_texts
            if now - stamp <= self.transcript_window
        ]

    def playback_recent(self, room_id: str, within_seconds: float = 3.0) -> bool:
        reference = self._references.get(room_id)
        if reference is None:
            return False
        return time.monotonic() <= reference.playback_ends_monotonic + within_seconds

    def echo_score(
        self,
        room_id: str,
        segment_pcm16: bytes,
        sample_rate: int = 16_000,
    ) -> float:
        """Best envelope correlation of a mic segment against recent playback."""

        reference = self._references.get(room_id)
        if reference is None or not reference.envelope:
            return 0.0
        if not self.playback_recent(room_id, within_seconds=4.0):
            return 0.0
        segment = _energy_envelope(segment_pcm16, sample_rate)
        return _normalized_peak_correlation(
            np.asarray(reference.envelope, dtype=np.float32), segment
        )

    def transcript_matches_response(self, room_id: str, transcript: str) -> bool:
        """True when a transcript is largely a repeat of a recent spoken reply."""

        reference = self._references.get(room_id)
        if reference is None or not reference.response_texts:
            return False
        now = time.monotonic()
        heard = {token for token in _tokenize(transcript)}
        if not heard:
            return False
        for stamp, text in reference.response_texts:
            if now - stamp > self.transcript_window:
                continue
            spoken = set(_tokenize(text))
            if not spoken:
                continue
            overlap = len(heard & spoken) / len(heard)
            if overlap >= 0.6:
                return True
        return False

    def health(self) -> dict:
        """Expose whether the server has a live Nova-output AEC reference.

        The reference is populated from the exact post-DSP PCM sent to every
        speaker in the room (``SatelliteAudioRuntime.note_playback``), not from
        a separately re-synthesized copy. Keeping this small status payload in
        ``/health`` makes self-echo diagnosis possible without exposing audio.
        """

        now = time.monotonic()
        active = 0
        references = 0
        for reference in self._references.values():
            references += 1
            if now <= reference.playback_ends_monotonic + 4.0:
                active += 1
        return {
            "enabled": True,
            "scope": "room",
            "references": references,
            "activeReferences": active,
            "correlationThreshold": self.correlation_threshold,
        }


def _tokenize(text: str) -> list[str]:
    return [token for token in "".join(
        char if char.isalpha() else " " for char in text.casefold()
    ).split() if len(token) > 1]
