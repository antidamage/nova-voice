from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from statistics import median

from nova_voice.audio.pcm import FRAME_MS, pcm16_to_float32, rms_db


@dataclass(frozen=True)
class CapturedAudioFrame:
    payload: bytes
    monotonic_ns: int
    playback_active: bool = False


class LocalActivityGate:
    """Low-cost satellite-side speech/activity gate.

    The gate deliberately keeps a pre-roll and transmits a silence tail. The
    pre-roll protects word onsets while the tail gives Iridium's higher-quality
    Silero VAD enough audio to close the utterance. This gate is only a transport
    filter; central VAD remains authoritative for deciding what is speech.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        threshold_db: float = -48.0,
        noise_margin_db: float = 6.0,
        trigger_ms: int = 60,
        pre_roll_ms: int = 400,
        hangover_ms: int = 800,
        calibration_ms: int = 1_000,
    ) -> None:
        self._lock = threading.RLock()
        self.enabled = enabled
        self.threshold_db = threshold_db
        self.noise_margin_db = noise_margin_db
        self.trigger_frames = max(1, trigger_ms // FRAME_MS)
        self.pre_roll_frames = max(1, pre_roll_ms // FRAME_MS)
        self.hangover_frames = max(1, hangover_ms // FRAME_MS)
        self.calibration_frames = max(0, calibration_ms // FRAME_MS)
        self._pre_roll: deque[CapturedAudioFrame] = deque(maxlen=self.pre_roll_frames)
        self.reset()

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active

    @property
    def is_enabled(self) -> bool:
        with self._lock:
            return self.enabled

    @property
    def noise_floor_db(self) -> float:
        with self._lock:
            return self._noise_floor_db

    @property
    def last_level_db(self) -> float:
        with self._lock:
            return self._last_level_db

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            self.enabled = enabled
            self.reset()

    def reset(self) -> None:
        with self._lock:
            self._pre_roll.clear()
            self._active = False
            self._speech_frames = 0
            self._silence_frames = 0
            self._noise_floor_db = -60.0
            self._last_level_db = -120.0
            self._calibration_remaining = self.calibration_frames
            self._calibration_levels: list[float] = []

    def accept(self, frame: CapturedAudioFrame) -> tuple[CapturedAudioFrame, ...]:
        with self._lock:
            return self._accept(frame)

    def _accept(self, frame: CapturedAudioFrame) -> tuple[CapturedAudioFrame, ...]:
        if not self.enabled:
            return (frame,)

        level_db = rms_db(pcm16_to_float32(frame.payload))
        self._last_level_db = level_db

        if not self._active and self._calibration_remaining > 0:
            self._pre_roll.append(frame)
            self._calibration_levels.append(level_db)
            self._calibration_remaining -= 1
            if self._calibration_remaining == 0:
                self._noise_floor_db = max(-100.0, min(-20.0, median(self._calibration_levels)))
                self._calibration_levels.clear()
            return ()

        activation_db = max(self.threshold_db, self._noise_floor_db + self.noise_margin_db)
        activity = level_db >= activation_db

        if self._active:
            if activity:
                self._silence_frames = 0
            else:
                self._silence_frames += 1
                self._learn_noise_floor(level_db)
            if self._silence_frames >= self.hangover_frames:
                self._active = False
                self._speech_frames = 0
                self._silence_frames = 0
                self._pre_roll.clear()
            return (frame,)

        self._pre_roll.append(frame)
        if activity:
            self._speech_frames += 1
        else:
            self._speech_frames = 0
            self._learn_noise_floor(level_db)

        if self._speech_frames < self.trigger_frames:
            return ()

        self._active = True
        self._speech_frames = 0
        self._silence_frames = 0
        buffered = tuple(self._pre_roll)
        self._pre_roll.clear()
        return buffered

    def _learn_noise_floor(self, level_db: float) -> None:
        # Slow attack avoids teaching a short speech/noise burst into the floor;
        # faster decay lets a newly quiet room recover sensitivity promptly.
        weight = 0.02 if level_db > self._noise_floor_db else 0.08
        self._noise_floor_db = max(
            -100.0,
            min(-20.0, (1.0 - weight) * self._noise_floor_db + weight * level_db),
        )
