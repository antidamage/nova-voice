from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from nova_voice.audio.pcm import FRAME_MS, pcm16_to_float32, rms_db
from nova_voice.audio.prosody import extract_acoustic_features
from nova_voice.domain import AcousticFeatures


@dataclass(frozen=True)
class SpeechSegment:
    pcm16: bytes
    acoustic: AcousticFeatures


class EnergyVad:
    """Diagnostic fallback only; production ambient execution requires Silero VAD."""

    def __init__(self, threshold_db: float = -42) -> None:
        self.threshold_db = threshold_db

    def score(self, frame: bytes) -> float:
        value = rms_db(pcm16_to_float32(frame))
        return max(0.0, min(1.0, (value - self.threshold_db + 12) / 12))


class SileroVad:
    _INFERENCE_SAMPLES = 1_280

    def __init__(self) -> None:
        try:
            import torch
            from silero_vad import load_silero_vad
        except ImportError as error:
            raise RuntimeError("install the audio extra and CUDA/CPU PyTorch") from error
        self._torch = torch
        self._model = load_silero_vad(onnx=False)
        self._samples = np.empty(0, dtype=np.float32)

    def score(self, frame: bytes) -> float:
        self._samples = np.concatenate((self._samples, pcm16_to_float32(frame)))
        # Running the recurrent model for every 20 ms frame cannot keep up
        # with continuous multi-room capture on CPU. Evaluate the freshest
        # 32 ms slice every 80 ms instead; SpeechSegmenter preserves the
        # 20 ms timing for onset and end-of-speech decisions.
        if self._samples.size < self._INFERENCE_SAMPLES:
            return 0
        window = self._samples[-512:]
        self._samples = np.empty(0, dtype=np.float32)
        return float(self._model(self._torch.from_numpy(window), 16_000).item())

    def reset(self) -> None:
        self._samples = np.empty(0, dtype=np.float32)
        self._model.reset_states()


class SpeechSegmenter:
    def __init__(
        self,
        score: Callable[[bytes], float],
        *,
        threshold: float = 0.5,
        pre_roll_ms: int = 400,
        end_silence_ms: int = 600,
        max_utterance_ms: int = 30_000,
    ) -> None:
        self.score = score
        self.threshold = threshold
        self.pre_roll_frames = max(1, pre_roll_ms // FRAME_MS)
        self.end_silence_frames = max(1, end_silence_ms // FRAME_MS)
        self.max_frames = max_utterance_ms // FRAME_MS
        self._pre_roll: deque[bytes] = deque(maxlen=self.pre_roll_frames)
        self._utterance: list[bytes] = []
        self._silence_frames = 0
        self._speaking = False

    def accept(self, frame: bytes) -> SpeechSegment | None:
        probability = self.score(frame)
        speech = probability >= self.threshold
        if not self._speaking:
            self._pre_roll.append(frame)
            if speech:
                self._speaking = True
                self._utterance = list(self._pre_roll)
                self._pre_roll.clear()
            return None

        self._utterance.append(frame)
        self._silence_frames = 0 if speech else self._silence_frames + 1
        if (
            self._silence_frames < self.end_silence_frames
            and len(self._utterance) < self.max_frames
        ):
            return None
        return self._finish()

    def flush(self) -> SpeechSegment | None:
        return self._finish() if self._utterance else None

    def _finish(self) -> SpeechSegment:
        payload = b"".join(self._utterance)
        segment = SpeechSegment(pcm16=payload, acoustic=extract_acoustic_features(payload))
        self._utterance = []
        self._silence_frames = 0
        self._speaking = False
        # Stateful VAD implementations (for example Silero's recurrent
        # model) must start each utterance from a clean state.  Keep the
        # segmenter interface callable-only while honoring an optional reset.
        reset = getattr(self.score, "reset", None)
        if not callable(reset):
            owner = getattr(self.score, "__self__", None)
            reset = getattr(owner, "reset", None)
        if callable(reset):
            reset()
        return segment
