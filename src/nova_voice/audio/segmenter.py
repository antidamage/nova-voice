from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from nova_voice.audio.endpointing import EndpointDecision, EndpointResult, SemanticEndpointDetector
from nova_voice.audio.pcm import FRAME_MS, pcm16_to_float32, rms_db
from nova_voice.audio.prosody import extract_acoustic_features
from nova_voice.domain import AcousticFeatures


@dataclass(frozen=True)
class SpeechSegment:
    pcm16: bytes
    acoustic: AcousticFeatures
    endpoint_decision: EndpointDecision = EndpointDecision.COMPLETE
    endpoint_wait_ms: int = 0
    endpoint_probability: float = 1.0


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
        endpoint_detector: SemanticEndpointDetector | None = None,
    ) -> None:
        self.score = score
        self.threshold = threshold
        self.pre_roll_frames = max(1, pre_roll_ms // FRAME_MS)
        self.end_silence_frames = max(1, end_silence_ms // FRAME_MS)
        self.max_frames = max_utterance_ms // FRAME_MS
        self.endpoint_detector = endpoint_detector
        self._pre_roll: deque[bytes] = deque(maxlen=self.pre_roll_frames)
        self._utterance: list[bytes] = []
        self._silence_frames = 0
        self._speaking = False
        self._endpoint_result: EndpointResult | None = None
        self._endpoint_target_frames = self.end_silence_frames
        self._endpoint_wait_announced = False
        self._interim_transcript = ""

    @property
    def speaking(self) -> bool:
        return self._speaking

    @property
    def endpoint_waiting(self) -> bool:
        return bool(
            self._endpoint_result is not None
            and self._silence_frames < self._endpoint_target_frames
        )

    def consume_endpoint_wait_started(self) -> bool:
        if not self.endpoint_waiting or self._endpoint_wait_announced:
            return False
        self._endpoint_wait_announced = True
        return True

    def set_interim_transcript(self, transcript: str) -> None:
        self._interim_transcript = " ".join(transcript.split()).strip()

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
        if speech:
            self._silence_frames = 0
            self._endpoint_result = None
            self._endpoint_target_frames = self.end_silence_frames
            self._endpoint_wait_announced = False
        else:
            self._silence_frames += 1
        if (
            self._silence_frames < self.end_silence_frames
            and len(self._utterance) < self.max_frames
        ):
            return None
        if (
            len(self._utterance) < self.max_frames
            and self.endpoint_detector is not None
            and self._silence_frames >= self.end_silence_frames
        ):
            if self._endpoint_result is None:
                self._endpoint_result = self.endpoint_detector.decide(
                    b"".join(self._utterance),
                    trailing_silence_ms=self._silence_frames * FRAME_MS,
                    interim_transcript=self._interim_transcript,
                )
                self._endpoint_target_frames = self.end_silence_frames + (
                    self._endpoint_result.additional_wait_ms // FRAME_MS
                )
            if self._silence_frames < self._endpoint_target_frames:
                return None
        return self._finish()

    def flush(self) -> SpeechSegment | None:
        return self._finish() if self._utterance else None

    def _finish(self) -> SpeechSegment:
        payload = b"".join(self._utterance)
        endpoint = self._endpoint_result or EndpointResult(
            EndpointDecision.COMPLETE,
            1.0,
            0,
        )
        segment = SpeechSegment(
            pcm16=payload,
            acoustic=extract_acoustic_features(payload),
            endpoint_decision=endpoint.decision,
            endpoint_wait_ms=max(
                0,
                self._silence_frames * FRAME_MS - self.end_silence_frames * FRAME_MS,
            ),
            endpoint_probability=endpoint.completion_probability,
        )
        self._utterance = []
        self._silence_frames = 0
        self._speaking = False
        self._endpoint_result = None
        self._endpoint_target_frames = self.end_silence_frames
        self._endpoint_wait_announced = False
        self._interim_transcript = ""
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
