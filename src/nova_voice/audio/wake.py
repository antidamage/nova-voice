from __future__ import annotations

import sys
import types
from collections.abc import Callable
from pathlib import Path

import numpy as np

from nova_voice.audio.pcm import pcm16_to_float32


class OpenWakeWordDetector:
    """openWakeWord adapter using ONNX or Google's current LiteRT interpreter."""

    def __init__(
        self,
        model_path: str,
        threshold: float = 0.5,
        feature_model_dir: str | None = None,
    ) -> None:
        framework = "onnx"
        if Path(model_path).suffix.casefold() == ".tflite":
            self._install_litert_compatibility()
            framework = "tflite"
        try:
            from openwakeword.model import Model
        except ImportError as error:
            raise RuntimeError("openwakeword ONNX wrapper is not installed") from error
        self.threshold = threshold
        feature_kwargs = {}
        if feature_model_dir:
            suffix = ".tflite" if framework == "tflite" else ".onnx"
            feature_root = Path(feature_model_dir)
            feature_kwargs = {
                "melspec_model_path": str(feature_root / f"melspectrogram{suffix}"),
                "embedding_model_path": str(feature_root / f"embedding_model{suffix}"),
            }
        self._model = Model(
            wakeword_models=[model_path],
            inference_framework=framework,
            **feature_kwargs,
        )
        self._buffer = np.empty(0, dtype=np.float32)

    def accept(self, frame: bytes) -> bool:
        self._buffer = np.concatenate((self._buffer, pcm16_to_float32(frame)))
        if self._buffer.size < 1_280:
            return False
        window = self._buffer[:1_280]
        self._buffer = self._buffer[1_280:]
        predictions = self._model.predict((window * 32767).astype(np.int16))
        return any(float(score) >= self.threshold for score in predictions.values())

    @staticmethod
    def _install_litert_compatibility() -> None:
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError as error:
            raise RuntimeError("install ai-edge-litert for a TFLite wake model") from error
        package = types.ModuleType("tflite_runtime")
        interpreter = types.ModuleType("tflite_runtime.interpreter")
        interpreter.Interpreter = Interpreter
        package.interpreter = interpreter
        sys.modules.setdefault("tflite_runtime", package)
        sys.modules.setdefault("tflite_runtime.interpreter", interpreter)


WakeDetectorFactory = Callable[[], OpenWakeWordDetector]
