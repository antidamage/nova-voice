from __future__ import annotations

import pickle

import numpy as np
import pytest

from nova_voice.inference.speaker import NemoSpeakerEmbedder, SpeakerRecognizer


class _ExplodingEmbedder:
    model_name = "fixture"
    device = "cpu"
    _model = None

    def embed(self, _pcm16: bytes):
        raise pickle.UnpicklingError("incompatible checkpoint")


class _FlakyEmbedder(NemoSpeakerEmbedder):
    """A real embedder whose backend import always succeeds but whose model
    build fails ``fail_times`` before succeeding — the shape of the intermittent
    in-process NeMo restore failure, without needing NeMo/PyTorch installed."""

    def __init__(self, *, fail_times: int) -> None:
        super().__init__("fixture")
        self._remaining_failures = fail_times
        self.import_calls = 0
        self.build_calls = 0

    def _import_backend(self):  # type: ignore[override]
        self.import_calls += 1
        return object(), object()  # stand-in nemo_asr, torch

    def _build_model(self, _nemo_asr, _torch):  # type: ignore[override]
        self.build_calls += 1
        if self._remaining_failures > 0:
            self._remaining_failures -= 1
            raise RuntimeError("'tuple' object is not callable")
        return object()  # a model sentinel; embed() is not exercised here


class _NoBackendEmbedder(NemoSpeakerEmbedder):
    def __init__(self) -> None:
        super().__init__("fixture")
        self.import_calls = 0

    def _import_backend(self):  # type: ignore[override]
        self.import_calls += 1
        raise ImportError("nemo is not installed")


class _RecoveringEmbedder:
    """Minimal recognizer-facing embedder: the first load fails, the next one
    succeeds. Mirrors a transient first-turn load that then self-heals."""

    model_name = "fixture"
    device = "cpu"

    def __init__(self, *, fail_times: int) -> None:
        self._remaining_failures = fail_times
        self._model = None
        self.embed_calls = 0

    def _ensure(self) -> None:
        if self._model is not None:
            return
        if self._remaining_failures > 0:
            self._remaining_failures -= 1
            raise RuntimeError("transient load failure")
        self._model = object()

    def warmup(self) -> bool:
        try:
            self._ensure()
        except Exception:
            return False
        return True

    def embed(self, _pcm16: bytes, sample_rate: int = 16_000) -> np.ndarray:
        self.embed_calls += 1
        self._ensure()
        return np.asarray([1.0, 0.0], dtype=np.float32)


@pytest.mark.asyncio
async def test_unexpected_model_error_fails_softly() -> None:
    recognizer = SpeakerRecognizer(
        object(),  # type: ignore[arg-type]
        "fixture",
        embedder=_ExplodingEmbedder(),  # type: ignore[arg-type]
    )

    embedding = await recognizer.extract(b"\x00\x00" * 20_000, duration_ms=1_250)

    assert embedding is None
    assert recognizer._last_error == "UnpicklingError: incompatible checkpoint"


def test_transient_model_load_failure_is_retried() -> None:
    # A load that fails once (e.g. NeMo's non-reentrant restore racing the audio
    # path) must not disable speaker recognition for the whole process: the next
    # attempt has to be allowed to try again.
    embedder = _FlakyEmbedder(fail_times=1)

    with pytest.raises(RuntimeError, match="failed to load"):
        embedder._load()
    assert embedder._model is None
    assert embedder._deps_missing is False

    embedder._load()  # retried, not latched
    assert embedder._model is not None
    assert embedder.build_calls == 2


def test_missing_dependencies_are_latched_and_not_retried() -> None:
    # Missing Python deps never resolve at runtime, so that failure *should*
    # latch — no point re-importing on every turn.
    embedder = _NoBackendEmbedder()

    with pytest.raises(RuntimeError, match="not installed"):
        embedder._load()
    with pytest.raises(RuntimeError, match="not installed"):
        embedder._load()

    assert embedder._deps_missing is True
    assert embedder.import_calls == 1


def test_warmup_loads_model_before_first_turn() -> None:
    embedder = _FlakyEmbedder(fail_times=0)

    assert embedder.warmup() is True
    assert embedder._model is not None
    assert embedder.build_calls == 1


def test_warmup_failure_is_soft_and_leaves_load_retryable() -> None:
    embedder = _FlakyEmbedder(fail_times=1)

    assert embedder.warmup() is False  # transient failure swallowed, not raised
    assert embedder._model is None
    assert embedder._deps_missing is False

    embedder._load()  # the real first turn still succeeds
    assert embedder._model is not None


@pytest.mark.asyncio
async def test_recognizer_recovers_after_transient_load_failure() -> None:
    # The user-visible regression: recognition (and therefore sample
    # accumulation) must resume on a later turn without restarting the service.
    embedder = _RecoveringEmbedder(fail_times=1)
    recognizer = SpeakerRecognizer(object(), "fixture", embedder=embedder)  # type: ignore[arg-type]

    first = await recognizer.extract(b"\x00\x00" * 20_000, duration_ms=1_250)
    assert first is None
    assert recognizer._last_error is not None

    second = await recognizer.extract(b"\x00\x00" * 20_000, duration_ms=1_250)
    assert second is not None
    assert recognizer._last_error is None


@pytest.mark.asyncio
async def test_recognizer_warmup_preloads_embedder() -> None:
    embedder = _RecoveringEmbedder(fail_times=0)
    recognizer = SpeakerRecognizer(object(), "fixture", embedder=embedder)  # type: ignore[arg-type]

    assert await recognizer.warmup() is True
    assert embedder._model is not None
    assert embedder.embed_calls == 0  # warmup loads without running inference


@pytest.mark.asyncio
async def test_recognizer_warmup_skips_when_disabled() -> None:
    embedder = _RecoveringEmbedder(fail_times=0)
    recognizer = SpeakerRecognizer(
        object(),  # type: ignore[arg-type]
        "fixture",
        enabled=False,
        embedder=embedder,
    )

    assert await recognizer.warmup() is False
    assert embedder._model is None
