from __future__ import annotations

import pickle

import pytest

from nova_voice.inference.speaker import NemoSpeakerEmbedder, SpeakerRecognizer


class _ExplodingEmbedder:
    model_name = "fixture"
    device = "cpu"
    _model = None

    def embed(self, _pcm16: bytes):
        raise pickle.UnpicklingError("incompatible checkpoint")


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


def test_failed_model_load_is_not_retried() -> None:
    embedder = NemoSpeakerEmbedder("fixture")
    cached = RuntimeError("cached failure")
    embedder._load_attempted = True
    embedder._load_error = cached

    with pytest.raises(RuntimeError, match="cached failure") as raised:
        embedder._load()

    assert raised.value is cached
