from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import MethodType

import pytest

from nova_voice.inference.stt import NemoSpeechToText, SpeechToText


class _FakeSpeechToText(SpeechToText):
    def __init__(self) -> None:
        self.calls: list[tuple[bytes, int]] = []

    async def transcribe(self, pcm16: bytes, sample_rate: int = 16_000) -> tuple[str, float]:
        self.calls.append((pcm16, sample_rate))
        return "combined", 0.75


async def _chunks() -> AsyncIterator[bytes]:
    yield b"one"
    yield b"two"


@pytest.mark.asyncio
async def test_stream_contract_preserves_sample_rate_and_chunk_order() -> None:
    adapter = _FakeSpeechToText()

    result = await adapter.transcribe_stream(_chunks(), sample_rate=16_000, stream_id="indium")

    assert result == ("combined", 0.75)
    assert adapter.calls == [(b"onetwo", 16_000)]


@pytest.mark.asyncio
async def test_nemo_stream_uses_complete_pcm16_frames() -> None:
    adapter = object.__new__(NemoSpeechToText)
    adapter._streaming_ready = True
    adapter._lock = asyncio.Lock()
    adapter._stream_states = {}
    adapter.stream_chunk_ms = 160
    calls: list[tuple[int, bool, int | None]] = []

    def record_chunk(
        self,
        stream_id: str,
        pcm16: bytes,
        *,
        final: bool = False,
        valid_samples: int | None = None,
    ) -> tuple[str, float]:
        del self, stream_id
        calls.append((len(pcm16), final, valid_samples))
        return "heard", 1.0

    adapter._stream_chunk = MethodType(record_chunk, adapter)

    async def audio() -> AsyncIterator[bytes]:
        yield b"\x01\x00" * 2_600

    result = await adapter.transcribe_stream(audio(), sample_rate=16_000, stream_id="test")

    assert result == ("heard", 1.0)
    assert calls == [
        (5_120, False, None),
        (5_120, True, 40),
    ]
