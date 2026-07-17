import asyncio
from collections import OrderedDict
from types import MethodType

import httpx
import pytest

from nova_voice.inference.tts import QwenTextToSpeech, VllmQwenTextToSpeech, select_tts_dtype


def test_auto_tts_dtype_uses_bfloat16_when_natively_supported() -> None:
    assert select_tts_dtype("auto", device="cuda", bf16_supported=True) == "bfloat16"


def test_auto_tts_dtype_uses_float32_on_turing() -> None:
    assert select_tts_dtype("auto", device="cuda", bf16_supported=False) == "float32"


def test_explicit_float16_requires_separate_operator_validation() -> None:
    assert select_tts_dtype("float16", device="cuda", bf16_supported=False) == "float16"


@pytest.mark.asyncio
async def test_repeated_tts_response_uses_bounded_pcm_cache() -> None:
    adapter = object.__new__(QwenTextToSpeech)
    adapter._lock = asyncio.Lock()
    adapter._cache = OrderedDict()
    adapter._cache_max_entries = 2
    adapter.speaker = "Ryan"
    adapter.language = "English"
    calls: list[tuple[str, str, str, str]] = []

    def synthesize(self, text: str, instruction: str) -> tuple[bytes, int]:
        calls.append((self.speaker, self.language, text, instruction))
        return f"{self.speaker}:{text}".encode(), 24_000

    adapter._synthesize = MethodType(synthesize, adapter)

    first = await adapter.synthesize("Done.", "Natural")
    second = await adapter.synthesize("Done.", "Natural")
    await adapter.configure(speaker="Aiden", language="English")
    changed_voice = await adapter.synthesize("Done.", "Natural")

    assert first == second == (b"Ryan:Done.", 24_000)
    assert changed_voice == (b"Aiden:Done.", 24_000)
    assert calls == [
        ("Ryan", "English", "Done.", "Natural"),
        ("Aiden", "English", "Done.", "Natural"),
    ]


@pytest.mark.asyncio
async def test_vllm_adapter_requests_true_pcm_stream_and_caches_complete_audio() -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=b"\x01\x00\x02\x00")

    adapter = VllmQwenTextToSpeech(
        "http://tts.local",
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "Serena",
        "English",
    )
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    try:
        first = [item async for item in adapter.synthesize_stream("Hello", "Natural")]
        second = [item async for item in adapter.synthesize_stream("Hello", "Natural")]
    finally:
        await adapter._client.aclose()

    assert first == second == [(b"\x01\x00\x02\x00", 24_000)]
    assert len(requests) == 1
    request = requests[0]
    assert request.url == "http://tts.local/v1/audio/speech"
    body = request.content.decode()
    assert '"voice":"serena"' in body
    assert '"stream":true' in body
    assert '"stream_format":"audio"' in body
    assert '"response_format":"pcm"' in body
