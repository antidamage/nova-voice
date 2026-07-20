from __future__ import annotations

import httpx

from nova_voice.api import create_app
from nova_voice.config import Settings
from nova_voice.voice_settings import VoiceSettings


class FakePreviewTts:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def synthesize(self, text: str, instruction: str) -> tuple[bytes, int]:
        self.calls.append((text, instruction))
        return b"\x00\x00" * 240, 24_000


class FakePreviewAudio:
    def __init__(self, tts: FakePreviewTts) -> None:
        self.tts = tts


class FakeVoiceService:
    def __init__(self, voice_settings: VoiceSettings | None, *, reply: str | None = None) -> None:
        self.voice_settings = voice_settings
        self._reply = reply
        self.asked: list[str | None] = []

    async def render_preview_reply(self, question: str | None = None) -> str | None:
        self.asked.append(question)
        return self._reply


async def request(app, method: str, path: str, **kwargs) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://voice-server.test"
    ) as client:
        return await client.request(method, path, **kwargs)


async def test_voice_preview_speaks_the_generated_reply_as_wav() -> None:
    tts = FakePreviewTts()
    voice = VoiceSettings(
        agentName="Nova",
        accent="british",
        emotion="cheerful",
        speechRate=110,
        pitch=0,
    )
    service = FakeVoiceService(voice, reply="Hello there — lovely to meet you!")
    app = create_app(Settings(), service=service, audio_runtime=FakePreviewAudio(tts))

    response = await request(app, "POST", "/v1/voices/preview", json={})

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.headers["cache-control"] == "no-store"
    assert response.content.startswith(b"RIFF")
    text, instruction = tts.calls[0]
    # The LLM-generated reply is what gets spoken, with the live style instruction.
    assert text == "Hello there — lovely to meet you!"
    assert instruction == voice.style_instruction()
    # A random question was asked (no explicit one supplied).
    assert service.asked == [None]


async def test_voice_preview_passes_through_an_explicit_question() -> None:
    tts = FakePreviewTts()
    service = FakeVoiceService(VoiceSettings(), reply="It is sunny.")
    app = create_app(Settings(), service=service, audio_runtime=FakePreviewAudio(tts))

    response = await request(
        app, "POST", "/v1/voices/preview", json={"question": "What is the weather?"}
    )

    assert response.status_code == 200
    assert service.asked == ["What is the weather?"]
    assert tts.calls[0][0] == "It is sunny."


async def test_voice_preview_falls_back_when_the_model_is_silent() -> None:
    tts = FakePreviewTts()
    service = FakeVoiceService(VoiceSettings(agentName="Aria"), reply=None)
    app = create_app(Settings(), service=service, audio_runtime=FakePreviewAudio(tts))

    response = await request(app, "POST", "/v1/voices/preview", json={})

    assert response.status_code == 200
    # An empty/failed generation still auditions the voice with a named line.
    assert "Aria" in tts.calls[0][0]


async def test_voice_preview_speaks_supplied_text_verbatim_and_bounded() -> None:
    tts = FakePreviewTts()
    service = FakeVoiceService(VoiceSettings(), reply="ignored — verbatim wins")
    app = create_app(Settings(), service=service, audio_runtime=FakePreviewAudio(tts))

    response = await request(
        app, "POST", "/v1/voices/preview", json={"text": "a" * 500}
    )

    assert response.status_code == 200
    # Verbatim text skips the model and is bounded to the preview cap.
    assert service.asked == []
    assert tts.calls[0][0] == "a" * 400


async def test_voice_preview_requires_audio_runtime() -> None:
    app = create_app(
        Settings(),
        service=FakeVoiceService(VoiceSettings()),
        audio_runtime=None,
    )

    response = await request(app, "POST", "/v1/voices/preview", json={})

    assert response.status_code == 503
