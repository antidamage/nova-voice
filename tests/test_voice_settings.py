from __future__ import annotations

import httpx
import pytest
from pydantic import ValidationError

from nova_voice.api import create_app
from nova_voice.config import Settings
from nova_voice.domain import Emotion, EmotionLabel
from nova_voice.persona import Persona
from nova_voice.voice_settings import VoiceAccent, VoiceSettings, VoiceSpeaker


def test_dashboard_voice_contract_compiles_deterministic_tts_instructions() -> None:
    settings = VoiceSettings.model_validate(
        {
            "speaker": "Aiden",
            "language": "English",
            "accent": "new-zealand",
            "speechRate": 115,
            "pitch": -8,
            "emotion": "dry",
            "emotionMirroring": 150,
            "updatedAt": "2026-07-15T01:02:03.000Z",
        }
    )

    assert settings.speaker is VoiceSpeaker.AIDEN
    assert settings.accent is VoiceAccent.NEW_ZEALAND
    assert settings.emotion_mirroring_strength == 1.5
    instruction = settings.style_instruction()
    assert "New Zealand accent" in instruction
    assert "115 percent" in instruction
    # Pitch is deliberately absent from the instruction: the TTS model ignores
    # pitch text, so the setting is applied as DSP instead (audio.pitch).
    assert "lower" not in instruction
    assert settings.pitch == -8
    assert "dry and understated" in instruction
    assert settings.model_dump(mode="json", by_alias=True)["speechRate"] == 115


def test_voice_contract_rejects_unknown_prompts_and_off_step_sliders() -> None:
    with pytest.raises(ValidationError):
        VoiceSettings.model_validate({"accent": "sound vaguely Kiwi"})
    with pytest.raises(ValidationError):
        VoiceSettings.model_validate({"speechRate": 117})


def test_persona_applies_baseline_style_and_scaled_conversation_emotion() -> None:
    persona = Persona(
        id="nova",
        display_name="Nova",
        summary="Household operator",
        complaint_budget_sentences=1,
        base_instruction="Natural conversational speech.",
        emotion_mirroring_strength=1,
    ).with_voice_settings(
        VoiceSettings(
            accent="australian",
            emotion="calm",
            emotionMirroring=150,
        )
    )

    instruction = persona.tone_instruction(
        Emotion(label=EmotionLabel.EXCITED, confidence=0.9, intensity=0.4)
    )
    assert "Australian accent" in instruction
    assert "baseline mood calm" in instruction
    assert "Energetic, bright" in instruction
    assert "Emotional intensity 0.60" in instruction


class _DashboardClient:
    def __init__(self, voice: dict) -> None:
        self.voice = voice
        self.collections = 0

    async def voice_settings(self) -> dict:
        self.collections += 1
        return {"voice": self.voice}


class _Provider:
    def __init__(self, client: _DashboardClient) -> None:
        self.client = client


class _Service:
    def __init__(self, client: _DashboardClient) -> None:
        self.nova_provider = _Provider(client)
        self.applied: VoiceSettings | None = None

    def apply_voice_settings(self, settings: VoiceSettings) -> None:
        self.applied = settings


class _Audio:
    def __init__(self) -> None:
        self.applied: VoiceSettings | None = None

    async def apply_voice_settings(self, settings: VoiceSettings) -> None:
        self.applied = settings


@pytest.mark.asyncio
async def test_refresh_signal_fetches_then_applies_nova_voice_settings() -> None:
    client = _DashboardClient(
        {
            "speaker": "Sohee",
            "language": "Korean",
            "accent": "voice-native",
            "speechRate": 90,
            "pitch": 4,
            "emotion": "empathetic",
            "emotionMirroring": 120,
            "temperature": 0.7,
            "agentName": "Beemo",
            "wakeWords": ["beemo", "bimo", "beamoh"],
            "wakePrefixes": "yo hey ok",
            "volumeDay": 85,
            "volumeNight": 25,
            "personality": "You are a dry, deadpan butler.",
        }
    )
    service = _Service(client)
    audio = _Audio()
    app = create_app(Settings(), service=service, audio_runtime=audio)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="https://voice-server.test") as session:
        response = await session.post("/v1/settings/refresh")

    assert response.status_code == 200
    assert response.json()["voice"]["speaker"] == "Sohee"
    assert client.collections == 1
    assert service.applied is not None and service.applied.speech_rate == 90
    assert service.applied.temperature == 0.7
    assert service.applied.agent_name == "Beemo"
    assert service.applied.wake_words == ["beemo", "bimo", "beamoh"]
    assert service.applied.wake_prefix_list() == ["yo", "hey", "ok"]
    assert service.applied.personality == "You are a dry, deadpan butler."
    assert audio.applied is not None and audio.applied.language.value == "Korean"
    assert audio.applied.volume_day == 85
    assert audio.applied.volume_night == 25


def test_voice_contract_rejects_off_step_volumes_and_oversized_personality() -> None:
    with pytest.raises(ValidationError):
        VoiceSettings.model_validate({"volumeDay": 47})
    with pytest.raises(ValidationError):
        VoiceSettings.model_validate({"volumeNight": 120})
    with pytest.raises(ValidationError):
        VoiceSettings.model_validate({"personality": "x" * 2001})
    defaults = VoiceSettings()
    assert defaults.volume_day == 100
    assert defaults.volume_night == 100
    assert defaults.personality == "You are a bright, bubbly helper!"


def test_voice_settings_keeps_current_temperature_scale_and_migrates_legacy_values() -> None:
    settings = VoiceSettings.model_validate({"temperature": 5})
    legacy_settings = VoiceSettings.model_validate({"temperature": 10})

    assert settings.temperature == 5
    assert legacy_settings.temperature == 1


def test_beemo_variants_are_the_default_wake_words() -> None:
    settings = VoiceSettings()
    assert settings.wake_word == "beemo"
    assert settings.wake_words == ["beemo", "bimo", "bemo", "beamo", "bmo"]
    assert settings.wake_prefix_list()[:3] == ["hey", "ok", "okay"]


def test_legacy_wake_word_is_migrated_to_the_list_contract() -> None:
    settings = VoiceSettings.model_validate({"wakeWord": "jarvis"})

    assert settings.wake_words == ["jarvis"]
    assert settings.model_dump(mode="json", by_alias=True)["wakeWords"] == ["jarvis"]


def test_conversation_idle_seconds_is_dashboard_tunable() -> None:
    assert VoiceSettings().conversation_idle_seconds == 60

    settings = VoiceSettings.model_validate({"conversationIdleSeconds": 120})
    assert settings.conversation_idle_seconds == 120

    with pytest.raises(ValidationError):
        VoiceSettings.model_validate({"conversationIdleSeconds": 7})
    with pytest.raises(ValidationError):
        VoiceSettings.model_validate({"conversationIdleSeconds": 63})


def test_conversation_idle_seconds_is_published_in_the_catalog() -> None:
    from nova_voice.voice_settings import voice_catalog

    ranges = voice_catalog()["ranges"]
    assert ranges["conversationIdleSeconds"] == {
        "min": 10,
        "max": 300,
        "step": 5,
        "default": 60,
    }


def test_satellite_rooms_are_normalized_and_forgiving() -> None:
    settings = VoiceSettings.model_validate(
        {"satelliteRooms": {"Indium": "lounge", "nocturnium": "lounge", "bad": "no spaces!"}}
    )
    assert settings.satellite_rooms == {"indium": "lounge", "nocturnium": "lounge"}

    assert VoiceSettings().satellite_rooms == {}
    assert VoiceSettings.model_validate({"satelliteRooms": None}).satellite_rooms == {}
