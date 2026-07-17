from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DEFAULT_WAKE_WORDS = ["beemo", "bimo", "bemo", "beamo", "bmo"]


def _to_camel(value: str) -> str:
    head, *tail = value.split("_")
    return head + "".join(part.capitalize() for part in tail)


class VoiceSpeaker(StrEnum):
    RYAN = "Ryan"
    AIDEN = "Aiden"
    VIVIAN = "Vivian"
    SERENA = "Serena"
    UNCLE_FU = "Uncle_Fu"
    DYLAN = "Dylan"
    ERIC = "Eric"
    ONO_ANNA = "Ono_Anna"
    SOHEE = "Sohee"


class VoiceLanguage(StrEnum):
    AUTO = "Auto"
    ENGLISH = "English"
    CHINESE = "Chinese"
    JAPANESE = "Japanese"
    KOREAN = "Korean"
    GERMAN = "German"
    FRENCH = "French"
    RUSSIAN = "Russian"
    PORTUGUESE = "Portuguese"
    SPANISH = "Spanish"
    ITALIAN = "Italian"


class VoiceAccent(StrEnum):
    VOICE_NATIVE = "voice-native"
    NEW_ZEALAND = "new-zealand"
    AUSTRALIAN = "australian"
    BRITISH = "british"
    AMERICAN = "american"
    IRISH = "irish"
    SCOTTISH = "scottish"


class VoiceEmotion(StrEnum):
    NATURAL = "natural"
    CALM = "calm"
    CHEERFUL = "cheerful"
    EMPATHETIC = "empathetic"
    SERIOUS = "serious"
    DRY = "dry"
    ENERGETIC = "energetic"


ACCENT_INSTRUCTIONS = {
    VoiceAccent.VOICE_NATIVE: "Keep the selected voice's natural accent.",
    VoiceAccent.NEW_ZEALAND: "Use a natural New Zealand accent.",
    VoiceAccent.AUSTRALIAN: "Use a natural Australian accent.",
    VoiceAccent.BRITISH: "Use a natural British accent.",
    VoiceAccent.AMERICAN: "Use a natural American accent.",
    VoiceAccent.IRISH: "Use a natural Irish accent.",
    VoiceAccent.SCOTTISH: "Use a natural Scottish accent.",
}

EMOTION_INSTRUCTIONS = {
    VoiceEmotion.NATURAL: "Keep a natural conversational baseline mood.",
    VoiceEmotion.CALM: "Keep the baseline mood calm and reassuring.",
    VoiceEmotion.CHEERFUL: "Keep the baseline mood cheerful and bright.",
    VoiceEmotion.EMPATHETIC: "Keep the baseline mood warm and empathetic.",
    VoiceEmotion.SERIOUS: "Keep the baseline mood serious and composed.",
    VoiceEmotion.DRY: "Keep the baseline mood dry and understated.",
    VoiceEmotion.ENERGETIC: "Keep the baseline mood energetic and engaged.",
}


SPEAKER_DETAILS = {
    VoiceSpeaker.RYAN: "Dynamic English voice with a strong rhythm",
    VoiceSpeaker.AIDEN: "Sunny American voice with a clear midrange",
    VoiceSpeaker.VIVIAN: "Bright young Chinese voice",
    VoiceSpeaker.SERENA: "Warm, gentle young Chinese voice",
    VoiceSpeaker.UNCLE_FU: "Seasoned, low and mellow Chinese voice",
    VoiceSpeaker.DYLAN: "Youthful Beijing voice with a natural timbre",
    VoiceSpeaker.ERIC: "Lively Chengdu voice with a husky brightness",
    VoiceSpeaker.ONO_ANNA: "Playful, light Japanese voice",
    VoiceSpeaker.SOHEE: "Warm Korean voice with rich emotion",
}


def voice_catalog() -> dict:
    """The voice-agent parameter surface published to the dashboard.

    Iridium is authoritative for what the TTS/LLM stack can actually do; the
    dashboard populates its Voice Agent controls from this payload instead of
    hard-coding model knowledge.
    """

    return {
        "voices": [
            {
                "value": speaker.value,
                "label": speaker.value.replace("_", " "),
                "detail": SPEAKER_DETAILS.get(speaker, ""),
            }
            for speaker in VoiceSpeaker
        ],
        "languages": [language.value for language in VoiceLanguage],
        "accents": [accent.value for accent in VoiceAccent],
        "emotions": [emotion.value for emotion in VoiceEmotion],
        "ranges": {
            "speechRate": {"min": 70, "max": 130, "step": 5, "default": 100},
            "pitch": {"min": -20, "max": 20, "step": 2, "default": 0},
            "emotionMirroring": {"min": 0, "max": 200, "step": 10, "default": 100},
            "temperature": {"min": 0.0, "max": 5.0, "step": 0.1, "default": 0.0},
            "volumeDay": {"min": 0, "max": 100, "step": 5, "default": 100},
            "volumeNight": {"min": 0, "max": 100, "step": 5, "default": 100},
            "conversationIdleSeconds": {"min": 10, "max": 300, "step": 5, "default": 60},
            "ttsPrerollMs": {"min": 200, "max": 2000, "step": 50, "default": 700},
            "ttsFrameMs": {"min": 20, "max": 200, "step": 10, "default": 100},
        },
        "wake": {
            "wordPattern": "^[A-Za-z]{2,24}$",
            "defaultWords": DEFAULT_WAKE_WORDS,
            "defaultPrefixes": "hey ok okay hi hello yo oi",
            "note": (
                "Wake words are matched transcript-first. Add common speech "
                "recognition spellings to the accepted list."
            ),
        },
    }


class VoiceSettings(BaseModel):
    model_config = ConfigDict(
        alias_generator=_to_camel,
        populate_by_name=True,
        extra="ignore",
    )

    agent_name: str = Field(
        default="Nova",
        min_length=1,
        max_length=40,
        pattern=r"^[\w][\w .'-]{0,39}$",
    )
    speaker: VoiceSpeaker = VoiceSpeaker.SERENA
    language: VoiceLanguage = VoiceLanguage.ENGLISH
    accent: VoiceAccent = VoiceAccent.NEW_ZEALAND
    speech_rate: int = Field(default=100, ge=70, le=130, multiple_of=5)
    pitch: int = Field(default=0, ge=-20, le=20, multiple_of=2)
    emotion: VoiceEmotion = VoiceEmotion.NATURAL
    emotion_mirroring: int = Field(default=100, ge=0, le=200, multiple_of=10)
    # LLM sampling temperature for the spoken-response renderer.  Zero keeps
    # replies deterministic (and TTS-cacheable); higher values add variety.
    temperature: float = Field(default=0.0, ge=0.0, le=5.0)
    # Accepted transcript spellings for the spoken wake phrase. Keeping ASR
    # near-misses explicit makes the matching surface visible and tunable.
    wake_words: list[str] = Field(
        default_factory=lambda: list(DEFAULT_WAKE_WORDS),
        min_length=1,
        max_length=12,
    )
    # Space-separated greeting prefixes accepted before the wake word.
    wake_prefixes: str = Field(default="hey ok okay hi hello yo oi", max_length=200)
    # Response playback loudness (percent), applied as DSP gain on synthesized
    # audio before it streams to satellites.  Daytime volume applies from
    # 08:00 local time, nighttime volume from 21:00.
    volume_day: int = Field(default=100, ge=0, le=100, multiple_of=5)
    volume_night: int = Field(default=100, ge=0, le=100, multiple_of=5)
    # How long a wake-opened conversation stays open without another usable
    # turn before context clears and the wake word is required again.
    conversation_idle_seconds: int = Field(default=60, ge=10, le=300, multiple_of=5)
    # The satellite holds this much streamed PCM before starting its player,
    # absorbing short TTS/network scheduling bursts. Lower values start audio
    # sooner at the cost of headroom against a stutter; live health reports
    # the pacing deficit so this can be tuned from evidence.
    tts_preroll_ms: int = Field(default=700, ge=200, le=2000, multiple_of=50)
    # Steady-state audio frame size sent to satellites once the fast-start
    # first chunk has gone out.
    tts_frame_ms: int = Field(default=100, ge=20, le=200, multiple_of=10)
    # Operator-authored personality description appended to the interpretation
    # and response-rendering system prompts.  Empty string disables it.
    personality: str = Field(default="You are a bright, bubbly helper!", max_length=2000)
    # Dashboard-configured satellite room assignments (satellite id -> room
    # id).  The dashboard roster is authoritative: a satellite's own env-file
    # room can be reset by redeploys and is only a fallback.
    satellite_rooms: dict[str, str] = Field(default_factory=dict)
    updated_at: str | None = None

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_wake_word(cls, value: object) -> object:
        if not isinstance(value, dict) or "wakeWords" in value or "wake_words" in value:
            return value
        legacy = value.get("wakeWord", value.get("wake_word"))
        if not isinstance(legacy, str) or not legacy.strip():
            return value
        migrated = dict(value)
        migrated["wakeWords"] = (
            list(DEFAULT_WAKE_WORDS)
            if legacy.strip().casefold() == "beemo"
            else [legacy]
        )
        return migrated

    @field_validator("wake_words", mode="before")
    @classmethod
    def normalize_wake_words(cls, value: object) -> object:
        if not isinstance(value, list):
            raise ValueError("wakeWords must be a list")
        words: list[str] = []
        for item in value:
            if not isinstance(item, str) or not item.strip().isalpha():
                raise ValueError("wake words must contain letters only")
            word = item.strip().casefold()
            if not 2 <= len(word) <= 24:
                raise ValueError("wake words must contain 2 to 24 letters")
            if word in words:
                raise ValueError("wake words must be unique")
            words.append(word)
        return words

    @field_validator("satellite_rooms", mode="before")
    @classmethod
    def normalize_satellite_rooms(cls, value: object) -> object:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("satelliteRooms must be an object")
        rooms: dict[str, str] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not isinstance(item, str):
                continue
            satellite = key.strip().casefold()
            room = item.strip()
            if satellite and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", room):
                rooms[satellite] = room
        return rooms

    @field_validator("temperature", mode="before")
    @classmethod
    def normalize_legacy_temperature(cls, value: object) -> object:
        """Migrate values from the former 0-10 renderer-temperature scale."""

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric = float(value)
            # Values through 5.0 are valid on the current scale. Only values
            # above the new bound can be unambiguously identified as legacy
            # 0-10 values and converted.
            if 5.0 < numeric <= 10:
                return numeric / 10
        return value

    def wake_prefix_list(self) -> list[str]:
        return [prefix for prefix in self.wake_prefixes.split() if prefix]

    @property
    def wake_word(self) -> str:
        """Primary wake word retained for single-label legacy consumers."""

        return self.wake_words[0]

    @property
    def emotion_mirroring_strength(self) -> float:
        return self.emotion_mirroring / 100

    def style_instruction(self) -> str:
        # Pitch is deliberately absent: the TTS model ignores pitch
        # instructions (measured <2% F0 movement for a ±20% request), so the
        # pitch setting is applied as DSP on the synthesized audio instead
        # (nova_voice.audio.pitch).
        return " ".join(
            (
                ACCENT_INSTRUCTIONS[self.accent],
                f"Speak at {self.speech_rate} percent of a natural conversational pace.",
                EMOTION_INSTRUCTIONS[self.emotion],
            )
        )
