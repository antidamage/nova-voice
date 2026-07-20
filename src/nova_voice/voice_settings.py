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
            "longResponseProbability": {"min": 0.0, "max": 1.0, "step": 0.05, "default": 0.0},
            "ttsPrerollMs": {"min": 20, "max": 2000, "step": 10, "default": 700},
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


class VoicePronouns(BaseModel):
    """The agent's third-person pronouns, in the three forms the language model
    is told to use for itself.

    Each form is stored (and, in the prompt, labelled) by its grammatical role
    so neo-pronoun sets — where one form cannot be inferred from another — are
    represented exactly rather than guessed by the model.
    """

    model_config = ConfigDict(extra="ignore")

    subjective: str = Field(default="they", min_length=1, max_length=20)
    objective: str = Field(default="them", min_length=1, max_length=20)
    possessive: str = Field(default="theirs", min_length=1, max_length=20)

    @model_validator(mode="before")
    @classmethod
    def clean(cls, value: object) -> object:
        # Be permissive: the dashboard already constrains these to short words,
        # but a stray/blank/non-string form must never break the settings pull.
        # Keep only usable strings (casefolded, trimmed, length-capped); any
        # missing form falls back to its neutral default.
        if value is None:
            return {}
        if not isinstance(value, dict):
            return value
        cleaned: dict[str, str] = {}
        for key in ("subjective", "objective", "possessive"):
            form = value.get(key)
            if isinstance(form, str) and form.strip():
                cleaned[key] = form.strip().casefold()[:20]
        return cleaned

    @property
    def slash_form(self) -> str:
        return f"{self.subjective}/{self.objective}/{self.possessive}"


class VoiceAffectations(BaseModel):
    """Deterministic speech quirks applied to the agent's finished reply text.

    Each flag is a dashboard "Affectations" checkbox saved with the voice
    personality. The transforms themselves live in ``nova_voice.affectations``
    and run on the final reply string (spoken and transcribed), not in the
    language-model prompt, so a quirk applies reliably on every turn.
    """

    model_config = ConfigDict(
        alias_generator=_to_camel,
        populate_by_name=True,
        extra="ignore",
    )

    # Drop the first "I"/"we" of each sentence from replies
    # ("I'm checking the weather" -> "Am checking the weather").
    pronoun_drop: bool = False

    @model_validator(mode="before")
    @classmethod
    def clean(cls, value: object) -> object:
        # Permissive like pronouns: a stray/malformed value must never break
        # the settings pull; it just means every quirk is off.
        if not isinstance(value, dict):
            return {}
        return value


class VoiceSettings(BaseModel):
    model_config = ConfigDict(
        alias_generator=_to_camel,
        populate_by_name=True,
        extra="ignore",
    )

    # Display name: what the dashboard shows (title bar, transcripts). Emoji and
    # symbols are allowed, so this is length-bounded only. The spoken/ASR-facing
    # identity comes from ``spoken_name`` (pronunciation, or this as a fallback).
    agent_name: str = Field(
        default="Nova",
        min_length=1,
        max_length=40,
    )
    # Optional plain-text pronunciation of the agent name. Empty means "use the
    # display name". This is the name the ASR is biased toward, the LLM persona
    # refers to, and the TTS speaks — see ``spoken_name``.
    agent_name_pronunciation: str = Field(default="", max_length=40)
    # System-wide voice killswitch. When false, the voice runtime drops every
    # incoming microphone frame and immediately closes any open conversation, so
    # voice is disabled for the entire household until it is turned back on.
    # Default true preserves existing behaviour.
    system_voice_enabled: bool = True
    # Learn local biometric voice templates and use confidently recognized
    # household profiles for conversational personalization.
    speaker_recognition_enabled: bool = True
    # Per-satellite killswitch: satellite ids whose microphone frames the runtime
    # drops. A soft, instant off-switch for a single satellite while testing other
    # devices — the process keeps running (no SSH stop, no fighting its watchdog).
    # Nocturnium is muted by default so a fresh/reset config processes only the
    # primary Indium microphone. An explicit empty list enables every satellite.
    disabled_satellites: list[str] = Field(default_factory=lambda: ["nocturnium"])
    # Global live override for the native satellites' local transport gate.
    # False is a diagnostic mode that streams every captured frame.
    satellite_noise_gate_enabled: bool = True
    speaker: VoiceSpeaker = VoiceSpeaker.RYAN
    language: VoiceLanguage = VoiceLanguage.ENGLISH
    accent: VoiceAccent = VoiceAccent.NEW_ZEALAND
    speech_rate: int = Field(default=100, ge=70, le=130, multiple_of=5)
    pitch: int = Field(default=0, ge=-20, le=20, multiple_of=2)
    emotion: VoiceEmotion = VoiceEmotion.NATURAL
    emotion_mirroring: int = Field(default=100, ge=0, le=200, multiple_of=10)
    # LLM sampling temperature for the spoken-response renderer.  Zero keeps
    # replies deterministic (and TTS-cacheable); higher values add variety.
    temperature: float = Field(default=0.0, ge=0.0, le=5.0)
    # Chance (0-1) that a conversational reply is rendered long-form (two to
    # four sentences) instead of the standard single sentence. Rolled per
    # response; zero keeps every reply short.
    long_response_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    # Maximum spoken-word length of a verified command acknowledgement. The
    # actual length is rolled fresh per reply as a random value in [0, this], so
    # zero yields a silent acknowledgement and higher values allow a short,
    # varied phrase instead of a fixed one-word confirmation.
    command_reply_max_words: int = Field(default=3, ge=0, le=10)
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
    tts_preroll_ms: int = Field(default=700, ge=20, le=2000, multiple_of=10)
    # Steady-state audio frame size sent to satellites once the fast-start
    # first chunk has gone out.
    tts_frame_ms: int = Field(default=100, ge=20, le=200, multiple_of=10)
    # Operator-authored personality description appended to the interpretation
    # and response-rendering system prompts.  Empty string disables it.
    personality: str = Field(default="You are a bright, bubbly helper!", max_length=2000)
    # The agent's third-person pronouns, passed to the language model so it
    # refers to itself correctly. Part of a saved voice personality.
    pronouns: VoicePronouns = Field(default_factory=VoicePronouns)
    # Deterministic speech quirks (dashboard "Affectations" checkboxes) applied
    # to the finished reply text. Part of a saved voice personality.
    affectations: VoiceAffectations = Field(default_factory=VoiceAffectations)
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

    @field_validator("disabled_satellites", mode="before")
    @classmethod
    def normalize_disabled_satellites(cls, value: object) -> object:
        # Permissive: a malformed value must never break the settings pull. It
        # falls back to the safe single-mic default; an explicit empty list still
        # enables every satellite. Ids are casefolded and de-duplicated in order.
        if not isinstance(value, (list, tuple)):
            return ["nocturnium"]
        seen: dict[str, None] = {}
        for item in value:
            if isinstance(item, str) and item.strip():
                seen.setdefault(item.strip().casefold(), None)
        return list(seen)

    @field_validator("agent_name_pronunciation", mode="before")
    @classmethod
    def normalize_pronunciation(cls, value: object) -> object:
        # Be permissive on input: coerce None/blank to "" and trim. The dashboard
        # already constrains this to plain text; a stray value must never break
        # the settings pull.
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        return value

    @property
    def spoken_name(self) -> str:
        """The plain, spoken/ASR-facing agent name.

        Uses the explicit pronunciation when set, otherwise falls back to the
        display name (unchanged behaviour for installs without a pronunciation).
        """

        return self.agent_name_pronunciation.strip() or self.agent_name

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

    def pronoun_instruction(self) -> str:
        """A one-line instruction telling the model its own pronouns.

        Each form is labelled by grammatical role so neo-pronoun sets are used
        correctly rather than inferred from an assumed common set.
        """

        p = self.pronouns
        return (
            "'I' is a valid pronoun for you. When speaking about yourself, use the "
            "first-person pronouns 'I', 'me', and 'my'. "
            f"Your third-person pronouns are {p.slash_form}: subjective '{p.subjective}', "
            f"objective '{p.objective}', possessive '{p.possessive}'; these forms are only for "
            "when the user or someone else refers to you."
        )

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
