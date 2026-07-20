from __future__ import annotations

from datetime import datetime, tzinfo
from functools import lru_cache
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="NOVA_VOICE_",
        extra="ignore",
        case_sensitive=False,
    )

    mode: Literal["development"] = "development"
    host: str = "127.0.0.1"
    port: int = Field(default=8766, ge=1, le=65535)
    database_path: Path = Path("data/transcripts.sqlite3")
    retention_hours: float = Field(default=24.0, gt=0, le=24)
    shadow_mode: bool = True
    passive_execution_enabled: bool = False
    audio_enabled: bool = False
    diagnostics_enabled: bool = False
    diagnostics_max_audio_seconds: int = Field(default=30, ge=1, le=120)

    # IANA timezone of the household. Day/night playback volume and the spoken
    # clock context use this rather than the host clock. An explicit empty
    # value falls back to the host's local timezone.
    household_timezone: str | None = "Pacific/Auckland"

    nova_base_url: str = "http://nova.local"
    nova_mcp_token: str | None = None
    nova_contract_version: str = "nova-provider-v1"
    provider_context_timeout_seconds: float = Field(default=0.75, gt=0, le=5)

    llm_base_url: str = "http://127.0.0.1:8765/v1"
    llm_model: str = "Qwen3.5-4B"
    llm_timeout_seconds: float = Field(default=20, gt=0, le=120)

    stt_model: str = "nvidia/nemotron-speech-streaming-en-0.6b"
    stt_model_path: Path | None = None
    stt_stream_chunk_ms: int = Field(default=160, ge=80, le=2000)
    tts_model: str = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    tts_model_path: Path | None = None
    tts_backend: Literal["qwen", "vllm"] = "qwen"
    tts_stream_base_url: str = "http://127.0.0.1:8091"
    tts_sample_rate: int = Field(default=24_000, ge=8_000, le=192_000)
    tts_speaker: str = "Serena"
    tts_language: str = "English"
    tts_device: Literal["cuda", "cpu"] = "cuda"
    # ``auto`` selects BF16 where it is natively supported and stable FP32 on
    # Turing/pre-BF16 CUDA devices. Explicit FP16 requires deployment testing.
    tts_dtype: Literal["auto", "float16", "bfloat16", "float32"] = "auto"
    # Input-handling pipeline: DeepFilterNet3 sidecar (stage 1), playback-echo
    # AEC (stage 2), the narrow simplified-English pass (stage 3), and the
    # wake-word conversation window (stage 4).
    # DeepFilterNet3 runs as a CPU-only sidecar so it cannot contend with the
    # resident STT/LLM/TTS CUDA workloads.  It is best-effort at runtime, but
    # enabled by default when the packaged systemd sidecar is present.
    denoise_base_url: str = "http://127.0.0.1:8092"
    denoise_timeout_seconds: float = Field(default=2.0, gt=0, le=10)
    echo_guard_enabled: bool = True
    echo_correlation_threshold: float = Field(default=0.55, ge=0, le=1)
    narrow_gate_enabled: bool = True
    narrow_gate_max_oov_ratio: float = Field(default=0.34, ge=0, le=1)
    conversation_idle_seconds: float = Field(default=60.0, ge=2, le=600)
    # Multiple satellites in earshot of each other form one arbitration scope:
    # exactly one of them may own an utterance and its spoken response.
    # ``household`` treats every satellite as sharing the same air; ``room``
    # falls back to per-room scoping for acoustically isolated installs.
    arbitration_scope: Literal["household", "room"] = "household"
    election_window_seconds: float = Field(default=0.25, ge=0.05, le=2)
    # How long a winning satellite holds the turn gate before playback length
    # is known, and the absolute ceiling once it is.  The gate silences the
    # other satellites' microphones for the duration of the handled turn.
    turn_gate_initial_hold_seconds: float = Field(default=25.0, ge=1, le=120)
    turn_gate_max_hold_seconds: float = Field(default=60.0, ge=5, le=300)
    # Near-duplicate transcripts inside this window are one utterance heard
    # twice (second microphone, or a single microphone hearing the household
    # repeat itself); only the first is handled.
    dedup_window_seconds: float = Field(default=6.0, ge=0, le=60)
    dedup_similarity: float = Field(default=0.82, ge=0, le=1)
    # Unaddressed speech (no wake word, no open conversation) shorter than
    # this many words is ambient noise: dropped without handling or logging.
    ambient_min_words: int = Field(default=2, ge=1, le=5)
    # An active exchange continues as long as turns arrive inside the idle
    # window.  Set an explicit ceiling only as an operator safety override.
    conversation_max_seconds: float | None = Field(default=None, ge=10, le=3600)

    # Voice dashboard announcements: speaking state and the bounded two-sided
    # transcript are POSTed to Nova without delaying the spoken turn.
    # The audible offset approximates how long after the announcement the
    # first audio actually leaves the satellite speaker (TTS first chunk +
    # WiFi hop + satellite jitter buffer).
    speech_announce_enabled: bool = True
    speech_audible_offset_ms: int = Field(default=450, ge=0, le=5000)
    # The satellite holds this much streamed PCM before starting its player.
    # It absorbs short vLLM/network scheduling bursts; live health reports the
    # pacing deficit so this can be tuned from evidence rather than guesswork.
    # Both this and tts_frame_ms are boot defaults only: the dashboard's
    # VoiceSettings (ttsPrerollMs/ttsFrameMs) override them live once the
    # first settings pull completes, with no service restart required.
    playback_preroll_ms: int = Field(default=700, ge=200, le=2000)
    # Steady-state audio frame size sent to satellites once the fast-start
    # first chunk has gone out. Below this, TCP/WS overhead per frame starts
    # to matter; above it, chunk-to-chunk latency grows.
    tts_frame_ms: int = Field(default=100, ge=20, le=200)

    persona_path: Path = Path("config/persona.example.yaml")
    skills_path: Path = Path("skills")
    tls_cert_path: Path | None = None
    tls_key_path: Path | None = None
    tls_ca_path: Path | None = None

    passive_addressed_threshold: float = Field(default=0.92, ge=0, le=1)
    active_addressed_threshold: float = Field(default=0.55, ge=0, le=1)
    passive_interpretation_threshold: float = Field(default=0.90, ge=0, le=1)
    active_interpretation_threshold: float = Field(default=0.65, ge=0, le=1)
    max_actions_per_turn: int = Field(default=4, ge=1, le=8)
    alias_refresh_seconds: float = Field(default=30, ge=1, le=3600)
    vad_threshold: float = Field(default=0.5, ge=0, le=1)
    vad_end_silence_ms: int = Field(default=600, ge=100, le=3000)
    vad_pre_roll_ms: int = Field(default=400, ge=0, le=2000)

    @field_validator("household_timezone")
    @classmethod
    def validate_household_timezone(cls, value: str | None) -> str | None:
        if value:
            ZoneInfo(value)
        return value

    def household_tzinfo(self) -> tzinfo:
        if self.household_timezone:
            return ZoneInfo(self.household_timezone)
        return datetime.now().astimezone().tzinfo or ZoneInfo("UTC")

    @model_validator(mode="after")
    def validate_execution_safety(self) -> Settings:
        if self.shadow_mode and self.passive_execution_enabled:
            raise ValueError("passive execution cannot be enabled while shadow mode is active")
        if (self.tls_cert_path is None) != (self.tls_key_path is None):
            raise ValueError("TLS certificate and key must be configured together")
        if self.host not in {"127.0.0.1", "localhost", "::1"}:
            if self.tls_cert_path is None or self.tls_ca_path is None:
                raise ValueError("LAN binding requires server TLS and a client CA")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
