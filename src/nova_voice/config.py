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
    durable_database_path: Path | None = None
    household_event_poll_seconds: float = Field(default=1.0, gt=0, le=60)
    household_event_batch_size: int = Field(default=200, ge=1, le=1_000)
    household_event_retention_days: float = Field(default=30, gt=0, le=365)
    structural_telemetry_path: Path | None = None
    retention_hours: float = Field(default=24.0, gt=0, le=24)
    shadow_mode: bool = True
    passive_execution_enabled: bool = False
    audio_enabled: bool = False
    diagnostics_enabled: bool = False
    diagnostics_max_audio_seconds: int = Field(default=30, ge=1, le=120)

    @property
    def effective_durable_database_path(self) -> Path:
        return self.durable_database_path or self.database_path.with_name(
            "durable-agent.sqlite3"
        )

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

    # Web access (optional). The voice LLM can rewrite a spoken request into a
    # clean query and either delegate it to a grounded cloud model (Google
    # Gemini free tier + Search grounding) or fall back to a keyless DuckDuckGo
    # search + local summarize. The live on/off switch, backend choice, and
    # spoken-answer length are dashboard-driven (see VoiceSettings); the fields
    # below are the iridium-side secret, model id, and limits. The Gemini key is
    # a free (no-billing) Google AI Studio key and stays on iridium — it is never
    # sent to, or stored by, the dashboard.
    web_gemini_api_key: str | None = None
    # Free-tier grounded model id. Kept in config (not hardcoded in logic) so it
    # can track Google's model line without a code change; verify current docs.
    web_gemini_model: str = "gemini-2.5-flash"
    web_gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    # Boot default backend before the first dashboard settings pull supplies one.
    # "brave" = the browser-scrape sidecar (Google-tier answers, keyless,
    # non-Google) is the best option and the default; "local" = keyless
    # DuckDuckGo (ddgs) fallback; "gemini" stays in code for anyone with billing
    # but is not the default and is not offered in the dashboard (no Google).
    web_backend_default: Literal["brave", "local", "gemini"] = "brave"
    web_request_timeout_seconds: float = Field(default=8, gt=0, le=30)
    # Brave Search browser-scrape sidecar (ops/websearch_server.py).
    web_search_service_url: str = "http://127.0.0.1:8093"
    web_search_service_timeout_seconds: float = Field(default=15, gt=0, le=60)
    # Keyless local (DuckDuckGo) result count and per-page fetch caps.
    web_search_results: int = Field(default=4, ge=1, le=10)
    web_fetch_max_bytes: int = Field(default=200_000, ge=4096, le=2_000_000)
    # Readable-text excerpt fed to the local summarize pass. Kept small so it
    # fits the LLM's 4096-token context alongside the prompt and history.
    web_max_result_chars: int = Field(default=2000, ge=500, le=40_000)

    stt_model: str = "nvidia/nemotron-speech-streaming-en-0.6b"
    stt_model_path: Path | None = None
    stt_stream_chunk_ms: int = Field(default=160, ge=80, le=2000)
    # Decode-time context biasing (NeMo GPU phrase-boosting tree): the
    # dashboard wake words and agent name are boosted during RNNT decoding so
    # invented names survive the model's internal language model. Alpha is the
    # shallow-fusion weight; zero or disabled keeps stock decoding.
    stt_context_biasing_enabled: bool = True
    stt_boost_alpha: float = Field(default=1.0, ge=0.0, le=10.0)
    # Text-independent household speaker identification. TitaNet stays on CPU
    # so it cannot consume the GPU headroom reserved for ASR/LLM/TTS.
    speaker_recognition_enabled: bool = True
    speaker_model: str = "nvidia/speakerverification_en_titanet_large"
    speaker_model_path: Path | None = Path(
        "/opt/nova-voice/models/speakerverification-en-titanet-large/"
        "speakerverification_en_titanet_large.nemo"
    )
    speaker_min_duration_ms: int = Field(default=1200, ge=500, le=10_000)
    speaker_timeout_seconds: float = Field(default=1.5, gt=0, le=10)
    speaker_candidate_retention_days: int = Field(default=30, ge=1, le=365)
    speaker_activation_samples: int = Field(default=3, ge=1, le=20)
    speaker_match_threshold: float = Field(default=0.65, ge=-1, le=1)
    speaker_match_margin: float = Field(default=0.03, ge=0, le=1)
    speaker_cluster_threshold: float = Field(default=0.60, ge=-1, le=1)
    # Once a wake-opened conversation has a speaker, keep ordinary embedding
    # variance on that template. Only a much lower score indicates a real
    # speaker hand-off within the same conversation.
    speaker_conversation_match_threshold: float = Field(default=0.35, ge=-1, le=1)
    # Very short commands are easy to trigger accidentally from media or a
    # guest's stray speech. Keep them non-executable until the current voice
    # resolves to an enrolled household profile. Longer explicit requests
    # still pass through the normal addressing and execution policy.
    speaker_short_execute_requires_recognition: bool = True
    speaker_short_execute_max_words: int = Field(default=4, ge=1, le=20)
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
    # AEC (stage 2), optional legacy narrow-English prefilter (stage 3), then
    # structured interpretation and the wake-word conversation window.
    # DeepFilterNet3 runs as a CPU-only sidecar so it cannot contend with the
    # resident STT/LLM/TTS CUDA workloads.  It is best-effort at runtime, but
    # enabled by default when the packaged systemd sidecar is present.
    denoise_base_url: str = "http://127.0.0.1:8092"
    denoise_timeout_seconds: float = Field(default=2.0, gt=0, le=10)
    echo_guard_enabled: bool = True
    echo_correlation_threshold: float = Field(default=0.55, ge=0, le=1)
    # Legacy pre-interpretation vocabulary filtering is opt-in. Command intent
    # belongs to the structured interpretation pass; dropping multiword speech
    # here prevents valid commands with names or unusual aliases from ever
    # reaching that classifier.
    narrow_gate_enabled: bool = False
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
    playback_preroll_ms: int = Field(default=700, ge=20, le=2000)
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
    endpointing_enabled: bool = True
    endpoint_wait_threshold: float = Field(default=0.65, ge=0, le=1)
    endpoint_continue_threshold: float = Field(default=0.35, ge=0, le=1)
    endpoint_intermediate_wait_ms: int = Field(default=300, ge=0, le=1000)
    endpoint_max_pause_ms: int = Field(default=1200, ge=100, le=3000)

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
