"""Single source of truth for Nova's TTS engine modules.

Nova hosts one GPU-resident TTS engine at a time, chosen by an engine-switch.
Engine identity used to be a binary derived four different ways (``tts_backend``
string, an adapter ``engine`` attribute, the API/UI ``classic``/``custom`` pair,
and the switcher's ``dots``/``classic`` targets). Adding a third engine that way
meant editing ~19 server sites and hoping none was missed — a regression trap.

This module makes every one of those sites *read* one ordered registry instead.
Each :class:`EngineSpec` binds an engine id to everything the rest of the system
needs to know about it:

* how to build its :class:`~nova_voice.inference.tts.TextToSpeech` adapter,
* which ``tts_backend`` values map to it,
* its systemd unit / drop-in profile / optional LLM VRAM profile,
* its health endpoint and readiness-check style, and
* the dashboard-facing label and capability flags.

The JSON-serialisable subset (:func:`engines_manifest`) is what the orchestrator
publishes to the dashboard *and* what the root-side ops scripts read (via
``python -m nova_voice.tts_engines dump``), so the app and the shell tooling can
never disagree about what an engine is.

Adding a fourth engine is one :class:`EngineSpec` entry here — not a hunt through
the codebase.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from nova_voice.inference.tts import (
    DotsStreamingTextToSpeech,
    GptSovitsTextToSpeech,
    QwenTextToSpeech,
    TextToSpeech,
    VllmQwenTextToSpeech,
)

if TYPE_CHECKING:
    from nova_voice.config import Settings
    from nova_voice.inference.scheduler import GpuExecutionGate
    from nova_voice.voice_settings import VoiceSettings


@dataclass(frozen=True)
class EngineCapabilities:
    """What voice controls the dashboard should render for an engine.

    Capability flags rather than an engine name keep the dashboard from
    hardcoding ``engine === "custom"`` branches: each engine declares what it
    supports and the UI renders accordingly.
    """

    uses_preset_speaker: bool
    uses_accent_mood: bool
    uses_custom_voice_dropdown: bool
    uses_num_steps: bool
    # "none" = no per-voice catalogue; "clips" = build from uploaded clips
    # (dots zero-shot); "bundle" = upload a trained checkpoint bundle.
    voice_catalogue: str

    def to_manifest(self) -> dict:
        return {
            "usesPresetSpeaker": self.uses_preset_speaker,
            "usesAccentMood": self.uses_accent_mood,
            "usesCustomVoiceDropdown": self.uses_custom_voice_dropdown,
            "usesNumSteps": self.uses_num_steps,
            "voiceCatalogue": self.voice_catalogue,
        }


@dataclass(frozen=True)
class EngineSpec:
    id: str
    """Stable engine id, the dashboard/API vocabulary: classic | custom | trained."""

    label: str
    """Human label shown in the engine picker."""

    backend: str
    """Canonical ``tts_backend`` value for deployment/ops (what the profile sets)."""

    backend_values: tuple[str, ...]
    """Every ``tts_backend`` value that resolves to this engine (classic ← qwen,vllm)."""

    unit: str
    """systemd unit that hosts this engine's model service."""

    profile: str
    """``zz-engine-<id>.conf`` drop-in installed on nova-voice.service for this engine."""

    vram_profile: str | None
    """LLM VRAM-rebalance drop-in to install while resident, or None to leave the LLM full."""

    stream_port: int
    """Localhost port the model service binds (used for the switcher's health poll)."""

    health_style: str
    """"ready-gate" (poll for ``"ready": true``) or "http-200" (any 200 is ready)."""

    voices_base_url_setting: str | None
    """Name of the Settings field holding this engine's voice-service base URL,
    or None if the engine has no per-voice catalogue (e.g. classic presets).
    api.py's generic /v1/voices/{engine_id} relay resolves through this rather
    than hardcoding which engine has which catalogue."""

    capabilities: EngineCapabilities

    build_adapter: Callable[["Settings", "GpuExecutionGate"], TextToSpeech]
    """Construct this engine's TTS adapter from settings (kept out of the manifest)."""

    resolve_speaker: Callable[["VoiceSettings"], str]
    """Pick this engine's voice id from voice settings — each engine has its own namespace."""

    @property
    def health_url(self) -> str:
        return f"http://127.0.0.1:{self.stream_port}/health"

    def to_manifest(self) -> dict:
        """The full ops view: everything the switcher/installer need to act.

        Consumed by ``python -m nova_voice.tts_engines dump`` (the root-side
        switcher and install scripts) — never sent to the browser.
        """
        return {
            "id": self.id,
            "label": self.label,
            "backend": self.backend,
            "backendValues": list(self.backend_values),
            "unit": self.unit,
            "profile": self.profile,
            "vramProfile": self.vram_profile,
            "streamPort": self.stream_port,
            "healthUrl": self.health_url,
            "healthStyle": self.health_style,
            "capabilities": self.capabilities.to_manifest(),
        }

    def to_dashboard_manifest(self) -> dict:
        """The browser-facing view: just id/label/capabilities.

        Ops internals (units, drop-in profiles, ports, health URLs) stay
        server-side — the dashboard only ever sees an abstract engine id plus
        what controls to render for it.
        """
        return {
            "id": self.id,
            "label": self.label,
            "capabilities": self.capabilities.to_manifest(),
        }


# --- Adapter builders -------------------------------------------------------
# Each mirrors exactly what audio/bootstrap.py constructed before this registry
# existed, so moving construction here is behaviour-preserving.


def _build_classic(settings: "Settings", execution_gate: "GpuExecutionGate") -> TextToSpeech:
    if settings.tts_backend == "vllm":
        # The remote OpenAI-compatible server validates its served model id; the
        # local checkpoint path is only meaningful to the in-process backend.
        return VllmQwenTextToSpeech(
            settings.tts_stream_base_url,
            settings.tts_model,
            settings.tts_speaker,
            settings.tts_language,
            sample_rate=settings.tts_sample_rate,
        )
    model_name = str(settings.tts_model_path) if settings.tts_model_path else settings.tts_model
    return QwenTextToSpeech(
        model_name,
        settings.tts_speaker,
        settings.tts_language,
        dtype=settings.tts_dtype,
        device=settings.tts_device,
        execution_gate=execution_gate,
    )


def _build_custom(settings: "Settings", execution_gate: "GpuExecutionGate") -> TextToSpeech:
    # The dots.tts service speaks the same streaming /v1/audio/speech PCM
    # contract, at native 48 kHz. ``tts_speaker`` is a custom-voice id resolved
    # by the service's registry.
    return DotsStreamingTextToSpeech(
        settings.dots_stream_base_url,
        settings.tts_model,
        settings.tts_speaker,
        settings.tts_language,
        sample_rate=settings.dots_sample_rate,
    )


def _build_trained(settings: "Settings", execution_gate: "GpuExecutionGate") -> TextToSpeech:
    # GPT-SoVITS fine-tuned engine: same streaming PCM contract at its native
    # rate. ``tts_speaker`` is a trained-checkpoint id resolved by the trained
    # service's registry.
    return GptSovitsTextToSpeech(
        settings.trained_stream_base_url,
        settings.tts_model,
        settings.tts_speaker,
        settings.tts_language,
        sample_rate=settings.trained_sample_rate,
    )


# --- The registry -----------------------------------------------------------

ENGINES: tuple[EngineSpec, ...] = (
    EngineSpec(
        id="classic",
        label="Classic presets",
        backend="vllm",
        backend_values=("qwen", "vllm"),
        unit="nova-voice-tts.service",
        profile="zz-engine-classic.conf",
        vram_profile=None,
        stream_port=8091,
        health_style="http-200",
        voices_base_url_setting=None,
        capabilities=EngineCapabilities(
            uses_preset_speaker=True,
            uses_accent_mood=True,
            uses_custom_voice_dropdown=False,
            uses_num_steps=False,
            voice_catalogue="none",
        ),
        build_adapter=_build_classic,
        resolve_speaker=lambda vs: vs.speaker.value,
    ),
    EngineSpec(
        id="custom",
        label="Custom voices",
        backend="dots",
        backend_values=("dots",),
        unit="nova-voice-dots-tts.service",
        profile="zz-engine-dots.conf",
        vram_profile="zz-engine-vram-dots.conf",
        stream_port=8095,
        health_style="ready-gate",
        voices_base_url_setting="dots_stream_base_url",
        capabilities=EngineCapabilities(
            uses_preset_speaker=False,
            uses_accent_mood=False,
            uses_custom_voice_dropdown=True,
            uses_num_steps=True,
            voice_catalogue="clips",
        ),
        build_adapter=_build_custom,
        resolve_speaker=lambda vs: vs.custom_speaker,
    ),
    EngineSpec(
        id="trained",
        label="Trained voices",
        backend="gptsovits",
        backend_values=("gptsovits",),
        unit="nova-voice-trained-tts.service",
        profile="zz-engine-trained.conf",
        # GPT-SoVITS serves in ~2 GB (lighter than dots), so the LLM likely need
        # not be trimmed. Left None until measured on the 2080 Ti; add a
        # zz-engine-vram-trained.conf here if a trim proves necessary.
        vram_profile=None,
        stream_port=8096,
        health_style="ready-gate",
        voices_base_url_setting="trained_stream_base_url",
        capabilities=EngineCapabilities(
            uses_preset_speaker=False,
            uses_accent_mood=False,
            uses_custom_voice_dropdown=True,
            uses_num_steps=False,
            voice_catalogue="bundle",
        ),
        build_adapter=_build_trained,
        resolve_speaker=lambda vs: vs.trained_speaker,
    ),
)

_BY_ID: dict[str, EngineSpec] = {spec.id: spec for spec in ENGINES}
_BY_BACKEND: dict[str, EngineSpec] = {
    backend: spec for spec in ENGINES for backend in spec.backend_values
}


def engine_ids() -> tuple[str, ...]:
    return tuple(spec.id for spec in ENGINES)


def engine_by_id(engine_id: str) -> EngineSpec:
    try:
        return _BY_ID[engine_id]
    except KeyError:
        raise KeyError(f"unknown engine id: {engine_id!r}") from None


def engine_for_backend(tts_backend: str) -> EngineSpec:
    """The engine a ``Settings.tts_backend`` value belongs to (classic ← qwen/vllm)."""
    try:
        return _BY_BACKEND[tts_backend]
    except KeyError:
        raise KeyError(f"no engine for tts_backend {tts_backend!r}") from None


def engines_manifest() -> list[dict]:
    """Ordered full JSON view of every engine, for the root-side ops scripts."""
    return [spec.to_manifest() for spec in ENGINES]


def dashboard_engines_manifest() -> list[dict]:
    """Ordered browser-safe JSON view of every engine, for the API/dashboard."""
    return [spec.to_dashboard_manifest() for spec in ENGINES]


def _main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "dump":
        json.dump(engines_manifest(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0
    sys.stderr.write("usage: python -m nova_voice.tts_engines dump\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(_main())
