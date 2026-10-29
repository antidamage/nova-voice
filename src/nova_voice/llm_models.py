"""Single source of truth for Nova's interpretation LLMs.

Exactly one llama.cpp model serves interpretation at a time. Each has its own
systemd unit and localhost port, so the root-side switcher can start the chosen
model while the current one keeps serving, route the orchestrator to it, then
stop the rest. The orchestrator, the dashboard and the root scripts all read
this registry (the scripts through ``python -m nova_voice.llm_models dump``), so
none of them can disagree about what a model is.

Adding a model is one :class:`LlmSpec` entry plus its unit file.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LlmSpec:
    id: str
    """Stable model id, the dashboard/API vocabulary: lfm3 | bonsai."""

    label: str
    """Human label shown in the model picker."""

    detail: str
    """One line for the picker describing the model."""

    alias: str
    """The ``--alias`` llama-server serves, sent as the request ``model``."""

    unit: str
    """systemd unit that hosts this model."""

    port: int
    """Localhost port the unit binds."""

    fits_with_vram_trim: bool
    """Whether the model still serves usefully when a TTS engine trims the LLM's
    GPU layers (Custom / dots.tts). A model that is unusable on the CPU is not
    offered while such an engine is resident."""

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    @property
    def health_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/health"

    def to_manifest(self) -> dict:
        """The full ops view, for the root-side switcher and installer."""
        return {
            "id": self.id,
            "label": self.label,
            "alias": self.alias,
            "unit": self.unit,
            "port": self.port,
            "healthUrl": self.health_url,
            "fitsWithVramTrim": self.fits_with_vram_trim,
        }

    def to_dashboard_manifest(self) -> dict:
        """The browser-facing view: no units, ports or URLs."""
        return {"id": self.id, "label": self.label, "detail": self.detail}


LLMS: tuple[LlmSpec, ...] = (
    LlmSpec(
        id="lfm3",
        label="LFM3 1.7B",
        detail="LiquidAI LFM3-1.7B, Q4_K_M. Small and fast; fits beside every TTS engine.",
        alias="LFM3-1.7B",
        unit="nova-voice-llm.service",
        port=8765,
        fits_with_vram_trim=True,
    ),
    LlmSpec(
        id="bonsai",
        label="Bonsai 9B",
        detail=(
            "PrismML Ternary Bonsai 2 9B (Qwen3.5-9B), Q2_0. Larger and slower to "
            "first word; needs the GPU to itself beside Classic or Trained TTS."
        ),
        alias="Ternary-Bonsai-2-9B",
        unit="nova-voice-llm-bonsai.service",
        port=8768,
        # Measured 2026-10-30: ~67 tok/s fully on the GPU, ~5 tok/s with half
        # its layers on the CPU, which is too slow to answer a spoken turn.
        fits_with_vram_trim=False,
    ),
)

DEFAULT_LLM_ID = "lfm3"

_BY_ID: dict[str, LlmSpec] = {spec.id: spec for spec in LLMS}


def llm_ids() -> tuple[str, ...]:
    return tuple(spec.id for spec in LLMS)


def llm_by_id(llm_id: str) -> LlmSpec:
    try:
        return _BY_ID[llm_id]
    except KeyError:
        raise KeyError(f"unknown llm id: {llm_id!r}") from None


def read_selected_llm(path: Path) -> LlmSpec | None:
    """The model named in the root switcher's selection file, if valid."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    llm_id = payload.get("llm") if isinstance(payload, dict) else None
    return _BY_ID.get(llm_id) if isinstance(llm_id, str) else None


def incompatibility(spec: LlmSpec, tts_trims_llm: bool) -> str | None:
    """Why ``spec`` cannot be selected beside the resident TTS engine, or None."""
    if tts_trims_llm and not spec.fits_with_vram_trim:
        return (
            f"{spec.label} needs the whole GPU and the Custom voice engine is "
            "loaded; switch the voice engine to Classic or Trained first."
        )
    return None


def llms_manifest() -> list[dict]:
    return [spec.to_manifest() for spec in LLMS]


def dashboard_llms_manifest(tts_trims_llm: bool) -> list[dict]:
    """Browser-safe registry, each entry marked available for the resident TTS engine."""
    entries = []
    for spec in LLMS:
        reason = incompatibility(spec, tts_trims_llm)
        entry = spec.to_dashboard_manifest()
        entry["available"] = reason is None
        if reason is not None:
            entry["unavailableReason"] = reason
        entries.append(entry)
    return entries


def _main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "dump":
        json.dump(llms_manifest(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0
    sys.stderr.write("usage: python -m nova_voice.llm_models dump\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(_main())
