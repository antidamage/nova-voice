from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from nova_voice.config import Settings


def test_optional_inference_adapters_are_importable_without_model_runtimes() -> None:
    """The control/API install must remain usable before CUDA models are installed."""

    stt = importlib.import_module("nova_voice.inference.stt")
    tts = importlib.import_module("nova_voice.inference.tts")
    audio = importlib.import_module("nova_voice.audio.bootstrap")

    assert stt.SpeechToText is not None
    assert tts.TextToSpeech is not None
    assert audio.build_audio_runtime is not None


def test_development_defaults_do_not_enable_audio_or_passive_execution() -> None:
    settings = Settings()

    assert settings.mode == "development"
    assert settings.audio_enabled is False
    assert settings.shadow_mode is True
    assert settings.passive_execution_enabled is False
    assert settings.narrow_gate_enabled is False
    assert settings.tts_speaker == "Serena"
    assert settings.conversation_idle_seconds == 60
    assert settings.denoise_base_url == "http://127.0.0.1:8092"
    assert settings.playback_preroll_ms == 700


def test_lan_binding_requires_mutual_tls() -> None:
    with pytest.raises(ValueError, match="LAN binding"):
        Settings(host="0.0.0.0")


def test_iridium_base_sync_preserves_pinned_inference_runtime() -> None:
    installer = (Path(__file__).resolve().parents[1] / "ops" / "install-iridium.sh").read_text()

    assert "sync --frozen --no-dev --inexact" in installer
    assert "nova-voice-dfn.service" in installer


def test_iridium_service_keeps_triton_cache_in_writable_cache_tree() -> None:
    service = (
        Path(__file__).resolve().parents[1] / "deploy" / "systemd" / "nova-voice.service"
    ).read_text()

    assert "TRITON_CACHE_DIR=/opt/nova-voice/cache/triton" in service
    assert "ReadWritePaths=/var/lib/nova-voice /opt/nova-voice/cache" in service


def test_iridium_voice_restart_brings_noise_suppression_back() -> None:
    service = (
        Path(__file__).resolve().parents[1] / "deploy" / "systemd" / "nova-voice.service"
    ).read_text()

    assert "Wants=network-online.target nova-voice-dfn.service" in service
    assert "After=" in service and "nova-voice-dfn.service" in service


def test_tier0_endurance_service_creates_its_writable_state_directory() -> None:
    service = (
        Path(__file__).resolve().parents[1]
        / "deploy"
        / "systemd"
        / "nova-voice-tier0-endurance.service"
    ).read_text()

    assert "StateDirectory=nova-voice/evaluation" in service
    assert "ReadWritePaths=/var/lib/nova-voice/evaluation" in service
