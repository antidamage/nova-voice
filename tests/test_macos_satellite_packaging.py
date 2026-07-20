from __future__ import annotations

import plistlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESOURCES = ROOT / "satellites" / "macos" / "Resources"


def test_macos_bundle_declares_required_privacy_reasons() -> None:
    with (RESOURCES / "Info.plist").open("rb") as file:
        info = plistlib.load(file)

    assert info["CFBundleIdentifier"] == "nz.co.skull.NovaVoiceSatellite"
    assert info["NSMicrophoneUsageDescription"]
    assert info["NSLocalNetworkUsageDescription"]


def test_legacy_launch_agent_is_attributed_to_the_signed_app() -> None:
    with (RESOURCES / "nz.co.skull.NovaVoiceSatellite.plist").open("rb") as file:
        agent = plistlib.load(file)

    assert agent["AssociatedBundleIdentifiers"] == "nz.co.skull.NovaVoiceSatellite"
    assert agent["RunAtLoad"] is True
    assert agent["KeepAlive"] is True


def test_macos_installer_registers_and_materializes_the_app() -> None:
    installer = (ROOT / "ops" / "install-macos-satellite.sh").read_text()

    assert '"$LSREGISTER" -f "$APP"' in installer
    assert "/usr/bin/plutil -remove ProgramArguments" in installer
    assert "/usr/bin/plutil -insert ProgramArguments.0 -string" in installer


def test_macos_satellite_flushes_audio_on_playback_cancel() -> None:
    source = (
        ROOT / "satellites" / "macos" / "Sources" / "NovaVoiceSatellite" / "main.swift"
    ).read_text()

    assert 'case "playback_cancel":' in source
    assert "audio.cancelPlaybackStream()" in source
    assert "func cancelPlaybackStream()" in source
    assert "teardownPlaybackEngineLocked()" in source


def test_macos_satellite_accepts_server_jitter_buffer_target() -> None:
    source = (
        ROOT / "satellites" / "macos" / "Sources" / "NovaVoiceSatellite" / "main.swift"
    ).read_text()

    assert "setPlaybackBufferMs" in source
    assert 'control["bufferMs"]' in source
    assert "initialPrerollSeconds = 0.7" in source


def test_macos_satellite_reports_actual_playback_edges() -> None:
    source = (
        ROOT / "satellites" / "macos" / "Sources" / "NovaVoiceSatellite" / "main.swift"
    ).read_text()

    assert "let playbackEvents = true" in source
    assert 'type: "playback_started"' in source
    assert 'type: "playback_finished"' in source
    assert "completionCallbackType: .dataPlayedBack" in source


def test_macos_satellite_gates_idle_audio_and_accepts_live_bypass() -> None:
    source = (
        ROOT / "satellites" / "macos" / "Sources" / "NovaVoiceSatellite" / "main.swift"
    ).read_text()

    assert "private final class LocalActivityGate" in source
    assert "preRollFrames" in source
    assert "hangoverFrames" in source
    assert 'case "local_vad":' in source
    assert "audio.setLocalVadEnabled(enabled)" in source
    assert "let localVad = true" in source
