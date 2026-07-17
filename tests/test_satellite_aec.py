from __future__ import annotations

import subprocess
import threading
from pathlib import Path

import pytest

from nova_voice.satellites.client import SatelliteAudio, SatelliteSettings


class FakeSoundDevice:
    def query_devices(self):
        return [
            {"name": "ALC256 Analog", "max_input_channels": 2, "max_output_channels": 2},
            {"name": "pipewire", "max_input_channels": 64, "max_output_channels": 64},
            {"name": "default", "max_input_channels": 64, "max_output_channels": 64},
        ]


class FakeOutputStream:
    def __init__(self) -> None:
        self.writes: list[bytes] = []
        self.started = False

    def start(self) -> None:
        self.started = True

    def write(self, payload: bytes) -> None:
        self.writes.append(payload)

    def stop(self) -> None:
        self.started = False

    def close(self) -> None:
        pass


class PlaybackSoundDevice(FakeSoundDevice):
    def __init__(self) -> None:
        self.output = FakeOutputStream()

    def RawOutputStream(self, **_kwargs) -> FakeOutputStream:
        return self.output


def settings() -> SatelliteSettings:
    return SatelliteSettings(
        satellite_id="nocturnium",
        display_name="Nocturnium",
        room_id="office",
        tls_ca_path=Path("ca.crt"),
        tls_cert_path=Path("client.crt"),
        tls_key_path=Path("client.key"),
        echo_cancellation=True,
    )


def audio() -> SatelliteAudio:
    value = object.__new__(SatelliteAudio)
    value._sd = FakeSoundDevice()
    value._settings = settings()
    return value


def playback_audio() -> tuple[SatelliteAudio, PlaybackSoundDevice]:
    sounddevice = PlaybackSoundDevice()
    value = object.__new__(SatelliteAudio)
    value._sd = sounddevice
    value._settings = settings().model_copy(
        update={"echo_cancellation": False, "output_device": 0}
    )
    value._input = None
    value._output = None
    value._output_rate = None
    value._playback_pending = bytearray()
    value._playback_target_bytes = 0
    value._stream_active = False
    value.playback_active = threading.Event()
    return value, sounddevice


def test_generic_portaudio_pipewire_device_requires_aec_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = audio()
    calls: list[tuple[str, ...]] = []

    def run(command, **_kwargs):
        calls.append(tuple(command))
        target = command[-1]
        node = "nova_voice_aec_sink" if "SINK" in target else "nova_voice_aec"
        return subprocess.CompletedProcess(command, 0, stdout=f'node.name = "{node}"\n')

    monkeypatch.setattr(subprocess, "run", run)
    input_device = value._resolve_device("default")
    output_device = value._resolve_device("default", output=True)

    assert input_device == 1
    assert output_device == 1
    value._require_aec_device(input_device)
    value._require_aec_device(output_device, output=True)
    assert calls == [
        ("wpctl", "inspect", "@DEFAULT_AUDIO_SOURCE@"),
        ("wpctl", "inspect", "@DEFAULT_AUDIO_SINK@"),
    ]


def test_satellite_fails_closed_when_pipewire_default_bypasses_aec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = audio()
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command,
            0,
            stdout='node.name = "alsa_input.usb-Microsoft_LifeCam"\n',
        ),
    )

    with pytest.raises(RuntimeError, match="AEC input device is unavailable"):
        value._require_aec_device(value._resolve_device("default"))


def test_linux_satellite_holds_server_controlled_room_playback_preroll() -> None:
    value, sounddevice = playback_audio()
    value.begin_playback(sample_rate=1_000, buffer_ms=700)

    value.enqueue_playback(b"a" * 600, 1_000)
    assert sounddevice.output.writes == []
    assert value.playback_active.is_set()

    value.enqueue_playback(b"b" * 800, 1_000)
    value.enqueue_playback(b"c" * 200, 1_000)
    assert sounddevice.output.writes == [b"a" * 600 + b"b" * 800, b"c" * 200]

    value.end_playback(1_000)
    assert not value.playback_active.is_set()


def test_linux_satellite_flushes_a_short_response_when_playback_finishes() -> None:
    value, sounddevice = playback_audio()
    value.begin_playback(sample_rate=1_000, buffer_ms=700)
    value.enqueue_playback(b"short", 1_000)

    value.end_playback(1_000)

    assert sounddevice.output.writes == [b"short"]
    assert not value.playback_active.is_set()
