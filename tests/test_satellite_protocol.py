from __future__ import annotations

import inspect

import pytest

from nova_voice import api
from nova_voice.audio.pcm import BYTES_PER_FRAME
from nova_voice.satellites.playback import SatellitePlaybackStream
from nova_voice.satellites.protocol import AudioFrame, FrameKind, SatelliteHello


def test_audio_frame_round_trip() -> None:
    frame = AudioFrame(
        kind=FrameKind.AUDIO_INPUT,
        flags=1,
        sequence=42,
        monotonic_ns=123456789,
        payload=b"\0" * BYTES_PER_FRAME,
    )

    assert AudioFrame.unpack(frame.pack()) == frame


def test_audio_frame_rejects_truncated_payload() -> None:
    frame = AudioFrame(
        kind=FrameKind.AUDIO_INPUT,
        sequence=1,
        monotonic_ns=2,
        payload=b"\0" * BYTES_PER_FRAME,
    )

    with pytest.raises(ValueError, match="payload length"):
        AudioFrame.unpack(frame.pack()[:-1])


def test_satellite_hello_requires_always_capture() -> None:
    hello = SatelliteHello.model_validate(
        {
            "protocolVersion": 1,
            "satelliteId": "nocturnium",
            "displayName": "Nocturnium",
            "roomId": "lounge",
            "client": "linux-native",
            "supervisor": "systemd",
            "capturePolicy": "foreground",
            "capabilities": {},
        }
    )

    with pytest.raises(ValueError, match="always capture"):
        hello.validate_protocol()


def test_macos_hello_accepts_camel_case_capabilities() -> None:
    hello = SatelliteHello.model_validate(
        {
            "protocolVersion": 1,
            "satelliteId": "indium",
            "displayName": "Indium",
            "roomId": "office",
            "client": "macos-native",
            "supervisor": "launchd",
            "capturePolicy": "always",
            "capabilities": {
                "microphone": True,
                "speaker": True,
                "echoCancellation": True,
                "noiseSuppression": True,
                "automaticGainControl": True,
                "playbackEvents": True,
            },
        }
    )

    hello.validate_protocol()
    assert hello.capabilities.echo_cancellation
    assert hello.capabilities.playback_events


def test_legacy_satellite_defaults_playback_events_off() -> None:
    hello = SatelliteHello.model_validate(
        {
            "protocolVersion": 1,
            "satelliteId": "nocturnium",
            "displayName": "Nocturnium",
            "roomId": "lounge",
            "client": "linux-native",
            "supervisor": "systemd",
            "capturePolicy": "always",
            "capabilities": {},
        }
    )

    assert hello.capabilities.playback_events is False
    assert hello.capabilities.local_vad is False


def test_server_closes_the_source_socket_for_legacy_playback_cancellation() -> None:
    source = inspect.getsource(SatellitePlaybackStream)

    assert '"type": "playback_cancel"' in source
    assert 'reason="playback interrupted"' in source


def test_server_routes_confirmed_playback_lifecycle_events() -> None:
    source = inspect.getsource(api.create_app) + inspect.getsource(SatellitePlaybackStream)

    assert 'control.get("type") == "playback_started"' in source
    assert 'control.get("type") == "playback_finished"' in source
    assert '"playbackId": self.playback_id' in source


def _browser_hello(capture_policy: str = "push-to-talk", supervisor: str = "none") -> dict:
    return {
        "protocolVersion": 1,
        "satelliteId": "web-lounge-tablet",
        "displayName": "Lounge Tablet",
        "roomId": "lounge",
        "client": "browser",
        "supervisor": supervisor,
        "capturePolicy": capture_policy,
        "capabilities": {"microphone": True, "speaker": True, "playbackEvents": True},
    }


def test_browser_hello_accepts_push_to_talk() -> None:
    hello = SatelliteHello.model_validate(_browser_hello())
    hello.validate_protocol()
    assert hello.is_browser


def test_browser_hello_accepts_always_capture() -> None:
    hello = SatelliteHello.model_validate(_browser_hello(capture_policy="always"))
    hello.validate_protocol()


def test_browser_hello_rejects_os_supervisor() -> None:
    hello = SatelliteHello.model_validate(_browser_hello(supervisor="systemd"))
    with pytest.raises(ValueError, match="no OS supervisor"):
        hello.validate_protocol()


def test_native_hello_rejects_none_supervisor() -> None:
    hello = SatelliteHello.model_validate(
        {
            "protocolVersion": 1,
            "satelliteId": "nocturnium",
            "displayName": "Nocturnium",
            "roomId": "lounge",
            "client": "linux-native",
            "supervisor": "none",
            "capturePolicy": "always",
            "capabilities": {},
        }
    )
    with pytest.raises(ValueError, match="OS supervisor"):
        hello.validate_protocol()


def test_server_arms_wake_on_browser_begin_turn() -> None:
    source = inspect.getsource(api.create_app)

    assert 'control.get("type") == "begin_turn"' in source
    assert 'turn_signal["wake_armed"] = True' in source
