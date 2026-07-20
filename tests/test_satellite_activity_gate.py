from __future__ import annotations

import struct

from nova_voice.audio.pcm import SAMPLES_PER_FRAME
from nova_voice.satellites.activity_gate import CapturedAudioFrame, LocalActivityGate


def pcm(amplitude: int) -> bytes:
    samples = [amplitude if index % 2 else -amplitude for index in range(SAMPLES_PER_FRAME)]
    return struct.pack(f"<{SAMPLES_PER_FRAME}h", *samples)


def captured(index: int, amplitude: int, *, playback: bool = False) -> CapturedAudioFrame:
    return CapturedAudioFrame(
        payload=pcm(amplitude),
        monotonic_ns=index * 20_000_000,
        playback_active=playback,
    )


def test_gate_suppresses_idle_audio_and_releases_preroll_on_sustained_activity() -> None:
    gate = LocalActivityGate(
        trigger_ms=60, pre_roll_ms=400, hangover_ms=800, calibration_ms=0
    )
    emitted: list[CapturedAudioFrame] = []

    for index in range(17):
        emitted.extend(gate.accept(captured(index, 20)))
    assert emitted == []

    for index in range(17, 20):
        emitted.extend(gate.accept(captured(index, 4_000, playback=index == 17)))

    assert len(emitted) == 20
    assert emitted[0].monotonic_ns == 0
    assert emitted[-1].monotonic_ns == 19 * 20_000_000
    assert emitted[17].playback_active is True
    assert gate.active


def test_gate_sends_silence_tail_then_returns_to_idle() -> None:
    gate = LocalActivityGate(
        trigger_ms=20, pre_roll_ms=20, hangover_ms=200, calibration_ms=0
    )

    assert len(gate.accept(captured(0, 4_000))) == 1
    for index in range(1, 10):
        assert len(gate.accept(captured(index, 20))) == 1
        assert gate.active
    assert len(gate.accept(captured(10, 20))) == 1
    assert not gate.active
    assert gate.accept(captured(11, 20)) == ()


def test_gate_can_be_disabled_for_diagnostics() -> None:
    gate = LocalActivityGate(enabled=False)
    frame = captured(0, 0)

    assert gate.accept(frame) == (frame,)


def test_gate_live_override_resets_buffered_state() -> None:
    gate = LocalActivityGate(trigger_ms=60, pre_roll_ms=400, calibration_ms=0)
    assert gate.accept(captured(0, 4_000)) == ()

    gate.set_enabled(False)
    frame = captured(1, 0)
    assert gate.accept(frame) == (frame,)

    gate.set_enabled(True)
    assert gate.accept(captured(2, 0)) == ()


def test_gate_calibrates_to_stationary_room_noise_before_triggering() -> None:
    gate = LocalActivityGate(
        trigger_ms=60,
        pre_roll_ms=400,
        calibration_ms=1_000,
        noise_margin_db=6,
    )

    for index in range(70):
        assert gate.accept(captured(index, 500)) == ()

    assert -37 < gate.noise_floor_db < -35
    emitted: list[CapturedAudioFrame] = []
    for index in range(70, 73):
        emitted.extend(gate.accept(captured(index, 4_000)))
    assert len(emitted) == 20
    assert gate.active
