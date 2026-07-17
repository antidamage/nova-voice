from __future__ import annotations

import numpy as np

from nova_voice.audio.arbitration import TurnArbiter
from nova_voice.audio.pcm import float32_to_pcm16
from nova_voice.audio.runtime import SatelliteAudioRuntime
from nova_voice.audio.segmenter import SpeechSegment
from nova_voice.domain import AcousticFeatures


class FakeSegmenter:
    """Closes a one-second speech segment on every frame."""

    def accept(self, frame: bytes) -> SpeechSegment:
        time = np.arange(16_000, dtype=np.float32) / 16_000
        samples = 0.4 * np.sin(2 * np.pi * 200 * time)
        return SpeechSegment(
            pcm16=float32_to_pcm16(samples),
            acoustic=AcousticFeatures(duration_ms=1000, rms_db=-8.0),
        )


class AlwaysWinsElection:
    """Every candidate wins, so the turn gate is the only mechanism tested."""

    async def elect(self, satellite_id, segment, *, wake_detected, room_id="", scope_id=None):
        return True


def make_runtime(arbitration_scope: str = "household") -> SatelliteAudioRuntime:
    return SatelliteAudioRuntime(
        service=None,
        stt=None,
        tts=None,
        segmenter_factory=FakeSegmenter,
        election=AlwaysWinsElection(),
        arbiter=TurnArbiter(initial_hold_seconds=25.0, max_hold_seconds=60.0),
        arbitration_scope=arbitration_scope,
    )


async def test_second_satellite_is_gated_until_the_first_turn_releases() -> None:
    runtime = make_runtime()
    frame = b"\x00\x00" * 320

    pending = await runtime.ingest(satellite_id="indium", room_id="lounge", frame=frame)
    assert pending is not None
    assert pending.arbiter_claim is not None

    gated = await runtime.ingest(satellite_id="nocturnium", room_id="office", frame=frame)
    assert gated is None

    # The claim owner's own follow-up still passes (barge-in stays possible).
    own = await runtime.ingest(satellite_id="indium", room_id="lounge", frame=frame)
    assert own is not None

    runtime.release_turn(own.arbiter_claim)
    after_release = await runtime.ingest(
        satellite_id="nocturnium", room_id="office", frame=frame
    )
    assert after_release is not None


async def test_room_scope_keeps_rooms_independent() -> None:
    runtime = make_runtime(arbitration_scope="room")
    frame = b"\x00\x00" * 320

    first = await runtime.ingest(satellite_id="indium", room_id="lounge", frame=frame)
    second = await runtime.ingest(satellite_id="nocturnium", room_id="office", frame=frame)

    assert first is not None
    assert second is not None
