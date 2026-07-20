from __future__ import annotations

from nova_voice.audio.arbitration import TurnArbiter
from nova_voice.audio.conversation import ConversationTracker
from nova_voice.audio.runtime import SatelliteAudioRuntime
from nova_voice.audio.segmenter import SpeechSegment
from nova_voice.domain import AcousticFeatures
from nova_voice.voice_settings import VoiceSettings


class FakeSegmenter:
    """Closes a one-second speech segment on every frame."""

    def accept(self, frame: bytes) -> SpeechSegment:
        return SpeechSegment(
            pcm16=b"\x00\x00" * 16_000,
            acoustic=AcousticFeatures(duration_ms=1000, rms_db=-8.0),
        )


class AlwaysWinsElection:
    async def elect(self, satellite_id, segment, *, wake_detected, room_id="", scope_id=None):
        return True


class FakeTts:
    async def configure(self, *, speaker: str, language: str) -> None:
        self.speaker = speaker
        self.language = language


def make_runtime() -> tuple[SatelliteAudioRuntime, ConversationTracker]:
    conversations = ConversationTracker(idle_seconds=60.0)
    runtime = SatelliteAudioRuntime(
        service=None,
        stt=None,
        tts=FakeTts(),
        segmenter_factory=FakeSegmenter,
        election=AlwaysWinsElection(),
        arbiter=TurnArbiter(initial_hold_seconds=25.0, max_hold_seconds=60.0),
        conversations=conversations,
    )
    return runtime, conversations


async def test_killswitch_drops_audio_and_closes_the_open_conversation() -> None:
    runtime, conversations = make_runtime()
    frame = b"\x00\x00" * 320

    # Voice is enabled by default: a frame produces an elected segment.
    assert await runtime.ingest(satellite_id="indium", room_id="lounge", frame=frame) is not None

    # An open conversation exists...
    conversations.start("lounge")
    assert conversations.active("lounge")

    # ...turning voice off closes it and drops all further audio.
    await runtime.apply_voice_settings(VoiceSettings(system_voice_enabled=False))
    assert not conversations.active("lounge")
    assert await runtime.ingest(satellite_id="indium", room_id="lounge", frame=frame) is None

    # Re-enabling resumes processing.
    await runtime.apply_voice_settings(VoiceSettings(system_voice_enabled=True))
    assert await runtime.ingest(satellite_id="indium", room_id="lounge", frame=frame) is not None


async def test_per_satellite_killswitch_drops_only_that_satellite() -> None:
    runtime, _ = make_runtime()
    frame = b"\x00\x00" * 320

    await runtime.apply_voice_settings(VoiceSettings(disabled_satellites=["indium"]))
    # A different, still-enabled satellite keeps working (first turn, nothing
    # else has claimed it yet).
    assert (
        await runtime.ingest(satellite_id="nocturnium", room_id="bedroom", frame=frame) is not None
    )
    # The disabled satellite's frames are dropped, case-insensitively.
    assert await runtime.ingest(satellite_id="indium", room_id="lounge", frame=frame) is None
    assert await runtime.ingest(satellite_id="INDIUM", room_id="lounge", frame=frame) is None


async def test_re_enabling_a_satellite_resumes_its_audio() -> None:
    runtime, _ = make_runtime()
    frame = b"\x00\x00" * 320

    await runtime.apply_voice_settings(VoiceSettings(disabled_satellites=["indium"]))
    assert await runtime.ingest(satellite_id="indium", room_id="lounge", frame=frame) is None

    await runtime.apply_voice_settings(VoiceSettings(disabled_satellites=[]))
    assert await runtime.ingest(satellite_id="indium", room_id="lounge", frame=frame) is not None


def test_disabled_satellites_parse_and_normalize() -> None:
    settings = VoiceSettings.model_validate(
        {"disabledSatellites": ["Indium", "indium", " nocturnium ", 5, ""]}
    )
    # Casefolded, de-duplicated, blanks/non-strings dropped, order preserved.
    assert settings.disabled_satellites == ["indium", "nocturnium"]
    # A malformed value never breaks the settings pull; it just disables nobody.
    assert VoiceSettings.model_validate({"disabledSatellites": "nope"}).disabled_satellites == [
        "nocturnium"
    ]
    assert VoiceSettings().disabled_satellites == ["nocturnium"]
    assert VoiceSettings.model_validate({"disabledSatellites": []}).disabled_satellites == []


def test_satellite_noise_gate_defaults_on_and_can_be_bypassed() -> None:
    assert VoiceSettings().satellite_noise_gate_enabled is True
    assert (
        VoiceSettings.model_validate({"satelliteNoiseGateEnabled": False})
        .satellite_noise_gate_enabled
        is False
    )
