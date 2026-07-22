from __future__ import annotations

import numpy as np

from nova_voice.audio.endpointing import (
    EndpointDecision,
    EndpointResult,
    SemanticEndpointDetector,
)
from nova_voice.audio.interruption import InterruptionKind, classify_interruption
from nova_voice.audio.listening_ack import ListeningAckController
from nova_voice.audio.prefetch import ForegroundPrefetch, StableInterimTracker, likely_tools
from nova_voice.audio.runtime import SatelliteAudioRuntime
from nova_voice.audio.segmenter import SpeechSegment, SpeechSegmenter
from nova_voice.audio.speech_units import sentence_speech_units
from nova_voice.domain import AcousticFeatures


def _recorded_cadence(amplitudes: tuple[float, float, float]) -> bytes:
    sections = []
    for amplitude in amplitudes:
        wave = np.sin(np.linspace(0, 20 * np.pi, 1_920)) * amplitude
        sections.append((wave * 32_767).astype("<i2"))
    sections.append(np.zeros(9_600, dtype="<i2"))
    return np.concatenate(sections).tobytes()


def test_semantic_endpointing_distinguishes_falling_and_continuing_cadence() -> None:
    detector = SemanticEndpointDetector()

    falling = detector.decide(
        _recorded_cadence((0.8, 0.4, 0.1)),
        trailing_silence_ms=600,
    )
    rising = detector.decide(
        _recorded_cadence((0.1, 0.4, 0.8)),
        trailing_silence_ms=600,
    )

    assert falling.decision == EndpointDecision.COMPLETE
    assert falling.additional_wait_ms == 0
    assert rising.decision == EndpointDecision.CONTINUE
    assert rising.additional_wait_ms == 1_200


class _ContinueDetector:
    def decide(self, _pcm16: bytes, *, trailing_silence_ms: int) -> EndpointResult:
        assert trailing_silence_ms >= 40
        return EndpointResult(EndpointDecision.CONTINUE, 0.2, 80)


def test_segmenter_announces_one_bounded_endpoint_wait() -> None:
    scores = iter([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    segmenter = SpeechSegmenter(
        lambda _frame: next(scores),
        pre_roll_ms=20,
        end_silence_ms=40,
        endpoint_detector=_ContinueDetector(),
    )
    frame = b"\x00\x00" * 320

    assert segmenter.accept(frame) is None
    assert segmenter.accept(frame) is None
    assert segmenter.accept(frame) is None
    assert segmenter.endpoint_waiting
    assert segmenter.consume_endpoint_wait_started()
    assert not segmenter.consume_endpoint_wait_started()
    assert segmenter.accept(frame) is None
    assert segmenter.accept(frame) is None
    assert segmenter.accept(frame) is None
    segment = segmenter.accept(frame)

    assert segment is not None
    assert segment.endpoint_decision == EndpointDecision.CONTINUE
    assert segment.endpoint_wait_ms == 80


def test_interruption_classifier_preserves_backchannels_and_rejects_cross_talk() -> None:
    acoustic = AcousticFeatures(duration_ms=700, rms_db=-24)

    backchannel = classify_interruption(
        "yeah",
        acoustic=acoustic,
        explicitly_addressed=True,
        explicit_stop=False,
    )
    cross_talk = classify_interruption(
        "did you put the kettle on",
        acoustic=acoustic,
        explicitly_addressed=False,
        explicit_stop=False,
    )
    barge_in = classify_interruption(
        "Nova stop",
        acoustic=acoustic,
        explicitly_addressed=True,
        explicit_stop=True,
    )

    assert backchannel.kind == InterruptionKind.BACKCHANNEL
    assert cross_talk.kind == InterruptionKind.CROSS_TALK
    assert barge_in.kind == InterruptionKind.TRUE_BARGE_IN


def test_stable_interim_prefetch_is_prefix_compatible_and_read_only_data() -> None:
    tracker = StableInterimTracker()
    assert tracker.observe("turn on") is None
    assert tracker.observe("turn on the") is None
    stable = tracker.observe("turn on the")
    assert stable == "turn on the"

    catalog = [
        {
            "function": {
                "name": "nova.light.set",
                "description": "turn a room light on or off",
            }
        },
        {"function": {"name": "web.search", "description": "look up facts"}},
    ]
    selected = likely_tools(stable, catalog)
    prefetch = ForegroundPrefetch.create(stable, {"room": "office"}, selected)

    assert selected == ("nova.light.set",)
    assert prefetch.compatible_with("turn on the office light")
    assert not prefetch.compatible_with("turn off the office light")
    assert prefetch.context_revision
    assert prefetch.llm_state_revision


def test_sentence_units_are_cancellable_and_listening_ack_never_confirms_work() -> None:
    units = sentence_speech_units(
        "The office light is on. I can also dim it, but I have not changed that yet."
    )
    acknowledgement = ListeningAckController(cooldown_seconds=60).choose(
        "office",
        addressed=True,
    )

    assert units == (
        "The office light is on.",
        "I can also dim it, but I have not changed that yet.",
    )
    assert acknowledgement is not None
    assert acknowledgement.implies_completion is False
    assert len(acknowledgement.pcm16) <= acknowledgement.sample_rate * 2 // 10


class _InterimStt:
    async def transcribe_chunk(self, _frame: bytes, **_kwargs) -> tuple[str, float]:
        return "turn on the", 0.8

    async def cancel_stream(self, _stream_id: str) -> None:
        return None


class _InterimService:
    def __init__(self) -> None:
        self.prefetch_calls = 0
        self.handle_calls = 0

    async def prefetch_foreground(self, room_id: str, text: str) -> ForegroundPrefetch:
        self.prefetch_calls += 1
        return ForegroundPrefetch.create(text, {"room": room_id}, ("nova.light.set",))

    async def handle(self, *_args, **_kwargs):
        self.handle_calls += 1
        raise AssertionError("the final-turn gate was not opened")


class _InterimSegmenter:
    def __init__(self) -> None:
        self.calls = 0
        self.speaking = False
        self._ack = False

    def accept(self, frame: bytes) -> SpeechSegment | None:
        self.calls += 1
        self.speaking = self.calls < 3
        if self.calls < 3:
            return None
        return SpeechSegment(frame, AcousticFeatures(duration_ms=20, rms_db=-20))

    def consume_endpoint_wait_started(self) -> bool:
        if self.calls != 2 or self._ack:
            return False
        self._ack = True
        return True


async def test_live_interim_prefetch_and_ack_commit_nothing_before_final_gate() -> None:
    service = _InterimService()
    runtime = SatelliteAudioRuntime(
        service,
        _InterimStt(),
        object(),
        _InterimSegmenter,
    )
    acknowledgements: list[bytes] = []

    async def ack_sink(chunk: bytes, _sample_rate: int) -> None:
        acknowledgements.append(chunk)

    values = {
        "satellite_id": "browser-office",
        "room_id": "office",
        "frame": b"\x01\x00" * 320,
        "wake_detected": True,
        "listening_ack_sink": ack_sink,
    }
    assert await runtime.ingest(**values) is None
    assert await runtime.ingest(**values) is None
    pending = await runtime.ingest(**values)
    assert pending is not None
    assert pending.prefetch_task is not None
    prefetch = await pending.prefetch_task

    assert prefetch.compatible_with("turn on the office light")
    assert service.prefetch_calls == 1
    assert service.handle_calls == 0
    assert len(acknowledgements) == 1
