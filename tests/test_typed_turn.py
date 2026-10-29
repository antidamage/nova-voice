import pytest

from nova_voice.audio.runtime import SatelliteAudioRuntime
from nova_voice.audio.typed_turn import playback_target, primary_speaker
from nova_voice.domain import (
    ActiveGoal,
    Decision,
    Emotion,
    EmotionLabel,
    GoalStatus,
    HandleResult,
    Interpretation,
    ResponsePlan,
    SpeechAct,
)


class _NoStt:
    async def transcribe(self, _pcm16: bytes, sample_rate: int = 16_000) -> tuple[str, float]:
        raise AssertionError("a typed turn must not reach STT")


class _Tts:
    def __init__(self) -> None:
        self.spoken: list[str] = []

    async def synthesize(self, text: str, _instruction: str) -> tuple[bytes, int]:
        self.spoken.append(text)
        return b"\x00\x00" * 100, 24_000


class _Service:
    def __init__(self) -> None:
        self.utterances = []

    async def handle(self, utterance, **_kwargs) -> HandleResult:
        self.utterances.append(utterance)
        return HandleResult(
            utterance_id=utterance.id,
            interpretation=Interpretation(
                emotion=Emotion(label=EmotionLabel.CALM, confidence=0.9, intensity=0.1),
                speech_act=SpeechAct.QUESTION,
                addressed_probability=1.0,
                decision=Decision.REPLY,
                confidence=0.9,
                active_goal=ActiveGoal(status=GoalStatus.SATISFIED),
                response_plan=ResponsePlan(),
            ),
            executed=False,
            shadowed=False,
            policy_reason="addressed reply",
            response_text="It is sunny.",
            response_tone_instruction="Natural conversational delivery.",
        )


async def _typed(runtime: SatelliteAudioRuntime, text: str):
    return await runtime.process_pcm(
        satellite_id="indium",
        room_id="lounge",
        pcm16=b"",
        wake_detected=True,
        typed_transcript=text,
    )


async def test_a_typed_turn_skips_stt_and_is_spoken() -> None:
    service, tts = _Service(), _Tts()
    runtime = SatelliteAudioRuntime(service, _NoStt(), tts, lambda: None)

    turn = await _typed(runtime, "  what's the weather  ")

    assert turn is not None
    assert service.utterances[0].transcript == "what's the weather"
    assert service.utterances[0].transcript_confidence == 1.0
    assert service.utterances[0].wake_detected is True
    assert tts.spoken == ["It is sunny."]


async def test_the_same_typed_line_twice_runs_twice() -> None:
    service = _Service()
    runtime = SatelliteAudioRuntime(service, _NoStt(), _Tts(), lambda: None)

    await _typed(runtime, "turn on the lounge lights")
    await _typed(runtime, "turn on the lounge lights")

    assert len(service.utterances) == 2


async def test_an_empty_typed_line_is_refused() -> None:
    runtime = SatelliteAudioRuntime(_Service(), _NoStt(), _Tts(), lambda: None)

    with pytest.raises(ValueError):
        await _typed(runtime, "   ")


def test_playback_prefers_the_requested_room_then_an_enabled_microphone() -> None:
    connected = [("lounge", "nocturnium"), ("lounge", "indium"), ("office", "desk")]

    assert playback_target(connected, room_id=None, disabled={"nocturnium"}) == (
        "lounge",
        "indium",
    )
    assert playback_target(connected, room_id="office", disabled=set()) == ("office", "desk")
    # An unknown room falls back to every speaker rather than to silence.
    assert playback_target(connected, room_id="attic", disabled=set()) == (
        "lounge",
        "nocturnium",
    )
    assert playback_target([], room_id=None, disabled=set()) is None


class _Store:
    def __init__(self, listing: dict) -> None:
        self.listing = listing

    async def list_profiles(self) -> dict:
        return self.listing


class _Recognizer:
    def __init__(self, listing: dict) -> None:
        self.store = _Store(listing)


async def test_the_primary_speaker_is_the_person_heard_most() -> None:
    recognizer = _Recognizer(
        {
            "profiles": [
                {
                    "id": "p-guest",
                    "displayName": "Guest",
                    "pronouns": None,
                    "templates": [{"id": "t-g", "sampleCount": 3}],
                },
                {
                    "id": "p-main",
                    "displayName": "Main",
                    "pronouns": "she/her",
                    "templates": [
                        {"id": "t-a", "sampleCount": 10},
                        {"id": "t-b", "sampleCount": 40},
                    ],
                },
            ]
        }
    )

    identity = await primary_speaker(recognizer)

    assert identity is not None
    assert identity.status == "recognized"
    assert identity.person_id == "p-main"
    assert identity.template_id == "t-b"


async def test_no_enrolled_person_leaves_the_turn_unattributed() -> None:
    assert await primary_speaker(None) is None
    assert await primary_speaker(_Recognizer({"profiles": []})) is None
    never_heard = {"id": "p-x", "displayName": "X", "pronouns": None, "templates": []}
    assert await primary_speaker(_Recognizer({"profiles": [never_heard]})) is None
