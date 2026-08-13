from __future__ import annotations

import pytest
from conftest import interpretation
from test_service_handle import (
    _action,
    _Interpreter,
    _Persona,
    _Provider,
    _Registry,
    _SpeakerProfiles,
    _Store,
)

from nova_voice.config import Settings
from nova_voice.domain import Decision, SpeakerIdentity
from nova_voice.service import NovaVoiceService
from nova_voice.voice_settings import VoiceSettings

RECOGNIZED = SpeakerIdentity(
    status="recognized",
    template_id="voice-a",
    person_id="person-a",
    display_name="Addie",
    confidence=0.91,
)

# Long enough to clear the short-command rule, so these tests exercise the
# training gate rather than the word-count one.
LONG_COMMAND = "Nova, please turn on the lounge lights for me now"


def _service(*, training: bool, recognition: bool = True) -> NovaVoiceService:
    provider = _Provider()
    service = NovaVoiceService(
        Settings(shadow_mode=False, passive_execution_enabled=True),
        _Interpreter(interpretation(decision=Decision.EXECUTE, actions=[_action()], addressed=1.0)),
        _Registry(provider),
        provider,
        _Store(),
        _Persona(),
        speaker_profiles=_SpeakerProfiles() if recognition else None,
    )
    service.voice_settings = VoiceSettings(
        voiceTrainingEnabled=training,
        speakerRecognitionEnabled=recognition,
    )
    service.nova_provider = provider
    return service


@pytest.mark.asyncio
async def test_training_off_ignores_a_command_from_an_unrecognized_voice(utterance) -> None:
    service = _service(training=False)
    spoken = utterance.model_copy(
        update={"transcript": LONG_COMMAND, "wake_detected": True}
    )

    result = await service.handle(spoken)

    assert result.interpretation.decision == Decision.IGNORE
    assert not result.executed


@pytest.mark.asyncio
async def test_training_on_accepts_a_command_from_an_unrecognized_voice(utterance) -> None:
    service = _service(training=True)
    spoken = utterance.model_copy(
        update={"transcript": LONG_COMMAND, "wake_detected": True}
    )

    result = await service.handle(spoken)

    assert result.interpretation.decision == Decision.EXECUTE
    assert result.executed


@pytest.mark.asyncio
async def test_training_off_still_accepts_a_recognized_voice(utterance) -> None:
    service = _service(training=False)
    spoken = utterance.model_copy(
        update={"transcript": LONG_COMMAND, "wake_detected": True, "speaker": RECOGNIZED}
    )

    result = await service.handle(spoken)

    assert result.interpretation.decision == Decision.EXECUTE
    assert result.executed


@pytest.mark.asyncio
async def test_training_off_is_bypassed_when_recognition_is_unavailable(utterance) -> None:
    # An identity that cannot be computed cannot be gated on. Gating anyway
    # would silence the household exactly when recognition has broken.
    service = _service(training=False, recognition=False)
    spoken = utterance.model_copy(
        update={"transcript": LONG_COMMAND, "wake_detected": True}
    )

    result = await service.handle(spoken)

    assert result.interpretation.decision == Decision.EXECUTE
    assert result.executed


@pytest.mark.asyncio
async def test_short_commands_still_require_recognition_while_training_is_on(
    utterance,
) -> None:
    service = _service(training=True)
    spoken = utterance.model_copy(
        update={"transcript": "Nova, lights on", "wake_detected": True}
    )

    result = await service.handle(spoken)

    assert result.interpretation.decision == Decision.IGNORE


def test_voice_training_defaults_on() -> None:
    assert VoiceSettings().voice_training_enabled is True
