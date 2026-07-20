from __future__ import annotations

import pytest

from nova_voice.audio.runtime import SatelliteAudioRuntime
from nova_voice.config import Settings
from nova_voice.inference.stt import boosting_phrase_variants
from nova_voice.voice_settings import VoiceSettings


def test_boosting_phrase_variants_cover_both_capitalizations() -> None:
    assert boosting_phrase_variants(["beemo"]) == ["beemo", "Beemo"]


def test_boosting_phrase_variants_normalize_and_dedupe() -> None:
    variants = boosting_phrase_variants(["Beemo", "  beemo ", "", "Nova"])
    assert variants == ["beemo", "Beemo", "nova", "Nova"]


def test_boosting_phrase_variants_title_case_multiword_names() -> None:
    assert boosting_phrase_variants(["mister bandit"]) == [
        "mister bandit",
        "Mister bandit",
        "Mister Bandit",
    ]


class _Tts:
    async def configure(self, **_kwargs) -> None:
        return None


class _BoostRecordingStt:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def set_boosted_phrases(self, phrases: list[str]) -> None:
        self.calls.append(list(phrases))


@pytest.mark.asyncio
async def test_apply_voice_settings_boosts_wake_words_and_agent_name() -> None:
    stt = _BoostRecordingStt()
    runtime = SatelliteAudioRuntime(
        service=None, stt=stt, tts=_Tts(), segmenter_factory=None
    )

    await runtime.apply_voice_settings(
        VoiceSettings(agentName="Bandit", wakeWords=["beemo", "bimo"])
    )

    assert stt.calls == [["beemo", "bimo", "Bandit"]]


@pytest.mark.asyncio
async def test_apply_voice_settings_boosts_the_spoken_pronunciation() -> None:
    # An emoji display name would be useless for ASR biasing; the plain
    # pronunciation is boosted instead.
    stt = _BoostRecordingStt()
    runtime = SatelliteAudioRuntime(
        service=None, stt=stt, tts=_Tts(), segmenter_factory=None
    )

    await runtime.apply_voice_settings(
        VoiceSettings(
            agentName="✨ Nova 🤖",
            agentNamePronunciation="Nova",
            wakeWords=["beemo", "bimo"],
        )
    )

    assert stt.calls == [["beemo", "bimo", "Nova"]]


@pytest.mark.asyncio
async def test_apply_voice_settings_tolerates_stt_without_boosting() -> None:
    runtime = SatelliteAudioRuntime(
        service=None, stt=None, tts=_Tts(), segmenter_factory=None
    )

    await runtime.apply_voice_settings(VoiceSettings())


def test_context_biasing_defaults_enabled_with_moderate_alpha() -> None:
    settings = Settings()

    assert settings.stt_context_biasing_enabled is True
    assert settings.stt_boost_alpha == 1.0
