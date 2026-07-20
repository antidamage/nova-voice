from __future__ import annotations

from nova_voice.audio.runtime import SatelliteAudioRuntime


class _StubRecognizer:
    def __init__(self) -> None:
        self.warmups = 0

    async def warmup(self) -> bool:
        self.warmups += 1
        return True


def _runtime(**kwargs) -> SatelliteAudioRuntime:
    return SatelliteAudioRuntime(
        service=None,  # type: ignore[arg-type]
        stt=None,  # type: ignore[arg-type]
        tts=None,  # type: ignore[arg-type]
        segmenter_factory=lambda: None,
        **kwargs,
    )


async def test_warmup_without_recognizer_is_a_noop() -> None:
    runtime = _runtime()
    await runtime.warmup()  # must not raise when speaker recognition is off


async def test_warmup_preloads_the_speaker_recognizer() -> None:
    recognizer = _StubRecognizer()
    runtime = _runtime(speaker_recognizer=recognizer)  # type: ignore[arg-type]

    await runtime.warmup()

    assert recognizer.warmups == 1
