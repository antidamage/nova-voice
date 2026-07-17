from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import numpy as np
import pytest

import nova_voice.audio.runtime as runtime_module
from nova_voice.audio.pcm import float32_to_pcm16, pcm16_to_float32, scale_pcm16
from nova_voice.audio.runtime import SatelliteAudioRuntime
from nova_voice.config import Settings
from nova_voice.service import current_clock_context
from nova_voice.voice_settings import VoiceSettings


def test_scale_pcm16_applies_linear_gain_and_keeps_full_volume_bit_exact() -> None:
    samples = np.linspace(-0.5, 0.5, 320, dtype=np.float32)
    payload = float32_to_pcm16(samples)

    assert scale_pcm16(payload, 100) is payload
    halved = pcm16_to_float32(scale_pcm16(payload, 50))
    assert np.allclose(halved, samples * 0.5, atol=1e-3)
    assert scale_pcm16(payload, 0) == b"\x00\x00" * 320


class _Tts:
    async def configure(self, **_kwargs) -> None:
        return None


@pytest.mark.asyncio
async def test_playback_volume_follows_the_8am_to_9pm_day_window() -> None:
    runtime = SatelliteAudioRuntime(
        service=None, stt=None, tts=_Tts(), segmenter_factory=None
    )
    await runtime.apply_voice_settings(VoiceSettings(volumeDay=80, volumeNight=20))

    day_hours = {8, 12, 20}
    night_hours = {21, 23, 0, 5, 7}
    assert {runtime.playback_volume_percent(hour) for hour in day_hours} == {80}
    assert {runtime.playback_volume_percent(hour) for hour in night_hours} == {20}
    # The no-argument form samples the real clock and must return one of the
    # two configured levels.
    assert runtime.playback_volume_percent() in {80, 20}


@pytest.mark.asyncio
async def test_playback_volume_uses_household_timezone(monkeypatch: pytest.MonkeyPatch) -> None:
    class _MidnightUtc(datetime):
        @classmethod
        def now(cls, tz=None):
            current = cls(2026, 7, 17, 0, 30, tzinfo=UTC)
            return current.astimezone(tz) if tz is not None else current.replace(tzinfo=None)

    monkeypatch.setattr(runtime_module, "datetime", _MidnightUtc)
    runtime = SatelliteAudioRuntime(
        service=None,
        stt=None,
        tts=_Tts(),
        segmenter_factory=None,
        playback_timezone=timezone(timedelta(hours=12)),
    )
    await runtime.apply_voice_settings(VoiceSettings(volumeDay=80, volumeNight=20))

    # 00:30 UTC is 12:30 in the household, so the daytime slider must win.
    assert runtime.playback_volume_percent() == 80


def test_household_timezone_defaults_to_auckland() -> None:
    assert Settings().household_timezone == "Pacific/Auckland"


def test_clock_context_is_speakable() -> None:
    context = current_clock_context()

    assert set(context) == {"iso", "date", "time"}
    assert context["time"].endswith(("am", "pm"))
    assert not context["time"].startswith("0")


@pytest.mark.parametrize(
    ("now", "expected_iso", "expected_time"),
    [
        ("2026-07-17T08:14:29", "2026-07-17T08:14+12:00", "8:14 am"),
        ("2026-07-17T08:14:30", "2026-07-17T08:15+12:00", "8:15 am"),
        ("2026-07-17T23:59:45", "2026-07-18T00:00+12:00", "12:00 am"),
        ("2026-07-17T12:00:00", "2026-07-17T12:00+12:00", "12:00 pm"),
    ],
)
def test_clock_context_rounds_to_minutes_and_includes_meridiem(
    now: str,
    expected_iso: str,
    expected_time: str,
) -> None:
    household_timezone = timezone(timedelta(hours=12))

    context = current_clock_context(
        household_timezone,
        now=datetime.fromisoformat(now).replace(tzinfo=household_timezone),
    )

    assert context["iso"] == expected_iso
    assert context["time"] == expected_time
    assert context["date"] == (
        "Saturday 18 July 2026" if "2026-07-18" in expected_iso else "Friday 17 July 2026"
    )
