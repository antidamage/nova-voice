from __future__ import annotations

import asyncio

import numpy as np

from nova_voice.audio.election import SegmentElection, energy_envelope, envelopes_match
from nova_voice.audio.pcm import float32_to_pcm16
from nova_voice.audio.segmenter import SpeechSegment
from nova_voice.domain import AcousticFeatures


def segment(amplitude: float) -> SpeechSegment:
    time = np.arange(16_000, dtype=np.float32) / 16_000
    modulation = 0.35 + 0.65 * np.square(np.sin(2 * np.pi * 3 * time))
    samples = amplitude * modulation * np.sin(2 * np.pi * 180 * time)
    return SpeechSegment(
        pcm16=float32_to_pcm16(samples),
        acoustic=AcousticFeatures(duration_ms=1000, rms_db=20 * np.log10(amplitude)),
    )


def test_energy_envelope_matches_gain_changed_copy() -> None:
    assert envelopes_match(energy_envelope(segment(0.2)), energy_envelope(segment(0.5)))


async def test_election_returns_only_best_source() -> None:
    election = SegmentElection(election_window_seconds=0.01)

    first, second = await asyncio.gather(
        election.elect("quiet", segment(0.2), wake_detected=False),
        election.elect("clear", segment(0.5), wake_detected=False),
    )

    assert not first
    assert second


async def test_election_does_not_merge_identical_speech_from_different_rooms() -> None:
    election = SegmentElection(election_window_seconds=0.01)

    first, second = await asyncio.gather(
        election.elect("lounge-mic", segment(0.4), wake_detected=False, room_id="lounge"),
        election.elect("office-mic", segment(0.4), wake_detected=False, room_id="office"),
    )

    assert first
    assert second


def offset_segment(amplitude: float, duration_ms: int = 1000) -> SpeechSegment:
    """A segment whose envelope shape differs from ``segment``'s."""

    samples_count = int(16 * duration_ms)
    time = np.arange(samples_count, dtype=np.float32) / 16_000
    modulation = 0.9 - 0.6 * np.abs(np.sin(2 * np.pi * 1.3 * time))
    samples = amplitude * modulation * np.sin(2 * np.pi * 240 * time)
    return SpeechSegment(
        pcm16=float32_to_pcm16(samples),
        acoustic=AcousticFeatures(duration_ms=duration_ms, rms_db=20 * np.log10(amplitude)),
    )


async def test_household_scope_elects_one_source_across_rooms() -> None:
    election = SegmentElection(election_window_seconds=0.01)

    first, second = await asyncio.gather(
        election.elect(
            "lounge-mic", segment(0.2), wake_detected=False, room_id="lounge",
            scope_id="household",
        ),
        election.elect(
            "office-mic", segment(0.5), wake_detected=False, room_id="office",
            scope_id="household",
        ),
    )

    assert not first
    assert second


async def test_overlapping_capture_intervals_group_despite_envelope_mismatch() -> None:
    election = SegmentElection(election_window_seconds=0.01)

    # Different microphones cut the same utterance with different envelopes
    # and lengths; the near-simultaneous capture intervals still group them.
    first, second = await asyncio.gather(
        election.elect(
            "quiet", offset_segment(0.2, duration_ms=1400), wake_detected=False,
            scope_id="household",
        ),
        election.elect(
            "clear", segment(0.5), wake_detected=False, scope_id="household",
        ),
    )

    assert not first
    assert second
