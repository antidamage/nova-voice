from __future__ import annotations

import json
import wave
from pathlib import Path

from nova_voice.evaluation.audio_replay import (
    AudioReplayRunner,
    ReplayObservation,
)


def _write_wave(path: Path, value: int) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(value.to_bytes(2, "little", signed=True) * 320)


def _manifest(tmp_path: Path) -> Path:
    categories = (
        "far_field",
        "echo",
        "interruption",
        "disfluency",
        "false_activation",
    )
    cases = []
    for index, category in enumerate(categories):
        filename = f"{category}.wav"
        _write_wave(tmp_path / filename, index + 1)
        cases.append(
            {
                "id": f"case-{category}",
                "category": category,
                "audio": filename,
                "maxLatencyMs": 25,
                "expected": {
                    "transcript": None if category == "false_activation" else category,
                    "terminalStatus": (
                        "ignored" if category == "false_activation" else "completed"
                    ),
                    "stages": [] if category == "false_activation" else ["capture", "commit"],
                    "monitorKinds": [category],
                },
            }
        )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"version": 1, "cases": cases}), encoding="utf-8")
    return path


class _SteppingClock:
    def __init__(self, step_seconds: float = 0.01) -> None:
        self.value = -step_seconds
        self.step_seconds = step_seconds

    def __call__(self) -> float:
        self.value += self.step_seconds
        return self.value


async def test_replay_manifest_covers_required_audio_classes_with_pinned_traces(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)

    async def target(case) -> ReplayObservation:
        return ReplayObservation(
            transcript=case.expected.transcript,
            terminal_status=case.expected.terminal_status,
            stages=case.expected.stages,
            monitor_kinds=case.expected.monitor_kinds,
        )

    runner = AudioReplayRunner(target, clock=_SteppingClock())
    results = await runner.run_manifest(manifest)

    assert {result.category.value for result in results} == {
        "far_field",
        "echo",
        "interruption",
        "disfluency",
        "false_activation",
    }
    assert all(result.passed for result in results)
    assert all(result.elapsed_ms == 10 for result in results)


async def test_replay_reports_structural_and_latency_regressions(tmp_path: Path) -> None:
    case = AudioReplayRunner.load_manifest(_manifest(tmp_path))[0]

    async def target(_case) -> ReplayObservation:
        return ReplayObservation(
            transcript="wrong",
            terminal_status="failed",
            stages=("capture",),
            monitor_kinds=(),
        )

    result = await AudioReplayRunner(target, clock=_SteppingClock(0.03)).run_case(case)

    assert not result.passed
    assert result.failures == (
        "transcript",
        "terminal_status",
        "stages",
        "monitor_kinds",
        "latency",
    )


def test_replay_manifest_rejects_path_escape_and_non_pcm16(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.wav"
    _write_wave(outside, 1)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "cases": [
                    {
                        "id": "escape",
                        "category": "echo",
                        "audio": "../outside.wav",
                        "maxLatencyMs": 1,
                        "expected": {"terminalStatus": "ignored", "stages": []},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    try:
        AudioReplayRunner.load_manifest(manifest)
    except ValueError as error:
        assert "escapes" in str(error)
    else:
        raise AssertionError("path escape was accepted")
