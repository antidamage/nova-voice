from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest

from nova_voice.evaluation.audio_replay import ReplayObservation
from nova_voice.evaluation.failure_replay import (
    FailureReplayStore,
    PinnedFailureReplayer,
    VersionPins,
    version_mismatches,
)


def _pins(suffix: str = "1") -> VersionPins:
    return VersionPins(
        models={"interpreter": f"model-{suffix}"},
        prompts={"system": f"prompt-{suffix}"},
        contracts={"nova": f"contract-{suffix}"},
        skills={"core": f"skill-{suffix}"},
        policies={"execution": f"policy-{suffix}"},
        providers={"nova": f"provider-{suffix}"},
    )


def _fixture(root: Path) -> Path:
    audio = root / "audio.wav"
    with wave.open(str(audio), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(b"\x01\x00" * 320)
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "cases": [
                    {
                        "id": "failed-case",
                        "category": "echo",
                        "audio": "audio.wav",
                        "maxLatencyMs": 10_000,
                        "expected": {
                            "transcript": None,
                            "terminalStatus": "ignored",
                            "stages": [],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest


async def test_failure_replay_requires_the_exact_version_environment(tmp_path: Path) -> None:
    _fixture(tmp_path)
    store = FailureReplayStore(tmp_path)
    store.save(
        failure_id="failure-1",
        replay_manifest="manifest.json",
        replay_case_id="failed-case",
        trace_id="trace-1",
        input_revision="input-hash",
        failure_codes=("terminal_status",),
        pins=_pins(),
    )
    replayer = PinnedFailureReplayer(store)

    async def target(_case) -> ReplayObservation:
        return ReplayObservation(None, "ignored", ())

    replayer.register_environment(_pins("different"), target)
    with pytest.raises(RuntimeError, match="exact pinned"):
        await replayer.replay("failure-1")

    replayer.register_environment(_pins(), target)
    result = await replayer.replay("failure-1")
    assert result.passed
    repeated = await replayer.replay_many(("failure-1",))
    assert len(repeated) == 1
    assert repeated[0].passed


def test_failure_artifact_contains_revisions_and_pins_not_content(tmp_path: Path) -> None:
    _fixture(tmp_path)
    store = FailureReplayStore(tmp_path)
    artifact = store.save(
        failure_id="failure-safe",
        replay_manifest="manifest.json",
        replay_case_id="failed-case",
        input_revision="input-hash",
        failure_codes=("latency",),
        pins=_pins(),
    )

    payload = (tmp_path / "failures" / "failure-safe.json").read_text(encoding="utf-8")
    assert artifact.pins.digest()
    assert "transcript" not in payload
    assert "pcm16" not in payload
    assert set(version_mismatches(_pins(), _pins("2"))) == {
        "models",
        "prompts",
        "contracts",
        "skills",
        "policies",
        "providers",
    }
