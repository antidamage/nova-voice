from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from nova_voice.evaluation.audio_replay import AudioReplayRunner, ReplayResult, ReplayTarget


class VersionPins(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    models: dict[str, str] = Field(min_length=1)
    prompts: dict[str, str] = Field(min_length=1)
    contracts: dict[str, str] = Field(min_length=1)
    skills: dict[str, str] = Field(min_length=1)
    policies: dict[str, str] = Field(min_length=1)
    providers: dict[str, str] = Field(min_length=1)

    def digest(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


class FailureArtifact(BaseModel):
    """Content-free pointer from a failure to an existing replay fixture."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    failure_id: str
    captured_at: datetime
    replay_manifest: str
    replay_case_id: str
    trace_id: str | None = None
    input_revision: str
    failure_codes: tuple[str, ...]
    pins: VersionPins


class FailureReplayStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def save(
        self,
        *,
        failure_id: str,
        replay_manifest: str,
        replay_case_id: str,
        input_revision: str,
        failure_codes: tuple[str, ...],
        pins: VersionPins,
        trace_id: str | None = None,
    ) -> FailureArtifact:
        artifact = FailureArtifact(
            failure_id=failure_id,
            captured_at=datetime.now(UTC),
            replay_manifest=replay_manifest,
            replay_case_id=replay_case_id,
            trace_id=trace_id,
            input_revision=input_revision,
            failure_codes=failure_codes,
            pins=pins,
        )
        path = self._artifact_path(failure_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
        return artifact

    def load(self, failure_id: str) -> FailureArtifact:
        return FailureArtifact.model_validate_json(
            self._artifact_path(failure_id).read_text(encoding="utf-8")
        )

    def manifest_path(self, artifact: FailureArtifact) -> Path:
        path = (self.root / artifact.replay_manifest).resolve()
        if self.root not in path.parents:
            raise ValueError("failure replay manifest escapes store root")
        return path

    def _artifact_path(self, failure_id: str) -> Path:
        safe_characters = "abcdefghijklmnopqrstuvwxyz0123456789-_"
        if not failure_id or any(
            character not in safe_characters for character in failure_id.casefold()
        ):
            raise ValueError("failure id contains unsafe characters")
        return self.root / "failures" / f"{failure_id}.json"


class PinnedFailureReplayer:
    def __init__(self, store: FailureReplayStore) -> None:
        self.store = store
        self._targets: dict[str, ReplayTarget] = {}

    def register_environment(self, pins: VersionPins, target: ReplayTarget) -> None:
        digest = pins.digest()
        if digest in self._targets:
            raise ValueError("failure replay environment is already registered")
        self._targets[digest] = target

    async def replay(self, failure_id: str) -> ReplayResult:
        artifact = self.store.load(failure_id)
        target = self._targets.get(artifact.pins.digest())
        if target is None:
            raise RuntimeError("exact pinned evaluation environment is unavailable")
        manifest = self.store.manifest_path(artifact)
        cases = AudioReplayRunner.load_manifest(manifest)
        try:
            case = next(case for case in cases if case.case_id == artifact.replay_case_id)
        except StopIteration as error:
            raise ValueError("failure replay case is missing from its pinned manifest") from error
        return await AudioReplayRunner(target).run_case(case)

    async def replay_many(self, failure_ids: tuple[str, ...]) -> tuple[ReplayResult, ...]:
        results = tuple([await self.replay(failure_id) for failure_id in failure_ids])
        if failed := [result.case_id for result in results if not result.passed]:
            raise RuntimeError(f"pinned failure replay gate failed: {failed}")
        return results


def version_mismatches(expected: VersionPins, actual: VersionPins) -> dict[str, Any]:
    mismatches: dict[str, Any] = {}
    expected_values = expected.model_dump(mode="json")
    actual_values = actual.model_dump(mode="json")
    for category, expected_pins in expected_values.items():
        if expected_pins != actual_values[category]:
            mismatches[category] = {
                "expected": expected_pins,
                "actual": actual_values[category],
            }
    return mismatches
