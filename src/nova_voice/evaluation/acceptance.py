from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from nova_voice.evaluation.endurance import EnduranceReport


class AcceptanceCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    category: Literal["far_field", "echo_barge_in", "false_activation", "spoken_number"]
    requirement: str


class AcceptanceEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    artifact_revision: str
    passed: bool
    metrics: dict[str, float | int | bool | str] = Field(default_factory=dict)


class CorpusResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    passed: bool
    required_cases: int
    passed_cases: int
    missing_cases: tuple[str, ...]
    failed_cases: tuple[str, ...]


class StreamingValidation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    backend: str
    production_backend: bool
    chunk_count: int = Field(ge=0)
    first_audio_ms: float = Field(ge=0)
    max_chunk_gap_ms: float = Field(ge=0)
    worst_pacing_deficit_ms: float = Field(ge=0)
    audio_bytes: int = Field(ge=0)
    passed: bool


class Tier0GateResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    passed: bool
    failures: tuple[str, ...]


class HouseholdAcceptanceCorpus:
    def __init__(self, cases: tuple[AcceptanceCase, ...]) -> None:
        if len({case.case_id for case in cases}) != len(cases):
            raise ValueError("acceptance corpus case ids must be unique")
        self.cases = cases

    @classmethod
    def load(cls, path: str | Path) -> HouseholdAcceptanceCorpus:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("version") != 1:
            raise ValueError("acceptance corpus must use version 1")
        return cls(tuple(AcceptanceCase.model_validate(case) for case in payload["cases"]))

    def evaluate(self, evidence: tuple[AcceptanceEvidence, ...]) -> CorpusResult:
        evidence_by_id = {item.case_id: item for item in evidence}
        required = {case.case_id for case in self.cases}
        missing = tuple(sorted(required - evidence_by_id.keys()))
        failed = tuple(
            sorted(
                case_id
                for case_id in required & evidence_by_id.keys()
                if not evidence_by_id[case_id].passed
            )
        )
        passed_count = len(required) - len(missing) - len(failed)
        categories = {case.category for case in self.cases}
        complete_categories = {
            "far_field",
            "echo_barge_in",
            "false_activation",
            "spoken_number",
        }
        return CorpusResult(
            passed=not missing and not failed and categories == complete_categories,
            required_cases=len(required),
            passed_cases=passed_count,
            missing_cases=missing,
            failed_cases=failed,
        )


def validate_streaming_chunks(
    *,
    backend: str,
    production_backend: bool,
    chunk_times_ms: tuple[float, ...],
    chunk_sizes: tuple[int, ...],
    audio_bytes: int,
    sample_rate: int = 24_000,
    playback_preroll_ms: float = 700,
    max_first_audio_ms: float = 3_000,
) -> StreamingValidation:
    if len(chunk_times_ms) != len(chunk_sizes):
        raise ValueError("streaming chunk times and sizes must have equal length")
    gaps = tuple(
        later - earlier
        for earlier, later in zip(chunk_times_ms, chunk_times_ms[1:], strict=False)
    )
    first_audio = chunk_times_ms[0] if chunk_times_ms else 0.0
    largest_gap = max(gaps, default=0.0)
    delivered_ms = 0.0
    worst_deficit = 0.0
    for arrived_ms, size in zip(chunk_times_ms, chunk_sizes, strict=True):
        delivered_ms += size / 2 / sample_rate * 1_000
        elapsed_after_first = max(0.0, arrived_ms - first_audio)
        worst_deficit = max(worst_deficit, elapsed_after_first - delivered_ms)
    passed = bool(
        production_backend
        and len(chunk_times_ms) >= 2
        and audio_bytes > 0
        and first_audio <= max_first_audio_ms
        and worst_deficit <= playback_preroll_ms
    )
    return StreamingValidation(
        backend=backend,
        production_backend=production_backend,
        chunk_count=len(chunk_times_ms),
        first_audio_ms=first_audio,
        max_chunk_gap_ms=largest_gap,
        worst_pacing_deficit_ms=worst_deficit,
        audio_bytes=audio_bytes,
        passed=passed,
    )


def evaluate_tier0_gate(
    endurance: EnduranceReport,
    corpus: CorpusResult,
    streaming: StreamingValidation,
    *,
    minimum_endurance_seconds: float = 86_400,
) -> Tier0GateResult:
    failures: list[str] = []
    if endurance.duration_seconds < minimum_endurance_seconds:
        failures.append("endurance_duration")
    if not endurance.passed:
        failures.extend(endurance.failures)
    latency_targets = {
        "finalSpeechToToolDispatch": (1_200, 2_000),
        "finalSpeechToFirstAudio": (1_800, 3_000),
        "novaMutation": (100, 300),
        "satelliteReconnect": (2_000, 5_000),
    }
    for name, (p50, p95) in latency_targets.items():
        measured = endurance.latency_percentiles_ms.get(name)
        if measured is None or measured["p50"] > p50 or measured["p95"] > p95:
            failures.append(f"latency:{name}")
    if not corpus.passed:
        failures.append("acceptance_corpus")
    if not streaming.passed:
        failures.append("production_streaming_tts")
    return Tier0GateResult(passed=not failures, failures=tuple(dict.fromkeys(failures)))
