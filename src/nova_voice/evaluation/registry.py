from __future__ import annotations

import inspect
import sqlite3
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from nova_voice.evaluation.failure_replay import VersionPins


class ScenarioExpectations(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_success: bool
    policy_allowed: bool
    required_stages: tuple[str, ...] = ()
    terminal_status: str = "completed"
    max_latency_ms: float = Field(gt=0)
    min_memory_precision: float | None = Field(default=None, ge=0, le=1)
    proactive_outcome: str | None = None


class EvaluationScenario(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str
    version: str
    pins: VersionPins
    expectations: ScenarioExpectations


class EvaluationObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_success: bool
    policy_allowed: bool
    policy_violations: int = Field(default=0, ge=0)
    stages: tuple[str, ...] = ()
    terminal_status: str
    latency_ms: float = Field(ge=0)
    memory_precision: float | None = Field(default=None, ge=0, le=1)
    proactive_outcome: str | None = None


class Grade(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    grader: Literal["outcome", "policy", "trace", "latency", "memory", "proactivity", "model"]
    passed: bool | None
    score: float = Field(ge=0, le=1)
    reason_code: str
    grader_version: str = "deterministic-v1"


class EvaluationRun(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    scenario_id: str
    scenario_version: str
    pins_digest: str
    created_at: datetime
    observation: EvaluationObservation
    grades: tuple[Grade, ...]
    passed: bool


class DeploymentGate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    eligible: bool
    pins_digest: str
    scenario_runs: dict[str, str] = Field(default_factory=dict)
    reasons: tuple[str, ...] = ()


ModelGrader = Callable[
    [EvaluationScenario, EvaluationObservation],
    Grade | Awaitable[Grade],
]


class EvaluationRegistry:
    """SQLite registry for version-pinned scenarios, metrics, and grades."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path)
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS scenarios (
                scenario_id TEXT NOT NULL,
                version TEXT NOT NULL,
                payload TEXT NOT NULL,
                PRIMARY KEY (scenario_id, version)
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluation_runs (
                run_id TEXT PRIMARY KEY,
                scenario_id TEXT NOT NULL,
                scenario_version TEXT NOT NULL,
                pins_digest TEXT NOT NULL,
                passed INTEGER NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self._connection.commit()

    def register(self, scenario: EvaluationScenario) -> None:
        self._connection.execute(
            """
            INSERT INTO scenarios (scenario_id, version, payload)
            VALUES (?, ?, ?)
            ON CONFLICT(scenario_id, version) DO UPDATE SET payload=excluded.payload
            """,
            (scenario.scenario_id, scenario.version, scenario.model_dump_json()),
        )
        self._connection.commit()

    def scenario(self, scenario_id: str, version: str) -> EvaluationScenario:
        row = self._connection.execute(
            "SELECT payload FROM scenarios WHERE scenario_id=? AND version=?",
            (scenario_id, version),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown evaluation scenario: {scenario_id}@{version}")
        return EvaluationScenario.model_validate_json(row[0])

    async def evaluate(
        self,
        scenario: EvaluationScenario,
        observation: EvaluationObservation,
        *,
        model_grader: ModelGrader | None = None,
    ) -> EvaluationRun:
        grades = list(deterministic_grades(scenario, observation))
        if model_grader is not None and any(grade.passed is None for grade in grades):
            pending = model_grader(scenario, observation)
            model_grade = await pending if inspect.isawaitable(pending) else pending
            if model_grade.grader != "model":
                raise ValueError("selective model grader returned the wrong grader kind")
            grades.append(model_grade)
        decisive = [grade for grade in grades if grade.passed is not None]
        inconclusive = [grade for grade in grades if grade.passed is None]
        model_decisions = [grade for grade in grades if grade.grader == "model"]
        run = EvaluationRun(
            run_id=uuid4().hex,
            scenario_id=scenario.scenario_id,
            scenario_version=scenario.version,
            pins_digest=scenario.pins.digest(),
            created_at=datetime.now(UTC),
            observation=observation,
            grades=tuple(grades),
            passed=(
                bool(decisive)
                and all(grade.passed for grade in decisive)
                and (
                    not inconclusive
                    or bool(model_decisions)
                    and all(grade.passed for grade in model_decisions)
                )
            ),
        )
        self._connection.execute(
            """
            INSERT INTO evaluation_runs
            (run_id, scenario_id, scenario_version, pins_digest, passed, payload, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.run_id,
                run.scenario_id,
                run.scenario_version,
                run.pins_digest,
                int(run.passed),
                run.model_dump_json(),
                run.created_at.isoformat(),
            ),
        )
        self._connection.commit()
        return run

    def history(self, scenario_id: str) -> tuple[EvaluationRun, ...]:
        rows = self._connection.execute(
            "SELECT payload FROM evaluation_runs WHERE scenario_id=? ORDER BY created_at, run_id",
            (scenario_id,),
        ).fetchall()
        return tuple(EvaluationRun.model_validate_json(row[0]) for row in rows)

    def deployment_gate(self, pins: VersionPins) -> DeploymentGate:
        digest = pins.digest()
        reasons: list[str] = []
        selected: dict[str, str] = {}
        rows = self._connection.execute(
            "SELECT scenario_id, version, payload FROM scenarios ORDER BY scenario_id, version"
        ).fetchall()
        if not rows:
            reasons.append("no_registered_scenarios")
        for scenario_id, version, payload in rows:
            scenario = EvaluationScenario.model_validate_json(payload)
            key = f"{scenario_id}@{version}"
            if scenario.pins.digest() != digest:
                reasons.append(f"pins_mismatch:{key}")
                continue
            run_row = self._connection.execute(
                """
                SELECT payload FROM evaluation_runs
                WHERE scenario_id=? AND scenario_version=? AND pins_digest=?
                ORDER BY created_at DESC, run_id DESC LIMIT 1
                """,
                (scenario_id, version, digest),
            ).fetchone()
            if run_row is None:
                reasons.append(f"missing_run:{key}")
                continue
            run = EvaluationRun.model_validate_json(run_row[0])
            selected[key] = run.run_id
            if not run.passed:
                reasons.append(f"failed_run:{key}")
        return DeploymentGate(
            eligible=not reasons,
            pins_digest=digest,
            scenario_runs=selected,
            reasons=tuple(reasons),
        )

    def close(self) -> None:
        self._connection.close()


def deterministic_grades(
    scenario: EvaluationScenario,
    observation: EvaluationObservation,
) -> tuple[Grade, ...]:
    expected = scenario.expectations
    stages_present = _ordered_subset(expected.required_stages, observation.stages)
    memory_passed = (
        None
        if expected.min_memory_precision is None or observation.memory_precision is None
        else observation.memory_precision >= expected.min_memory_precision
    )
    proactive_passed = (
        None
        if expected.proactive_outcome is None or observation.proactive_outcome is None
        else observation.proactive_outcome == expected.proactive_outcome
    )
    return (
        _grade("outcome", observation.task_success == expected.task_success),
        _grade(
            "policy",
            observation.policy_allowed == expected.policy_allowed
            and observation.policy_violations == 0,
        ),
        _grade(
            "trace",
            stages_present and observation.terminal_status == expected.terminal_status,
        ),
        _grade("latency", observation.latency_ms <= expected.max_latency_ms),
        _grade("memory", memory_passed),
        _grade("proactivity", proactive_passed),
    )


def _grade(grader: str, passed: bool | None) -> Grade:
    return Grade(
        grader=grader,  # type: ignore[arg-type]
        passed=passed,
        score=0.5 if passed is None else float(passed),
        reason_code="not_applicable" if passed is None else "pass" if passed else "fail",
    )


def _ordered_subset(required: tuple[str, ...], actual: tuple[str, ...]) -> bool:
    position = 0
    for stage in actual:
        if position < len(required) and stage == required[position]:
            position += 1
    return position == len(required)
