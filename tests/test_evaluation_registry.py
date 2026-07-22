from __future__ import annotations

from pathlib import Path

from nova_voice.evaluation.failure_replay import VersionPins
from nova_voice.evaluation.registry import (
    EvaluationObservation,
    EvaluationRegistry,
    EvaluationScenario,
    Grade,
    ScenarioExpectations,
    deterministic_grades,
)


def _scenario() -> EvaluationScenario:
    return EvaluationScenario(
        scenario_id="office-light",
        version="1.0.0",
        pins=VersionPins(
            models={"interpreter": "qwen-test"},
            prompts={"system": "sha256:prompt"},
            contracts={"nova": "nova-provider-v1"},
            skills={"home": "sha256:skill"},
            policies={"execution": "policy-v1"},
            providers={"nova": "provider-v1"},
        ),
        expectations=ScenarioExpectations(
            task_success=True,
            policy_allowed=True,
            required_stages=("capture", "authorize", "commit"),
            max_latency_ms=1_000,
            min_memory_precision=0.8,
            proactive_outcome="not_applicable",
        ),
    )


def _observation(**updates) -> EvaluationObservation:
    values = {
        "task_success": True,
        "policy_allowed": True,
        "policy_violations": 0,
        "stages": ("capture", "endpoint", "authorize", "commit"),
        "terminal_status": "completed",
        "latency_ms": 250,
        "memory_precision": 0.9,
        "proactive_outcome": "not_applicable",
    }
    values.update(updates)
    return EvaluationObservation(**values)


async def test_registry_persists_versioned_metrics_and_deterministic_grades(
    tmp_path: Path,
) -> None:
    path = tmp_path / "evaluations.sqlite3"
    scenario = _scenario()
    registry = EvaluationRegistry(path)
    registry.register(scenario)

    run = await registry.evaluate(scenario, _observation())

    assert run.passed
    assert [grade.grader for grade in run.grades] == [
        "outcome",
        "policy",
        "trace",
        "latency",
        "memory",
        "proactivity",
    ]
    assert registry.scenario("office-light", "1.0.0") == scenario
    assert registry.history("office-light") == (run,)
    gate = registry.deployment_gate(scenario.pins)
    assert gate.eligible
    assert gate.scenario_runs == {"office-light@1.0.0": run.run_id}
    registry.close()

    reopened = EvaluationRegistry(path)
    assert reopened.history("office-light") == (run,)
    reopened.close()


async def test_selective_model_grader_runs_only_for_inconclusive_metrics(tmp_path: Path) -> None:
    registry = EvaluationRegistry(tmp_path / "evaluations.sqlite3")
    scenario = _scenario()
    calls = 0

    async def model_grader(_scenario, _observation) -> Grade:
        nonlocal calls
        calls += 1
        return Grade(
            grader="model",
            passed=True,
            score=0.8,
            reason_code="structural_review_pass",
            grader_version="fixture-model-v1",
        )

    conclusive = await registry.evaluate(scenario, _observation(), model_grader=model_grader)
    inconclusive = await registry.evaluate(
        scenario,
        _observation(memory_precision=None),
        model_grader=model_grader,
    )

    assert calls == 1
    assert all(grade.grader != "model" for grade in conclusive.grades)
    assert inconclusive.grades[-1].grader == "model"
    registry.close()


async def test_inconclusive_required_metric_blocks_gate_without_model_grade(
    tmp_path: Path,
) -> None:
    registry = EvaluationRegistry(tmp_path / "evaluations.sqlite3")

    run = await registry.evaluate(_scenario(), _observation(memory_precision=None))

    assert not run.passed
    assert next(grade for grade in run.grades if grade.grader == "memory").passed is None
    registry.close()


def test_deterministic_graders_report_policy_latency_memory_and_trace_failures() -> None:
    grades = deterministic_grades(
        _scenario(),
        _observation(
            policy_violations=1,
            stages=("capture", "commit"),
            latency_ms=2_000,
            memory_precision=0.2,
        ),
    )

    failed = {grade.grader for grade in grades if grade.passed is False}
    assert failed == {"policy", "trace", "latency", "memory"}


def test_deployment_gate_requires_registered_matching_passing_runs(tmp_path: Path) -> None:
    registry = EvaluationRegistry(tmp_path / "evaluations.sqlite3")
    empty = registry.deployment_gate(_scenario().pins)
    registry.register(_scenario())
    missing = registry.deployment_gate(_scenario().pins)
    mismatched = registry.deployment_gate(
        _scenario().pins.model_copy(update={"models": {"interpreter": "other"}})
    )

    assert empty.reasons == ("no_registered_scenarios",)
    assert missing.reasons == ("missing_run:office-light@1.0.0",)
    assert mismatched.reasons == ("pins_mismatch:office-light@1.0.0",)
    registry.close()
