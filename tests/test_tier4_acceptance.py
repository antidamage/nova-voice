import pytest

from nova_voice.evaluation.tier4_acceptance import (
    REQUIRED_TIER4_REQUIREMENTS,
    Tier4AcceptanceEvidence,
    evaluate_tier4_gate,
)


def _evidence(requirement, **updates) -> Tier4AcceptanceEvidence:
    values = {
        "requirement": requirement,
        "artifact_revision": f"sha256:{'e' * 64}",
        "passed": True,
        "scenario_count": 3,
    }
    values.update(updates)
    return Tier4AcceptanceEvidence(**values)


def test_tier4_gate_requires_unique_passing_evidence() -> None:
    result = evaluate_tier4_gate(
        tuple(_evidence(requirement) for requirement in REQUIRED_TIER4_REQUIREMENTS)
    )

    assert result.passed
    assert result.counters["duplex_production_turns"] == 0


def test_tier4_gate_fails_closed_for_missing_failed_and_duplicate_evidence() -> None:
    result = evaluate_tier4_gate(
        (
            _evidence("multimodal_privacy", passed=False),
            _evidence("multimodal_privacy"),
        )
    )

    assert not result.passed
    assert "multimodal_privacy" in result.failed_requirements
    assert "multimodal_privacy" in result.duplicate_requirements
    assert "cascade_retention" in result.missing_requirements


@pytest.mark.parametrize(
    "counter",
    [
        "privacy_violations",
        "twin_side_effects",
        "optimizer_applied_changes",
        "foreground_optimizer_calls",
        "duplex_production_turns",
        "unprovenanced_visual_contexts",
    ],
)
def test_tier4_gate_has_zero_tolerance_for_frontier_failures(counter: str) -> None:
    evidence = [_evidence(requirement) for requirement in REQUIRED_TIER4_REQUIREMENTS]
    evidence[0] = evidence[0].model_copy(update={counter: 1})

    result = evaluate_tier4_gate(tuple(evidence))

    assert not result.passed
    assert counter in result.failures
