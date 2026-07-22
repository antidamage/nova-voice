from __future__ import annotations

import pytest
from pydantic import ValidationError

from nova_voice.evaluation.tier1_acceptance import (
    REQUIRED_TIER1_REQUIREMENTS,
    Tier1AcceptanceEvidence,
    evaluate_tier1_gate,
)


def _evidence(requirement, **updates) -> Tier1AcceptanceEvidence:
    values = {
        "requirement": requirement,
        "artifact_revision": f"sha256:{'a' * 64}",
        "passed": True,
        "scenario_count": 3,
    }
    values.update(updates)
    return Tier1AcceptanceEvidence(**values)


def test_tier1_gate_requires_unique_passing_evidence_for_every_invariant() -> None:
    complete = tuple(_evidence(requirement) for requirement in REQUIRED_TIER1_REQUIREMENTS)

    result = evaluate_tier1_gate(complete)

    assert result.passed
    assert result.required_evidence == 6
    assert result.passed_evidence == 6
    assert result.scenario_count == 18
    assert not result.failures


def test_tier1_gate_fails_closed_for_missing_failed_and_duplicate_evidence() -> None:
    result = evaluate_tier1_gate(
        (
            _evidence("restart_exactly_once", passed=False),
            _evidence("restart_exactly_once"),
            _evidence("permission_correctness"),
        )
    )

    assert not result.passed
    assert "restart_exactly_once" in result.failed_requirements
    assert "restart_exactly_once" in result.duplicate_requirements
    assert "pre_activation_simulation" in result.missing_requirements


@pytest.mark.parametrize(
    ("field", "failure"),
    [
        ("duplicate_mutations", "duplicate_mutations"),
        ("unapproved_high_impact_mutations", "unapproved_high_impact_mutations"),
    ],
)
def test_tier1_gate_has_zero_tolerance_for_unsafe_mutations(field: str, failure: str) -> None:
    evidence = [_evidence(requirement) for requirement in REQUIRED_TIER1_REQUIREMENTS]
    evidence[0] = evidence[0].model_copy(update={field: 1})

    result = evaluate_tier1_gate(tuple(evidence))

    assert not result.passed
    assert failure in result.failures


def test_tier1_evidence_requires_content_addressed_artifacts() -> None:
    with pytest.raises(ValidationError):
        _evidence("mempalace_recovery", artifact_revision="latest")
