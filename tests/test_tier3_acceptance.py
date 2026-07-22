from __future__ import annotations

import pytest
from pydantic import ValidationError

from nova_voice.evaluation.tier3_acceptance import (
    REQUIRED_TIER3_REQUIREMENTS,
    Tier3AcceptanceEvidence,
    evaluate_tier3_gate,
)


def _evidence(requirement, **updates) -> Tier3AcceptanceEvidence:
    values = {
        "requirement": requirement,
        "artifact_revision": f"sha256:{'c' * 64}",
        "passed": True,
        "scenario_count": 4,
    }
    values.update(updates)
    return Tier3AcceptanceEvidence(**values)


def test_tier3_gate_requires_unique_passing_evidence_for_every_invariant() -> None:
    result = evaluate_tier3_gate(
        tuple(_evidence(requirement) for requirement in REQUIRED_TIER3_REQUIREMENTS)
    )

    assert result.passed
    assert result.required_evidence == 6
    assert result.passed_evidence == 6
    assert result.scenario_count == 24


def test_tier3_gate_fails_closed_for_missing_failed_and_duplicate_evidence() -> None:
    result = evaluate_tier3_gate(
        (
            _evidence("longitudinal_precision", passed=False),
            _evidence("longitudinal_precision"),
            _evidence("privacy_boundary"),
        )
    )

    assert not result.passed
    assert "longitudinal_precision" in result.failed_requirements
    assert "longitudinal_precision" in result.duplicate_requirements
    assert "naturalness" in result.missing_requirements


@pytest.mark.parametrize(
    "counter",
    [
        "false_recollections",
        "uncorrected_contradictions",
        "privacy_disclosures",
        "misattributed_turns",
        "naturalness_failures",
        "unauthorized_actions",
    ],
)
def test_tier3_gate_has_zero_tolerance_for_failures(counter: str) -> None:
    evidence = [_evidence(requirement) for requirement in REQUIRED_TIER3_REQUIREMENTS]
    evidence[0] = evidence[0].model_copy(update={counter: 1})

    result = evaluate_tier3_gate(tuple(evidence))

    assert not result.passed
    assert counter in result.failures


def test_tier3_evidence_requires_content_addressed_artifacts() -> None:
    with pytest.raises(ValidationError):
        _evidence("naturalness", artifact_revision="latest")
