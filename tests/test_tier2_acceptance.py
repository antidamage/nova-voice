from __future__ import annotations

import pytest
from pydantic import ValidationError

from nova_voice.evaluation.tier2_acceptance import (
    REQUIRED_TIER2_REQUIREMENTS,
    Tier2AcceptanceEvidence,
    evaluate_tier2_gate,
)


def _evidence(requirement, **updates) -> Tier2AcceptanceEvidence:
    values = {
        "requirement": requirement,
        "artifact_revision": f"sha256:{'b' * 64}",
        "passed": True,
        "scenario_count": 4,
    }
    values.update(updates)
    return Tier2AcceptanceEvidence(**values)


def test_tier2_gate_requires_unique_passing_evidence_for_every_invariant() -> None:
    result = evaluate_tier2_gate(
        tuple(_evidence(requirement) for requirement in REQUIRED_TIER2_REQUIREMENTS)
    )

    assert result.passed
    assert result.required_evidence == 6
    assert result.passed_evidence == 6
    assert result.scenario_count == 24
    assert not result.failures


def test_tier2_gate_fails_closed_for_missing_failed_and_duplicate_evidence() -> None:
    result = evaluate_tier2_gate(
        (
            _evidence("multi_day_restart", passed=False),
            _evidence("multi_day_restart"),
            _evidence("timezone_correctness"),
        )
    )

    assert not result.passed
    assert "multi_day_restart" in result.failed_requirements
    assert "multi_day_restart" in result.duplicate_requirements
    assert "recipient_verification" in result.missing_requirements


@pytest.mark.parametrize(
    "counter",
    [
        "duplicate_external_effects",
        "unverified_recipients",
        "amount_mismatches",
        "unaudited_external_effects",
        "invisible_cancellations_or_undo",
    ],
)
def test_tier2_gate_has_zero_tolerance_for_safety_failures(counter: str) -> None:
    evidence = [_evidence(requirement) for requirement in REQUIRED_TIER2_REQUIREMENTS]
    evidence[0] = evidence[0].model_copy(update={counter: 1})

    result = evaluate_tier2_gate(tuple(evidence))

    assert not result.passed
    assert counter in result.failures


def test_tier2_evidence_requires_content_addressed_artifacts() -> None:
    with pytest.raises(ValidationError):
        _evidence("external_effect_audit", artifact_revision="latest")
