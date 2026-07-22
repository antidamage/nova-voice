from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Tier3Requirement = Literal[
    "longitudinal_precision",
    "contradiction_correction",
    "privacy_boundary",
    "multi_party_dialogue",
    "naturalness",
    "authority_separation",
]

REQUIRED_TIER3_REQUIREMENTS: tuple[Tier3Requirement, ...] = (
    "longitudinal_precision",
    "contradiction_correction",
    "privacy_boundary",
    "multi_party_dialogue",
    "naturalness",
    "authority_separation",
)


class Tier3AcceptanceEvidence(BaseModel):
    """Content-free evidence for one required continuity invariant."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    requirement: Tier3Requirement
    artifact_revision: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    passed: bool
    scenario_count: int = Field(ge=1)
    false_recollections: int = Field(default=0, ge=0)
    uncorrected_contradictions: int = Field(default=0, ge=0)
    privacy_disclosures: int = Field(default=0, ge=0)
    misattributed_turns: int = Field(default=0, ge=0)
    naturalness_failures: int = Field(default=0, ge=0)
    unauthorized_actions: int = Field(default=0, ge=0)
    metrics: dict[str, float | int | bool | str] = Field(default_factory=dict)


class Tier3GateResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    passed: bool
    required_evidence: int
    passed_evidence: int
    scenario_count: int
    missing_requirements: tuple[Tier3Requirement, ...]
    failed_requirements: tuple[Tier3Requirement, ...]
    duplicate_requirements: tuple[Tier3Requirement, ...]
    false_recollections: int
    uncorrected_contradictions: int
    privacy_disclosures: int
    misattributed_turns: int
    naturalness_failures: int
    unauthorized_actions: int
    failures: tuple[str, ...]


def evaluate_tier3_gate(
    evidence: tuple[Tier3AcceptanceEvidence, ...],
) -> Tier3GateResult:
    """Fail closed unless every Tier 3 invariant has unique passing evidence."""

    grouped: dict[Tier3Requirement, list[Tier3AcceptanceEvidence]] = {
        requirement: [] for requirement in REQUIRED_TIER3_REQUIREMENTS
    }
    for item in evidence:
        grouped[item.requirement].append(item)
    missing = tuple(item for item in REQUIRED_TIER3_REQUIREMENTS if not grouped[item])
    duplicates = tuple(item for item in REQUIRED_TIER3_REQUIREMENTS if len(grouped[item]) > 1)
    failed = tuple(
        item
        for item in REQUIRED_TIER3_REQUIREMENTS
        if grouped[item] and not all(evidence_item.passed for evidence_item in grouped[item])
    )
    counter_names = (
        "false_recollections",
        "uncorrected_contradictions",
        "privacy_disclosures",
        "misattributed_turns",
        "naturalness_failures",
        "unauthorized_actions",
    )
    counters = {
        name: sum(getattr(item, name) for item in evidence) for name in counter_names
    }
    failures = [*(f"missing:{item}" for item in missing)]
    failures.extend(f"duplicate_evidence:{item}" for item in duplicates)
    failures.extend(f"failed:{item}" for item in failed)
    failures.extend(name for name, count in counters.items() if count)
    return Tier3GateResult(
        passed=not failures,
        required_evidence=len(REQUIRED_TIER3_REQUIREMENTS),
        passed_evidence=sum(
            len(items) == 1 and items[0].passed for items in grouped.values()
        ),
        scenario_count=sum(item.scenario_count for item in evidence),
        missing_requirements=missing,
        failed_requirements=failed,
        duplicate_requirements=duplicates,
        failures=tuple(failures),
        **counters,
    )
