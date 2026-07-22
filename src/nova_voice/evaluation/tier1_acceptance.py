from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Tier1Requirement = Literal[
    "restart_exactly_once",
    "permission_correctness",
    "pre_activation_simulation",
    "proactive_reason_audit",
    "mempalace_recovery",
    "high_impact_safety",
]

REQUIRED_TIER1_REQUIREMENTS: tuple[Tier1Requirement, ...] = (
    "restart_exactly_once",
    "permission_correctness",
    "pre_activation_simulation",
    "proactive_reason_audit",
    "mempalace_recovery",
    "high_impact_safety",
)


class Tier1AcceptanceEvidence(BaseModel):
    """Content-free evidence for one required Tier 1 safety property."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    requirement: Tier1Requirement
    artifact_revision: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    passed: bool
    scenario_count: int = Field(ge=1)
    duplicate_mutations: int = Field(default=0, ge=0)
    unapproved_high_impact_mutations: int = Field(default=0, ge=0)
    metrics: dict[str, float | int | bool | str] = Field(default_factory=dict)


class Tier1GateResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    passed: bool
    required_evidence: int
    passed_evidence: int
    scenario_count: int
    missing_requirements: tuple[Tier1Requirement, ...]
    failed_requirements: tuple[Tier1Requirement, ...]
    duplicate_requirements: tuple[Tier1Requirement, ...]
    duplicate_mutations: int
    unapproved_high_impact_mutations: int
    failures: tuple[str, ...]


def evaluate_tier1_gate(
    evidence: tuple[Tier1AcceptanceEvidence, ...],
) -> Tier1GateResult:
    """Fail closed unless every Tier 1 invariant has unique passing evidence."""

    grouped: dict[Tier1Requirement, list[Tier1AcceptanceEvidence]] = {
        requirement: [] for requirement in REQUIRED_TIER1_REQUIREMENTS
    }
    for item in evidence:
        grouped[item.requirement].append(item)

    missing = tuple(
        requirement for requirement in REQUIRED_TIER1_REQUIREMENTS if not grouped[requirement]
    )
    duplicates = tuple(
        requirement for requirement in REQUIRED_TIER1_REQUIREMENTS if len(grouped[requirement]) > 1
    )
    failed = tuple(
        requirement
        for requirement in REQUIRED_TIER1_REQUIREMENTS
        if grouped[requirement] and not all(item.passed for item in grouped[requirement])
    )
    duplicate_mutations = sum(item.duplicate_mutations for item in evidence)
    unapproved_high_impact = sum(item.unapproved_high_impact_mutations for item in evidence)

    failures: list[str] = []
    failures.extend(f"missing:{requirement}" for requirement in missing)
    failures.extend(f"duplicate_evidence:{requirement}" for requirement in duplicates)
    failures.extend(f"failed:{requirement}" for requirement in failed)
    if duplicate_mutations:
        failures.append("duplicate_mutations")
    if unapproved_high_impact:
        failures.append("unapproved_high_impact_mutations")

    passed_requirements = sum(
        len(items) == 1 and items[0].passed for items in grouped.values()
    )
    return Tier1GateResult(
        passed=not failures,
        required_evidence=len(REQUIRED_TIER1_REQUIREMENTS),
        passed_evidence=passed_requirements,
        scenario_count=sum(item.scenario_count for item in evidence),
        missing_requirements=missing,
        failed_requirements=failed,
        duplicate_requirements=duplicates,
        duplicate_mutations=duplicate_mutations,
        unapproved_high_impact_mutations=unapproved_high_impact,
        failures=tuple(failures),
    )
