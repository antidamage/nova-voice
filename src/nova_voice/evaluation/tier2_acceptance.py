from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Tier2Requirement = Literal[
    "multi_day_restart",
    "timezone_correctness",
    "recipient_verification",
    "amount_verification",
    "external_effect_audit",
    "visible_cancellation_undo",
]

REQUIRED_TIER2_REQUIREMENTS: tuple[Tier2Requirement, ...] = (
    "multi_day_restart",
    "timezone_correctness",
    "recipient_verification",
    "amount_verification",
    "external_effect_audit",
    "visible_cancellation_undo",
)


class Tier2AcceptanceEvidence(BaseModel):
    """Content-free evidence for one required personal-assistant invariant."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    requirement: Tier2Requirement
    artifact_revision: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    passed: bool
    scenario_count: int = Field(ge=1)
    duplicate_external_effects: int = Field(default=0, ge=0)
    unverified_recipients: int = Field(default=0, ge=0)
    amount_mismatches: int = Field(default=0, ge=0)
    unaudited_external_effects: int = Field(default=0, ge=0)
    invisible_cancellations_or_undo: int = Field(default=0, ge=0)
    metrics: dict[str, float | int | bool | str] = Field(default_factory=dict)


class Tier2GateResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    passed: bool
    required_evidence: int
    passed_evidence: int
    scenario_count: int
    missing_requirements: tuple[Tier2Requirement, ...]
    failed_requirements: tuple[Tier2Requirement, ...]
    duplicate_requirements: tuple[Tier2Requirement, ...]
    duplicate_external_effects: int
    unverified_recipients: int
    amount_mismatches: int
    unaudited_external_effects: int
    invisible_cancellations_or_undo: int
    failures: tuple[str, ...]


def evaluate_tier2_gate(
    evidence: tuple[Tier2AcceptanceEvidence, ...],
) -> Tier2GateResult:
    """Fail closed unless every Tier 2 invariant has unique passing evidence."""

    grouped: dict[Tier2Requirement, list[Tier2AcceptanceEvidence]] = {
        requirement: [] for requirement in REQUIRED_TIER2_REQUIREMENTS
    }
    for item in evidence:
        grouped[item.requirement].append(item)
    missing = tuple(item for item in REQUIRED_TIER2_REQUIREMENTS if not grouped[item])
    duplicates = tuple(item for item in REQUIRED_TIER2_REQUIREMENTS if len(grouped[item]) > 1)
    failed = tuple(
        item
        for item in REQUIRED_TIER2_REQUIREMENTS
        if grouped[item] and not all(evidence_item.passed for evidence_item in grouped[item])
    )
    counters = {
        "duplicate_external_effects": sum(item.duplicate_external_effects for item in evidence),
        "unverified_recipients": sum(item.unverified_recipients for item in evidence),
        "amount_mismatches": sum(item.amount_mismatches for item in evidence),
        "unaudited_external_effects": sum(item.unaudited_external_effects for item in evidence),
        "invisible_cancellations_or_undo": sum(
            item.invisible_cancellations_or_undo for item in evidence
        ),
    }
    failures = [*(f"missing:{item}" for item in missing)]
    failures.extend(f"duplicate_evidence:{item}" for item in duplicates)
    failures.extend(f"failed:{item}" for item in failed)
    failures.extend(name for name, count in counters.items() if count)
    passed_evidence = sum(len(items) == 1 and items[0].passed for items in grouped.values())
    return Tier2GateResult(
        passed=not failures,
        required_evidence=len(REQUIRED_TIER2_REQUIREMENTS),
        passed_evidence=passed_evidence,
        scenario_count=sum(item.scenario_count for item in evidence),
        missing_requirements=missing,
        failed_requirements=failed,
        duplicate_requirements=duplicates,
        failures=tuple(failures),
        **counters,
    )
