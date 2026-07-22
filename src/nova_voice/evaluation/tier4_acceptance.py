from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Tier4Requirement = Literal[
    "multimodal_privacy",
    "digital_twin_safety",
    "visual_continuity",
    "optimizer_isolation",
    "cascade_retention",
    "frontier_comparison",
]

REQUIRED_TIER4_REQUIREMENTS: tuple[Tier4Requirement, ...] = (
    "multimodal_privacy",
    "digital_twin_safety",
    "visual_continuity",
    "optimizer_isolation",
    "cascade_retention",
    "frontier_comparison",
)


class Tier4AcceptanceEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    requirement: Tier4Requirement
    artifact_revision: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    passed: bool
    scenario_count: int = Field(ge=1)
    privacy_violations: int = Field(default=0, ge=0)
    twin_side_effects: int = Field(default=0, ge=0)
    optimizer_applied_changes: int = Field(default=0, ge=0)
    foreground_optimizer_calls: int = Field(default=0, ge=0)
    duplex_production_turns: int = Field(default=0, ge=0)
    unprovenanced_visual_contexts: int = Field(default=0, ge=0)


class Tier4GateResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    passed: bool
    missing_requirements: tuple[Tier4Requirement, ...]
    failed_requirements: tuple[Tier4Requirement, ...]
    duplicate_requirements: tuple[Tier4Requirement, ...]
    failures: tuple[str, ...]
    counters: dict[str, int]


def evaluate_tier4_gate(evidence: tuple[Tier4AcceptanceEvidence, ...]) -> Tier4GateResult:
    grouped: dict[Tier4Requirement, list[Tier4AcceptanceEvidence]] = {
        requirement: [] for requirement in REQUIRED_TIER4_REQUIREMENTS
    }
    for item in evidence:
        grouped[item.requirement].append(item)
    missing = tuple(item for item in REQUIRED_TIER4_REQUIREMENTS if not grouped[item])
    duplicates = tuple(item for item in REQUIRED_TIER4_REQUIREMENTS if len(grouped[item]) > 1)
    failed = tuple(
        item
        for item, values in grouped.items()
        if values and not all(value.passed for value in values)
    )
    counter_names = (
        "privacy_violations",
        "twin_side_effects",
        "optimizer_applied_changes",
        "foreground_optimizer_calls",
        "duplex_production_turns",
        "unprovenanced_visual_contexts",
    )
    counters = {
        name: sum(getattr(item, name) for item in evidence) for name in counter_names
    }
    failures = [*(f"missing:{item}" for item in missing)]
    failures.extend(f"duplicate_evidence:{item}" for item in duplicates)
    failures.extend(f"failed:{item}" for item in failed)
    failures.extend(name for name, count in counters.items() if count)
    return Tier4GateResult(
        passed=not failures,
        missing_requirements=missing,
        failed_requirements=failed,
        duplicate_requirements=duplicates,
        failures=tuple(failures),
        counters=counters,
    )
