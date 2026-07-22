"""Deterministic replay and household-simulation support."""

from nova_voice.evaluation.audio_replay import AudioReplayRunner
from nova_voice.evaluation.failure_replay import PinnedFailureReplayer, VersionPins
from nova_voice.evaluation.household import HouseholdSimulator, SimulatedHouseholdProvider
from nova_voice.evaluation.registry import EvaluationRegistry
from nova_voice.evaluation.tier1_acceptance import (
    Tier1AcceptanceEvidence,
    Tier1GateResult,
    evaluate_tier1_gate,
)

__all__ = [
    "AudioReplayRunner",
    "EvaluationRegistry",
    "HouseholdSimulator",
    "PinnedFailureReplayer",
    "SimulatedHouseholdProvider",
    "Tier1AcceptanceEvidence",
    "Tier1GateResult",
    "VersionPins",
    "evaluate_tier1_gate",
]
