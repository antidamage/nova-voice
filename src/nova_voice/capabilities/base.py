from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from nova_voice.domain import PlannedAction, ToolResult


class ToolPolicy(BaseModel):
    """Deterministic execution metadata owned by a provider, not the LLM."""

    model_config = ConfigDict(extra="forbid")

    risk: Literal["low", "confirmation", "blocked"] = "low"
    reversible: bool = True
    idempotent: bool = False
    parallel_safe: bool = False
    # Provider-owned lock names for durable plan concurrency. Templates may
    # reference scalar action arguments, for example ``entity:{entity_id}``.
    # No declaration conservatively locks the complete provider.
    resource_templates: tuple[str, ...] = ()
    requires_confirmation: bool = False
    # "before_side_effects" is the safe default for mutations: a queued call
    # may be skipped, but an in-flight provider call must finish and verify.
    # Read-only providers may opt into "anytime" cancellation.
    cancellation: Literal["never", "before_side_effects", "anytime"] = (
        "before_side_effects"
    )


class CapabilityManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    version: str
    contract_version: str
    execution_class: Literal["iridium_local", "household_lan_service"]
    tools: list[dict]
    skill_files: list[str]
    tool_policies: dict[str, ToolPolicy] = Field(default_factory=dict)


class CapabilityProvider(ABC):
    @abstractmethod
    def manifest(self) -> CapabilityManifest: ...

    @abstractmethod
    async def execute(self, action: PlannedAction) -> ToolResult: ...

    @abstractmethod
    async def health(self) -> dict: ...

    async def close(self) -> None:
        return None
