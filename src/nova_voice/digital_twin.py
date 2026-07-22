from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from nova_voice.capabilities.base import CapabilityManifest, CapabilityProvider, ToolPolicy
from nova_voice.domain import PlannedAction, ToolResult

StateSupplier = Callable[[], Awaitable[dict[str, Any]]]


class TwinChange(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    entity_id: str
    before: str | None
    after: str
    cause: str


class TwinScenario(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    baseline_revision: str
    changes: tuple[TwinChange, ...]
    explanations: tuple[str, ...]
    baseline_watts: float = Field(ge=0)
    projected_watts: float = Field(ge=0)
    duration_hours: float = Field(ge=0)
    projected_energy_kwh: float = Field(ge=0)
    warnings: tuple[str, ...] = ()
    side_effects: int = 0


class HouseholdDigitalTwin:
    """Read-only, deterministic projection over one live household snapshot."""

    @staticmethod
    def _revision(state: dict[str, Any]) -> str:
        encoded = json.dumps(state, sort_keys=True, separators=(",", ":"), default=str)
        return f"sha256:{hashlib.sha256(encoded.encode()).hexdigest()}"

    @staticmethod
    def _entities(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
        raw = state.get("entities", [])
        if isinstance(raw, dict):
            return {
                str(key): value if isinstance(value, dict) else {"state": value}
                for key, value in raw.items()
            }
        entities: dict[str, dict[str, Any]] = {}
        if isinstance(raw, list):
            for item in raw:
                if not isinstance(item, dict):
                    continue
                entity_id = item.get("entity_id") or item.get("entityId") or item.get("id")
                if entity_id:
                    entities[str(entity_id)] = item
        return entities

    @staticmethod
    def _watts(entity: dict[str, Any], state: str | None = None) -> float:
        attributes = entity.get("attributes") if isinstance(entity.get("attributes"), dict) else {}
        value = (
            attributes.get("power_w")
            or attributes.get("powerWatts")
            or attributes.get("power")
            or entity.get("power_w")
            or 0
        )
        try:
            watts = max(0.0, float(value))
        except (TypeError, ValueError):
            return 0.0
        selected_state = str(state if state is not None else entity.get("state", "")).casefold()
        return watts if selected_state not in {"off", "closed", "idle", "unavailable"} else 0.0

    def simulate(
        self,
        state: dict[str, Any],
        actions: list[dict[str, Any]],
        *,
        duration_hours: float = 1.0,
    ) -> TwinScenario:
        entities = self._entities(state)
        baseline_watts = sum(self._watts(entity) for entity in entities.values())
        projected = {key: dict(value) for key, value in entities.items()}
        changes: list[TwinChange] = []
        explanations: list[str] = []
        warnings: list[str] = []
        for action in actions:
            entity_id = str(action.get("entityId") or action.get("entity_id") or "")
            after = str(action.get("state") or "")
            if not entity_id or not after:
                warnings.append("ignored an action without entity and state")
                continue
            entity = projected.get(entity_id)
            if entity is None:
                warnings.append(f"unknown entity: {entity_id}")
                continue
            before = str(entity.get("state")) if entity.get("state") is not None else None
            entity["state"] = after
            cause = str(action.get("cause") or "proposed scenario action")
            changes.append(TwinChange(entity_id=entity_id, before=before, after=after, cause=cause))
            explanations.append(
                f"Because {cause}, {entity_id} changes from {before or 'unknown'} to {after}."
            )
        projected_watts = sum(self._watts(entity) for entity in projected.values())
        duration = max(0.0, min(float(duration_hours), 168.0))
        return TwinScenario(
            baseline_revision=self._revision(state),
            changes=tuple(changes),
            explanations=tuple(explanations),
            baseline_watts=baseline_watts,
            projected_watts=projected_watts,
            duration_hours=duration,
            projected_energy_kwh=projected_watts * duration / 1000,
            warnings=tuple(warnings),
        )

    def rehearse_automation(
        self, state: dict[str, Any], automation: dict[str, Any]
    ) -> dict[str, Any]:
        trigger = automation.get("trigger") if isinstance(automation.get("trigger"), dict) else {}
        actions = automation.get("actions") or automation.get("proposedActions") or []
        trigger_kind = str(trigger.get("kind") or "")
        known_event_kinds = set(state.get("eventKinds", ()))
        return {
            "baselineRevision": self._revision(state),
            "triggerMatched": bool(trigger_kind and trigger_kind in known_event_kinds),
            "triggerKind": trigger_kind,
            "proposedActionCount": len(actions) if isinstance(actions, list) else 0,
            "safe": isinstance(actions, list) and all(isinstance(item, dict) for item in actions),
            "sideEffects": 0,
        }


class DigitalTwinProvider(CapabilityProvider):
    def __init__(self, state_supplier: StateSupplier) -> None:
        self.state_supplier = state_supplier
        self.twin = HouseholdDigitalTwin()

    def manifest(self) -> CapabilityManifest:
        tools = [
            _tool("twin.simulate", "Simulate household state changes without applying them."),
            _tool("twin.explain", "Explain causes and effects in a proposed household scenario."),
            _tool("twin.energy", "Estimate scenario power and energy without applying changes."),
            _tool("twin.rehearse_automation", "Rehearse an automation with no side effects."),
        ]
        return CapabilityManifest(
            id="household_digital_twin",
            version="1.0.0",
            contract_version="digital-twin-v1",
            execution_class="iridium_local",
            tools=tools,
            skill_files=[],
            tool_policies={
                item["function"]["name"]: ToolPolicy(
                    idempotent=True,
                    parallel_safe=True,
                    cancellation="anytime",
                    resource_templates=("digital-twin",),
                )
                for item in tools
            },
        )

    async def execute(self, action: PlannedAction) -> ToolResult:
        state = await self.state_supplier()
        arguments = action.call.arguments
        if action.call.tool == "twin.rehearse_automation":
            observed = self.twin.rehearse_automation(
                state,
                (
                    arguments.get("automation")
                    if isinstance(arguments.get("automation"), dict)
                    else {}
                ),
            )
        elif action.call.tool in {"twin.simulate", "twin.explain", "twin.energy"}:
            raw_actions = arguments.get("actions", [])
            actions = (
                [item for item in raw_actions if isinstance(item, dict)]
                if isinstance(raw_actions, list)
                else []
            )
            observed = self.twin.simulate(
                state,
                actions,
                duration_hours=float(arguments.get("durationHours", 1)),
            ).model_dump(mode="json")
        else:
            return ToolResult(
                action_id=action.id,
                ok=False,
                code="invalid",
                message="unknown digital twin tool",
            )
        return ToolResult(
            action_id=action.id,
            ok=True,
            code="ok",
            observed=observed,
            message="side-effect-free household scenario evaluated",
        )

    async def health(self) -> dict:
        try:
            state = await self.state_supplier()
            return {"ok": True, "baselineRevision": self.twin._revision(state)}
        except Exception as error:
            return {"ok": False, "error": type(error).__name__}


def _tool(name: str, description: str) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "entityId": {"type": "string"},
                    "state": {"type": "string"},
                    "cause": {"type": "string"},
                },
                "required": ["entityId", "state"],
                "additionalProperties": False,
            },
        },
        "durationHours": {"type": "number", "minimum": 0, "maximum": 168},
    }
    required: list[str] = ["actions"]
    if name == "twin.rehearse_automation":
        properties = {"automation": {"type": "object"}}
        required = ["automation"]
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }
