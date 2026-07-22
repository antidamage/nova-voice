from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Literal

from nova_voice.capabilities.base import CapabilityManifest, CapabilityProvider, ToolPolicy
from nova_voice.domain import CapabilityToolCall, PlannedAction, ToolResult
from nova_voice.providers.nova import verify_loop
from nova_voice.providers.nova.client import NovaDashboardClient, NovaDashboardError

logger = logging.getLogger(__name__)


def normalize_alias(value: str) -> str:
    value = value.casefold().replace("air conditioner", "air con").replace("aircon", "air con")
    return " ".join(re.findall(r"[a-z0-9]+", value))


_NON_ROOM_ZONE_IDS = {"everything", "climate", "heating", "network", "outside"}
_CLIMATE_OFF_STATES = {"off", "unknown", "unavailable", ""}
_AIRCON_RE = re.compile(r"\b(?:air\s*con(?:ditioner)?|gree)\b", re.IGNORECASE)
_PANEL_HEATER_RE = re.compile(r"\bpanel[\s_]*heater\b", re.IGNORECASE)


def climate_control_kind(entity: dict[str, Any]) -> Literal["aircon", "panel_heater"] | None:
    """Identify the two household climate controls from their stable IDs/names."""

    if str(entity.get("domain") or "") != "climate":
        return None
    text = f"{entity.get('entity_id', '')} {entity.get('name', '')}"
    if _PANEL_HEATER_RE.search(text):
        return "panel_heater"
    if _AIRCON_RE.search(text):
        return "aircon"
    return None


def logical_entity_room(entity: dict[str, Any]) -> str | None:
    """Map organisational climate entities back to the rooms they actually serve."""

    kind = climate_control_kind(entity)
    if kind == "aircon":
        return "lounge"
    if kind == "panel_heater":
        return "bedroom"
    return str(entity.get("area_id") or "") or None


def climate_is_on(entity: dict[str, Any]) -> bool:
    return str(entity.get("state") or "").casefold() not in _CLIMATE_OFF_STATES


def _describe_objective(target_name: str, action: str, args: dict[str, Any]) -> str:
    """Short, human-readable "done" statement for the LLM confirmation pass."""

    if action == "turn_on":
        return f"{target_name} is turned on"
    if action == "turn_off":
        return f"{target_name} is turned off"
    if action == "toggle":
        return f"{target_name} has changed power state"
    if action == "set_level":
        return f"{target_name} brightness is set to {args.get('value')}%"
    if action == "set_temperature":
        return f"{target_name} target temperature is set to {args.get('value')}°C"
    if action == "set_mode":
        return f"{target_name} mode is set to {args.get('value')}"
    if action == "set_color":
        return f"{target_name} color is updated"
    return f"{target_name} reflects the requested {action} change"


@dataclass(frozen=True)
class AliasTarget:
    kind: Literal["zone", "entity"]
    id: str
    name: str
    domain: str | None = None
    room: str | None = None


class AliasIndex:
    def __init__(self) -> None:
        self._aliases: dict[str, set[AliasTarget]] = {}

    def _add(self, alias: str | None, target: AliasTarget) -> None:
        if not alias:
            return
        normalized = normalize_alias(alias)
        if normalized:
            self._aliases.setdefault(normalized, set()).add(target)

    def rebuild(self, state: dict) -> None:
        self._aliases.clear()
        for zone in state.get("zones", []):
            if not isinstance(zone, dict) or not zone.get("id"):
                continue
            target = AliasTarget("zone", str(zone["id"]), str(zone.get("name") or zone["id"]))
            self._add(str(zone["id"]), target)
            self._add(str(zone.get("name") or ""), target)
            if zone["id"] == "everything":
                for alias in ("all lights", "every light", "whole house", "all the lights"):
                    self._add(alias, target)

        for entity in _device_tree_entities(state):
            if not isinstance(entity, dict) or not entity.get("entity_id"):
                continue
            entity_id = str(entity["entity_id"])
            name = str(entity.get("name") or entity_id)
            room = logical_entity_room(entity)
            target = AliasTarget(
                "entity",
                entity_id,
                name,
                str(entity.get("domain") or entity_id.split(".", 1)[0]),
                room,
            )
            self._add(entity_id, target)
            self._add(name, target)
            self._add(str(entity.get("attributes", {}).get("friendly_name") or ""), target)
            if room:
                self._add(f"{room} {name}", target)
            kind = climate_control_kind(entity)
            if kind == "aircon":
                self._add("lounge air conditioner", target)
                self._add("lounge aircon", target)
            elif kind == "panel_heater":
                self._add("bedroom heater", target)
                self._add("bedroom panel heater", target)

    def resolve(self, value: str, *, room: str | None = None) -> list[AliasTarget]:
        normalized = normalize_alias(value)
        exact = list(self._aliases.get(normalized, set()))
        if exact:
            return self._prefer_room(exact, room)

        scored: list[tuple[float, AliasTarget]] = []
        for alias, targets in self._aliases.items():
            score = SequenceMatcher(None, normalized, alias).ratio()
            if score >= 0.84:
                scored.extend((score, target) for target in targets)
        if not scored:
            return []
        best = max(score for score, _ in scored)
        candidates = list({target for score, target in scored if score >= best - 0.025})
        return self._prefer_room(candidates, room)

    @staticmethod
    def _prefer_room(candidates: list[AliasTarget], room: str | None) -> list[AliasTarget]:
        if not room:
            return candidates
        normalized_room = normalize_alias(room)
        in_room = [
            target for target in candidates if normalize_alias(target.room or "") == normalized_room
        ]
        return in_room or candidates


def _device_tree_entities(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten the dashboard's entity genealogy without assuming its shape.

    The state contract currently exposes entities both as a top-level catalogue
    and below zone nodes.  Future dashboard device groups may add further
    nesting, so aliasing and command admission walk the tree rather than rely
    on a light/climate-only projection.
    """

    entities: dict[str, dict[str, Any]] = {}

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            entity_id = value.get("entity_id")
            if isinstance(entity_id, str) and entity_id:
                entities.setdefault(entity_id, value)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(state.get("entities", []))
    visit(state.get("zones", []))
    return list(entities.values())


NOVA_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "nova.lighting_shortcut",
            "description": (
                "Apply Nova's configured broad lighting controls. Indoor on uses the "
                "adaptive daylight/sunset preset."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "scope": {"enum": ["indoors", "all", "outside"]},
                    "action": {"enum": ["on", "off"]},
                },
                "required": ["scope", "action"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "nova.query",
            "description": "Read normalized Nova household state or tasks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "scope": {
                        "enum": [
                            "home",
                            "room",
                            "target",
                            "tasks",
                            "health",
                            "energy",
                            "occupancy",
                            "device_health",
                            "maintenance",
                            "media",
                            "timers",
                            "schedules",
                            "diagnostics",
                        ]
                    },
                    "query": {"type": "string"},
                    "room": {"type": "string"},
                },
                "required": ["scope"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "nova.control",
            "description": "Control an allowlisted Nova household target using friendly names.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {"type": "string"},
                    "action": {
                        "enum": [
                            "turn_on",
                            "turn_off",
                            "set_level",
                            "set_temperature",
                            "set_color",
                            "set_timer",
                            "wake",
                            "sleep",
                        ]
                    },
                    "value": {},
                    "unit": {"type": "string"},
                    "durationMinutes": {"type": "integer", "minimum": 1, "maximum": 1440},
                    "room": {"type": "string"},
                },
                "required": ["target", "action"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "nova.task",
            "description": "List, add, update, complete, dismiss, or remove a Nova task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "enum": ["list", "add", "update", "complete", "dismiss", "remove"]
                    },
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "start": {"type": "string"},
                    "end": {"type": "string"},
                },
                "required": ["operation"],
                "additionalProperties": False,
            },
        },
    },
]


class NovaProvider(CapabilityProvider):
    def __init__(
        self,
        client: NovaDashboardClient,
        *,
        contract_version: str = "nova-provider-v1",
        alias_refresh_seconds: float = 30,
        climate_verify_timeout_seconds: float = 8,
        climate_verify_interval_seconds: float = 0.5,
        verification_loop_enabled: bool = True,
        verification_max_iterations: int = 20,
        verification_thinking_threshold_seconds: float = 2.5,
        verification_llm_verify_enabled: bool = True,
        verification_llm_verify_min_interval_seconds: float = 1.5,
        verification_llm_confirm_timeout_seconds: float = 3.0,
    ) -> None:
        self.client = client
        self.agent_name = "Nova"
        self.contract_version = contract_version
        self.alias_refresh_seconds = alias_refresh_seconds
        # The old constructor names are retained for compatibility, but the
        # verification loop now covers every stateful smart-home target rather
        # than climate devices alone.
        self.verification_loop_enabled = bool(verification_loop_enabled)
        self.verification_failure_seconds = max(0.0, float(climate_verify_timeout_seconds))
        self.verification_sleep_seconds = max(0.0, float(climate_verify_interval_seconds))
        self.verification_max_iterations = max(1, int(verification_max_iterations))
        self.verification_thinking_threshold_seconds = max(
            0.0, float(verification_thinking_threshold_seconds)
        )
        self.verification_llm_verify_enabled = bool(verification_llm_verify_enabled)
        self.verification_llm_verify_min_interval_seconds = max(
            0.0, float(verification_llm_verify_min_interval_seconds)
        )
        self.verification_llm_confirm_timeout_seconds = max(
            0.1, float(verification_llm_confirm_timeout_seconds)
        )
        self.aliases = AliasIndex()
        self._state: dict = {}
        self._last_refresh = 0.0
        self._refresh_lock = asyncio.Lock()

    def configure_verification_loop(
        self,
        *,
        enabled: bool,
        max_iterations: int,
        sleep_seconds: float,
        failure_seconds: float,
        thinking_threshold_seconds: float | None = None,
        llm_verify_enabled: bool | None = None,
        llm_verify_min_interval_seconds: float | None = None,
        llm_confirm_timeout_seconds: float | None = None,
    ) -> None:
        """Apply live bounded-loop controls collected from the dashboard."""

        self.verification_loop_enabled = bool(enabled)
        self.verification_max_iterations = max(1, int(max_iterations))
        self.verification_sleep_seconds = max(0.0, float(sleep_seconds))
        self.verification_failure_seconds = max(0.0, float(failure_seconds))
        if thinking_threshold_seconds is not None:
            self.verification_thinking_threshold_seconds = max(
                0.0, float(thinking_threshold_seconds)
            )
        if llm_verify_enabled is not None:
            self.verification_llm_verify_enabled = bool(llm_verify_enabled)
        if llm_verify_min_interval_seconds is not None:
            self.verification_llm_verify_min_interval_seconds = max(
                0.0, float(llm_verify_min_interval_seconds)
            )
        if llm_confirm_timeout_seconds is not None:
            self.verification_llm_confirm_timeout_seconds = max(
                0.1, float(llm_confirm_timeout_seconds)
            )

    def _verification_config(self) -> verify_loop.VerificationLoopConfig:
        return verify_loop.VerificationLoopConfig(
            enabled=self.verification_loop_enabled,
            max_iterations=self.verification_max_iterations,
            sleep_seconds=self.verification_sleep_seconds,
            failure_seconds=self.verification_failure_seconds,
            thinking_threshold_seconds=self.verification_thinking_threshold_seconds,
            llm_verify_enabled=self.verification_llm_verify_enabled,
            llm_verify_min_interval_seconds=self.verification_llm_verify_min_interval_seconds,
            llm_confirm_timeout_seconds=self.verification_llm_confirm_timeout_seconds,
        )

    async def _poll(self) -> dict[str, Any]:
        return await self.refresh(force=True)

    async def _run_verification(
        self,
        action: PlannedAction,
        state: dict[str, Any],
        *,
        label: str,
        objective: str,
        verify: Callable[[dict[str, Any]], tuple[bool, dict[str, Any] | None]],
    ) -> tuple[dict[str, Any], dict[str, Any] | None, bool]:
        """Run the bounded wiggum loop for a single verification target."""

        task = verify_loop.VerificationTask(
            action_id=action.id, label=label, objective=objective, verify=verify
        )
        result = await verify_loop.run(
            [task],
            state,
            poll=self._poll,
            config=self._verification_config(),
            context=verify_loop.current_turn_context(),
        )
        item = result.item(action.id)
        observed = item.observed if item is not None else None
        verified = item.confirmed if item is not None else False
        return result.final_state, observed, verified

    def manifest(self) -> CapabilityManifest:
        return CapabilityManifest(
            id="nova",
            version="0.1.0",
            contract_version=self.contract_version,
            execution_class="household_lan_service",
            tools=NOVA_TOOLS,
            skill_files=["nova-home-control/SKILL.md", "nova-tasks/SKILL.md"],
            tool_policies={
                "nova.query": ToolPolicy(
                    risk="low",
                    reversible=True,
                    idempotent=True,
                    parallel_safe=True,
                    cancellation="anytime",
                ),
                "nova.control": ToolPolicy(
                    risk="low", reversible=True, idempotent=True, parallel_safe=False
                ),
                "nova.lighting_shortcut": ToolPolicy(
                    risk="low", reversible=True, idempotent=True, parallel_safe=False
                ),
                "nova.task": ToolPolicy(
                    risk="low",
                    reversible=True,
                    idempotent=False,
                ),
            },
        )

    async def refresh(self, *, force: bool = False) -> dict:
        if (
            not force
            and self._state
            and time.monotonic() - self._last_refresh < self.alias_refresh_seconds
        ):
            return self._state
        async with self._refresh_lock:
            if (
                not force
                and self._state
                and time.monotonic() - self._last_refresh < self.alias_refresh_seconds
            ):
                return self._state
            state = await self.client.state()
            self.aliases.rebuild(state)
            self._state = state
            self._last_refresh = time.monotonic()
            return state

    async def prompt_context(self, room: str) -> dict[str, Any]:
        state = await self.refresh()
        normalized_room = normalize_alias(room)
        indoor_rooms = self._indoor_rooms(state)
        climate_controls = self._climate_controls(state)
        entities = []
        for entity in state.get("entities", []):
            if not isinstance(entity, dict):
                continue
            entity_room = logical_entity_room(entity) or ""
            if normalized_room and normalize_alias(entity_room) != normalized_room:
                continue
            if str(entity.get("domain") or "") == "climate":
                continue
            entities.append(
                {
                    "name": entity.get("name"),
                    "room": entity_room or None,
                    "domain": entity.get("domain"),
                    "state": entity.get("state"),
                }
            )
        entities.extend(
            control
            for control in climate_controls
            if normalize_alias(control["room"]) == normalized_room
        )
        indoor = self._indoor_temperatures(state)
        return {
            "room": room,
            # These are physical rooms inside the home. Organisational/synthetic
            # dashboard zones such as Climate, Network, Home, and Outside are not
            # mixed into this list.
            "indoorRooms": indoor_rooms,
            "nearbyTargets": entities[:30],
            # Non-climate devices keep their compact raw state. Climate has a
            # separate household-level contract below so HVAC implementation
            # modes never masquerade as the controls the assistant should offer.
            "deviceStates": self._device_states(state),
            "climateControls": climate_controls,
            # Indoor temperature for this satellite's room; null means no sensor
            # is configured for it (unknown), distinct from the outdoor weather.
            "indoorTemperatureC": indoor.get(normalize_alias(room)),
            "indoorTemperaturesByRoom": indoor,
            # Outdoor conditions only — never conflate with an indoor reading.
            "weather": self._weather_brief(state),
        }

    async def invalid_device_action_ids(self, actions: list[PlannedAction]) -> set[str]:
        """Return planned device mutations that have no current tree target.

        This is deliberately an admission check, before policy/execution.  The
        LLM may produce a schema-valid target string for television dialogue or
        another non-household phrase; that is not a dashboard command unless a
        real, unambiguous target exists in the freshly-read dashboard tree.
        AliasIndex is populated from every entity domain, so new supported
        device classes are admitted automatically once the dashboard exposes
        them.  Task and query tools are not device mutations and remain valid.
        """

        device_actions = [
            action
            for action in actions
            if action.call.provider == "nova"
            and action.call.tool in {"nova.control", "nova.lighting_shortcut"}
        ]
        if not device_actions:
            return set()

        await self.refresh(force=True)
        invalid: set[str] = set()
        for action in device_actions:
            if action.call.tool == "nova.lighting_shortcut":
                scope = str(action.call.arguments.get("scope") or "")
                if not self._lighting_scope_has_target(scope):
                    invalid.add(action.id)
                continue

            arguments = action.call.arguments
            requested_action = str(arguments.get("action") or "")
            # Managed-computer wake/sleep is a distinct dashboard operation,
            # not an HA entity mutation. Its endpoint owns computer lookup.
            if requested_action in {"wake", "sleep"}:
                continue
            target_name = str(arguments.get("target") or "")
            room = str(arguments.get("room") or "") or None
            if len(self.aliases.resolve(target_name, room=room)) != 1:
                invalid.add(action.id)
        return invalid

    def _lighting_scope_has_target(self, scope: str) -> bool:
        state = self._state
        if scope in {"indoors", "outside"}:
            zone_id = "everything" if scope == "indoors" else "outside"
            zones = [
                zone
                for zone in state.get("zones", [])
                if isinstance(zone, dict)
                and (
                    zone.get("id") == zone_id
                    or (scope == "outside" and str(zone.get("name") or "").casefold() == "outside")
                )
            ]
            entities = _device_tree_entities({"zones": zones})
        else:
            entities = _device_tree_entities(state)
        return any(
            (entity.get("domain") == "light" or entity.get("isIllumination"))
            and str(entity.get("state") or "").casefold() not in {"unavailable", "unknown"}
            for entity in entities
        )

    @staticmethod
    def _device_states(state: dict) -> list[dict[str, Any]]:
        """Compact home-wide table of controllable devices for command context.

        Bounded and limited to stateful, controllable domains so the interpreter
        can resolve and reason about devices in any room without the prompt
        ballooning with every diagnostic sensor in the house.
        """

        controllable = {"light", "switch", "cover", "fan", "media_player", "lock", "input_boolean"}
        rows: list[dict[str, Any]] = []
        for entity in state.get("entities", []):
            if not isinstance(entity, dict):
                continue
            domain = str(entity.get("domain") or "")
            if domain not in controllable:
                continue
            if normalize_alias(str(entity.get("area_id") or "")) in {"climate", "heating"}:
                # Climate-area switches are implementation details of the
                # aircon. The assistant gets one canonical power/target view.
                continue
            row: dict[str, Any] = {
                "name": entity.get("name"),
                "room": str(entity.get("area_id") or "") or None,
                "domain": domain,
                "state": entity.get("state"),
            }
            rows.append(row)
            if len(rows) >= 60:
                break
        return rows

    @staticmethod
    def _indoor_rooms(state: dict) -> list[str]:
        rooms: list[str] = []
        for zone in state.get("zones", []):
            if not isinstance(zone, dict):
                continue
            zone_id = normalize_alias(str(zone.get("id") or ""))
            zone_name = normalize_alias(str(zone.get("name") or ""))
            special = normalize_alias(str(zone.get("special") or ""))
            if zone_id in _NON_ROOM_ZONE_IDS or zone_name in _NON_ROOM_ZONE_IDS:
                continue
            if special in {"power", "tasks", "world"}:
                continue
            room = zone_id or zone_name
            if room and room not in rooms:
                rooms.append(room)
        return rooms

    @classmethod
    def _climate_controls(cls, state: dict) -> list[dict[str, Any]]:
        """Household-level climate controls, deliberately hiding raw HVAC modes."""

        indoor = cls._indoor_temperatures(state)
        preferences = state.get("preferences") if isinstance(state.get("preferences"), dict) else {}
        aircon_preferences = (
            preferences.get("aircon") if isinstance(preferences.get("aircon"), dict) else {}
        )
        controls: list[dict[str, Any]] = []
        for entity in state.get("entities", []):
            if not isinstance(entity, dict):
                continue
            kind = climate_control_kind(entity)
            if kind is None:
                continue
            room = "lounge" if kind == "aircon" else "bedroom"
            state_name = str(entity.get("state") or "").casefold()
            if state_name in {"unknown", "unavailable", ""}:
                power = "unavailable"
            else:
                # Manual aircon operation still means the appliance is on, but
                # an assistant turn_on command normalises it to dashboard Auto.
                power = (
                    "on"
                    if climate_is_on(entity)
                    or (kind == "aircon" and aircon_preferences.get("autoMode") is True)
                    else "off"
                )
            attributes = (
                entity.get("attributes")
                if isinstance(entity.get("attributes"), dict)
                else {}
            )
            target = (
                aircon_preferences.get("temperature")
                if kind == "aircon" and aircon_preferences.get("temperature") is not None
                else attributes.get("temperature")
            )
            controls.append(
                {
                    "name": entity.get("name")
                    or ("Air Conditioner" if kind == "aircon" else "Panel Heater"),
                    "room": room,
                    "power": power,
                    "targetTemperatureC": cls._numeric_value(target),
                    "roomTemperatureC": indoor.get(room),
                    "supportedActions": ["turn_on", "turn_off", "set_temperature"],
                }
            )
        return controls

    @staticmethod
    def _numeric_value(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if number == number else None

    @staticmethod
    def _indoor_temperatures(state: dict) -> dict[str, float | None]:
        """Authoritative indoor temperature per room, or None when unknown.

        Resolution mirrors the dashboard so the trusted source wins: a room's
        configured temperature sensor (its HA area temperature entity) first —
        which is why the lounge reads its Tuya sensor rather than the aircon's
        unreliable one — then a climate device's own measured temperature in the
        room (e.g. the bedroom panel heater), then a plain temperature sensor in
        the room. A room with none of these is unknown (None).
        """

        entities = [entity for entity in state.get("entities", []) if isinstance(entity, dict)]
        by_id = {str(entity.get("entity_id")): entity for entity in entities}

        def numeric(entity: dict | None) -> float | None:
            if entity is None:
                return None
            try:
                return float(entity.get("state"))
            except (TypeError, ValueError):
                return None

        result: dict[str, float | None] = {
            room: None for room in NovaProvider._indoor_rooms(state)
        }
        for zone in state.get("zones", []):
            if not isinstance(zone, dict):
                continue
            room = normalize_alias(str(zone.get("id") or zone.get("name") or ""))
            if not room or room not in result:
                continue
            temperature: float | None = None
            environment = zone.get("environment")
            if isinstance(environment, dict) and environment.get("temperatureEntityId"):
                temperature = numeric(by_id.get(str(environment["temperatureEntityId"])))
            if temperature is None:
                for entity in entities:
                    if str(entity.get("domain")) != "sensor":
                        continue
                    if normalize_alias(str(entity.get("area_id") or "")) != room:
                        continue
                    attributes = entity.get("attributes")
                    if not isinstance(attributes, dict):
                        continue
                    if str(attributes.get("device_class")) != "temperature":
                        continue
                    temperature = numeric(entity)
                    if temperature is not None:
                        break
            result[room] = temperature

        # The dashboard groups climate devices under its organisational Climate
        # area. Their logical rooms are explicit household semantics, not HA
        # area membership: the aircon serves the lounge and the panel heater's
        # published current_temperature measures the bedroom.
        for entity in entities:
            kind = climate_control_kind(entity)
            if kind is None:
                continue
            logical_room = "lounge" if kind == "aircon" else "bedroom"
            if logical_room not in result or result[logical_room] is not None:
                continue
            attributes = entity.get("attributes")
            if isinstance(attributes, dict):
                result[logical_room] = NovaProvider._numeric_value(
                    attributes.get("current_temperature")
                )
        return result

    @staticmethod
    def _weather_brief(state: dict) -> dict[str, Any] | None:
        """Compact current outdoor conditions for the LLM context packet."""

        weather = state.get("weather")
        if not isinstance(weather, dict):
            return None
        brief = {
            "condition": weather.get("condition"),
            "temperatureC": weather.get("temperature"),
            "todayHighC": weather.get("high"),
            "todayLowC": weather.get("low"),
            "humidityPct": weather.get("humidity"),
        }
        return brief if any(value is not None for value in brief.values()) else None

    async def execute(self, action: PlannedAction) -> ToolResult:
        try:
            if action.call.tool == "nova.query":
                return await self._query(action)
            if action.call.tool == "nova.control":
                return await self._control(action)
            if action.call.tool == "nova.lighting_shortcut":
                return await self._lighting_shortcut(action)
            if action.call.tool == "nova.task":
                return await self._task(action)
            return self._result(
                action, False, "blocked", f"Unknown {self.agent_name} semantic tool"
            )
        except NovaDashboardError:
            return self._result(
                action,
                False,
                "backend_error",
                f"{self.agent_name} dashboard request failed",
            )
        except (TypeError, ValueError, KeyError):
            return self._result(
                action,
                False,
                "invalid",
                f"{self.agent_name} action arguments were invalid",
            )

    @staticmethod
    def deterministic_lighting_shortcut(
        transcript: str,
        *,
        # Keep the helper's historical default for direct callers; the live
        # service passes the dashboard-selected Beemo value explicitly.
        wake_words: tuple[str, ...] | list[str] = ("bandit",),
    ) -> PlannedAction | None:
        """Map a deliberately small broad-light grammar without LLM guesswork."""

        tokens = normalize_alias(transcript).split()
        wakes = {normalize_alias(word) for word in wake_words if normalize_alias(word)}
        wake_position = next(
            (index for index, token in enumerate(tokens[:4]) if token in wakes),
            None,
        )
        if wake_position is not None:
            tokens = tokens[wake_position + 1 :]
        while tokens and tokens[0] in {"please", "can", "could", "would", "you"}:
            tokens.pop(0)
        if len(tokens) < 3 or tokens[0] not in {"turn", "switch"}:
            return None
        action = tokens[-1]
        if action not in {"on", "off"}:
            return None
        target = tokens[1:-1]
        if target and target[0] == "the":
            target = target[1:]

        indoor_targets = {
            ("all", "lights"),
            ("all", "of", "the", "lights"),
            ("every", "light"),
            ("everything",),
        }
        outside_targets = {
            ("outside", "light"),
            ("outside", "lights"),
            ("the", "outside", "light"),
            ("the", "outside", "lights"),
        }
        target_tuple = tuple(target)
        if target_tuple in outside_targets:
            scope = "outside"
        elif target_tuple in indoor_targets:
            scope = "indoors"
        elif (
            "outside" in target
            and any(token in target for token in {"all", "every", "everything"})
            and "including" in target
        ):
            scope = "all"
        else:
            return None
        return PlannedAction(
            id="nova-lighting-shortcut",
            order=0,
            call=CapabilityToolCall(
                provider="nova",
                tool="nova.lighting_shortcut",
                arguments={"scope": scope, "action": action},
            ),
        )

    async def _lighting_shortcut(self, action: PlannedAction) -> ToolResult:
        scope = str(action.call.arguments["scope"])
        requested_action = str(action.call.arguments["action"])
        returned_action = await self.client.lighting_shortcut(scope, requested_action)
        if returned_action != requested_action:
            return self._result(
                action,
                False,
                "unverified",
                f"{self.agent_name} returned an unexpected lighting shortcut result",
                requested={"scope": scope, "action": requested_action},
                observed={"response": returned_action},
            )
        state = await self.refresh(force=True)
        names = {"indoors": "Home lights", "all": "All lights", "outside": "Outside lights"}
        label = names[scope]
        state, observed, verified = await self._run_verification(
            action,
            state,
            label=label,
            objective=f"{label} are turned {requested_action}",
            verify=lambda candidate: self._verify_lighting_shortcut(
                candidate, scope, requested_action
            ),
        )
        return self._result(
            action,
            verified,
            "ok" if verified else "unverified",
            "Lighting shortcut verified"
            if verified
            else f"{self.agent_name} returned but the requested lighting state was not observed",
            target=label,
            requested={"scope": scope, "action": requested_action},
            observed=observed,
        )

    @staticmethod
    def _verify_lighting_shortcut(
        state: dict[str, Any], scope: str, action: str
    ) -> tuple[bool, dict[str, Any]]:
        if scope in {"indoors", "outside"}:
            zone_id = "everything" if scope == "indoors" else "outside"
            zone = next(
                (
                    item
                    for item in state.get("zones", [])
                    if item.get("id") == zone_id
                    or (scope == "outside" and str(item.get("name", "")).casefold() == "outside")
                ),
                None,
            )
            if zone is None:
                return False, {"zone": zone_id, "available": False}
            is_on = bool(zone.get("isOn"))
            lights = [
                item
                for item in zone.get("entities", [])
                if (item.get("domain") == "light" or item.get("isIllumination"))
                and str(item.get("state", "")).casefold() not in {"unavailable", "unknown"}
            ]
            on_count = sum(str(item.get("state", "")).casefold() == "on" for item in lights)
            # Dashboard zone.isOn means "any entity on": false verifies a full
            # off, but true cannot prove a requested all-on result. The live
            # state contract includes zone entities, so require every available
            # lighting entity for the on case and fail closed if they are absent.
            verified = (
                bool(lights) and on_count == len(lights)
                if action == "on"
                else not is_on
            )
            return verified, {
                "zone": zone_id,
                "lights": len(lights),
                "on": on_count,
                "isOn": is_on,
            }

        lights = [
            item
            for item in state.get("entities", [])
            if (item.get("domain") == "light" or item.get("isIllumination"))
            and str(item.get("state", "")).casefold() not in {"unavailable", "unknown"}
        ]
        on_count = sum(str(item.get("state", "")).casefold() == "on" for item in lights)
        verified = bool(lights) and (on_count == len(lights) if action == "on" else on_count == 0)
        return verified, {"lights": len(lights), "on": on_count}

    async def _query(self, action: PlannedAction) -> ToolResult:
        args = action.call.arguments
        scope = str(args.get("scope", ""))
        if scope == "tasks":
            tasks = await self.client.list_tasks()
            return self._result(action, True, "ok", "Tasks read", observed=tasks)
        if scope == "health":
            version, state = await asyncio.gather(self.client.version(), self.refresh(force=True))
            observed = {"version": version, "generatedAt": state.get("generatedAt")}
            return self._result(
                action, True, "ok", f"{self.agent_name} is healthy", observed=observed
            )

        state = await self.refresh()
        if scope in {
            "energy",
            "occupancy",
            "device_health",
            "maintenance",
            "media",
            "timers",
            "schedules",
            "diagnostics",
        }:
            observed = self._extended_household_observed(state, scope)
            return self._result(
                action,
                True,
                "ok",
                f"{scope.replace('_', ' ').capitalize()} read",
                observed=observed,
            )
        query = str(args.get("query") or args.get("room") or "")
        if not query or scope == "home":
            observed = {
                "generatedAt": state.get("generatedAt"),
                "totals": state.get("totals", {}),
                "zones": [
                    {"id": zone.get("id"), "name": zone.get("name"), "isOn": zone.get("isOn")}
                    for zone in state.get("zones", [])
                ],
            }
            return self._result(action, True, "ok", "State read", observed=observed)
        candidates = self.aliases.resolve(query, room=str(args.get("room") or "") or None)
        if len(candidates) != 1:
            return self._resolution_failure(action, candidates)
        target = candidates[0]
        observed = self._observed_target(state, target)
        if target.domain == "climate":
            observed = self._canonical_climate_observed(state, target)
        return self._result(
            action, True, "ok", "Target state read", target=target.name, observed=observed
        )

    @staticmethod
    def _extended_household_observed(state: dict[str, Any], scope: str) -> dict[str, Any]:
        """Return bounded, read-only extended household operational context.

        The dashboard state contract is intentionally the sole source here.
        New device domains become available only when present in that contract;
        Nova never fabricates a scene, schedule, or diagnostic target.
        """

        entities = [item for item in state.get("entities", []) if isinstance(item, dict)]
        unavailable = {"unavailable", "unknown"}

        def brief(item: dict[str, Any]) -> dict[str, Any]:
            return {
                "id": item.get("entity_id"),
                "name": item.get("name"),
                "room": logical_entity_room(item),
                "domain": item.get("domain"),
                "state": item.get("state"),
                "attributes": item.get("attributes", {}),
            }

        if scope == "energy":
            return {
                "generatedAt": state.get("generatedAt"),
                "entities": [
                    brief(item)
                    for item in entities
                    if str(item.get("attributes", {}).get("device_class", "")).casefold()
                    in {"energy", "power"}
                    or str(item.get("attributes", {}).get("unit_of_measurement", ""))
                    in {"W", "kW", "Wh", "kWh"}
                ],
            }
        if scope == "occupancy":
            return {
                "generatedAt": state.get("generatedAt"),
                "entities": [
                    brief(item)
                    for item in entities
                    if str(item.get("domain", "")) == "person"
                    or str(item.get("attributes", {}).get("device_class", "")).casefold()
                    in {"occupancy", "motion", "presence"}
                ],
            }
        if scope == "device_health":
            return {
                "generatedAt": state.get("generatedAt"),
                "unavailable": [
                    brief(item)
                    for item in entities
                    if str(item.get("state", "")).casefold() in unavailable
                ],
            }
        if scope == "maintenance":
            return {
                "generatedAt": state.get("generatedAt"),
                "attention": [
                    brief(item)
                    for item in entities
                    if str(item.get("attributes", {}).get("device_class", "")).casefold()
                    in {"battery", "problem", "update"}
                    or str(item.get("state", "")).casefold() in unavailable
                ],
            }
        if scope == "media":
            return {
                "generatedAt": state.get("generatedAt"),
                "players": [
                    brief(item) for item in entities if item.get("domain") == "media_player"
                ],
            }
        if scope in {"timers", "schedules"}:
            keys = ("timers", "schedules") if scope == "schedules" else ("timers",)
            return {
                "generatedAt": state.get("generatedAt"),
                **{key: state.get(key, []) for key in keys},
                "preferences": state.get("preferences", {}),
            }
        return {
            "generatedAt": state.get("generatedAt"),
            "totals": state.get("totals", {}),
            "unavailable": [
                brief(item)
                for item in entities
                if str(item.get("state", "")).casefold() in unavailable
            ],
            "entities": len(entities),
        }

    async def _control(self, action: PlannedAction) -> ToolResult:
        args = action.call.arguments
        target_name = str(args["target"])
        requested_action = str(args["action"])
        room = str(args.get("room") or "") or None
        if requested_action == "set_level":
            level = int(args["value"])
            if not 0 <= level <= 100:
                raise ValueError("brightness must be between 0 and 100")
        if requested_action == "set_temperature":
            temperature = float(args["value"])
            if not 5 <= temperature <= 35:
                raise ValueError("temperature is outside the configured 5-35C bound")
        if requested_action in {"wake", "sleep"}:
            target = target_name or "desktop"
            observed = await self.client.desktop_action(requested_action, {"target": target})
            return self._result(
                action,
                True,
                "ok",
                f"Desktop {requested_action} requested",
                target=target,
                requested={"operation": requested_action},
                observed=observed,
            )
        await self.refresh()
        candidates = self.aliases.resolve(target_name, room=room)
        if len(candidates) != 1:
            return self._resolution_failure(action, candidates)
        target = candidates[0]

        if requested_action == "set_timer":
            body = {
                "entityId": target.id,
                "durationMinutes": int(args["durationMinutes"]),
            }
            target_text = f"{target.id} {target.name} {target.domain or ''}".casefold()
            if "heater" in target_text:
                returned_state = await self.client.panel_heater_timer(body)
            elif any(value in target_text for value in ("climate", "aircon", "air con", "ac")):
                returned_state = await self.client.aircon_timer(body)
            else:
                return self._result(
                    action,
                    False,
                    "blocked",
                    "Timers are only supported for climate targets",
                    target=target.name,
                )
            return self._result(
                action,
                True,
                "ok",
                "Timer scheduled",
                target=target.name,
                requested=body,
                observed=returned_state,
            )
        if target.kind == "zone":
            body = self._zone_body(target, requested_action, args)
            returned_state = await self.client.zone_action(body)
        else:
            body = self._entity_body(target, requested_action, args)
            returned_state = await self.client.entity_action(body)

        # Some dashboard builds return only an operation acknowledgement. Keep
        # the pre-action snapshot as the initial (almost certainly stale) check;
        # the bounded verification loop below performs authoritative refreshes.
        if not isinstance(returned_state.get("entities"), list) and not isinstance(
            returned_state.get("zones"), list
        ):
            returned_state = self._state
        else:
            self._store_state(returned_state)
        returned_state, raw_observed, verified = await self._wait_for_verified_state(
            action,
            returned_state,
            target,
            requested_action,
            args,
        )
        observed = (
            self._canonical_climate_observed(returned_state, target)
            if target.domain == "climate"
            else raw_observed
        )
        return self._result(
            action,
            verified,
            "ok" if verified else "unverified",
            "Action verified"
            if verified
            else f"{self.agent_name} returned but the requested state was not observed",
            target=target.name,
            requested=body,
            observed=observed,
        )

    async def _task(self, action: PlannedAction) -> ToolResult:
        args = action.call.arguments
        operation = str(args.get("operation", ""))
        if operation in {"remove", "dismiss"}:
            return self._result(
                action,
                False,
                "blocked",
                "Destructive task edits require an explicit confirmation turn",
            )
        if operation == "list":
            observed = await self.client.list_tasks()
        else:
            observed = await self.client.tasks(
                operation, {key: value for key, value in args.items() if key != "operation"}
            )
        return self._result(action, True, "ok", "Task operation completed", observed=observed)

    def _store_state(self, state: dict[str, Any]) -> None:
        self._state = state
        self._last_refresh = time.monotonic()
        self.aliases.rebuild(state)

    async def _wait_for_verified_state(
        self,
        action: PlannedAction,
        state: dict[str, Any],
        target: AliasTarget,
        action_kind: str,
        args: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any] | None, bool]:
        """Wait for any stateful target to publish the requested state."""

        def verify(candidate: dict[str, Any]) -> tuple[bool, dict[str, Any] | None]:
            observed = self._observed_target(candidate, target)
            return (
                self._target_verified(candidate, target, action_kind, args, observed),
                observed,
            )

        return await self._run_verification(
            action,
            state,
            label=target.name,
            objective=_describe_objective(target.name, action_kind, args),
            verify=verify,
        )

    @classmethod
    def _target_verified(
        cls,
        state: dict[str, Any],
        target: AliasTarget,
        action: str,
        args: dict[str, Any],
        observed: dict[str, Any] | None,
    ) -> bool:
        if target.domain == "climate" and observed and climate_control_kind(observed) == "aircon":
            preferences = state.get("preferences")
            aircon = preferences.get("aircon") if isinstance(preferences, dict) else None
            auto_mode = aircon.get("autoMode") if isinstance(aircon, dict) else None
            raw_state = str(observed.get("state") or "").casefold()
            if action == "turn_on":
                return auto_mode is True and raw_state not in {"unknown", "unavailable", ""}
            if action == "turn_off":
                return auto_mode is not True and raw_state == "off"
        return cls._verify(action, args, observed)

    @staticmethod
    def _zone_body(target: AliasTarget, action: str, args: dict[str, Any]) -> dict:
        mapping = {
            "turn_on": "on",
            "turn_off": "off",
            "toggle": "toggle",
            "set_level": "brightness",
            "set_color": "color",
        }
        if action not in mapping:
            raise ValueError("zone operation is not supported by the current contract")
        body: dict[str, Any] = {"zoneId": target.id, "action": mapping[action]}
        if action == "set_level":
            body["brightnessPct"] = int(args["value"])
        if action == "set_color":
            body["rgb"] = args["value"]
        return body

    @staticmethod
    def _entity_body(target: AliasTarget, action: str, args: dict[str, Any]) -> dict:
        service_map = {
            "turn_on": "turn_on",
            "turn_off": "turn_off",
            "toggle": "toggle",
            "set_level": "turn_on",
            "set_temperature": "set_temperature",
            "set_mode": "set_hvac_mode",
            "set_color": "turn_on",
        }
        service = service_map[action]
        data: dict[str, Any] = {}
        if action == "set_level":
            data["brightness_pct"] = int(args["value"])
        elif action == "set_temperature":
            data["temperature"] = float(args["value"])
        elif action == "set_mode":
            data["hvac_mode"] = str(args["value"])
        elif action == "set_color":
            data["rgb_color"] = args["value"]
        body: dict[str, Any] = {
            "entityId": target.id,
            "domain": target.domain,
            "service": service,
            "data": data,
        }
        # Aircon on/off must go through the dashboard's wired routine so its
        # remembered auto state stays in sync: OFF clears autoMode (and any off
        # timer) exactly like the dashboard's Off button; ON re-engages the
        # dashboard's auto loop. Panel heaters and other climate devices keep
        # their own controls, so they are excluded.
        domain = (target.domain or "").casefold()
        target_text = f"{target.id} {target.name}".casefold()
        is_aircon = domain == "climate" and not _PANEL_HEATER_RE.search(target_text)
        if is_aircon:
            if action == "turn_off":
                body["remember"] = {"aircon": {"autoMode": False, "offTimerEndsAt": None}}
            elif action == "turn_on":
                body["remember"] = {"aircon": {"autoMode": True}}
            elif action == "set_temperature":
                body["remember"] = {"aircon": {"temperature": float(args["value"])}}
        return body

    @staticmethod
    def _observed_target(state: dict, target: AliasTarget) -> dict[str, Any] | None:
        collection = state.get("zones" if target.kind == "zone" else "entities", [])
        key = "id" if target.kind == "zone" else "entity_id"
        return next(
            (candidate for candidate in collection if candidate.get(key) == target.id), None
        )

    @classmethod
    def _canonical_climate_observed(
        cls, state: dict[str, Any], target: AliasTarget
    ) -> dict[str, Any] | None:
        raw = cls._observed_target(state, target)
        if raw is None:
            return None
        kind = climate_control_kind(raw)
        if kind is None:
            return None
        room = "lounge" if kind == "aircon" else "bedroom"
        return next(
            (control for control in cls._climate_controls(state) if control["room"] == room),
            None,
        )

    @staticmethod
    def _verify(action: str, args: dict[str, Any], observed: dict[str, Any] | None) -> bool:
        if not observed:
            return False
        state = str(observed.get("state", "")).casefold()
        is_climate = (
            str(observed.get("domain") or "") == "climate"
            or str(observed.get("entity_id") or "").startswith("climate.")
        )
        if action == "turn_on":
            if "isOn" in observed:
                return bool(observed.get("isOn"))
            return state not in _CLIMATE_OFF_STATES if is_climate else state == "on"
        if action == "turn_off":
            return not bool(observed.get("isOn")) if "isOn" in observed else state == "off"
        if action == "toggle":
            return state in {"on", "off"}
        if action == "set_level":
            expected = int(args["value"])
            if "brightnessPct" in observed:
                actual = int(observed.get("brightnessPct") or 0)
            else:
                actual = round(
                    int(observed.get("attributes", {}).get("brightness") or 0) * 100 / 255
                )
            return abs(actual - expected) <= 2
        if action == "set_temperature":
            actual = observed.get("attributes", {}).get("temperature")
            return actual is not None and abs(float(actual) - float(args["value"])) <= 0.5
        if action == "set_mode":
            return state == str(args["value"]).casefold()
        if action == "set_color":
            return observed.get("attributes", {}).get("rgb_color") is not None
        return False

    def _resolution_failure(
        self, action: PlannedAction, candidates: list[AliasTarget]
    ) -> ToolResult:
        if not candidates:
            return self._result(
                action, False, "not_found", f"No matching {self.agent_name} target"
            )
        return self._result(
            action,
            False,
            "ambiguous",
            f"{self.agent_name} target is ambiguous",
            candidates=sorted({candidate.name for candidate in candidates}),
        )

    @staticmethod
    def _result(
        action: PlannedAction,
        ok: bool,
        code: str,
        message: str,
        **kwargs: Any,
    ) -> ToolResult:
        return ToolResult(action_id=action.id, ok=ok, code=code, message=message, **kwargs)

    async def health(self) -> dict:
        try:
            # Do not force a full dashboard-state rebuild here: /health is polled
            # every few seconds by the status strip, and forcing `/api/state`
            # (the whole HA snapshot) each time is what made the round trip
            # 70-140 ms. `version()` still proves the dashboard is reachable, and
            # a non-forced refresh reports the last snapshot's generatedAt from
            # cache (fetching only when it has aged past alias_refresh_seconds).
            version, state = await asyncio.gather(self.client.version(), self.refresh())
            return {
                "ok": True,
                "contractVersion": self.contract_version,
                "dashboardBuildId": version.get("buildId"),
                "stateGeneratedAt": state.get("generatedAt"),
            }
        except NovaDashboardError:
            return {"ok": False, "contractVersion": self.contract_version}

    async def close(self) -> None:
        await self.client.close()
