from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Literal

from nova_voice.capabilities.base import CapabilityManifest, CapabilityProvider, ToolPolicy
from nova_voice.domain import CapabilityToolCall, PlannedAction, ToolResult
from nova_voice.providers.nova.client import NovaDashboardClient, NovaDashboardError


def normalize_alias(value: str) -> str:
    value = value.casefold().replace("air conditioner", "air con").replace("aircon", "air con")
    return " ".join(re.findall(r"[a-z0-9]+", value))


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

        for entity in state.get("entities", []):
            if not isinstance(entity, dict) or not entity.get("entity_id"):
                continue
            entity_id = str(entity["entity_id"])
            name = str(entity.get("name") or entity_id)
            room = str(entity.get("area_id") or "") or None
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
                    "scope": {"enum": ["home", "room", "target", "tasks", "health"]},
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
                            "set_mode",
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
    ) -> None:
        self.client = client
        self.agent_name = "Nova"
        self.contract_version = contract_version
        self.alias_refresh_seconds = alias_refresh_seconds
        self.aliases = AliasIndex()
        self._state: dict = {}
        self._last_refresh = 0.0
        self._refresh_lock = asyncio.Lock()

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
                    risk="low", reversible=True, idempotent=True, parallel_safe=True
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
        zones = [str(zone.get("name") or zone.get("id")) for zone in state.get("zones", [])]
        entities = []
        for entity in state.get("entities", []):
            if not isinstance(entity, dict):
                continue
            entity_room = str(entity.get("area_id") or "")
            if normalized_room and normalize_alias(entity_room) != normalized_room:
                continue
            entities.append(
                {
                    "name": entity.get("name"),
                    "room": entity_room or None,
                    "domain": entity.get("domain"),
                    "state": entity.get("state"),
                }
            )
        return {
            "room": room,
            "zones": zones,
            "nearbyTargets": entities[:30],
            "weather": self._weather_brief(state),
        }

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
        verified, observed = self._verify_lighting_shortcut(state, scope, requested_action)
        names = {"indoors": "Home lights", "all": "All lights", "outside": "Outside lights"}
        return self._result(
            action,
            verified,
            "ok" if verified else "unverified",
            "Lighting shortcut verified"
            if verified
            else f"{self.agent_name} returned but the requested lighting state was not observed",
            target=names[scope],
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
        return self._result(
            action, True, "ok", "Target state read", target=target.name, observed=observed
        )

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
            domain = (target.domain or target.name).casefold()
            if "heater" in domain:
                returned_state = await self.client.panel_heater_timer(body)
            elif any(value in domain for value in ("climate", "aircon", "air con", "ac")):
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

        # Some dashboard builds return only an operation acknowledgement.  Do
        # not treat that as verified state: refresh the normalized snapshot
        # before reporting success.
        if not isinstance(returned_state.get("entities"), list) and not isinstance(
            returned_state.get("zones"), list
        ):
            returned_state = await self.refresh(force=True)
        else:
            self._state = returned_state
            self._last_refresh = time.monotonic()
            self.aliases.rebuild(returned_state)
        observed = self._observed_target(returned_state, target)
        verified = self._verify(requested_action, args, observed)
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
        return {"entityId": target.id, "domain": target.domain, "service": service, "data": data}

    @staticmethod
    def _observed_target(state: dict, target: AliasTarget) -> dict[str, Any] | None:
        collection = state.get("zones" if target.kind == "zone" else "entities", [])
        key = "id" if target.kind == "zone" else "entity_id"
        return next(
            (candidate for candidate in collection if candidate.get(key) == target.id), None
        )

    @staticmethod
    def _verify(action: str, args: dict[str, Any], observed: dict[str, Any] | None) -> bool:
        if not observed:
            return False
        state = str(observed.get("state", "")).casefold()
        if action == "turn_on":
            return bool(observed.get("isOn")) if "isOn" in observed else state == "on"
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
            version, state = await asyncio.gather(self.client.version(), self.refresh(force=True))
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
