from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from nova_voice.capabilities.base import CapabilityManifest, CapabilityProvider, ToolPolicy
from nova_voice.domain import PlannedAction, ToolResult


@dataclass(order=True)
class _Scheduled:
    due_seconds: float
    sequence: int
    kind: str = field(compare=False)
    payload: dict[str, Any] = field(compare=False)


class SimulatedClock:
    def __init__(self, start: datetime | None = None) -> None:
        selected = start or datetime(2026, 1, 1, tzinfo=UTC)
        if selected.tzinfo is None:
            raise ValueError("simulated clock start must be timezone-aware")
        self._start = selected
        self._seconds = 0.0

    def __call__(self) -> float:
        return self._seconds

    def now(self) -> datetime:
        return self._start + timedelta(seconds=self._seconds)

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("simulated time cannot move backwards")
        self._seconds += seconds


class HouseholdSimulator:
    """Repeatable home state, failure, occupancy, and speaker event model."""

    def __init__(self, *, clock: SimulatedClock | None = None) -> None:
        self.clock = clock or SimulatedClock()
        self.entities: dict[str, dict[str, Any]] = {}
        self.occupants: dict[str, str] = {}
        self.speaker_events: list[dict[str, Any]] = []
        self.event_log: list[dict[str, Any]] = []
        self._scheduled: list[_Scheduled] = []
        self._sequence = 0
        self._failures: dict[str, list[str]] = {}

    def add_entity(self, entity_id: str, state: str, **attributes: Any) -> None:
        self.entities[entity_id] = {"state": state, "attributes": dict(attributes)}

    def fail_next(self, tool: str, *, count: int = 1, message: str = "simulated failure") -> None:
        if count < 1:
            raise ValueError("failure count must be positive")
        self._failures.setdefault(tool, []).extend([message] * count)

    def consume_failure(self, tool: str) -> str | None:
        pending = self._failures.get(tool)
        if not pending:
            return None
        message = pending.pop(0)
        if not pending:
            self._failures.pop(tool, None)
        return message

    def set_entity(self, entity_id: str, state: str, *, delay_seconds: float = 0) -> None:
        if entity_id not in self.entities:
            raise KeyError(entity_id)
        self.schedule(
            delay_seconds,
            "entity_state",
            {"entityId": entity_id, "state": state},
        )

    def set_occupancy(self, person: str, room: str, *, delay_seconds: float = 0) -> None:
        self.schedule(
            delay_seconds,
            "occupancy",
            {"person": person, "room": room},
        )

    def speak(
        self,
        speaker: str,
        room: str,
        transcript: str,
        *,
        delay_seconds: float = 0,
    ) -> None:
        self.schedule(
            delay_seconds,
            "speaker",
            {"speaker": speaker, "room": room, "transcript": transcript},
        )

    def schedule(self, delay_seconds: float, kind: str, payload: dict[str, Any]) -> None:
        if delay_seconds < 0:
            raise ValueError("event delay cannot be negative")
        self._sequence += 1
        heapq.heappush(
            self._scheduled,
            _Scheduled(self.clock() + delay_seconds, self._sequence, kind, dict(payload)),
        )

    def advance(self, seconds: float) -> tuple[dict[str, Any], ...]:
        target = self.clock() + seconds
        if seconds < 0:
            raise ValueError("simulated time cannot move backwards")
        applied: list[dict[str, Any]] = []
        while self._scheduled and self._scheduled[0].due_seconds <= target:
            event = heapq.heappop(self._scheduled)
            self.clock.advance(event.due_seconds - self.clock())
            applied.append(self._apply(event))
        self.clock.advance(target - self.clock())
        return tuple(applied)

    def snapshot(self) -> dict[str, Any]:
        return {
            "at": self.clock.now().isoformat(),
            "entities": {
                entity_id: {
                    "state": value["state"],
                    "attributes": dict(value["attributes"]),
                }
                for entity_id, value in sorted(self.entities.items())
            },
            "occupancy": dict(sorted(self.occupants.items())),
            "pendingEvents": len(self._scheduled),
        }

    def _apply(self, event: _Scheduled) -> dict[str, Any]:
        record = {
            "sequence": event.sequence,
            "at": self.clock.now().isoformat(),
            "kind": event.kind,
            **event.payload,
        }
        if event.kind == "entity_state":
            self.entities[event.payload["entityId"]]["state"] = event.payload["state"]
        elif event.kind == "occupancy":
            self.occupants[event.payload["person"]] = event.payload["room"]
        elif event.kind == "speaker":
            self.speaker_events.append(record)
        else:
            raise ValueError(f"unknown simulated event kind: {event.kind}")
        self.event_log.append(record)
        return record


class SimulatedHouseholdProvider(CapabilityProvider):
    def __init__(self, simulator: HouseholdSimulator) -> None:
        self.simulator = simulator

    def manifest(self) -> CapabilityManifest:
        return CapabilityManifest(
            id="sim_household",
            version="1.0.0",
            contract_version="sim-household-v1",
            execution_class="iridium_local",
            tools=[
                _tool("sim.household.query", {}, "Read the deterministic household state."),
                _tool(
                    "sim.household.set",
                    {
                        "entityId": {"type": "string"},
                        "state": {"type": "string"},
                        "delaySeconds": {"type": "number", "minimum": 0},
                    },
                    "Schedule deterministic entity state convergence.",
                    required=("entityId", "state"),
                ),
                _tool(
                    "sim.household.occupancy",
                    {
                        "person": {"type": "string"},
                        "room": {"type": "string"},
                        "delaySeconds": {"type": "number", "minimum": 0},
                    },
                    "Schedule deterministic occupancy state.",
                    required=("person", "room"),
                ),
            ],
            skill_files=[],
            tool_policies={
                "sim.household.query": ToolPolicy(
                    idempotent=True,
                    parallel_safe=True,
                    cancellation="anytime",
                ),
                "sim.household.set": ToolPolicy(),
                "sim.household.occupancy": ToolPolicy(),
            },
        )

    async def execute(self, action: PlannedAction) -> ToolResult:
        failure = self.simulator.consume_failure(action.call.tool)
        if failure is not None:
            return ToolResult(
                action_id=action.id,
                ok=False,
                code="backend_error",
                message=failure,
            )
        arguments = action.call.arguments
        if action.call.tool == "sim.household.query":
            return ToolResult(
                action_id=action.id,
                ok=True,
                code="ok",
                observed=self.simulator.snapshot(),
                message="simulated household state",
            )
        if action.call.tool == "sim.household.set":
            entity_id = str(arguments["entityId"])
            try:
                self.simulator.set_entity(
                    entity_id,
                    str(arguments["state"]),
                    delay_seconds=float(arguments.get("delaySeconds", 0)),
                )
            except KeyError:
                return ToolResult(
                    action_id=action.id,
                    ok=False,
                    code="not_found",
                    target=entity_id,
                    message="simulated entity not found",
                )
            return ToolResult(
                action_id=action.id,
                ok=True,
                code="ok",
                target=entity_id,
                requested={"state": str(arguments["state"])},
                observed=self.simulator.snapshot(),
                message="simulated convergence scheduled",
            )
        if action.call.tool == "sim.household.occupancy":
            self.simulator.set_occupancy(
                str(arguments["person"]),
                str(arguments["room"]),
                delay_seconds=float(arguments.get("delaySeconds", 0)),
            )
            return ToolResult(
                action_id=action.id,
                ok=True,
                code="ok",
                requested={
                    "person": str(arguments["person"]),
                    "room": str(arguments["room"]),
                },
                observed=self.simulator.snapshot(),
                message="simulated occupancy scheduled",
            )
        return ToolResult(
            action_id=action.id,
            ok=False,
            code="invalid",
            message="unknown simulator tool",
        )

    async def health(self) -> dict:
        return {"ok": True, "clock": self.simulator.clock.now().isoformat()}


def _tool(
    name: str,
    properties: dict[str, Any],
    description: str,
    *,
    required: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(required),
                "additionalProperties": False,
            },
        },
    }
