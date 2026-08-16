"""Calendar, reminders, location and Health, executed on the owner's phone.

An ordinary `CapabilityProvider` from Nova's point of view: the planner sees
tools, `ToolPolicy` decides what needs confirming, and the registry validates
arguments — none of which know or care that the work happens on a device
across the room. What is different is only what happens when the device is not
there, and the answer is a typed *unavailable* result rather than an exception,
because a phone in a pocket is a normal state of the world and not a fault.

Two rules shape the tool list:

**Every read is bounded by construction.** No tool here can ask for "my
calendar" — a window and an item cap are required arguments, not optional
niceties. An unbounded query against a decade of personal records is not a
feature anyone asked for, and it is the shape that turns one careless plan into
a large personal payload crossing the network.

**Every mutation is separate from every read**, and every one of them is
`requires_confirmation`. That is what makes the approval gate reachable: the
provider refuses, the proposal becomes a durable approval, and the write
happens once afterwards under a stored idempotency key.
"""

from __future__ import annotations

from typing import Any, Protocol

from nova_voice.capabilities.base import CapabilityManifest, CapabilityProvider, ToolPolicy
from nova_voice.domain import PlannedAction, ToolResult


class CompanionSessions(Protocol):
    """The part of the session manager this provider needs."""

    async def personal_call(
        self,
        tool: str,
        arguments: dict,
        *,
        max_items: int = 50,
        deadline_seconds: float = 15.0,
    ) -> Any: ...

    def snapshot(self, *, now: float | None = None) -> Any: ...


def _schema(properties: dict, required: list[str]) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": _schema(properties, required),
        },
    }


_TIME = {"type": "string", "format": "date-time"}
_ID = {"type": "string", "minLength": 1, "maxLength": 200}
_TITLE = {"type": "string", "minLength": 1, "maxLength": 500}
# The cap is required rather than defaulted, so a plan has to state how much
# personal data it intends to move.
_MAX_ITEMS = {"type": "integer", "minimum": 1, "maximum": 200}
_LIST_NAME = {"type": "string", "minLength": 1, "maxLength": 200}
# The item's last-modified time when the proposal was made. An approval
# describes an item as it was; if it has moved on since, the thing the owner
# agreed to is not the thing that would be changed, and applying it anyway
# would silently overwrite whatever happened in between.
_UNCHANGED_SINCE = {"type": "string", "format": "date-time"}

# Health writes are an explicit allowlist with units and plausible ranges, not
# a generic sample writer. Anything outside this is refused before an approval
# is even offered, so the owner is never asked to agree to a write that could
# not have been performed.
HEALTH_WRITE_ALLOWLIST: dict[str, dict] = {
    "bodyMass": {"unit": "kg", "minimum": 2.0, "maximum": 400.0},
    "dietaryWater": {"unit": "mL", "minimum": 0.0, "maximum": 5000.0},
    "mindfulSession": {"unit": "min", "minimum": 1.0, "maximum": 600.0},
}

# Reads are similarly enumerated. "Whatever HealthKit has" is not a query.
HEALTH_READ_ALLOWLIST: tuple[str, ...] = (
    "stepCount",
    "activeEnergyBurned",
    "sleepAnalysis",
    "restingHeartRate",
    "bodyMass",
)


COMPANION_PERSONAL_TOOLS: list[dict] = [
    _tool(
        "companion.calendar.list",
        "List calendar events in an explicit window on the owner's phone.",
        {
            "start": _TIME,
            "end": _TIME,
            "maxItems": _MAX_ITEMS,
            "calendars": {
                "type": "array",
                "items": _LIST_NAME,
                "maxItems": 20,
            },
        },
        ["start", "end", "maxItems"],
    ),
    _tool(
        "companion.calendar.create",
        "Propose creating a calendar event. Requires the owner's approval.",
        {
            "title": _TITLE,
            "start": _TIME,
            "end": _TIME,
            "calendar": _LIST_NAME,
            "allDay": {"type": "boolean"},
            "notes": {"type": "string", "maxLength": 2000},
        },
        ["title", "start", "end"],
    ),
    _tool(
        "companion.calendar.update",
        "Propose changing an existing calendar event. Requires approval.",
        {
            "eventId": _ID,
            "title": _TITLE,
            "start": _TIME,
            "end": _TIME,
            "notes": {"type": "string", "maxLength": 2000},
            "ifUnchangedSince": _UNCHANGED_SINCE,
        },
        ["eventId"],
    ),
    _tool(
        "companion.calendar.delete",
        "Propose deleting a calendar event. Requires approval.",
        {"eventId": _ID, "ifUnchangedSince": _UNCHANGED_SINCE},
        ["eventId"],
    ),
    _tool(
        "companion.reminders.list",
        "List reminders on the owner's phone, bounded by count and completion.",
        {
            "maxItems": _MAX_ITEMS,
            "list": _LIST_NAME,
            "includeCompleted": {"type": "boolean"},
            "dueBefore": _TIME,
        },
        ["maxItems"],
    ),
    _tool(
        "companion.reminders.create",
        "Propose creating a reminder. Requires the owner's approval.",
        {
            "title": _TITLE,
            "list": _LIST_NAME,
            "due": _TIME,
            "notes": {"type": "string", "maxLength": 2000},
        },
        ["title"],
    ),
    _tool(
        "companion.reminders.update",
        "Propose changing a reminder. Requires approval.",
        {
            "reminderId": _ID,
            "title": _TITLE,
            "due": _TIME,
            "ifUnchangedSince": _UNCHANGED_SINCE,
        },
        ["reminderId"],
    ),
    _tool(
        "companion.reminders.complete",
        "Propose marking a reminder complete. Requires approval.",
        {"reminderId": _ID},
        ["reminderId"],
    ),
    _tool(
        "companion.reminders.uncomplete",
        "Propose marking a completed reminder as not done. Requires approval.",
        {"reminderId": _ID},
        ["reminderId"],
    ),
    _tool(
        "companion.reminders.delete",
        "Propose deleting a reminder. Requires approval.",
        {"reminderId": _ID},
        ["reminderId"],
    ),
    _tool(
        "companion.location.current",
        "The phone's current or last-known location, with its age and accuracy.",
        {
            # Coarse by default is not enough: the plan has to say which it
            # needs, so a request for street-level precision is visible in the
            # plan rather than implied by a default.
            "accuracy": {"type": "string", "enum": ["coarse", "fine"]},
            "maxAgeSeconds": {"type": "integer", "minimum": 0, "maximum": 86400},
        },
        ["accuracy", "maxAgeSeconds"],
    ),
    _tool(
        "companion.health.summary",
        "Summarise explicitly named Health metrics over an explicit window.",
        {
            "metrics": {
                "type": "array",
                "items": {"type": "string", "enum": list(HEALTH_READ_ALLOWLIST)},
                "minItems": 1,
                "maxItems": len(HEALTH_READ_ALLOWLIST),
            },
            "start": _TIME,
            "end": _TIME,
        },
        ["metrics", "start", "end"],
    ),
    _tool(
        "companion.health.write",
        "Propose writing one allowlisted Health sample. Requires approval.",
        {
            "metric": {"type": "string", "enum": list(HEALTH_WRITE_ALLOWLIST)},
            "value": {"type": "number"},
            "unit": {"type": "string", "maxLength": 16},
            "recordedAt": _TIME,
        },
        ["metric", "value", "unit", "recordedAt"],
    ),
]


def _read_policy() -> ToolPolicy:
    return ToolPolicy(
        risk="low",
        reversible=True,
        idempotent=True,
        # Reads of different things do not contend, and a planner that wants
        # three of them should not pay for them serially.
        parallel_safe=True,
        cancellation="anytime",
    )


def _write_policy(resource: str, *, reversible: bool = True) -> ToolPolicy:
    return ToolPolicy(
        risk="confirmation",
        reversible=reversible,
        idempotent=False,
        parallel_safe=False,
        resource_templates=(resource,),
        requires_confirmation=True,
        cancellation="before_side_effects",
    )


_POLICIES: dict[str, ToolPolicy] = {
    "companion.calendar.list": _read_policy(),
    "companion.calendar.create": _write_policy("companion:calendar"),
    "companion.calendar.update": _write_policy("companion:calendar:{eventId}"),
    "companion.calendar.delete": _write_policy(
        "companion:calendar:{eventId}", reversible=False
    ),
    "companion.reminders.list": _read_policy(),
    "companion.reminders.create": _write_policy("companion:reminder"),
    "companion.reminders.update": _write_policy("companion:reminder:{reminderId}"),
    "companion.reminders.complete": _write_policy("companion:reminder:{reminderId}"),
    "companion.reminders.uncomplete": _write_policy("companion:reminder:{reminderId}"),
    "companion.reminders.delete": _write_policy(
        "companion:reminder:{reminderId}", reversible=False
    ),
    "companion.location.current": _read_policy(),
    "companion.health.summary": _read_policy(),
    # Not reversible: a Health sample that has been written and then deleted is
    # not the same as one never written, and the owner should be told so.
    "companion.health.write": _write_policy("companion:health:{metric}", reversible=False),
}

_READ_TOOLS = frozenset(name for name, policy in _POLICIES.items() if policy.risk == "low")

# Codes a device may use that Nova's own result type already understands.
_KNOWN_RESULT_CODES = frozenset(
    {"not_found", "blocked", "invalid", "timeout", "unavailable", "backend_error"}
)


# iCloud tools this provider stands in for while the device is answering.
#
# The pairs are semantic, not textual: `companion.calendar.list` and
# `icloud.calendar.list` answer the same question about the same household
# calendar, reached two different ways. Offering both would put a genuinely
# ambiguous choice in front of the planner, and the failure it produces —
# a reminder created in the account the owner does not check — is invisible
# until someone misses something.
#
# Deliberately *not* including the iCloud mutations. Those go through
# CalDAV and work whether or not a phone is present, and an approval that
# has already been agreed to should execute by whichever route still works.
SUPERSEDED_ICLOUD_TOOLS: frozenset[str] = frozenset(
    {"icloud.calendar.list", "icloud.reminders.list"}
)


class CompanionPersonalProvider(CapabilityProvider):
    def __init__(
        self,
        sessions: CompanionSessions,
        *,
        contract_version: str = "1",
        deadline_seconds: float = 15.0,
    ) -> None:
        self._sessions = sessions
        self.contract_version = contract_version
        self._deadline = deadline_seconds

    @property
    def supersedes(self) -> frozenset[str]:
        """Read by the registry when it builds a planner catalogue."""

        return SUPERSEDED_ICLOUD_TOOLS

    def manifest(self) -> CapabilityManifest:
        return CapabilityManifest(
            id="companion",
            version="0.1.0",
            contract_version=self.contract_version,
            # The work happens on a device on the household network, not in
            # this process. Classifying it honestly is what lets the planner
            # and the durable runner reason about its availability.
            execution_class="household_lan_service",
            tools=COMPANION_PERSONAL_TOOLS,
            skill_files=[],
            tool_policies=dict(_POLICIES),
        )

    async def execute(self, action: PlannedAction) -> ToolResult:
        tool = action.call.tool
        policy = _POLICIES.get(tool)
        if policy is None:
            return self._failed(action, "invalid", "unknown companion personal tool")

        # A mutation must never reach the phone from here. The approval gate
        # owns writes: it stores the exact proposal, gets a signed decision,
        # and executes once afterwards. Letting this path perform one would be
        # a second, unapproved way for personal records to change.
        if tool not in _READ_TOOLS:
            return self._failed(
                action,
                "blocked",
                "this change needs the owner's approval before it can run",
            )

        if tool == "companion.health.summary":
            problem = self._invalid_metrics(action.call.arguments)
            if problem is not None:
                return self._failed(action, "invalid", problem)

        snapshot = self._sessions.snapshot()
        if not getattr(snapshot, "connected", False):
            # Typed unavailability, not an exception. A phone in a pocket is a
            # normal state of the world, and a plan that meets it should be
            # able to say so rather than fail.
            return self._failed(action, "unavailable", "the companion device is not connected")

        result = await self._sessions.personal_call(
            tool,
            dict(action.call.arguments),
            max_items=int(action.call.arguments.get("maxItems", 50) or 50),
            deadline_seconds=self._deadline,
        )
        if result is None:
            return self._failed(
                action, "unavailable", "the companion device did not answer in time"
            )
        if not getattr(result, "ok", False):
            # The device's own code is not passed straight through: `ToolResult`
            # has a closed set, and a device inventing a new one would fail
            # validation here and turn a handled refusal into a crash. Unknown
            # codes become `backend_error` with the original named in the
            # message, so nothing is lost from the operator's point of view.
            device_code = str(getattr(result, "code", "") or "")
            message = getattr(result, "message", "the companion could not complete this")
            if device_code in _KNOWN_RESULT_CODES:
                return self._failed(action, device_code, message)
            return self._failed(
                action,
                "backend_error",
                f"{message} (companion reported {device_code or 'no code'})",
            )
        return ToolResult(
            action_id=action.id,
            ok=True,
            code="ok",
            requested=action.call.arguments,
            message=getattr(result, "message", "done"),
            observed={
                "items": list(getattr(result, "items", []) or []),
                "truncated": bool(getattr(result, "truncated", False)),
            },
        )

    async def health(self) -> dict:
        """Reflect the live session, so 'degraded' means something specific."""

        snapshot = self._sessions.snapshot()
        connected = bool(getattr(snapshot, "connected", False))
        tools = sorted(getattr(snapshot, "personal_tools", frozenset()) or frozenset())
        return {
            "ok": connected and bool(tools),
            "connected": connected,
            "locality": getattr(snapshot, "locality", "other"),
            # What the device says it can currently do, which changes as
            # permissions are granted and revoked in iOS Settings.
            "availableTools": tools,
            "detail": (
                "connected"
                if connected and tools
                else "no companion session"
                if not connected
                else "the device grants no personal permissions"
            ),
        }

    @staticmethod
    def _invalid_metrics(arguments: dict) -> str | None:
        metrics = arguments.get("metrics") or []
        unknown = [
            metric for metric in metrics if metric not in HEALTH_READ_ALLOWLIST
        ]
        if unknown:
            return f"unsupported Health metric: {', '.join(sorted(unknown))}"
        return None

    @staticmethod
    def _failed(action: PlannedAction, code: str, message: str) -> ToolResult:
        return ToolResult(
            action_id=action.id,
            ok=False,
            code=code,
            requested=action.call.arguments,
            message=message,
        )


def validate_health_write(metric: str, value: float, unit: str) -> str | None:
    """Check a proposed Health write before an approval is offered.

    Refusing here rather than after approval matters: the owner should never be
    asked to agree to a write that could not have been performed anyway.
    """

    spec = HEALTH_WRITE_ALLOWLIST.get(metric)
    if spec is None:
        return f"{metric} is not an allowlisted Health sample type"
    if unit != spec["unit"]:
        return f"{metric} must be written in {spec['unit']}, not {unit}"
    if not spec["minimum"] <= value <= spec["maximum"]:
        return (
            f"{value} {unit} is outside the plausible range for {metric} "
            f"({spec['minimum']}-{spec['maximum']} {spec['unit']})"
        )
    return None
