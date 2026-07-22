from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from nova_voice.capabilities.base import CapabilityManifest, CapabilityProvider, ToolPolicy
from nova_voice.commitments import CommitmentManager
from nova_voice.domain import PlannedAction, ToolResult
from nova_voice.durable.models import CommitmentRecord


def _schema(properties: dict, required: list[str]) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


_ID = {"type": "string", "minLength": 1, "maxLength": 160}
COMMITMENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "commitments.create",
            "description": "Create a durable reminder or wait-until commitment.",
            "parameters": _schema(
                {
                    "summary": {"type": "string", "minLength": 1, "maxLength": 1000},
                    "due": {"type": "string"},
                    "timezone": {"type": "string"},
                    "deadline": {"type": "string"},
                    "recurrence": {"type": "string", "pattern": "^FREQ="},
                    "eventKey": {"type": "string"},
                    "channels": {
                        "type": "array",
                        "items": {"enum": ["voice", "dashboard", "notification"]},
                        "uniqueItems": True,
                    },
                },
                ["summary"],
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "commitments.list",
            "description": "List durable commitments on any authenticated device.",
            "parameters": _schema({}, []),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "commitments.snooze",
            "description": "Continue a due commitment on this device at a later time.",
            "parameters": _schema(
                {
                    "commitmentId": _ID,
                    "until": {"type": "string"},
                    "timezone": {"type": "string"},
                    "device": {"type": "string"},
                },
                ["commitmentId", "until", "device"],
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "commitments.complete",
            "description": "Acknowledge and complete or advance a recurring commitment.",
            "parameters": _schema(
                {"commitmentId": _ID, "device": {"type": "string"}}, ["commitmentId", "device"]
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "commitments.cancel",
            "description": "Cancel a durable commitment.",
            "parameters": _schema(
                {"commitmentId": _ID, "device": {"type": "string"}}, ["commitmentId", "device"]
            ),
        },
    },
]


class CommitmentsProvider(CapabilityProvider):
    def __init__(
        self, manager: CommitmentManager, *, contract_version: str = "commitments-provider-v1"
    ) -> None:
        self.manager = manager
        self.contract_version = contract_version

    def manifest(self) -> CapabilityManifest:
        read = ToolPolicy(
            risk="low", reversible=True, idempotent=True, parallel_safe=True, cancellation="anytime"
        )
        write = ToolPolicy(
            risk="low",
            reversible=True,
            idempotent=True,
            resource_templates=("commitments:data",),
            cancellation="before_side_effects",
        )
        return CapabilityManifest(
            id="commitments",
            version="0.1.0",
            contract_version=self.contract_version,
            execution_class="iridium_local",
            tools=COMMITMENT_TOOLS,
            skill_files=[],
            tool_policies={
                "commitments.list": read,
                **{
                    tool["function"]["name"]: write
                    for tool in COMMITMENT_TOOLS
                    if tool["function"]["name"] != "commitments.list"
                },
            },
        )

    @staticmethod
    def _time(value, timezone_name: str | None = None):
        if value is None:
            return None
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            if not timezone_name:
                raise ValueError("local commitment time requires an IANA timezone")
            parsed = parsed.replace(tzinfo=ZoneInfo(timezone_name))
        return parsed

    async def execute(self, action: PlannedAction) -> ToolResult:
        tool, args = action.call.tool, action.call.arguments
        try:
            if tool == "commitments.list":
                records = await self.manager.list()
                return ToolResult(
                    action_id=action.id,
                    ok=True,
                    code="ok",
                    observed={
                        "commitments": [record.model_dump(mode="json") for record in records]
                    },
                    message=f"Listed {len(records)} commitments",
                )
            if tool == "commitments.create":
                record = await self.manager.create(
                    CommitmentRecord(
                        id=f"commitment-nova-{action.id}",
                        owner_id="voice-owner",
                        summary=str(args["summary"]),
                        due_at=self._time(args.get("due"), args.get("timezone")),
                        deadline=self._time(args.get("deadline"), args.get("timezone")),
                        recurrence=args.get("recurrence"),
                        wait_event_key=args.get("eventKey"),
                        channels=tuple(args.get("channels") or ("dashboard",)),
                    ),
                    actor_id="voice",
                )
            elif tool == "commitments.snooze":
                record = await self.manager.snooze(
                    str(args["commitmentId"]),
                    until=self._time(args["until"], args.get("timezone")),
                    device=str(args["device"]),
                )
            elif tool == "commitments.complete":
                record = await self.manager.acknowledge(
                    str(args["commitmentId"]), device=str(args["device"])
                )
            else:
                record = await self.manager.cancel(
                    str(args["commitmentId"]), device=str(args["device"])
                )
            return ToolResult(
                action_id=action.id,
                ok=True,
                code="ok",
                target=record.id,
                observed={"commitment": record.model_dump(mode="json")},
                message="Commitment updated and verified",
            )
        except KeyError:
            return ToolResult(action_id=action.id, ok=False, code="not_found")
        except ValueError as error:
            return ToolResult(action_id=action.id, ok=False, code="invalid", message=str(error))

    async def health(self) -> dict:
        records = await self.manager.list()
        return {"ok": True, "commitments": len(records), "contractVersion": self.contract_version}
