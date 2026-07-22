from __future__ import annotations

from uuid import uuid4

from nova_voice.briefings import BriefingManager
from nova_voice.capabilities.base import CapabilityManifest, CapabilityProvider, ToolPolicy
from nova_voice.domain import PlannedAction, ToolResult
from nova_voice.durable.models import BriefingScheduleRecord, EventSubscriptionRecord


def _schema(properties: dict, required: list[str]) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


BRIEFING_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "briefings.schedule",
            "description": "Schedule a restart-safe morning or evening briefing.",
            "parameters": _schema(
                {
                    "ownerId": {"type": "string"},
                    "period": {"type": "string", "enum": ["morning", "evening"]},
                    "localTime": {"type": "string", "pattern": r"^([01]\d|2[0-3]):[0-5]\d$"},
                    "timezone": {"type": "string"},
                    "channel": {"type": "string", "enum": ["voice", "dashboard", "notification"]},
                },
                ["ownerId", "period", "localTime", "timezone"],
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "briefings.list",
            "description": "List briefing schedules and recent generated briefings.",
            "parameters": _schema({}, []),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "subscriptions.create",
            "description": (
                "Tell the household owner when a matching durable household event occurs."
            ),
            "parameters": _schema(
                {
                    "ownerId": {"type": "string"},
                    "summary": {"type": "string"},
                    "eventKind": {"type": "string"},
                    "match": {"type": "object"},
                    "oneShot": {"type": "boolean"},
                    "channel": {"type": "string", "enum": ["voice", "dashboard", "notification"]},
                },
                ["ownerId", "summary", "eventKind"],
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "subscriptions.list",
            "description": "List event subscriptions and their trigger status.",
            "parameters": _schema({}, []),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "subscriptions.cancel",
            "description": "Cancel an event subscription.",
            "parameters": _schema(
                {"subscriptionId": {"type": "string"}, "actorId": {"type": "string"}},
                ["subscriptionId", "actorId"],
            ),
        },
    },
]


class BriefingsProvider(CapabilityProvider):
    def __init__(self, manager: BriefingManager) -> None:
        self.manager = manager

    def manifest(self) -> CapabilityManifest:
        return CapabilityManifest(
            id="briefings",
            version="0.1.0",
            contract_version="briefings-provider-v1",
            execution_class="iridium_local",
            tools=BRIEFING_TOOLS,
            skill_files=[],
            tool_policies={
                tool["function"]["name"]: ToolPolicy(
                    risk="low",
                    reversible=True,
                    idempotent=tool["function"]["name"].endswith(".list"),
                    parallel_safe=True,
                    cancellation="anytime",
                )
                for tool in BRIEFING_TOOLS
            },
        )

    async def execute(self, action: PlannedAction) -> ToolResult:
        args = action.call.arguments
        tool = action.call.tool
        try:
            if tool == "briefings.schedule":
                record = await self.manager.create_schedule(
                    BriefingScheduleRecord(
                        id=f"briefing-schedule:{uuid4()}",
                        owner_id=str(args.get("ownerId") or ""),
                        period=str(args.get("period") or ""),
                        local_time=str(args.get("localTime") or ""),
                        timezone=str(args.get("timezone") or ""),
                        channels=(str(args.get("channel") or "dashboard"),),
                    )
                )
                observed = {"schedule": record.model_dump(mode="json")}
                message = f"The {record.period} briefing is scheduled."
            elif tool == "briefings.list":
                observed = {
                    "schedules": [
                        item.model_dump(mode="json") for item in await self.manager.schedules()
                    ],
                    "briefings": [
                        item.model_dump(mode="json") for item in await self.manager.briefings()
                    ],
                }
                message = "Briefings retrieved."
            elif tool == "subscriptions.create":
                record = await self.manager.create_subscription(
                    EventSubscriptionRecord(
                        id=f"subscription:{uuid4()}",
                        owner_id=str(args.get("ownerId") or ""),
                        summary=str(args.get("summary") or ""),
                        event_kind=str(args.get("eventKind") or ""),
                        match=dict(args.get("match") or {}),
                        one_shot=bool(args.get("oneShot", True)),
                        channels=(str(args.get("channel") or "dashboard"),),
                    )
                )
                observed = {"subscription": record.model_dump(mode="json")}
                message = "I'll tell you when that happens."
            elif tool == "subscriptions.list":
                observed = {
                    "subscriptions": [
                        item.model_dump(mode="json") for item in await self.manager.subscriptions()
                    ]
                }
                message = "Subscriptions retrieved."
            elif tool == "subscriptions.cancel":
                record = await self.manager.cancel_subscription(
                    str(args.get("subscriptionId") or ""), actor_id=str(args.get("actorId") or "")
                )
                observed = {"subscription": record.model_dump(mode="json")}
                message = "Subscription cancelled."
            else:
                return ToolResult(
                    action_id=action.id, ok=False, code="blocked", message="Unknown briefing tool"
                )
        except KeyError:
            return ToolResult(
                action_id=action.id, ok=False, code="not_found", message="Briefing item not found"
            )
        except ValueError as error:
            return ToolResult(action_id=action.id, ok=False, code="invalid", message=str(error))
        return ToolResult(
            action_id=action.id,
            ok=True,
            code="ok",
            target="briefings",
            requested=args,
            observed=observed,
            message=message,
        )

    async def health(self) -> dict:
        return {
            "ok": True,
            "schedules": len(await self.manager.schedules()),
            "subscriptions": len(await self.manager.subscriptions()),
            "contractVersion": "briefings-provider-v1",
        }
