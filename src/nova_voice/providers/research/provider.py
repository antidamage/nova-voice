from __future__ import annotations

from nova_voice.capabilities.base import CapabilityManifest, CapabilityProvider, ToolPolicy
from nova_voice.domain import PlannedAction, ToolResult
from nova_voice.research import ResearchManager


def _schema(properties: dict, required: list[str]) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


RESEARCH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "research.start",
            "description": (
                "Start a longer web research job in the background and return immediately."
            ),
            "parameters": _schema(
                {
                    "query": {"type": "string", "minLength": 1, "maxLength": 1000},
                    "ownerId": {"type": "string", "minLength": 1, "maxLength": 160},
                },
                ["query", "ownerId"],
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "research.status",
            "description": "Check a background research job and retrieve its cited result.",
            "parameters": _schema(
                {"researchId": {"type": "string", "minLength": 1, "maxLength": 200}},
                ["researchId"],
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "research.cancel",
            "description": "Cancel queued or running background research.",
            "parameters": _schema(
                {
                    "researchId": {"type": "string", "minLength": 1, "maxLength": 200},
                    "actorId": {"type": "string", "minLength": 1, "maxLength": 160},
                },
                ["researchId", "actorId"],
            ),
        },
    },
]


class ResearchProvider(CapabilityProvider):
    def __init__(self, manager: ResearchManager) -> None:
        self.manager = manager

    def manifest(self) -> CapabilityManifest:
        policies = {
            name: ToolPolicy(
                risk="low",
                reversible=name != "research.status",
                idempotent=name == "research.status",
                parallel_safe=True,
                cancellation="anytime",
            )
            for name in ("research.start", "research.status", "research.cancel")
        }
        return CapabilityManifest(
            id="research",
            version="0.1.0",
            contract_version="research-provider-v1",
            execution_class="iridium_local",
            tools=RESEARCH_TOOLS,
            skill_files=[],
            tool_policies=policies,
        )

    async def execute(self, action: PlannedAction) -> ToolResult:
        arguments = action.call.arguments
        try:
            if action.call.tool == "research.start":
                record = await self.manager.create(
                    owner_id=str(arguments.get("ownerId") or ""),
                    query=str(arguments.get("query") or ""),
                )
                message = "I started that research. I'll tell you when it's ready."
            elif action.call.tool == "research.status":
                record = await self.manager.get(str(arguments.get("researchId") or ""))
                message = record.spoken_summary or f"That research is {record.status.value}."
            elif action.call.tool == "research.cancel":
                record = await self.manager.cancel(
                    str(arguments.get("researchId") or ""),
                    actor_id=str(arguments.get("actorId") or ""),
                )
                message = "Research cancelled."
            else:
                return ToolResult(
                    action_id=action.id, ok=False, code="blocked", message="Unknown research tool"
                )
        except KeyError:
            return ToolResult(
                action_id=action.id, ok=False, code="not_found", message="Research job not found"
            )
        except ValueError as error:
            return ToolResult(action_id=action.id, ok=False, code="invalid", message=str(error))
        return ToolResult(
            action_id=action.id,
            ok=True,
            code="ok",
            target=record.id,
            requested=arguments,
            observed={"research": record.model_dump(mode="json")},
            message=message,
        )

    async def health(self) -> dict:
        return {
            "ok": True,
            "jobs": len(await self.manager.list()),
            "contractVersion": "research-provider-v1",
        }
