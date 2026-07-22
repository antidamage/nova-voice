from __future__ import annotations

from nova_voice.capabilities.base import CapabilityManifest, CapabilityProvider, ToolPolicy
from nova_voice.dialogue import MultiPartyDialogueManager
from nova_voice.domain import PlannedAction, ToolResult


def _schema(properties: dict, required: list[str]) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


DIALOGUE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "dialogue.relay",
            "description": "Create a durable ask/tell relay for one person or the household.",
            "parameters": _schema(
                {
                    "senderId": {"type": "string"},
                    "recipientScope": {"type": "string", "enum": ["person", "household"]},
                    "recipientId": {"type": "string"},
                    "recipientName": {"type": "string"},
                    "speechAct": {"type": "string", "enum": ["tell", "ask"]},
                    "content": {"type": "string", "minLength": 1, "maxLength": 2000},
                    "conversationId": {"type": "string"},
                },
                ["senderId", "recipientScope", "speechAct", "content"],
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dialogue.pending",
            "description": (
                "List relay messages addressed to the current recognized person or household."
            ),
            "parameters": _schema(
                {"personId": {"type": "string"}, "displayName": {"type": "string"}}, ["personId"]
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dialogue.acknowledge",
            "description": "Acknowledge delivery of an addressed relay message.",
            "parameters": _schema(
                {
                    "messageId": {"type": "string"},
                    "personId": {"type": "string"},
                    "displayName": {"type": "string"},
                },
                ["messageId", "personId"],
            ),
        },
    },
]


class DialogueProvider(CapabilityProvider):
    def __init__(self, manager: MultiPartyDialogueManager) -> None:
        self.manager = manager

    def manifest(self) -> CapabilityManifest:
        return CapabilityManifest(
            id="dialogue",
            version="0.1.0",
            contract_version="dialogue-provider-v1",
            execution_class="iridium_local",
            tools=DIALOGUE_TOOLS,
            skill_files=[],
            tool_policies={
                tool["function"]["name"]: ToolPolicy(
                    risk="low",
                    reversible=True,
                    idempotent=tool["function"]["name"] != "dialogue.relay",
                    parallel_safe=True,
                    cancellation="anytime",
                )
                for tool in DIALOGUE_TOOLS
            },
        )

    async def execute(self, action: PlannedAction) -> ToolResult:
        args = action.call.arguments
        try:
            if action.call.tool == "dialogue.relay":
                record = await self.manager.create(
                    sender_id=str(args.get("senderId") or ""),
                    recipient_scope=str(args.get("recipientScope") or ""),
                    recipient_id=str(args.get("recipientId") or "") or None,
                    recipient_name=str(args.get("recipientName") or "") or None,
                    speech_act=str(args.get("speechAct") or ""),
                    content=str(args.get("content") or ""),
                    conversation_id=str(args.get("conversationId") or "") or None,
                )
                observed = {"message": record.model_dump(mode="json")}
                message = "I'll pass that on."
            elif action.call.tool == "dialogue.pending":
                records = await self.manager.pending_for(
                    person_id=str(args.get("personId") or ""),
                    display_name=str(args.get("displayName") or "") or None,
                )
                observed = {"messages": [record.model_dump(mode="json") for record in records]}
                message = f"There are {len(records)} pending messages."
            elif action.call.tool == "dialogue.acknowledge":
                record = await self.manager.acknowledge(
                    str(args.get("messageId") or ""),
                    person_id=str(args.get("personId") or ""),
                    display_name=str(args.get("displayName") or "") or None,
                )
                observed = {"message": record.model_dump(mode="json")}
                message = "Message acknowledged."
            else:
                return ToolResult(
                    action_id=action.id, ok=False, code="blocked", message="Unknown dialogue tool"
                )
        except KeyError:
            return ToolResult(
                action_id=action.id,
                ok=False,
                code="not_found",
                message="Dialogue message not found",
            )
        except (ValueError, PermissionError) as error:
            return ToolResult(action_id=action.id, ok=False, code="invalid", message=str(error))
        return ToolResult(
            action_id=action.id,
            ok=True,
            code="ok",
            target="dialogue",
            requested=args,
            observed=observed,
            message=message,
        )

    async def health(self) -> dict:
        return {
            "ok": True,
            "messages": len(await self.manager.list()),
            "contractVersion": "dialogue-provider-v1",
        }
