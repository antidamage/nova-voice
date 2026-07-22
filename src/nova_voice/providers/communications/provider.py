from __future__ import annotations

import re

from nova_voice.capabilities.base import CapabilityManifest, CapabilityProvider, ToolPolicy
from nova_voice.communications import CommunicationManager
from nova_voice.domain import PlannedAction, ToolResult


def _schema(properties: dict, required: list[str]) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


_ID = {"type": "string", "minLength": 1, "maxLength": 500}
COMMUNICATION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "communications.draft",
            "description": "Draft, but never send, an email, message, or invitation.",
            "parameters": _schema(
                {
                    "channel": {"enum": ["email", "message", "invitation"]},
                    "recipient": _ID,
                    "subject": {"type": "string", "maxLength": 500},
                    "body": {"type": "string", "minLength": 1, "maxLength": 20_000},
                    "invitation": {"type": "object"},
                },
                ["channel", "recipient", "body"],
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "communications.preview",
            "description": "Preview a draft without exposing its dashboard approval secret.",
            "parameters": _schema({"draftId": _ID}, ["draftId"]),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "communications.cancel",
            "description": "Cancel a pending or delivered communication when supported.",
            "parameters": _schema({"draftId": _ID}, ["draftId"]),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "communications.send",
            "description": "Send only through the separately authenticated approval workflow.",
            "parameters": _schema({"draftId": _ID}, ["draftId"]),
        },
    },
]


class CommunicationsProvider(CapabilityProvider):
    def __init__(
        self,
        manager: CommunicationManager,
        *,
        contract_version: str = "communications-provider-v1",
    ) -> None:
        self.manager = manager
        self.contract_version = contract_version

    def manifest(self) -> CapabilityManifest:
        return CapabilityManifest(
            id="communications",
            version="0.1.0",
            contract_version=self.contract_version,
            execution_class="iridium_local",
            tools=COMMUNICATION_TOOLS,
            skill_files=[],
            tool_policies={
                "communications.draft": ToolPolicy(
                    risk="low",
                    reversible=True,
                    idempotent=True,
                    resource_templates=("communications:drafts",),
                    cancellation="before_side_effects",
                ),
                "communications.preview": ToolPolicy(
                    risk="low",
                    reversible=True,
                    idempotent=True,
                    parallel_safe=True,
                    cancellation="anytime",
                ),
                "communications.cancel": ToolPolicy(
                    risk="low",
                    reversible=False,
                    idempotent=True,
                    resource_templates=("communications:drafts",),
                    cancellation="before_side_effects",
                ),
                # The immediate voice executor always blocks confirmation tools.
                # Authenticated API approval calls the manager directly instead.
                "communications.send": ToolPolicy(
                    risk="confirmation",
                    reversible=False,
                    idempotent=True,
                    resource_templates=("communications:drafts",),
                    requires_confirmation=True,
                    cancellation="never",
                ),
            },
        )

    async def execute(self, action: PlannedAction) -> ToolResult:
        tool = action.call.tool
        args = action.call.arguments
        if tool == "communications.draft":
            draft_id = f"draft-nova-{re.sub(r'[^a-zA-Z0-9_.-]', '-', action.id)}"
            draft, candidates = await self.manager.create_draft(
                draft_id=draft_id,
                channel=args["channel"],
                recipient=str(args["recipient"]),
                subject=args.get("subject"),
                body=str(args["body"]),
                invitation=args.get("invitation"),
            )
            if draft is None:
                return ToolResult(
                    action_id=action.id,
                    ok=False,
                    code="ambiguous" if candidates else "not_found",
                    observed={"candidates": [item.model_dump(mode="json") for item in candidates]},
                    message="Choose one contact with exactly one address for this channel",
                )
            return ToolResult(
                action_id=action.id,
                ok=True,
                code="ok",
                target=draft.id,
                observed={"draft": draft.model_dump(mode="json"), "approvalRequired": True},
                message="Communication drafted; review and approve it before sending",
            )
        draft_id = str(args["draftId"])
        if tool == "communications.preview":
            draft = await self.manager.get(draft_id)
            if draft is None:
                return ToolResult(action_id=action.id, ok=False, code="not_found")
            return ToolResult(
                action_id=action.id,
                ok=True,
                code="ok",
                target=draft.id,
                observed={"draft": draft.model_dump(mode="json"), "approvalRequired": True},
                message="Draft previewed; authenticated approval is still required",
            )
        if tool == "communications.cancel":
            try:
                draft = await self.manager.cancel(draft_id, actor="voice")
            except KeyError:
                return ToolResult(action_id=action.id, ok=False, code="not_found")
            except (RuntimeError, ValueError):
                return ToolResult(action_id=action.id, ok=False, code="unverified")
            return ToolResult(
                action_id=action.id,
                ok=True,
                code="ok",
                target=draft.id,
                observed={"draft": draft.model_dump(mode="json")},
                message="Communication cancellation verified",
            )
        return ToolResult(
            action_id=action.id,
            ok=False,
            code="blocked",
            message="Sending requires authenticated preview approval",
        )

    async def health(self) -> dict:
        return {**await self.manager.health(), "contractVersion": self.contract_version}

    async def close(self) -> None:
        await self.manager.close()
