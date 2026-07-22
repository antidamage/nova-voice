from __future__ import annotations

import re
from datetime import UTC, datetime

from nova_voice.capabilities.base import CapabilityManifest, CapabilityProvider, ToolPolicy
from nova_voice.domain import PlannedAction, ToolResult
from nova_voice.transactions import TransactionManager, TransactionProposal


def _schema(properties: dict, required: list[str]) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


_ID = {"type": "string", "minLength": 1, "maxLength": 500}
TRANSACTION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "transactions.propose",
            "description": (
                "Create a non-executing travel, shopping, booking, finance, or purchase proposal."
            ),
            "parameters": _schema(
                {
                    "category": {"enum": ["travel", "shopping", "booking", "finance", "purchase"]},
                    "counterparty": _ID,
                    "amount": {"type": "number", "exclusiveMinimum": 0, "maximum": 1000000},
                    "currency": {"type": "string", "pattern": "^[A-Za-z]{3}$"},
                    "summary": {"type": "string", "minLength": 1, "maxLength": 2000},
                    "details": {"type": "object"},
                },
                ["category", "counterparty", "amount", "currency", "summary"],
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "transactions.preview",
            "description": "Preview a proposal without exposing its approval secret.",
            "parameters": _schema({"proposalId": _ID}, ["proposalId"]),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "transactions.commit",
            "description": (
                "Commit only through authenticated approval or a matching standing budget."
            ),
            "parameters": _schema({"proposalId": _ID}, ["proposalId"]),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "transactions.cancel",
            "description": (
                "Cancel a proposal or compensate a committed transaction when supported."
            ),
            "parameters": _schema({"proposalId": _ID}, ["proposalId"]),
        },
    },
]


class TransactionsProvider(CapabilityProvider):
    def __init__(
        self, manager: TransactionManager, *, contract_version: str = "transactions-provider-v1"
    ) -> None:
        self.manager = manager
        self.contract_version = contract_version

    def manifest(self) -> CapabilityManifest:
        return CapabilityManifest(
            id="transactions",
            version="0.1.0",
            contract_version=self.contract_version,
            execution_class="iridium_local",
            tools=TRANSACTION_TOOLS,
            skill_files=[],
            tool_policies={
                "transactions.propose": ToolPolicy(
                    risk="low",
                    reversible=True,
                    idempotent=True,
                    resource_templates=("transactions:proposals",),
                    cancellation="before_side_effects",
                ),
                "transactions.preview": ToolPolicy(
                    risk="low",
                    reversible=True,
                    idempotent=True,
                    parallel_safe=True,
                    cancellation="anytime",
                ),
                "transactions.commit": ToolPolicy(
                    risk="confirmation",
                    reversible=False,
                    idempotent=True,
                    resource_templates=("transactions:proposals",),
                    requires_confirmation=True,
                    cancellation="never",
                ),
                "transactions.cancel": ToolPolicy(
                    risk="low",
                    reversible=False,
                    idempotent=True,
                    resource_templates=("transactions:proposals",),
                    cancellation="before_side_effects",
                ),
            },
        )

    async def execute(self, action: PlannedAction) -> ToolResult:
        tool, args = action.call.tool, action.call.arguments
        if tool == "transactions.propose":
            now = datetime.now(UTC)
            proposal = await self.manager.propose(
                TransactionProposal(
                    id=f"proposal-nova-{re.sub(r'[^a-zA-Z0-9_.-]', '-', action.id)}",
                    category=args["category"],
                    counterparty=str(args["counterparty"]),
                    amount=args["amount"],
                    currency=args["currency"],
                    summary=str(args["summary"]),
                    details=args.get("details", {}),
                    created_at=now,
                    updated_at=now,
                ),
                actor="voice",
            )
            return ToolResult(
                action_id=action.id,
                ok=True,
                code="ok",
                target=proposal.id,
                observed={"proposal": proposal.model_dump(mode="json"), "approvalRequired": True},
                message="Transaction proposed; no commitment was made",
            )
        proposal_id = str(args["proposalId"])
        if tool == "transactions.preview":
            proposal = await self.manager.get(proposal_id)
            return ToolResult(
                action_id=action.id,
                ok=proposal is not None,
                code="ok" if proposal else "not_found",
                target=proposal_id,
                observed={
                    "proposal": proposal.model_dump(mode="json") if proposal else None,
                    "approvalRequired": True,
                },
            )
        if tool == "transactions.cancel":
            try:
                proposal = await self.manager.cancel(proposal_id, actor="voice")
                return ToolResult(
                    action_id=action.id,
                    ok=True,
                    code="ok",
                    target=proposal.id,
                    observed={"proposal": proposal.model_dump(mode="json")},
                    message="Transaction cancellation verified",
                )
            except KeyError:
                return ToolResult(action_id=action.id, ok=False, code="not_found")
            except (RuntimeError, ValueError):
                return ToolResult(action_id=action.id, ok=False, code="unverified")
        return ToolResult(
            action_id=action.id,
            ok=False,
            code="blocked",
            message="Commit requires authenticated approval or a standing budget",
        )

    async def health(self) -> dict:
        return {**await self.manager.health(), "contractVersion": self.contract_version}

    async def close(self) -> None:
        await self.manager.close()
