from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

import httpx
import pytest

from nova_voice.api import create_app
from nova_voice.capabilities.registry import CapabilityRegistry
from nova_voice.config import Settings
from nova_voice.domain import CapabilityToolCall, PlannedAction
from nova_voice.providers.transactions.provider import TransactionsProvider
from nova_voice.transactions import TransactionManager, TransactionProposal


class _Transport:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.commits = []
        self.cancellations = []

    async def commit(self, proposal):
        self.commits.append(proposal.id)
        if self.fail:
            raise RuntimeError("failed")
        return {"committed": True, "receipt": f"receipt-{proposal.id}"}

    async def cancel(self, proposal):
        self.cancellations.append(proposal.receipt)
        return {"cancelled": True, "receipt": f"cancel-{proposal.id}"}

    async def health(self):
        return {"ok": True, "configured": True}

    async def close(self):
        return None


class _BlockingTransport(_Transport):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def commit(self, proposal):
        self.commits.append(proposal.id)
        self.started.set()
        await self.release.wait()
        return {"committed": True, "receipt": f"receipt-{proposal.id}"}


def _proposal(proposal_id: str = "proposal-1", *, amount: float = 25) -> TransactionProposal:
    now = datetime.now(UTC)
    return TransactionProposal(
        id=proposal_id,
        category="shopping",
        counterparty="Local shop",
        amount=amount,
        currency="nzd",
        summary="Buy household supplies",
        created_at=now,
        updated_at=now,
    )


def _action(tool: str, arguments: dict, action_id: str) -> PlannedAction:
    return PlannedAction(
        id=action_id,
        order=0,
        call=CapabilityToolCall(provider="transactions", tool=tool, arguments=arguments),
    )


async def test_owner_preview_token_commits_once_with_verified_receipt_and_cancel(tmp_path) -> None:
    transport = _Transport()
    manager = TransactionManager(tmp_path / "transactions.sqlite3", transport)
    await manager.propose(_proposal(), actor="voice")
    _, token = await manager.preview("proposal-1", actor="dashboard-admin")

    with pytest.raises(PermissionError):
        await manager.commit("proposal-1", actor="dashboard-admin", approval_token="wrong")
    committed = await manager.commit("proposal-1", actor="dashboard-admin", approval_token=token)
    with pytest.raises(ValueError):
        await manager.commit("proposal-1", actor="dashboard-admin", approval_token=token)
    cancelled = await manager.cancel("proposal-1", actor="dashboard-admin")

    assert committed.state == "committed"
    assert committed.receipt == "receipt-proposal-1"
    assert cancelled.state == "cancelled"
    assert transport.commits == ["proposal-1"]
    assert transport.cancellations == ["receipt-proposal-1"]


async def test_standing_budget_matches_category_currency_counterparty_and_remaining(
    tmp_path,
) -> None:
    manager = TransactionManager(tmp_path / "transactions.sqlite3", _Transport())
    await manager.create_budget(
        budget_id="groceries",
        category="shopping",
        currency="NZD",
        limit_amount=40,
        counterparty="Local shop",
    )
    await manager.propose(_proposal(amount=25), actor="voice")

    committed = await manager.commit("proposal-1", actor="durable-plan", budget_id="groceries")
    budget = await manager.budget("groceries")

    assert committed.state == "committed"
    assert budget["remaining_amount"] == 15

    await manager.propose(_proposal("proposal-2", amount=20), actor="voice")
    with pytest.raises(PermissionError):
        await manager.commit("proposal-2", actor="durable-plan", budget_id="groceries")


async def test_failed_bridge_never_commits_and_refunds_reserved_budget(tmp_path) -> None:
    manager = TransactionManager(tmp_path / "transactions.sqlite3", _Transport(fail=True))
    await manager.create_budget(
        budget_id="travel",
        category="travel",
        currency="NZD",
        limit_amount=100,
    )
    proposal = _proposal(amount=80).model_copy(update={"category": "travel"})
    await manager.propose(proposal, actor="voice")

    failed = await manager.commit(proposal.id, actor="durable-plan", budget_id="travel")
    budget = await manager.budget("travel")

    assert failed.state == "failed" and failed.receipt is None
    assert budget["remaining_amount"] == 100


async def test_in_flight_commit_lease_prevents_duplicate_budget_execution(tmp_path) -> None:
    transport = _BlockingTransport()
    manager = TransactionManager(tmp_path / "transactions.sqlite3", transport)
    await manager.create_budget(
        budget_id="shopping",
        category="shopping",
        currency="NZD",
        limit_amount=100,
    )
    await manager.propose(_proposal(), actor="voice")
    first = asyncio.create_task(
        manager.commit("proposal-1", actor="plan-one", budget_id="shopping")
    )
    await transport.started.wait()

    with pytest.raises(ValueError, match="not pending"):
        await manager.commit("proposal-1", actor="plan-two", budget_id="shopping")
    transport.release.set()
    await first

    assert transport.commits == ["proposal-1"]


def test_immediate_commit_tool_is_confirmation_blocked(tmp_path) -> None:
    registry = CapabilityRegistry(allowlist={"transactions"})
    registry.register(TransactionsProvider(TransactionManager(tmp_path / "transactions.sqlite3")))
    policy = registry.policy_for("transactions", "transactions.commit")
    assert policy.risk == "confirmation" and policy.requires_confirmation
    assert policy.cancellation == "never"


async def test_authenticated_transaction_api_approval_and_audit(tmp_path) -> None:
    manager = TransactionManager(tmp_path / "transactions.sqlite3", _Transport())
    await manager.propose(_proposal(), actor="voice")
    app = create_app(Settings(), service=SimpleNamespace(transactions=manager))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="https://voice.test") as client:
        preview = await client.post(
            "/v1/transactions/proposal-1/preview", json={"actor": "dashboard-admin"}
        )
        committed = await client.post(
            "/v1/transactions/proposal-1/commit",
            json={"approval_token": preview.json()["approvalToken"]},
        )
        audit = await client.get("/v1/transactions/proposal-1/audit")

    assert committed.json()["proposal"]["state"] == "committed"
    assert audit.json()["events"][-1]["action"] == "commit_verified"
