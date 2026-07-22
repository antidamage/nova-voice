from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

TransactionCategory = Literal["travel", "shopping", "booking", "finance", "purchase"]
TransactionState = Literal["proposed", "committing", "committed", "failed", "cancelled"]


class TransactionProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    category: TransactionCategory
    counterparty: str = Field(min_length=1, max_length=500)
    amount: float = Field(gt=0, le=1_000_000)
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    summary: str = Field(min_length=1, max_length=2000)
    details: dict = Field(default_factory=dict)
    state: TransactionState = "proposed"
    revision: int = 1
    receipt: str | None = None
    created_at: datetime
    updated_at: datetime

    @field_validator("currency", mode="before")
    @classmethod
    def normalize_currency(cls, value: object) -> str:
        return str(value).strip().upper()


class TransactionTransport(Protocol):
    async def commit(self, proposal: TransactionProposal) -> dict: ...
    async def cancel(self, proposal: TransactionProposal) -> dict: ...
    async def health(self) -> dict: ...
    async def close(self) -> None: ...


class DisabledTransactionTransport:
    async def commit(self, proposal: TransactionProposal) -> dict:
        raise RuntimeError("transaction bridge is not configured")

    async def cancel(self, proposal: TransactionProposal) -> dict:
        raise RuntimeError("transaction bridge is not configured")

    async def health(self) -> dict:
        return {"ok": True, "configured": False}

    async def close(self) -> None:
        return None


class WebhookTransactionTransport:
    def __init__(self, url: str, token: str, *, timeout_seconds: float = 15) -> None:
        self._client = httpx.AsyncClient(
            base_url=url.rstrip("/"),
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout_seconds,
        )

    async def _verified(self, path: str, payload: dict, key: str) -> dict:
        response = await self._client.post(path, json=payload)
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, dict) or not result.get(key) or not result.get("receipt"):
            raise RuntimeError(f"transaction bridge did not verify {key}")
        return result

    async def commit(self, proposal: TransactionProposal) -> dict:
        return await self._verified("/commit", proposal.model_dump(mode="json"), "committed")

    async def cancel(self, proposal: TransactionProposal) -> dict:
        return await self._verified(
            "/cancel", {"proposalId": proposal.id, "receipt": proposal.receipt}, "cancelled"
        )

    async def health(self) -> dict:
        try:
            response = await self._client.get("/health")
            return {"ok": response.status_code == 200, "configured": True}
        except httpx.HTTPError:
            return {"ok": False, "configured": True}

    async def close(self) -> None:
        await self._client.aclose()


class TransactionManager:
    """Fail-closed proposal, budget, approval, receipt, and compensation store."""

    def __init__(self, path: Path, transport: TransactionTransport | None = None) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.transport = transport or DisabledTransactionTransport()
        self._lock = asyncio.Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS transaction_proposals (
                    id TEXT PRIMARY KEY, payload_json TEXT NOT NULL,
                    approval_hash TEXT, approval_revision INTEGER,
                    approval_used INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS transaction_budgets (
                    id TEXT PRIMARY KEY, category TEXT NOT NULL, counterparty TEXT,
                    currency TEXT NOT NULL, limit_amount REAL NOT NULL,
                    remaining_amount REAL NOT NULL, active INTEGER NOT NULL DEFAULT 1
                );
                CREATE TABLE IF NOT EXISTS transaction_audit (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    proposal_id TEXT NOT NULL, action TEXT NOT NULL,
                    actor TEXT NOT NULL, detail_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    @staticmethod
    def _decode(row: sqlite3.Row) -> TransactionProposal:
        return TransactionProposal.model_validate_json(row["payload_json"])

    @staticmethod
    def _hash(token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()

    @staticmethod
    def _audit(connection, proposal_id: str, action: str, actor: str, detail: dict) -> None:
        connection.execute(
            """
            INSERT INTO transaction_audit
                (proposal_id, action, actor, detail_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                proposal_id,
                action,
                actor,
                json.dumps(detail, sort_keys=True, separators=(",", ":")),
                datetime.now(UTC).isoformat(),
            ),
        )

    async def propose(self, proposal: TransactionProposal, *, actor: str) -> TransactionProposal:
        async with self._lock:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT * FROM transaction_proposals WHERE id=?", (proposal.id,)
                ).fetchone()
                if row:
                    return self._decode(row)
                connection.execute(
                    "INSERT INTO transaction_proposals(id, payload_json) VALUES (?, ?)",
                    (proposal.id, proposal.model_dump_json()),
                )
                self._audit(
                    connection,
                    proposal.id,
                    "proposed",
                    actor,
                    {"amount": proposal.amount, "currency": proposal.currency},
                )
        return proposal

    async def get(self, proposal_id: str) -> TransactionProposal | None:
        def read():
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT * FROM transaction_proposals WHERE id=?", (proposal_id,)
                ).fetchone()
            return self._decode(row) if row else None

        return await asyncio.to_thread(read)

    async def preview(self, proposal_id: str, *, actor: str) -> tuple[TransactionProposal, str]:
        token = secrets.token_urlsafe(24)
        async with self._lock:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT * FROM transaction_proposals WHERE id=?", (proposal_id,)
                ).fetchone()
                if row is None:
                    raise KeyError(proposal_id)
                proposal = self._decode(row)
                if proposal.state != "proposed":
                    raise ValueError("only pending proposals can be approved")
                connection.execute(
                    """
                    UPDATE transaction_proposals
                    SET approval_hash=?, approval_revision=?, approval_used=0
                    WHERE id=?
                    """,
                    (self._hash(token), proposal.revision, proposal.id),
                )
                self._audit(
                    connection, proposal.id, "previewed", actor, {"revision": proposal.revision}
                )
        return proposal, token

    async def create_budget(
        self,
        *,
        budget_id: str,
        category: TransactionCategory,
        currency: str,
        limit_amount: float,
        counterparty: str | None = None,
    ) -> dict:
        if limit_amount <= 0:
            raise ValueError("budget must be positive")
        normalized_currency = currency.upper()
        async with self._lock:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO transaction_budgets
                        (id, category, counterparty, currency, limit_amount, remaining_amount)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        budget_id,
                        category,
                        counterparty,
                        normalized_currency,
                        limit_amount,
                        limit_amount,
                    ),
                )
        return {"id": budget_id, "remaining": limit_amount, "currency": normalized_currency}

    async def budget(self, budget_id: str) -> dict | None:
        def read():
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT * FROM transaction_budgets WHERE id=?", (budget_id,)
                ).fetchone()
            return dict(row) if row else None

        return await asyncio.to_thread(read)

    async def commit(
        self,
        proposal_id: str,
        *,
        actor: str,
        approval_token: str | None = None,
        budget_id: str | None = None,
    ) -> TransactionProposal:
        async with self._lock:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT * FROM transaction_proposals WHERE id=?", (proposal_id,)
                ).fetchone()
                if row is None:
                    raise KeyError(proposal_id)
                proposal = self._decode(row)
                if proposal.state != "proposed":
                    raise ValueError("proposal is not pending")
                authorized = bool(
                    approval_token
                    and not row["approval_used"]
                    and row["approval_revision"] == proposal.revision
                    and secrets.compare_digest(
                        str(row["approval_hash"] or ""), self._hash(approval_token)
                    )
                )
                budget = None
                if not authorized and budget_id:
                    budget = connection.execute(
                        "SELECT * FROM transaction_budgets WHERE id=? AND active=1",
                        (budget_id,),
                    ).fetchone()
                    authorized = bool(
                        budget
                        and budget["category"] == proposal.category
                        and budget["currency"] == proposal.currency
                        and (
                            not budget["counterparty"]
                            or budget["counterparty"].casefold() == proposal.counterparty.casefold()
                        )
                        and budget["remaining_amount"] >= proposal.amount
                    )
                if not authorized:
                    raise PermissionError("owner approval or matching standing budget required")
                if budget is not None:
                    connection.execute(
                        """
                        UPDATE transaction_budgets
                        SET remaining_amount=remaining_amount-? WHERE id=?
                        """,
                        (proposal.amount, budget_id),
                    )
                connection.execute(
                    "UPDATE transaction_proposals SET approval_used=1 WHERE id=?", (proposal.id,)
                )
                committing = proposal.model_copy(
                    update={"state": "committing", "updated_at": datetime.now(UTC)}
                )
                connection.execute(
                    "UPDATE transaction_proposals SET payload_json=? WHERE id=?",
                    (committing.model_dump_json(), proposal.id),
                )
                self._audit(
                    connection,
                    proposal.id,
                    "commit_authorized",
                    actor,
                    {"budgetId": budget_id},
                )
        try:
            receipt = await self.transport.commit(proposal)
            updated = proposal.model_copy(
                update={
                    "state": "committed",
                    "receipt": str(receipt["receipt"]),
                    "revision": proposal.revision + 1,
                    "updated_at": datetime.now(UTC),
                }
            )
            action = "commit_verified"
        except Exception as error:
            updated = proposal.model_copy(
                update={
                    "state": "failed",
                    "revision": proposal.revision + 1,
                    "updated_at": datetime.now(UTC),
                }
            )
            receipt = {"error": type(error).__name__}
            action = "commit_failed"
            if budget_id:
                with self._connect() as connection:
                    connection.execute(
                        """
                        UPDATE transaction_budgets
                        SET remaining_amount=remaining_amount+? WHERE id=?
                        """,
                        (proposal.amount, budget_id),
                    )
        async with self._lock:
            with self._connect() as connection:
                connection.execute(
                    "UPDATE transaction_proposals SET payload_json=? WHERE id=?",
                    (updated.model_dump_json(), updated.id),
                )
                self._audit(connection, updated.id, action, actor, receipt)
        return updated

    async def cancel(self, proposal_id: str, *, actor: str) -> TransactionProposal:
        proposal = await self.get(proposal_id)
        if proposal is None:
            raise KeyError(proposal_id)
        if proposal.state == "committed":
            await self.transport.cancel(proposal)
        elif proposal.state not in {"proposed", "failed"}:
            raise ValueError("transaction is already cancelled")
        updated = proposal.model_copy(
            update={
                "state": "cancelled",
                "revision": proposal.revision + 1,
                "updated_at": datetime.now(UTC),
            }
        )
        async with self._lock:
            with self._connect() as connection:
                connection.execute(
                    "UPDATE transaction_proposals SET payload_json=?, approval_used=1 WHERE id=?",
                    (updated.model_dump_json(), updated.id),
                )
                self._audit(connection, updated.id, "cancelled", actor, {})
        return updated

    async def audit(self, proposal_id: str) -> tuple[dict, ...]:
        def read():
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT action, actor, detail_json, created_at
                    FROM transaction_audit
                    WHERE proposal_id=? ORDER BY sequence
                    """,
                    (proposal_id,),
                ).fetchall()
            return tuple(
                {
                    "action": row["action"],
                    "actor": row["actor"],
                    "detail": json.loads(row["detail_json"]),
                    "createdAt": row["created_at"],
                }
                for row in rows
            )

        return await asyncio.to_thread(read)

    async def health(self) -> dict:
        return {**await self.transport.health(), "enabled": True}

    async def close(self) -> None:
        await self.transport.close()
