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
from pydantic import BaseModel, ConfigDict

from nova_voice.providers.personal.store import PersonalDataStore, PersonalRecord

CommunicationChannel = Literal["email", "message", "invitation"]
CommunicationState = Literal["draft", "sent", "delivery_failed", "cancelled"]


class CommunicationDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    channel: CommunicationChannel
    recipient_id: str
    recipient_name: str
    recipient_address: str
    subject: str | None = None
    body: str
    invitation: dict | None = None
    state: CommunicationState = "draft"
    revision: int = 1
    delivery_receipt: str | None = None
    created_at: datetime
    updated_at: datetime


class DeliveryTransport(Protocol):
    async def send(self, draft: CommunicationDraft) -> dict: ...
    async def cancel(self, draft: CommunicationDraft) -> dict: ...
    async def health(self) -> dict: ...
    async def close(self) -> None: ...


class DisabledDeliveryTransport:
    async def send(self, draft: CommunicationDraft) -> dict:
        raise RuntimeError("communication delivery bridge is not configured")

    async def cancel(self, draft: CommunicationDraft) -> dict:
        raise RuntimeError("communication delivery bridge is not configured")

    async def health(self) -> dict:
        return {"ok": True, "configured": False}

    async def close(self) -> None:
        return None


class WebhookDeliveryTransport:
    def __init__(self, url: str, token: str, *, timeout_seconds: float = 10) -> None:
        self._client = httpx.AsyncClient(
            base_url=url.rstrip("/"),
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout_seconds,
        )

    async def send(self, draft: CommunicationDraft) -> dict:
        response = await self._client.post("/send", json=draft.model_dump(mode="json"))
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, dict) or not result.get("delivered") or not result.get("receipt"):
            raise RuntimeError("delivery bridge did not verify delivery")
        return result

    async def cancel(self, draft: CommunicationDraft) -> dict:
        response = await self._client.post(
            "/cancel", json={"draftId": draft.id, "receipt": draft.delivery_receipt}
        )
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, dict) or not result.get("cancelled"):
            raise RuntimeError("delivery bridge did not verify cancellation")
        return result

    async def health(self) -> dict:
        try:
            response = await self._client.get("/health")
            return {"ok": response.status_code == 200, "configured": True}
        except httpx.HTTPError:
            return {"ok": False, "configured": True}

    async def close(self) -> None:
        await self._client.aclose()


class CommunicationManager:
    """Draft-first communication lifecycle with dashboard-only approval tokens."""

    def __init__(
        self,
        path: Path,
        contacts: PersonalDataStore,
        transport: DeliveryTransport | None = None,
    ) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.contacts = contacts
        self.transport = transport or DisabledDeliveryTransport()
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
                CREATE TABLE IF NOT EXISTS communication_drafts (
                    id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    approval_hash TEXT,
                    approval_revision INTEGER,
                    approval_used INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS communication_audit (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    draft_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    detail_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    @staticmethod
    def _decode(row: sqlite3.Row) -> CommunicationDraft:
        return CommunicationDraft.model_validate_json(row["payload_json"])

    @staticmethod
    def _token_hash(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def _audit(
        self, connection: sqlite3.Connection, draft_id: str, action: str, actor: str, detail: dict
    ) -> None:
        connection.execute(
            """
            INSERT INTO communication_audit
                (draft_id, action, actor, detail_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                draft_id,
                action,
                actor,
                json.dumps(detail, sort_keys=True, separators=(",", ":")),
                datetime.now(UTC).isoformat(),
            ),
        )

    async def _recipient(
        self, selector: str, channel: CommunicationChannel
    ) -> tuple[PersonalRecord | None, str | None, tuple[PersonalRecord, ...]]:
        matches = await self.contacts.resolve("contact", selector)
        if len(matches) != 1:
            return None, None, matches
        contact = matches[0]
        field = "emails" if channel in {"email", "invitation"} else "phones"
        addresses = [
            str(value).strip() for value in contact.data.get(field, []) if str(value).strip()
        ]
        return contact, addresses[0] if len(addresses) == 1 else None, matches

    async def create_draft(
        self,
        *,
        draft_id: str,
        channel: CommunicationChannel,
        recipient: str,
        body: str,
        subject: str | None = None,
        invitation: dict | None = None,
        actor: str = "voice",
    ) -> tuple[CommunicationDraft | None, tuple[PersonalRecord, ...]]:
        contact, address, matches = await self._recipient(recipient, channel)
        if contact is None or address is None:
            return None, matches
        now = datetime.now(UTC)
        draft = CommunicationDraft(
            id=draft_id,
            channel=channel,
            recipient_id=contact.id,
            recipient_name=contact.label,
            recipient_address=address,
            subject=subject,
            body=body,
            invitation=invitation,
            created_at=now,
            updated_at=now,
        )
        async with self._lock:
            with self._connect() as connection:
                existing = connection.execute(
                    "SELECT * FROM communication_drafts WHERE id=?", (draft_id,)
                ).fetchone()
                if existing:
                    return self._decode(existing), ()
                connection.execute(
                    "INSERT INTO communication_drafts(id, payload_json) VALUES (?, ?)",
                    (draft.id, draft.model_dump_json()),
                )
                self._audit(connection, draft.id, "drafted", actor, {"channel": channel})
        return draft, ()

    async def get(self, draft_id: str) -> CommunicationDraft | None:
        def read() -> CommunicationDraft | None:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT * FROM communication_drafts WHERE id=?", (draft_id,)
                ).fetchone()
            return self._decode(row) if row else None

        return await asyncio.to_thread(read)

    async def preview(self, draft_id: str, *, actor: str) -> tuple[CommunicationDraft, str]:
        token = secrets.token_urlsafe(24)
        async with self._lock:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT * FROM communication_drafts WHERE id=?", (draft_id,)
                ).fetchone()
                if row is None:
                    raise KeyError(draft_id)
                draft = self._decode(row)
                if draft.state != "draft":
                    raise ValueError("only pending drafts can be approved")
                connection.execute(
                    """
                    UPDATE communication_drafts
                    SET approval_hash=?, approval_revision=?, approval_used=0
                    WHERE id=?
                    """,
                    (self._token_hash(token), draft.revision, draft.id),
                )
                self._audit(connection, draft.id, "previewed", actor, {"revision": draft.revision})
        return draft, token

    async def send_approved(self, draft_id: str, token: str, *, actor: str) -> CommunicationDraft:
        async with self._lock:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT * FROM communication_drafts WHERE id=?", (draft_id,)
                ).fetchone()
                if row is None:
                    raise KeyError(draft_id)
                draft = self._decode(row)
                if (
                    draft.state != "draft"
                    or row["approval_used"]
                    or row["approval_revision"] != draft.revision
                    or not secrets.compare_digest(
                        str(row["approval_hash"] or ""), self._token_hash(token)
                    )
                ):
                    raise PermissionError("valid preview approval is required")
                connection.execute(
                    "UPDATE communication_drafts SET approval_used=1 WHERE id=?", (draft.id,)
                )
                self._audit(connection, draft.id, "send_authorized", actor, {})
        try:
            receipt = await self.transport.send(draft)
            updated = draft.model_copy(
                update={
                    "state": "sent",
                    "delivery_receipt": str(receipt["receipt"]),
                    "revision": draft.revision + 1,
                    "updated_at": datetime.now(UTC),
                }
            )
            action = "delivery_verified"
        except Exception as error:
            updated = draft.model_copy(
                update={
                    "state": "delivery_failed",
                    "revision": draft.revision + 1,
                    "updated_at": datetime.now(UTC),
                }
            )
            action = "delivery_failed"
            receipt = {"error": type(error).__name__}
        async with self._lock:
            with self._connect() as connection:
                connection.execute(
                    "UPDATE communication_drafts SET payload_json=? WHERE id=?",
                    (updated.model_dump_json(), updated.id),
                )
                self._audit(connection, updated.id, action, actor, receipt)
        return updated

    async def cancel(self, draft_id: str, *, actor: str) -> CommunicationDraft:
        draft = await self.get(draft_id)
        if draft is None:
            raise KeyError(draft_id)
        if draft.state == "sent":
            await self.transport.cancel(draft)
        elif draft.state not in {"draft", "delivery_failed"}:
            raise ValueError("communication is already cancelled")
        updated = draft.model_copy(
            update={
                "state": "cancelled",
                "revision": draft.revision + 1,
                "updated_at": datetime.now(UTC),
            }
        )
        async with self._lock:
            with self._connect() as connection:
                connection.execute(
                    "UPDATE communication_drafts SET payload_json=?, approval_used=1 WHERE id=?",
                    (updated.model_dump_json(), updated.id),
                )
                self._audit(connection, updated.id, "cancelled", actor, {})
        return updated

    async def audit(self, draft_id: str) -> tuple[dict, ...]:
        def read() -> tuple[dict, ...]:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT action, actor, detail_json, created_at
                    FROM communication_audit
                    WHERE draft_id=? ORDER BY sequence
                    """,
                    (draft_id,),
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
