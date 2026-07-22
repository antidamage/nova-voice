from __future__ import annotations

import asyncio
import json
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

RecordKind = Literal["note", "list", "contact"]


class PersonalRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    kind: RecordKind
    label: str = Field(min_length=1, max_length=500)
    data: dict
    revision: int = Field(default=1, ge=1)
    created_at: datetime
    updated_at: datetime


class MutationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    record: PersonalRecord | None
    undo_token: str
    changed: bool


def _normalized(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


class PersonalDataStore:
    """Revisioned personal data with atomic, conflict-safe one-step undo."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
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
                CREATE TABLE IF NOT EXISTS personal_records (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    label TEXT NOT NULL,
                    normalized_label TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    deleted INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS personal_records_lookup
                    ON personal_records(kind, normalized_label, deleted);
                CREATE TABLE IF NOT EXISTS personal_undo (
                    token TEXT PRIMARY KEY,
                    record_id TEXT NOT NULL,
                    before_json TEXT,
                    after_revision INTEGER NOT NULL,
                    used INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );
                """
            )

    @staticmethod
    def _decode(row: sqlite3.Row) -> PersonalRecord:
        return PersonalRecord(
            id=row["id"],
            kind=row["kind"],
            label=row["label"],
            data=json.loads(row["payload_json"]),
            revision=row["revision"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    @staticmethod
    def _snapshot(row: sqlite3.Row) -> str:
        return json.dumps(dict(row), sort_keys=True, separators=(",", ":"))

    def _search_sync(self, kind: RecordKind, query: str) -> tuple[PersonalRecord, ...]:
        normalized = _normalized(query)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM personal_records WHERE kind=? AND deleted=0 ORDER BY label, id",
                (kind,),
            ).fetchall()
        records = [self._decode(row) for row in rows]
        if not normalized:
            return tuple(records)
        exact = [record for record in records if _normalized(record.label) == normalized]
        if exact:
            return tuple(exact)
        return tuple(record for record in records if normalized in _normalized(record.label))

    async def search(self, kind: RecordKind, query: str = "") -> tuple[PersonalRecord, ...]:
        return await asyncio.to_thread(self._search_sync, kind, query)

    async def resolve(self, kind: RecordKind, selector: str) -> tuple[PersonalRecord, ...]:
        direct = await self.get(selector)
        if direct is not None and direct.kind == kind:
            return (direct,)
        return await self.search(kind, selector)

    def _get_sync(self, record_id: str) -> PersonalRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM personal_records WHERE id=? AND deleted=0", (record_id,)
            ).fetchone()
        return self._decode(row) if row else None

    async def get(self, record_id: str) -> PersonalRecord | None:
        return await asyncio.to_thread(self._get_sync, record_id)

    def _mutate_sync(
        self,
        *,
        record_id: str,
        kind: RecordKind,
        label: str,
        data: dict,
        undo_token: str,
        delete: bool = False,
    ) -> MutationResult:
        now = datetime.now(UTC)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            prior_undo = connection.execute(
                "SELECT record_id FROM personal_undo WHERE token=?", (undo_token,)
            ).fetchone()
            current = connection.execute(
                "SELECT * FROM personal_records WHERE id=?", (record_id,)
            ).fetchone()
            if prior_undo is not None:
                visible = current if current is not None and not current["deleted"] else None
                return MutationResult(
                    record=self._decode(visible) if visible else None,
                    undo_token=undo_token,
                    changed=False,
                )
            before = self._snapshot(current) if current is not None else None
            revision = int(current["revision"] if current is not None else 0) + 1
            created_at = datetime.fromisoformat(current["created_at"]) if current else now
            connection.execute(
                """
                INSERT INTO personal_records
                    (id, kind, label, normalized_label, payload_json, revision,
                     deleted, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    kind=excluded.kind, label=excluded.label,
                    normalized_label=excluded.normalized_label,
                    payload_json=excluded.payload_json, revision=excluded.revision,
                    deleted=excluded.deleted, updated_at=excluded.updated_at
                """,
                (
                    record_id,
                    kind,
                    label,
                    _normalized(label),
                    json.dumps(data, sort_keys=True, separators=(",", ":")),
                    revision,
                    int(delete),
                    created_at.isoformat(),
                    now.isoformat(),
                ),
            )
            connection.execute(
                """
                INSERT INTO personal_undo
                    (token, record_id, before_json, after_revision, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (undo_token, record_id, before, revision, now.isoformat()),
            )
            row = connection.execute(
                "SELECT * FROM personal_records WHERE id=?", (record_id,)
            ).fetchone()
        return MutationResult(
            record=None if delete else self._decode(row),
            undo_token=undo_token,
            changed=True,
        )

    async def mutate(self, **arguments) -> MutationResult:
        async with self._lock:
            return await asyncio.to_thread(self._mutate_sync, **arguments)

    def _undo_sync(self, token: str) -> PersonalRecord | None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            undo = connection.execute(
                "SELECT * FROM personal_undo WHERE token=?", (token,)
            ).fetchone()
            if undo is None:
                raise KeyError("unknown undo token")
            if undo["used"]:
                raise ValueError("undo token has already been used")
            current = connection.execute(
                "SELECT * FROM personal_records WHERE id=?", (undo["record_id"],)
            ).fetchone()
            if current is None or current["revision"] != undo["after_revision"]:
                raise ValueError("record changed after this undo token was issued")
            before = json.loads(undo["before_json"]) if undo["before_json"] else None
            if before is None:
                connection.execute(
                    "UPDATE personal_records SET deleted=1, revision=revision+1 WHERE id=?",
                    (undo["record_id"],),
                )
                restored = None
            else:
                connection.execute(
                    """
                    UPDATE personal_records SET kind=?, label=?, normalized_label=?,
                        payload_json=?, revision=revision+1, deleted=?, updated_at=?
                    WHERE id=?
                    """,
                    (
                        before["kind"],
                        before["label"],
                        before["normalized_label"],
                        before["payload_json"],
                        before["deleted"],
                        datetime.now(UTC).isoformat(),
                        undo["record_id"],
                    ),
                )
                row = connection.execute(
                    "SELECT * FROM personal_records WHERE id=?", (undo["record_id"],)
                ).fetchone()
                restored = None if row["deleted"] else self._decode(row)
            connection.execute("UPDATE personal_undo SET used=1 WHERE token=?", (token,))
        return restored

    async def undo(self, token: str) -> PersonalRecord | None:
        async with self._lock:
            return await asyncio.to_thread(self._undo_sync, token)

    async def health(self) -> dict:
        def probe() -> int:
            with self._connect() as connection:
                return int(
                    connection.execute(
                        "SELECT COUNT(*) FROM personal_records WHERE deleted=0"
                    ).fetchone()[0]
                )

        return {"ok": True, "records": await asyncio.to_thread(probe)}
