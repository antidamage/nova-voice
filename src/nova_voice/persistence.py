from __future__ import annotations

import asyncio
import sqlite3
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path

from nova_voice.domain import Interpretation, Utterance


class TranscriptStore:
    def __init__(self, path: Path, retention_hours: float = 24) -> None:
        self.path = path
        self.retention = timedelta(hours=retention_hours)
        self._stop = asyncio.Event()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA secure_delete=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def initialize_sync(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS transcripts (
                    utterance_id TEXT PRIMARY KEY,
                    satellite_id TEXT NOT NULL,
                    room_id TEXT NOT NULL,
                    transcribed_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    transcript TEXT NOT NULL,
                    transcript_confidence REAL NOT NULL,
                    wake_detected INTEGER NOT NULL,
                    interpretation_json TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_transcripts_expires_at
                    ON transcripts(expires_at);
                """
            )

    async def initialize(self) -> None:
        await asyncio.to_thread(self.initialize_sync)

    def add_sync(self, utterance: Utterance, interpretation: Interpretation | None = None) -> None:
        transcribed_at = utterance.ended_at.astimezone(UTC)
        expires_at = transcribed_at + self.retention
        interpretation_json = interpretation.model_dump_json() if interpretation else None
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO transcripts (
                    utterance_id, satellite_id, room_id, transcribed_at, expires_at,
                    transcript, transcript_confidence, wake_detected, interpretation_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    utterance.id,
                    utterance.satellite_id,
                    utterance.room_id,
                    transcribed_at.isoformat(),
                    expires_at.isoformat(),
                    utterance.transcript,
                    utterance.transcript_confidence,
                    int(utterance.wake_detected),
                    interpretation_json,
                ),
            )

    async def add(self, utterance: Utterance, interpretation: Interpretation | None = None) -> None:
        await asyncio.to_thread(self.add_sync, utterance, interpretation)

    def delete_expired_sync(self, now: datetime | None = None) -> int:
        boundary = (now or datetime.now(UTC)).astimezone(UTC).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM transcripts WHERE expires_at <= ?", (boundary,)
            )
            deleted = cursor.rowcount
            connection.commit()
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        return deleted

    async def delete_expired(self, now: datetime | None = None) -> int:
        return await asyncio.to_thread(self.delete_expired_sync, now)

    def next_expiry_sync(self) -> datetime | None:
        with self._connect() as connection:
            row = connection.execute("SELECT MIN(expires_at) AS expiry FROM transcripts").fetchone()
        if not row or row["expiry"] is None:
            return None
        return datetime.fromisoformat(row["expiry"]).astimezone(UTC)

    async def next_expiry(self) -> datetime | None:
        return await asyncio.to_thread(self.next_expiry_sync)

    def count_sync(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM transcripts").fetchone()[0])

    async def count(self) -> int:
        return await asyncio.to_thread(self.count_sync)

    async def run_janitor(self, safety_interval_seconds: float = 300) -> None:
        await self.delete_expired()
        while not self._stop.is_set():
            expiry = await self.next_expiry()
            if expiry is None:
                delay = safety_interval_seconds
            else:
                delay = max(
                    0.1, min(safety_interval_seconds, (expiry - datetime.now(UTC)).total_seconds())
                )
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._stop.wait(), timeout=delay)
            if not self._stop.is_set():
                await self.delete_expired()

    def stop(self) -> None:
        self._stop.set()
