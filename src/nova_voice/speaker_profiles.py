from __future__ import annotations

import asyncio
import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import numpy as np

from nova_voice.domain import SelfProfileUpdate, SpeakerIdentity


def _name_key(value: str) -> str:
    return " ".join(value.casefold().split())


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = " ".join(value.split()).strip()
    return cleaned or None


_DISCLOSURE_WORD_RE = re.compile(
    r"[^\W_]+(?:['\N{RIGHT SINGLE QUOTATION MARK}.\-][^\W_]+)*",
    re.UNICODE,
)


def _words(value: str) -> list[str]:
    return _DISCLOSURE_WORD_RE.findall(value.casefold())


def _terms_appear_in_order(terms: list[str], evidence: str) -> bool:
    if not terms:
        return False
    evidence_words = _words(evidence)
    cursor = 0
    for term in terms:
        try:
            cursor = evidence_words.index(term, cursor) + 1
        except ValueError:
            return False
    return True


def _normalize_pronouns(value: str | None) -> str | None:
    cleaned = _clean(value)
    if cleaned is None:
        return None
    terms = _words(cleaned)
    if terms and terms[-1] in {"pronoun", "pronouns"}:
        terms = terms[:-1]
    if 2 <= len(terms) <= 3:
        return "/".join(terms)
    return cleaned


def validated_self_profile_update(
    update: SelfProfileUpdate,
    transcript: str,
) -> SelfProfileUpdate | None:
    """Keep only current-turn identity fields supported by exact evidence."""

    evidence = " ".join(update.evidence.casefold().split())
    spoken = " ".join(transcript.casefold().split())
    if not evidence or evidence not in spoken:
        return None
    proposed_name = _clean(update.name)
    proposed_pronouns = _normalize_pronouns(update.pronouns)
    name = (
        proposed_name
        if proposed_name is not None
        and proposed_name.casefold() in evidence
        and any(
            cue in evidence
            for cue in (
                "my name",
                "call me",
                "this is",
                "i'm",
                "i am",
                "i go by",
                "i'm called",
            )
        )
        else None
    )
    pronoun_terms = _words(proposed_pronouns or "")
    pronouns = (
        proposed_pronouns
        if proposed_pronouns is not None
        and _terms_appear_in_order(pronoun_terms, evidence)
        and (
            any(
                cue in evidence
                for cue in ("my pronouns", "i use", "i go by", "pronouns are")
            )
            or (name is not None and name.casefold() in evidence)
        )
        else None
    )
    if name is None and pronouns is None:
        return None
    return SelfProfileUpdate(
        name=name,
        pronouns=pronouns,
        evidence=update.evidence,
    )


def _normalize(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("speaker embedding must have a finite non-zero norm")
    return value / norm


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return -1.0
    return float(np.dot(left, right))


@dataclass(frozen=True)
class _Template:
    id: str
    person_id: str | None
    state: str
    centroid: np.ndarray
    sample_count: int
    claimed_name: str | None
    claimed_pronouns: str | None


class SpeakerProfileStore:
    """Local biometric-template store; raw enrollment audio never reaches it."""

    def __init__(
        self,
        path: Path,
        *,
        retention_days: int = 30,
        activation_samples: int = 3,
        match_threshold: float = 0.65,
        match_margin: float = 0.03,
        cluster_threshold: float = 0.60,
    ) -> None:
        self.path = path
        self.retention = timedelta(days=retention_days)
        self.activation_samples = max(1, activation_samples)
        self.match_threshold = match_threshold
        self.match_margin = match_margin
        self.cluster_threshold = cluster_threshold
        self._lock = asyncio.Lock()

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
                CREATE TABLE IF NOT EXISTS speaker_people (
                    id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    name_key TEXT NOT NULL,
                    pronouns TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_speaker_people_name
                    ON speaker_people(name_key);

                CREATE TABLE IF NOT EXISTS speaker_templates (
                    id TEXT PRIMARY KEY,
                    person_id TEXT REFERENCES speaker_people(id) ON DELETE CASCADE,
                    state TEXT NOT NULL CHECK(state IN ('provisional', 'pending', 'active')),
                    centroid BLOB NOT NULL,
                    dimensions INTEGER NOT NULL,
                    sample_count INTEGER NOT NULL,
                    claimed_name TEXT,
                    claimed_pronouns TEXT,
                    created_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    expires_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_speaker_templates_person
                    ON speaker_templates(person_id);
                CREATE INDEX IF NOT EXISTS idx_speaker_templates_expiry
                    ON speaker_templates(expires_at);
                """
            )

    async def initialize(self) -> None:
        await asyncio.to_thread(self.initialize_sync)

    @staticmethod
    def _vector(row: sqlite3.Row) -> np.ndarray:
        vector = np.frombuffer(row["centroid"], dtype=np.float32).copy()
        if vector.size != int(row["dimensions"]):
            raise ValueError("stored speaker embedding has invalid dimensions")
        return _normalize(vector)

    def _templates(self, connection: sqlite3.Connection) -> list[_Template]:
        rows = connection.execute("SELECT * FROM speaker_templates").fetchall()
        return [
            _Template(
                id=row["id"],
                person_id=row["person_id"],
                state=row["state"],
                centroid=self._vector(row),
                sample_count=int(row["sample_count"]),
                claimed_name=row["claimed_name"],
                claimed_pronouns=row["claimed_pronouns"],
            )
            for row in rows
        ]

    @staticmethod
    def _person(connection: sqlite3.Connection, person_id: str) -> sqlite3.Row | None:
        return connection.execute(
            "SELECT * FROM speaker_people WHERE id = ?", (person_id,)
        ).fetchone()

    def _identity(
        self,
        connection: sqlite3.Connection,
        template: _Template,
        confidence: float,
    ) -> SpeakerIdentity:
        if template.person_id is None:
            return SpeakerIdentity(
                status="pending" if template.state == "pending" else "provisional",
                template_id=template.id,
                confidence=confidence,
            )
        person = self._person(connection, template.person_id)
        if person is None:
            return SpeakerIdentity(template_id=template.id, confidence=confidence)
        return SpeakerIdentity(
            status="recognized",
            template_id=template.id,
            person_id=person["id"],
            display_name=person["display_name"],
            pronouns=person["pronouns"],
            confidence=confidence,
        )

    @staticmethod
    def _updated_centroid(template: _Template, embedding: np.ndarray) -> np.ndarray:
        weight = min(template.sample_count, 20)
        return _normalize((template.centroid * weight + embedding) / (weight + 1))

    def _update_template(
        self,
        connection: sqlite3.Connection,
        template: _Template,
        embedding: np.ndarray,
        now: datetime,
    ) -> _Template:
        centroid = self._updated_centroid(template, embedding)
        sample_count = template.sample_count + 1
        expiry = None if template.person_id else (now + self.retention).isoformat()
        connection.execute(
            """
            UPDATE speaker_templates
            SET centroid = ?, dimensions = ?, sample_count = ?, last_seen_at = ?, expires_at = ?
            WHERE id = ?
            """,
            (
                centroid.tobytes(),
                int(centroid.size),
                sample_count,
                now.isoformat(),
                expiry,
                template.id,
            ),
        )
        return _Template(
            id=template.id,
            person_id=template.person_id,
            state=template.state,
            centroid=centroid,
            sample_count=sample_count,
            claimed_name=template.claimed_name,
            claimed_pronouns=template.claimed_pronouns,
        )

    def _create_template(
        self, connection: sqlite3.Connection, embedding: np.ndarray, now: datetime
    ) -> _Template:
        template = _Template(
            id=uuid4().hex,
            person_id=None,
            state="provisional",
            centroid=embedding,
            sample_count=1,
            claimed_name=None,
            claimed_pronouns=None,
        )
        connection.execute(
            """
            INSERT INTO speaker_templates (
                id, person_id, state, centroid, dimensions, sample_count,
                claimed_name, claimed_pronouns, created_at, last_seen_at, expires_at
            ) VALUES (?, NULL, 'provisional', ?, ?, 1, NULL, NULL, ?, ?, ?)
            """,
            (
                template.id,
                embedding.tobytes(),
                int(embedding.size),
                now.isoformat(),
                now.isoformat(),
                (now + self.retention).isoformat(),
            ),
        )
        return template

    @staticmethod
    def _best_active(
        templates: list[_Template], embedding: np.ndarray
    ) -> tuple[_Template | None, float, float]:
        best_by_person: dict[str, tuple[_Template, float]] = {}
        for template in templates:
            if template.person_id is None or template.state != "active":
                continue
            score = _cosine(template.centroid, embedding)
            current = best_by_person.get(template.person_id)
            if current is None or score > current[1]:
                best_by_person[template.person_id] = (template, score)
        ranked = sorted(best_by_person.values(), key=lambda item: item[1], reverse=True)
        if not ranked:
            return None, -1.0, -1.0
        runner_up = ranked[1][1] if len(ranked) > 1 else -1.0
        return ranked[0][0], ranked[0][1], runner_up

    def _activate_pending(
        self, connection: sqlite3.Connection, template: _Template, now: datetime
    ) -> _Template:
        if (
            template.state != "pending"
            or template.sample_count < self.activation_samples
            or not template.claimed_name
        ):
            return template
        key = _name_key(template.claimed_name)
        person = connection.execute(
            "SELECT * FROM speaker_people WHERE name_key = ? ORDER BY created_at LIMIT 1",
            (key,),
        ).fetchone()
        if person is None:
            person_id = uuid4().hex
            connection.execute(
                """
                INSERT INTO speaker_people (
                    id, display_name, name_key, pronouns, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    person_id,
                    template.claimed_name,
                    key,
                    template.claimed_pronouns,
                    now.isoformat(),
                    now.isoformat(),
                ),
            )
        else:
            person_id = person["id"]
            if template.claimed_pronouns and not person["pronouns"]:
                connection.execute(
                    "UPDATE speaker_people SET pronouns = ?, updated_at = ? WHERE id = ?",
                    (template.claimed_pronouns, now.isoformat(), person_id),
                )
        connection.execute(
            """
            UPDATE speaker_templates
            SET person_id = ?, state = 'active', expires_at = NULL
            WHERE id = ?
            """,
            (person_id, template.id),
        )
        return _Template(
            id=template.id,
            person_id=person_id,
            state="active",
            centroid=template.centroid,
            sample_count=template.sample_count,
            claimed_name=template.claimed_name,
            claimed_pronouns=template.claimed_pronouns,
        )

    def recognize_sync(
        self,
        embedding: np.ndarray,
        *,
        now: datetime | None = None,
        preferred_template_id: str | None = None,
        preferred_threshold: float | None = None,
    ) -> SpeakerIdentity:
        current = (now or datetime.now(UTC)).astimezone(UTC)
        value = _normalize(embedding)
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM speaker_templates WHERE person_id IS NULL AND expires_at <= ?",
                (current.isoformat(),),
            )
            templates = self._templates(connection)
            # A wake-opened conversation is overwhelmingly one person speaking
            # across several short turns. Reuse its bound template through
            # ordinary microphone/phrase drift, but fall back to global matching
            # when the embedding is genuinely far away (a speaker hand-off).
            preferred = next(
                (item for item in templates if item.id == preferred_template_id),
                None,
            )
            if preferred is not None and preferred_threshold is not None:
                preferred_score = _cosine(preferred.centroid, value)
                if preferred_score >= preferred_threshold:
                    preferred = self._update_template(
                        connection, preferred, value, current
                    )
                    preferred = self._activate_pending(
                        connection, preferred, current
                    )
                    return self._identity(connection, preferred, preferred_score)
            active, active_score, runner_up = self._best_active(templates, value)
            if (
                active is not None
                and active_score >= self.match_threshold
                and active_score - runner_up >= self.match_margin
            ):
                # Do not let marginal matches slowly pull a known template toward
                # another voice. High-confidence matches may adapt to microphones.
                if active_score >= min(0.99, self.match_threshold + self.match_margin):
                    active = self._update_template(connection, active, value, current)
                return self._identity(connection, active, active_score)

            candidates = [item for item in templates if item.person_id is None]
            candidate_scores = [(_cosine(item.centroid, value), item) for item in candidates]
            score, candidate = max(candidate_scores, default=(-1.0, None), key=lambda item: item[0])
            if candidate is not None and score >= self.cluster_threshold:
                candidate = self._update_template(connection, candidate, value, current)
                candidate = self._activate_pending(connection, candidate, current)
                return self._identity(connection, candidate, score)
            candidate = self._create_template(connection, value, current)
            return self._identity(connection, candidate, 1.0)

    async def recognize(
        self,
        embedding: np.ndarray,
        *,
        preferred_template_id: str | None = None,
        preferred_threshold: float | None = None,
    ) -> SpeakerIdentity:
        async with self._lock:
            return await asyncio.to_thread(
                self.recognize_sync,
                embedding,
                preferred_template_id=preferred_template_id,
                preferred_threshold=preferred_threshold,
            )

    def apply_disclosure_sync(
        self,
        identity: SpeakerIdentity,
        update: SelfProfileUpdate,
        transcript: str,
        *,
        now: datetime | None = None,
    ) -> SpeakerIdentity:
        validated = validated_self_profile_update(update, transcript)
        if validated is None or identity.template_id is None:
            return identity
        current = (now or datetime.now(UTC)).astimezone(UTC)
        name = validated.name
        pronouns = validated.pronouns
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM speaker_templates WHERE id = ?", (identity.template_id,)
            ).fetchone()
            if row is None:
                return identity
            template = _Template(
                id=row["id"],
                person_id=row["person_id"],
                state=row["state"],
                centroid=self._vector(row),
                sample_count=int(row["sample_count"]),
                claimed_name=row["claimed_name"],
                claimed_pronouns=row["claimed_pronouns"],
            )
            if template.person_id is not None:
                person = self._person(connection, template.person_id)
                if person is None:
                    return identity
                selected_name = name or person["display_name"]
                selected_pronouns = pronouns if pronouns is not None else person["pronouns"]
                connection.execute(
                    """
                    UPDATE speaker_people
                    SET display_name = ?, name_key = ?, pronouns = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        selected_name,
                        _name_key(selected_name),
                        selected_pronouns,
                        current.isoformat(),
                        template.person_id,
                    ),
                )
                return SpeakerIdentity(
                    status="recognized",
                    template_id=template.id,
                    person_id=template.person_id,
                    display_name=selected_name,
                    pronouns=selected_pronouns,
                    confidence=identity.confidence,
                )

            claimed_name = name or template.claimed_name
            claimed_pronouns = pronouns or template.claimed_pronouns
            connection.execute(
                """
                UPDATE speaker_templates
                SET state = 'pending', claimed_name = ?, claimed_pronouns = ?
                WHERE id = ?
                """,
                (claimed_name, claimed_pronouns, template.id),
            )
            pending = _Template(
                id=template.id,
                person_id=None,
                state="pending",
                centroid=template.centroid,
                sample_count=template.sample_count,
                claimed_name=claimed_name,
                claimed_pronouns=claimed_pronouns,
            )
            pending = self._activate_pending(connection, pending, current)
            return self._identity(connection, pending, identity.confidence or 0.0)

    async def apply_disclosure(
        self,
        identity: SpeakerIdentity,
        update: SelfProfileUpdate,
        transcript: str,
    ) -> SpeakerIdentity:
        async with self._lock:
            return await asyncio.to_thread(
                self.apply_disclosure_sync, identity, update, transcript
            )

    def list_profiles_sync(self) -> dict:
        with self._connect() as connection:
            people = connection.execute(
                "SELECT * FROM speaker_people ORDER BY display_name COLLATE NOCASE, created_at"
            ).fetchall()
            templates = connection.execute(
                """
                SELECT id, person_id, state, sample_count, claimed_name, claimed_pronouns,
                       created_at, last_seen_at, expires_at
                FROM speaker_templates ORDER BY created_at
                """
            ).fetchall()
        by_person: dict[str, list[dict]] = {}
        provisional: list[dict] = []
        for row in templates:
            payload = {
                "id": row["id"],
                "state": row["state"],
                "sampleCount": int(row["sample_count"]),
                "claimedName": row["claimed_name"],
                "claimedPronouns": row["claimed_pronouns"],
                "createdAt": row["created_at"],
                "lastSeenAt": row["last_seen_at"],
                "expiresAt": row["expires_at"],
            }
            if row["person_id"]:
                by_person.setdefault(row["person_id"], []).append(payload)
            else:
                provisional.append(payload)
        return {
            "profiles": [
                {
                    "id": row["id"],
                    "displayName": row["display_name"],
                    "pronouns": row["pronouns"],
                    "createdAt": row["created_at"],
                    "updatedAt": row["updated_at"],
                    "templates": by_person.get(row["id"], []),
                }
                for row in people
            ],
            "provisionalTemplates": provisional,
        }

    async def list_profiles(self) -> dict:
        return await asyncio.to_thread(self.list_profiles_sync)

    def update_person_sync(
        self, person_id: str, *, display_name: str | None, pronouns: str | None
    ) -> bool:
        with self._connect() as connection:
            person = self._person(connection, person_id)
            if person is None:
                return False
            name = _clean(display_name) or person["display_name"]
            selected_pronouns = _clean(pronouns) if pronouns is not None else person["pronouns"]
            connection.execute(
                """
                UPDATE speaker_people
                SET display_name = ?, name_key = ?, pronouns = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    name,
                    _name_key(name),
                    selected_pronouns,
                    datetime.now(UTC).isoformat(),
                    person_id,
                ),
            )
            return True

    async def update_person(
        self, person_id: str, *, display_name: str | None, pronouns: str | None
    ) -> bool:
        async with self._lock:
            return await asyncio.to_thread(
                self.update_person_sync,
                person_id,
                display_name=display_name,
                pronouns=pronouns,
            )

    def delete_person_sync(self, person_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM speaker_people WHERE id = ?", (person_id,))
            return cursor.rowcount > 0

    async def delete_person(self, person_id: str) -> bool:
        async with self._lock:
            return await asyncio.to_thread(self.delete_person_sync, person_id)

    def delete_template_sync(self, template_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM speaker_templates WHERE id = ?", (template_id,)
            )
            return cursor.rowcount > 0

    async def delete_template(self, template_id: str) -> bool:
        async with self._lock:
            return await asyncio.to_thread(self.delete_template_sync, template_id)

    def delete_all_templates_sync(self) -> int:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM speaker_templates")
            return max(0, cursor.rowcount)

    async def delete_all_templates(self) -> int:
        async with self._lock:
            return await asyncio.to_thread(self.delete_all_templates_sync)

    def assign_template_sync(self, template_id: str, person_id: str) -> bool:
        with self._connect() as connection:
            if self._person(connection, person_id) is None:
                return False
            cursor = connection.execute(
                """
                UPDATE speaker_templates
                SET person_id = ?, state = 'active', expires_at = NULL
                WHERE id = ?
                """,
                (person_id, template_id),
            )
            return cursor.rowcount > 0

    async def assign_template(self, template_id: str, person_id: str) -> bool:
        async with self._lock:
            return await asyncio.to_thread(
                self.assign_template_sync, template_id, person_id
            )

    async def health(self) -> dict:
        payload = await self.list_profiles()
        return {
            "ok": True,
            "profiles": len(payload["profiles"]),
            "provisionalTemplates": len(payload["provisionalTemplates"]),
            "retentionDays": self.retention.days,
            "activationSamples": self.activation_samples,
        }
