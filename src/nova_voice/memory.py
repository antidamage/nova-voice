"""Selective, user-controlled conversational memory contract for Nova Voice.

The Voice service owns admission and use policy.  MemPalace owns persistence
and semantic retrieval behind the small private HTTP API in ``memory_server``.
Routine device commands and transient device state are intentionally excluded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

import httpx
from pydantic import BaseModel, Field


class MemoryType(StrEnum):
    PROFILE = "profile"
    PREFERENCE = "preference"
    EPISODIC = "episodic"
    COMMITMENT = "commitment"
    RELATIONSHIP = "relationship"
    PROCEDURAL = "procedural"
    HOUSEHOLD_FACT = "household_fact"


class MemorySensitivity(StrEnum):
    NORMAL = "normal"
    SENSITIVE = "sensitive"
    RESTRICTED = "restricted"


class MemoryStatus(StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    DELETED = "deleted"


class MemoryOperation(StrEnum):
    RETRIEVE = "retrieve"
    WRITE = "write"
    CALLBACK = "callback"
    CORRECT = "correct"
    EXPORT = "export"
    FORGET = "forget"


class MemoryRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    text: str = Field(min_length=1, max_length=2000)
    memory_type: MemoryType
    owner_id: str | None = None
    audience: list[str] = Field(default_factory=list)
    provenance: str = Field(min_length=1, max_length=160)
    source_turn_id: str | None = None
    confidence: float = Field(default=0.8, ge=0, le=1)
    sensitivity: MemorySensitivity = MemorySensitivity.NORMAL
    status: MemoryStatus = MemoryStatus.ACTIVE
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    reviewed_at: datetime | None = None
    accessed_at: datetime | None = None
    expires_at: datetime | None = None
    supersedes: str | None = None
    pinned: bool = False
    needs_confirmation: bool = False


@dataclass(frozen=True)
class MemoryAccessContext:
    actor_id: str | None
    recognized: bool = False
    participant_ids: tuple[str, ...] = ()
    administrative: bool = False

    def model_payload(self) -> dict[str, Any]:
        return {
            "actor_id": self.actor_id,
            "recognized": self.recognized,
            "participant_ids": list(self.participant_ids),
            "administrative": self.administrative,
        }


class MemoryAudiencePolicy:
    """One fail-closed audience policy for every MemPalace operation."""

    @staticmethod
    def _audience_ids(memory: MemoryRecord) -> set[str]:
        return {
            item.removeprefix("person:") for item in memory.audience if item and item != "household"
        }

    def can_access(
        self,
        memory: MemoryRecord,
        context: MemoryAccessContext,
        operation: MemoryOperation,
    ) -> bool:
        if context.administrative:
            return True
        actor = context.actor_id
        if not context.recognized or not actor:
            return False
        if operation in {MemoryOperation.CORRECT, MemoryOperation.FORGET}:
            return memory.owner_id == actor
        return (
            memory.owner_id == actor
            or "household" in memory.audience
            or actor in self._audience_ids(memory)
        )

    def can_create(self, memory: MemoryRecord, context: MemoryAccessContext) -> bool:
        if context.administrative:
            return True
        actor = context.actor_id
        if not context.recognized or not actor or memory.owner_id != actor:
            return False
        permitted = {actor, *context.participant_ids}
        audience_ids = self._audience_ids(memory)
        if not audience_ids.issubset(permitted):
            return False
        if memory.sensitivity != MemorySensitivity.NORMAL:
            return "household" not in memory.audience and audience_ids.issubset({actor})
        return True

    def filter(
        self,
        memories: list[MemoryRecord],
        context: MemoryAccessContext,
        operation: MemoryOperation,
    ) -> list[MemoryRecord]:
        return [item for item in memories if self.can_access(item, context, operation)]


class MemoryCandidate(BaseModel):
    text: str
    memory_type: MemoryType
    sensitivity: MemorySensitivity = MemorySensitivity.NORMAL
    needs_confirmation: bool = False


class MemoryIntentKind(StrEnum):
    NONE = "none"
    SAVE = "save"
    QUERY_ALL = "query_all"
    QUERY_TOPIC = "query_topic"
    CONTROL = "control"


@dataclass(frozen=True)
class MemoryIntent:
    kind: MemoryIntentKind
    query: str | None = None


_MEMORY_CONTROL = re.compile(
    r"^(?:please )?(?:pin|forget)\s+.+$|"
    r"^(?:please )?correct memory .+? to .+$|"
    r"^(?:please )?expire memory .+? in \d{1,3} days?$",
    re.I,
)
_MEMORY_SAVE = re.compile(
    r"^(?:please\s+)?(?:remember|save|note)(?:\s+(?:down|that))?\s+(.+)$|"
    r"^(?:please\s+)?(?:don't|do not)\s+forget(?:\s+that)?\s+(.+)$",
    re.I,
)
_MEMORY_QUERY_ALL = re.compile(
    r"\b(?:what\s+(?:do|did)\s+you\s+remember(?:\s+about\s+me)?|"
    r"what\s+have\s+you\s+saved(?:\s+about\s+me)?|"
    r"(?:show|list|check)(?:\s+me)?\s+(?:your|my|the)?\s*(?:saved\s+)?memories|"
    r"(?:show|list|check)\s+(?:your|my|the)\s+memory|"
    r"what(?:'s|\s+is)\s+in\s+(?:your|my|the)\s+memory)\b",
    re.I,
)
_MEMORY_QUERY_TOPIC = re.compile(
    r"\b(?:do\s+you\s+remember|"
    r"what\s+(?:do|did|have)\s+you\s+(?:remember(?:ed)?|save(?:d)?)\s+about|"
    r"what(?:'s|\s+is)\s+my\s+saved)\s+(.+)$",
    re.I,
)
_MEMORY_WORD = re.compile(r"\b(?:remember|remembered|memory|memories|saved)\b", re.I)


def classify_memory_intent(transcript: str) -> MemoryIntent:
    """Classify explicit MemPalace operations before general interpretation."""

    text = " ".join(transcript.split()).strip()
    if not text:
        return MemoryIntent(MemoryIntentKind.NONE)
    if _MEMORY_CONTROL.match(text):
        return MemoryIntent(MemoryIntentKind.CONTROL, text)
    save = _MEMORY_SAVE.match(text)
    if save:
        topic = next((group for group in save.groups() if group), text)
        return MemoryIntent(MemoryIntentKind.SAVE, topic.strip())
    if _MEMORY_QUERY_ALL.fullmatch(text.rstrip(" ?.!")):
        return MemoryIntent(MemoryIntentKind.QUERY_ALL)
    topic = _MEMORY_QUERY_TOPIC.search(text)
    if topic:
        return MemoryIntent(MemoryIntentKind.QUERY_TOPIC, topic.group(1).strip(" ?."))
    if _MEMORY_WORD.search(text):
        return MemoryIntent(MemoryIntentKind.QUERY_TOPIC, text)
    return MemoryIntent(MemoryIntentKind.NONE)


_ROUTINE = re.compile(r"\b(?:turn|switch|set|dim|brighten|lights?|thermostat|volume)\b", re.I)
_TRANSIENT = re.compile(r"\b(?:is|were?)\s+(?:on|off|open|closed)\b", re.I)
_SENSITIVE = re.compile(
    r"\b(?:health|medical|diagnos|medication|pregnan|bank|finance|income|password|security|"
    r"address|phone number|third party|someone else)\b",
    re.I,
)


def salient_memory_candidate(transcript: str) -> MemoryCandidate | None:
    """Deterministic, deliberately conservative first-pass memory admission.

    Explicit requests and obvious longer-lived preferences/commitments are
    eligible.  The model is never allowed to silently convert casual ambient
    speech or a routine automation command into a durable fact.
    """

    text = " ".join(transcript.split()).strip()
    if not text or (_ROUTINE.search(text) and not re.search(r"\bremember\b", text, re.I)):
        return None
    if _TRANSIENT.search(text) and not re.search(r"\bremember\b", text, re.I):
        return None
    lowered = text.casefold()
    explicit = classify_memory_intent(text).kind == MemoryIntentKind.SAVE
    if "i prefer" in lowered or "i like" in lowered or "my favorite" in lowered:
        kind = MemoryType.PREFERENCE
    elif re.search(r"\b(?:i(?:'ll| will)|we(?:'ll| will)|promise|need to)\b", text, re.I):
        kind = MemoryType.COMMITMENT
    elif re.search(r"\b(?:my (?:partner|child|mother|father)|relationship)\b", text, re.I):
        kind = MemoryType.RELATIONSHIP
    elif explicit:
        kind = MemoryType.EPISODIC
    else:
        return None
    sensitive = bool(_SENSITIVE.search(text))
    return MemoryCandidate(
        text=text,
        memory_type=kind,
        sensitivity=MemorySensitivity.SENSITIVE if sensitive else MemorySensitivity.NORMAL,
        needs_confirmation=sensitive,
    )


class MemPalaceClient:
    def __init__(self, base_url: str, token: str | None, *, timeout_seconds: float = 0.6) -> None:
        self._enabled = bool(token)
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
            headers={"Authorization": f"Bearer {token}"} if token else {},
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def search(
        self,
        query: str,
        *,
        owner_id: str | None = None,
        access: MemoryAccessContext | None = None,
        operation: MemoryOperation = MemoryOperation.RETRIEVE,
    ) -> list[MemoryRecord]:
        context = access or MemoryAccessContext(actor_id=owner_id, recognized=bool(owner_id))
        if not self._enabled or (not context.actor_id and not context.administrative):
            return []
        try:
            response = await self._client.post(
                "/v1/search",
                json={"query": query, "operation": operation.value, **context.model_payload()},
            )
            response.raise_for_status()
            return [
                MemoryRecord.model_validate(item) for item in response.json().get("memories", [])
            ]
        except (httpx.HTTPError, ValueError):
            return []

    async def create(
        self, memory: MemoryRecord, *, access: MemoryAccessContext | None = None
    ) -> MemoryRecord | None:
        if not self._enabled:
            return None
        try:
            context = access or MemoryAccessContext(
                actor_id=memory.owner_id, recognized=bool(memory.owner_id)
            )
            response = await self._client.post(
                "/v1/memories",
                json={"memory": memory.model_dump(mode="json"), "access": context.model_payload()},
            )
            response.raise_for_status()
            return MemoryRecord.model_validate(response.json()["memory"])
        except (httpx.HTTPError, ValueError):
            return None

    async def list(
        self,
        *,
        owner_id: str | None = None,
        access: MemoryAccessContext | None = None,
        operation: MemoryOperation = MemoryOperation.RETRIEVE,
    ) -> list[MemoryRecord]:
        if not self._enabled:
            return []
        try:
            context = access or MemoryAccessContext(actor_id=owner_id, recognized=bool(owner_id))
            response = await self._client.get(
                "/v1/memories",
                params={
                    "actor_id": context.actor_id or "",
                    "recognized": str(context.recognized).lower(),
                    "administrative": str(context.administrative).lower(),
                    "participant_ids": ",".join(context.participant_ids),
                    "operation": operation.value,
                },
            )
            response.raise_for_status()
            return [
                MemoryRecord.model_validate(item) for item in response.json().get("memories", [])
            ]
        except (httpx.HTTPError, ValueError):
            return []

    async def request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        if not self._enabled:
            return None
        try:
            response = await self._client.request(
                method,
                path,
                params=payload if method.upper() == "GET" else None,
                json=None if method.upper() == "GET" else payload,
            )
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError):
            return None

    async def health(self) -> dict[str, Any]:
        if not self._enabled:
            return {"ok": True, "enabled": False}
        try:
            response = await self._client.get("/health")
            response.raise_for_status()
            return {**response.json(), "enabled": True}
        except httpx.HTTPError:
            return {"ok": False, "enabled": True}

    async def close(self) -> None:
        await self._client.aclose()
