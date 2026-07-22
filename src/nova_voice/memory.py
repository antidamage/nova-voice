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

    async def search(self, query: str, *, owner_id: str | None) -> list[MemoryRecord]:
        if not self._enabled or not owner_id:
            return []
        try:
            response = await self._client.post(
                "/v1/search", json={"query": query, "owner_id": owner_id}
            )
            response.raise_for_status()
            return [
                MemoryRecord.model_validate(item) for item in response.json().get("memories", [])
            ]
        except (httpx.HTTPError, ValueError):
            return []

    async def create(self, memory: MemoryRecord) -> MemoryRecord | None:
        if not self._enabled:
            return None
        try:
            response = await self._client.post("/v1/memories", json=memory.model_dump(mode="json"))
            response.raise_for_status()
            return MemoryRecord.model_validate(response.json()["memory"])
        except (httpx.HTTPError, ValueError):
            return None

    async def list(self, *, owner_id: str | None = None) -> list[MemoryRecord]:
        if not self._enabled:
            return []
        try:
            response = await self._client.get(
                "/v1/memories", params={"owner_id": owner_id} if owner_id else None
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
            response = await self._client.request(method, path, json=payload)
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
