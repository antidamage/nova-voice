"""Selective, user-controlled conversational memory contract for Nova Voice.

The Voice service owns admission and use policy.  MemPalace owns persistence
and semantic retrieval behind the small private HTTP API in ``memory_server``.
Routine device commands and transient device state are intentionally excluded.
"""

from __future__ import annotations

import re
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
    explicit = bool(re.search(r"\b(?:remember|don't forget|please note)\b", text, re.I))
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
