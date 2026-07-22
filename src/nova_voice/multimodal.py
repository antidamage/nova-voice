from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MultimodalInputKind(StrEnum):
    DASHBOARD_SCREEN = "dashboard_screen"
    USER_IMAGE = "user_image"
    CAMERA_SNAPSHOT = "camera_snapshot"
    DOCUMENT = "document"
    DEVICE_DIAGRAM = "device_diagram"


class MultimodalProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_uri: str = Field(min_length=1, max_length=1000)
    content_revision: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    acquired_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    supplied_by: str = Field(min_length=1, max_length=160)
    device_id: str | None = Field(default=None, max_length=160)
    room_id: str | None = Field(default=None, max_length=160)


class MultimodalPermission(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    actor_id: str = Field(min_length=1, max_length=160)
    purpose: str = Field(min_length=1, max_length=240)
    audience: tuple[str, ...] = Field(min_length=1, max_length=16)
    explicit_user_share: bool = False
    grant_id: str | None = Field(default=None, max_length=160)
    retain_until: datetime | None = None
    expires_at: datetime | None = None


class MultimodalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str = Field(min_length=1, max_length=160)
    kind: MultimodalInputKind
    source_uri: str = Field(min_length=1, max_length=1000)
    permission: MultimodalPermission
    expected_mime_types: tuple[str, ...] = ()
    max_bytes: int = Field(default=10_000_000, ge=1, le=50_000_000)

    @model_validator(mode="after")
    def enforce_acquisition_authority(self) -> MultimodalRequest:
        user_supplied = {
            MultimodalInputKind.DASHBOARD_SCREEN,
            MultimodalInputKind.USER_IMAGE,
            MultimodalInputKind.DOCUMENT,
            MultimodalInputKind.DEVICE_DIAGRAM,
        }
        if self.kind in user_supplied and not self.permission.explicit_user_share:
            raise ValueError("user-supplied multimodal input requires an explicit share")
        if (
            self.kind == MultimodalInputKind.CAMERA_SNAPSHOT
            and not self.permission.grant_id
        ):
            raise ValueError("camera snapshots require a permission grant")
        return self


class MultimodalAsset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    asset_id: str = Field(min_length=1, max_length=160)
    kind: MultimodalInputKind
    mime_type: str = Field(min_length=1, max_length=160)
    byte_count: int = Field(ge=1)
    local_uri: str = Field(min_length=1, max_length=1000)
    provenance: MultimodalProvenance
    permission: MultimodalPermission


class MultimodalObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    asset_id: str
    summary: str = Field(min_length=1, max_length=4000)
    claims: tuple[str, ...] = Field(default=(), max_length=32)
    confidence: float = Field(ge=0, le=1)
    source_revision: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    memory_eligible: bool = False
    sensitivity: Literal["normal", "sensitive", "restricted"] = "normal"


class MultimodalProvider(ABC):
    """Replaceable acquisition/observation boundary for visual and document input."""

    @abstractmethod
    async def acquire(self, request: MultimodalRequest) -> MultimodalAsset: ...

    @abstractmethod
    async def observe(
        self, asset: MultimodalAsset, *, question: str
    ) -> MultimodalObservation: ...

    @abstractmethod
    async def health(self) -> dict: ...

    async def close(self) -> None:
        return None
