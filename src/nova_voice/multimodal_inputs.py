from __future__ import annotations

import asyncio
import hashlib
import re
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import httpx

from nova_voice.multimodal import (
    MultimodalAsset,
    MultimodalInputKind,
    MultimodalObservation,
    MultimodalProvenance,
    MultimodalProvider,
    MultimodalRequest,
)


class LocalMultimodalInputProvider(MultimodalProvider):
    """Bounded local store for explicit shares and granted dashboard snapshots."""

    def __init__(self, root: Path, *, dashboard_base_url: str, timeout_seconds: float = 10) -> None:
        self.root = root
        self.dashboard_base_url = dashboard_base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _revision(content: bytes) -> str:
        return f"sha256:{hashlib.sha256(content).hexdigest()}"

    @staticmethod
    def _extension(mime_type: str) -> str:
        return {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
            "application/pdf": ".pdf",
            "application/json": ".json",
            "text/plain": ".txt",
            "video/mp2t": ".ts",
        }.get(mime_type, ".bin")

    async def _persist(
        self, request: MultimodalRequest, content: bytes, mime_type: str
    ) -> MultimodalAsset:
        if not content or len(content) > request.max_bytes:
            raise ValueError("multimodal input is empty or exceeds its byte limit")
        if request.expected_mime_types and mime_type not in request.expected_mime_types:
            raise ValueError("multimodal input MIME type was not permitted")
        self.root.mkdir(parents=True, exist_ok=True)
        await self.purge_expired()
        asset_id = f"asset-{uuid4().hex}"
        path = self.root / f"{asset_id}{self._extension(mime_type)}"
        await asyncio.to_thread(path.write_bytes, content)
        asset = MultimodalAsset(
            asset_id=asset_id,
            kind=request.kind,
            mime_type=mime_type,
            byte_count=len(content),
            local_uri=str(path.resolve()),
            provenance=MultimodalProvenance(
                source_uri=request.source_uri,
                content_revision=self._revision(content),
                supplied_by=request.permission.actor_id,
                device_id=(
                    request.source_uri.split("/", 3)[2]
                    if request.source_uri.startswith("camera://")
                    else None
                ),
            ),
            permission=request.permission,
        )
        await asyncio.to_thread(
            path.with_suffix(path.suffix + ".json").write_text,
            asset.model_dump_json(indent=2),
            "utf-8",
        )
        return asset

    async def purge_expired(self, *, now: datetime | None = None) -> int:
        current = (now or datetime.now(UTC)).astimezone(UTC)
        if not self.root.exists():
            return 0
        removed = 0
        for metadata_path in self.root.glob("asset-*.json"):
            try:
                asset = MultimodalAsset.model_validate_json(
                    await asyncio.to_thread(metadata_path.read_text, encoding="utf-8")
                )
            except (OSError, ValueError):
                continue
            retain_until = asset.permission.retain_until
            if retain_until is None or retain_until > current:
                continue
            content_path = Path(asset.local_uri)
            for path in (content_path, metadata_path):
                try:
                    await asyncio.to_thread(path.unlink, missing_ok=True)
                except OSError:
                    pass
            removed += 1
        return removed

    async def store_share(
        self, request: MultimodalRequest, content: bytes, mime_type: str
    ) -> MultimodalAsset:
        if request.kind == MultimodalInputKind.CAMERA_SNAPSHOT:
            raise ValueError("camera content must use granted acquisition")
        return await self._persist(request, content, mime_type)

    async def acquire(self, request: MultimodalRequest) -> MultimodalAsset:
        match = re.fullmatch(
            r"camera://([A-Za-z0-9_-]{1,64})/snapshots/([A-Za-z0-9_-]{1,120})",
            request.source_uri,
        )
        if request.kind != MultimodalInputKind.CAMERA_SNAPSHOT or match is None:
            raise ValueError("unsupported multimodal acquisition source")
        camera_id, snapshot_id = match.groups()
        url = (
            f"{self.dashboard_base_url}/api/camera/{camera_id}/snapshots/{snapshot_id}"
        )
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.get(url)
            response.raise_for_status()
        mime_type = response.headers.get(
            "content-type", "application/octet-stream"
        ).split(";", 1)[0]
        return await self._persist(request, response.content, mime_type)

    async def observe(
        self, asset: MultimodalAsset, *, question: str
    ) -> MultimodalObservation:
        path = Path(asset.local_uri)
        summary = f"{asset.kind.value} acquired for: {question.strip()}"
        claims: tuple[str, ...] = ()
        confidence = 0.0
        if asset.mime_type in {"text/plain", "application/json"}:
            content = await asyncio.to_thread(path.read_text, encoding="utf-8", errors="replace")
            excerpt = " ".join(content.split())[:3000]
            summary = excerpt or summary
            claims = (excerpt,) if excerpt else ()
            confidence = 1.0
        return MultimodalObservation(
            asset_id=asset.asset_id,
            summary=summary,
            claims=claims,
            confidence=confidence,
            source_revision=asset.provenance.content_revision,
        )

    async def health(self) -> dict:
        try:
            await asyncio.to_thread(self.root.mkdir, parents=True, exist_ok=True)
            expired_removed = await self.purge_expired()
            return {
                "ok": True,
                "root": str(self.root),
                "expiredRemoved": expired_removed,
            }
        except OSError as error:
            return {"ok": False, "error": type(error).__name__}
