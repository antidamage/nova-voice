"""Private authenticated MemPalace service for Nova Voice on Iridium."""

from __future__ import annotations

import asyncio
import hmac
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException

from nova_voice.config import Settings
from nova_voice.memory import MemoryRecord, MemoryStatus


class MemoryPalace:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = asyncio.Lock()
        self._collection = None

    def collection(self):
        if self._collection is None:
            self.path.mkdir(parents=True, exist_ok=True)
            # This uses MemPalace's backend, not a separate vector database.
            from mempalace.palace import get_collection

            self._collection = get_collection(str(self.path), collection_name="nova_voice_memory")
        return self._collection

    @staticmethod
    def metadata(memory: MemoryRecord) -> dict[str, str | float | bool]:
        return {
            "memory_type": memory.memory_type.value,
            "owner_id": memory.owner_id or "",
            "status": memory.status.value,
            "sensitivity": memory.sensitivity.value,
            "provenance": memory.provenance,
            "source_turn_id": memory.source_turn_id or "",
            "created_at": memory.created_at.isoformat(),
            "updated_at": memory.updated_at.isoformat(),
            "expires_at": memory.expires_at.isoformat() if memory.expires_at else "",
            "pinned": memory.pinned,
            "needs_confirmation": memory.needs_confirmation,
            "record": memory.model_dump_json(),
        }

    async def put(self, memory: MemoryRecord) -> MemoryRecord:
        async with self._lock:
            await asyncio.to_thread(
                self.collection().upsert,
                documents=[memory.text],
                ids=[memory.id],
                metadatas=[self.metadata(memory)],
            )
        return memory

    async def search(self, query: str, owner_id: str, limit: int = 5) -> list[MemoryRecord]:
        async with self._lock:
            raw = await asyncio.to_thread(
                self.collection().query,
                query_texts=[query],
                n_results=max(1, min(limit, 10)),
                where={"owner_id": owner_id},
                include=["metadatas"],
            )
        results: list[MemoryRecord] = []
        now = datetime.now(UTC)
        for metadata in (raw.get("metadatas") or [[]])[0]:
            try:
                memory = MemoryRecord.model_validate_json(str(metadata["record"]))
            except (KeyError, ValueError):
                continue
            if memory.status != MemoryStatus.ACTIVE or memory.needs_confirmation:
                continue
            if memory.expires_at and memory.expires_at <= now:
                continue
            results.append(memory.model_copy(update={"accessed_at": now}))
        return results

    async def list(self, owner_id: str | None = None) -> list[MemoryRecord]:
        async with self._lock:
            raw = await asyncio.to_thread(
                self.collection().get,
                where={"owner_id": owner_id} if owner_id else None,
                include=["metadatas"],
            )
        records: list[MemoryRecord] = []
        for metadata in raw.get("metadatas") or []:
            try:
                records.append(MemoryRecord.model_validate_json(str(metadata["record"])))
            except (KeyError, ValueError):
                continue
        return sorted(records, key=lambda item: item.updated_at, reverse=True)

    async def get(self, memory_id: str) -> MemoryRecord | None:
        async with self._lock:
            raw = await asyncio.to_thread(
                self.collection().get, ids=[memory_id], include=["metadatas"]
            )
        values = raw.get("metadatas") or []
        if not values:
            return None
        try:
            return MemoryRecord.model_validate_json(str(values[0]["record"]))
        except (KeyError, ValueError):
            return None

    async def backup(self) -> Path:
        async with self._lock:
            # A point-in-time filesystem copy is recoverable without exposing
            # data on the network. The follow-up count check proves the copied
            # MemPalace index can be opened before we report success.
            destination = (
                self.path.parent
                / "mempalace-backups"
                / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            )
            await asyncio.to_thread(shutil.copytree, self.path, destination)
            from mempalace.palace import get_collection

            await asyncio.to_thread(
                get_collection(str(destination), "nova_voice_memory", False).count
            )
        return destination

    async def consolidate(self) -> dict[str, int]:
        """Merge exact duplicate facts; never invent or auto-apply procedures."""

        records = await self.list()
        groups: dict[tuple[str | None, str, str], list[MemoryRecord]] = {}
        for record in records:
            if record.status != MemoryStatus.ACTIVE:
                continue
            key = (
                record.owner_id,
                record.memory_type.value,
                " ".join(record.text.casefold().split()),
            )
            groups.setdefault(key, []).append(record)
        merged = 0
        for duplicates in groups.values():
            if len(duplicates) < 2:
                continue
            duplicates.sort(key=lambda item: (item.pinned, item.updated_at), reverse=True)
            winner = duplicates[0]
            for duplicate in duplicates[1:]:
                await self.put(
                    duplicate.model_copy(
                        update={
                            "status": MemoryStatus.SUPERSEDED,
                            "supersedes": winner.id,
                            "updated_at": datetime.now(UTC),
                        }
                    )
                )
                merged += 1
        return {"merged": merged, "conflicts": 0, "proceduresProposed": 0}


def create_app(settings: Settings | None = None) -> FastAPI:
    selected = settings or Settings()
    expected_token = selected.mempalace_token or ""
    palace = MemoryPalace(selected.mempalace_data_path)
    app = FastAPI(title="Nova Voice MemPalace", version="1")

    def require_token(authorization: Annotated[str | None, Header()] = None) -> None:
        supplied = authorization.removeprefix("Bearer ") if authorization else ""
        if not expected_token or not hmac.compare_digest(supplied, expected_token):
            raise HTTPException(status_code=401, detail="Unauthorized")

    @app.get("/health")
    async def health(_: None = Depends(require_token)) -> dict:
        count = await asyncio.to_thread(palace.collection().count)
        return {"ok": True, "backend": "mempalace", "memories": count}

    @app.post("/v1/memories")
    async def create(memory: MemoryRecord, _: None = Depends(require_token)) -> dict:
        return {"memory": (await palace.put(memory)).model_dump(mode="json")}

    @app.post("/v1/search")
    async def search(payload: dict, _: None = Depends(require_token)) -> dict:
        owner_id, query = str(payload.get("owner_id") or ""), str(payload.get("query") or "")
        if not owner_id or not query.strip():
            raise HTTPException(status_code=422, detail="owner_id and query are required")
        memories = await palace.search(query, owner_id, int(payload.get("limit", 5)))
        return {"memories": [item.model_dump(mode="json") for item in memories]}

    @app.get("/v1/memories")
    async def list_memories(owner_id: str | None = None, _: None = Depends(require_token)) -> dict:
        memories = await palace.list(owner_id)
        return {"memories": [item.model_dump(mode="json") for item in memories]}

    @app.patch("/v1/memories/{memory_id}")
    async def update(memory_id: str, payload: dict, _: None = Depends(require_token)) -> dict:
        existing = await palace.get(memory_id)
        if existing is None:
            raise HTTPException(status_code=404, detail="Unknown memory")
        allowed = {
            "text",
            "expires_at",
            "pinned",
            "status",
            "supersedes",
            "reviewed_at",
            "needs_confirmation",
        }
        changes = {key: value for key, value in payload.items() if key in allowed}
        if "needs_confirmation" in changes and changes["needs_confirmation"] is False:
            changes["reviewed_at"] = datetime.now(UTC)
        updated = existing.model_copy(update={**changes, "updated_at": datetime.now(UTC)})
        return {"memory": (await palace.put(updated)).model_dump(mode="json")}

    @app.delete("/v1/memories/{memory_id}")
    async def forget(memory_id: str, _: None = Depends(require_token)) -> dict:
        existing = await palace.get(memory_id)
        if existing is None:
            raise HTTPException(status_code=404, detail="Unknown memory")
        deleted = existing.model_copy(
            update={"status": MemoryStatus.DELETED, "updated_at": datetime.now(UTC)}
        )
        await palace.put(deleted)
        return {"ok": True, "memory": deleted.model_dump(mode="json")}

    @app.get("/v1/export")
    async def export(owner_id: str | None = None, _: None = Depends(require_token)) -> dict:
        memories = await palace.list(owner_id)
        return {"version": 1, "memories": [item.model_dump(mode="json") for item in memories]}

    @app.post("/v1/backup")
    async def backup(_: None = Depends(require_token)) -> dict:
        path = await palace.backup()
        return {"ok": True, "backup": path.name, "restoreVerified": True}

    @app.post("/v1/consolidate")
    async def consolidate(_: None = Depends(require_token)) -> dict:
        return {"ok": True, **(await palace.consolidate())}

    return app


app = create_app()


def main() -> None:
    import uvicorn

    port = int(os.environ.get("MEMPALACE_PORT", "8094"))
    uvicorn.run(app, host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
