from __future__ import annotations

import asyncio
import logging
import re
import sqlite3
from typing import Any, cast
from uuid import uuid4

from nova_voice.domain import CapabilityToolCall, PlannedAction
from nova_voice.durable.models import (
    ProactiveInterventionRecord,
    ProactiveInterventionState,
    ResearchRecord,
    ResearchState,
    utc_now,
)
from nova_voice.durable.store import ConcurrentRecordUpdate, DurableAgentStore
from nova_voice.providers.web.provider import WebProvider

logger = logging.getLogger(__name__)


def _citations(observed: dict[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    for result in observed.get("results") or []:
        if isinstance(result, dict) and isinstance(result.get("url"), str):
            values.append(result["url"])
    for source in observed.get("sources") or []:
        if isinstance(source, str):
            values.append(source)
    return tuple(dict.fromkeys(value for value in values if value))[:12]


def _spoken_summary(query: str, observed: dict[str, Any]) -> str:
    answer = str(observed.get("answer") or "").strip()
    if not answer:
        results = observed.get("results") or []
        if results and isinstance(results[0], dict):
            answer = str(results[0].get("snippet") or results[0].get("title") or "").strip()
    if not answer:
        return (
            f"I finished researching {query}, but the available sources did not support an answer."
        )
    sentences = re.split(r"(?<=[.!?])\s+", answer)
    concise = " ".join(sentences[:3]).strip()
    return concise[:2000]


class ResearchManager:
    """Durable, restart-safe research kept off the latency-critical speech path."""

    def __init__(self, store: DurableAgentStore, web: WebProvider, *, concurrency: int = 2) -> None:
        self.store = store
        self.web = web
        self._semaphore = asyncio.Semaphore(max(1, concurrency))
        self._tasks: dict[str, asyncio.Task[None]] = {}

    async def start(self) -> None:
        for item in await self.store.list(ResearchRecord):
            record = cast(ResearchRecord, item.record)
            if record.status in {ResearchState.QUEUED, ResearchState.RUNNING}:
                if record.status == ResearchState.RUNNING:
                    record = await self._save(
                        record.model_copy(
                            update={"status": ResearchState.QUEUED, "updated_at": utc_now()}
                        ),
                        actor_id="research-recovery",
                    )
                self._schedule(record.id)

    async def create(self, *, owner_id: str, query: str) -> ResearchRecord:
        record = ResearchRecord(id=f"research:{uuid4()}", owner_id=owner_id, query=query.strip())
        stored = await self.store.create(record, actor_id=owner_id)
        created = cast(ResearchRecord, stored.record)
        self._schedule(created.id)
        return created

    async def list(self) -> tuple[ResearchRecord, ...]:
        return tuple(
            cast(ResearchRecord, item.record) for item in await self.store.list(ResearchRecord)
        )

    async def get(self, research_id: str) -> ResearchRecord:
        stored = await self.store.get(ResearchRecord, research_id)
        if stored is None:
            raise KeyError(research_id)
        return cast(ResearchRecord, stored.record)

    async def cancel(self, research_id: str, *, actor_id: str) -> ResearchRecord:
        record = await self.get(research_id)
        if record.status in {ResearchState.COMPLETED, ResearchState.FAILED}:
            raise ValueError("completed research cannot be cancelled")
        task = self._tasks.get(research_id)
        if task is not None:
            task.cancel()
        return await self._save(
            record.model_copy(
                update={
                    "status": ResearchState.CANCELLED,
                    "completed_at": utc_now(),
                    "updated_at": utc_now(),
                }
            ),
            actor_id=actor_id,
        )

    def _schedule(self, research_id: str) -> None:
        if research_id not in self._tasks or self._tasks[research_id].done():
            self._tasks[research_id] = asyncio.create_task(
                self._run(research_id), name=f"nova-research:{research_id}"
            )

    async def _save(self, record: ResearchRecord, *, actor_id: str) -> ResearchRecord:
        stored = await self.store.get(ResearchRecord, record.id)
        if stored is None:
            raise KeyError(record.id)
        saved = await self.store.save(record, expected_revision=stored.revision, actor_id=actor_id)
        return cast(ResearchRecord, saved.record)

    async def _run(self, research_id: str) -> None:
        try:
            async with self._semaphore:
                record = await self.get(research_id)
                if record.status == ResearchState.CANCELLED:
                    return
                record = await self._save(
                    record.model_copy(
                        update={
                            "status": ResearchState.RUNNING,
                            "started_at": utc_now(),
                            "updated_at": utc_now(),
                        }
                    ),
                    actor_id="research-worker",
                )
                result = await self.web.execute(
                    PlannedAction(
                        id=f"{research_id}:web",
                        order=0,
                        call=CapabilityToolCall(
                            provider="web", tool="web.ask", arguments={"query": record.query}
                        ),
                    )
                )
                latest = await self.get(research_id)
                if latest.status == ResearchState.CANCELLED:
                    return
                if not result.ok:
                    await self._save(
                        latest.model_copy(
                            update={
                                "status": ResearchState.FAILED,
                                "error": result.message,
                                "completed_at": utc_now(),
                                "updated_at": utc_now(),
                            }
                        ),
                        actor_id="research-worker",
                    )
                    return
                observed = dict(result.observed or {})
                citations = _citations(observed)
                summary = _spoken_summary(record.query, observed)
                uncertainty = (
                    "low"
                    if len(citations) >= 2 and observed.get("answer")
                    else "medium"
                    if citations
                    else "high"
                )
                intervention = ProactiveInterventionRecord(
                    id=f"intervention:{research_id}",
                    reason_code="research_complete",
                    reason_detail=summary,
                    channel="voice",
                    status=ProactiveInterventionState.PROPOSED,
                    deduplication_key=f"research:{research_id}",
                )
                try:
                    await self.store.create(intervention, actor_id="research-worker")
                except sqlite3.IntegrityError:
                    # The deterministic id makes restart completion announcements idempotent.
                    if await self.store.get(ProactiveInterventionRecord, intervention.id) is None:
                        raise
                await self._save(
                    latest.model_copy(
                        update={
                            "status": ResearchState.COMPLETED,
                            "spoken_summary": summary,
                            "detail": observed,
                            "citations": citations,
                            "uncertainty": uncertainty,
                            "backend": str(observed.get("backend") or "unknown"),
                            "completed_at": utc_now(),
                            "updated_at": utc_now(),
                        }
                    ),
                    actor_id="research-worker",
                )
        except asyncio.CancelledError:
            raise
        except (ConcurrentRecordUpdate, KeyError):
            return
        except Exception as error:
            logger.exception("background research failed id=%s", research_id)
            try:
                latest = await self.get(research_id)
                if latest.status not in {
                    ResearchState.CANCELLED,
                    ResearchState.COMPLETED,
                    ResearchState.FAILED,
                }:
                    await self._save(
                        latest.model_copy(
                            update={
                                "status": ResearchState.FAILED,
                                "error": str(error)[:1000],
                                "completed_at": utc_now(),
                                "updated_at": utc_now(),
                            }
                        ),
                        actor_id="research-worker",
                    )
            except (ConcurrentRecordUpdate, KeyError):
                pass
        finally:
            self._tasks.pop(research_id, None)

    async def close(self) -> None:
        tasks = tuple(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
