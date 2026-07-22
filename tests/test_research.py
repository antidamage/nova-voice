from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx

from nova_voice.api import create_app
from nova_voice.config import Settings
from nova_voice.domain import CapabilityToolCall, PlannedAction, ToolResult
from nova_voice.durable.models import (
    ProactiveInterventionRecord,
    ResearchRecord,
    ResearchState,
)
from nova_voice.durable.store import DurableAgentStore
from nova_voice.providers.research.provider import ResearchProvider
from nova_voice.research import ResearchManager


class FakeWeb:
    def __init__(self, *, wait: asyncio.Event | None = None) -> None:
        self.wait = wait
        self.started = asyncio.Event()

    async def execute(self, action: PlannedAction) -> ToolResult:
        self.started.set()
        if self.wait is not None:
            await self.wait.wait()
        return ToolResult(
            action_id=action.id,
            ok=True,
            code="ok",
            observed={
                "backend": "test",
                "answer": "The supported answer. Additional detail. A third point. A fourth point.",
                "results": [
                    {"title": "One", "url": "https://one.test/a", "snippet": "Evidence one"},
                    {"title": "Two", "url": "https://two.test/b", "snippet": "Evidence two"},
                ],
                "sources": ["one.test", "two.test"],
            },
            message="done",
        )


async def _store(tmp_path) -> DurableAgentStore:
    store = DurableAgentStore(tmp_path / "durable.sqlite3")
    await store.initialize()
    return store


async def _wait_for(
    manager: ResearchManager, research_id: str, state: ResearchState
) -> ResearchRecord:
    for _ in range(100):
        record = await manager.get(research_id)
        if record.status == state:
            return record
        await asyncio.sleep(0.01)
    raise AssertionError(f"research did not reach {state}")


async def test_research_completes_with_citations_uncertainty_and_announcement(tmp_path) -> None:
    store = await _store(tmp_path)
    manager = ResearchManager(store, FakeWeb())  # type: ignore[arg-type]
    created = await manager.create(owner_id="owner", query="Compare two supported facts")

    completed = await _wait_for(manager, created.id, ResearchState.COMPLETED)
    interventions = await store.list(ProactiveInterventionRecord)

    assert completed.spoken_summary == "The supported answer. Additional detail. A third point."
    assert completed.citations[:2] == ("https://one.test/a", "https://two.test/b")
    assert completed.uncertainty == "low"
    assert completed.detail["results"][0]["title"] == "One"
    assert len(interventions) == 1
    assert interventions[0].record.reason_code == "research_complete"
    await manager.close()


async def test_research_recovers_running_job_after_restart(tmp_path) -> None:
    store = await _store(tmp_path)
    await store.create(
        ResearchRecord(
            id="research:restart",
            owner_id="owner",
            query="Recover me",
            status=ResearchState.RUNNING,
        ),
        actor_id="owner",
    )
    manager = ResearchManager(store, FakeWeb())  # type: ignore[arg-type]

    await manager.start()
    completed = await _wait_for(manager, "research:restart", ResearchState.COMPLETED)

    assert completed.started_at is not None
    assert completed.backend == "test"
    await manager.close()


async def test_running_research_can_be_cancelled(tmp_path) -> None:
    store = await _store(tmp_path)
    release = asyncio.Event()
    web = FakeWeb(wait=release)
    manager = ResearchManager(store, web)  # type: ignore[arg-type]
    created = await manager.create(owner_id="owner", query="Wait for this")
    await web.started.wait()

    cancelled = await manager.cancel(created.id, actor_id="owner")
    release.set()

    assert cancelled.status == ResearchState.CANCELLED
    assert (await manager.get(created.id)).status == ResearchState.CANCELLED
    await manager.close()


async def test_research_provider_and_api_expose_dashboard_detail(tmp_path) -> None:
    store = await _store(tmp_path)
    manager = ResearchManager(store, FakeWeb())  # type: ignore[arg-type]
    provider = ResearchProvider(manager)
    result = await provider.execute(
        PlannedAction(
            id="research-start",
            order=0,
            call=CapabilityToolCall(
                provider="research",
                tool="research.start",
                arguments={"ownerId": "owner", "query": "A useful question"},
            ),
        )
    )
    research_id = result.observed["research"]["id"]
    await _wait_for(manager, research_id, ResearchState.COMPLETED)
    app = create_app(Settings(), service=SimpleNamespace(research=manager))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://voice.test"
    ) as client:
        response = await client.get(f"/v1/research/{research_id}")

    payload = response.json()["research"]
    assert response.status_code == 200
    assert payload["citations"][0] == "https://one.test/a"
    assert payload["detail"]["backend"] == "test"
    await manager.close()
