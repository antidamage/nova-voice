from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from nova_voice.durable.models import utc_now


class OptimizerKind(StrEnum):
    RESEARCH = "research"
    AUTOMATION = "automation"
    MEMORY = "memory"
    PLAN = "plan"


class OptimizerProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1, max_length=160)
    kind: OptimizerKind
    input_revision: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    objective: str = Field(min_length=1, max_length=1000)
    payload: dict = Field(default_factory=dict)
    submitted_at: datetime = Field(default_factory=utc_now)


class OptimizerEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    proposal_id: str
    kind: OptimizerKind
    input_revision: str
    accepted_for_review: bool
    score: float = Field(ge=0, le=1)
    issues: tuple[str, ...] = ()
    recommendation: dict = Field(default_factory=dict)
    applied_changes: int = 0
    completed_at: datetime = Field(default_factory=utc_now)


Evaluator = Callable[[OptimizerProposal], Awaitable[OptimizerEvaluation]]


async def structural_evaluator(proposal: OptimizerProposal) -> OptimizerEvaluation:
    issues: list[str] = []
    if not proposal.payload:
        issues.append("empty_payload")
    if any(key.casefold() in {"apply", "execute", "send", "activate"} for key in proposal.payload):
        issues.append("side_effect_request")
    return OptimizerEvaluation(
        proposal_id=proposal.id,
        kind=proposal.kind,
        input_revision=proposal.input_revision,
        accepted_for_review=not issues,
        score=1.0 if not issues else 0.0,
        issues=tuple(issues),
        recommendation=dict(proposal.payload) if not issues else {},
    )


class OfflineOptimizerPool:
    """Bounded recommendation-only workers, isolated from foreground execution."""

    def __init__(
        self,
        evaluator: Evaluator = structural_evaluator,
        *,
        queue_size: int = 32,
        result_limit: int = 100,
    ) -> None:
        self.evaluator = evaluator
        self.queues = {
            kind: asyncio.Queue[OptimizerProposal](maxsize=max(1, queue_size))
            for kind in OptimizerKind
        }
        self.result_limit = max(1, result_limit)
        self.results: dict[str, OptimizerEvaluation] = {}
        self._tasks: list[asyncio.Task] = []
        self._stopping = asyncio.Event()

    def start(self) -> None:
        if self._tasks:
            return
        self._stopping.clear()
        self._tasks = [
            asyncio.create_task(self._run(kind), name=f"optimizer:{kind.value}")
            for kind in OptimizerKind
        ]

    async def submit(self, proposal: OptimizerProposal) -> None:
        self.queues[proposal.kind].put_nowait(proposal)

    async def _run(self, kind: OptimizerKind) -> None:
        queue = self.queues[kind]
        while not self._stopping.is_set():
            try:
                proposal = await asyncio.wait_for(queue.get(), timeout=1)
            except TimeoutError:
                continue
            try:
                result = await self.evaluator(proposal)
                if result.applied_changes:
                    result = result.model_copy(
                        update={
                            "accepted_for_review": False,
                            "score": 0.0,
                            "issues": (*result.issues, "optimizer_applied_changes"),
                            "applied_changes": 0,
                        }
                    )
                self.results[proposal.id] = result
                while len(self.results) > self.result_limit:
                    self.results.pop(next(iter(self.results)))
            finally:
                queue.task_done()

    async def close(self) -> None:
        self._stopping.set()
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    def health(self) -> dict:
        return {
            "ok": bool(self._tasks) and all(not task.done() for task in self._tasks),
            "mode": "recommendation_only",
            "workers": len(self._tasks),
            "queued": {kind.value: queue.qsize() for kind, queue in self.queues.items()},
            "results": len(self.results),
            "foregroundHooks": 0,
            "providerAccess": False,
            "storeWriteAccess": False,
        }


def compare_frontier_candidate(
    *,
    cascade_metrics: dict[str, float],
    candidate_metrics: dict[str, float],
) -> dict:
    keys = sorted(set(cascade_metrics) | set(candidate_metrics))
    return {
        "productionRuntime": "cascade",
        "candidateRole": "evaluation_only",
        "candidateSelectedForProduction": False,
        "deltas": {
            key: candidate_metrics.get(key, 0.0) - cascade_metrics.get(key, 0.0)
            for key in keys
        },
    }
