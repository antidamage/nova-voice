from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Literal
from uuid import uuid4

from nova_voice.domain import (
    PlannedAction,
    ToolResult,
    TurnCancellationRecord,
    TurnPolicyDecision,
    TurnResponseRevision,
    TurnStage,
    TurnStageRecord,
    TurnStageStatus,
    TurnTerminalStatus,
    TurnToolJournalEntry,
    TurnTrace,
    TurnVerificationEvidence,
    Utterance,
)

_STAGE_ORDER = tuple(TurnStage)
ResponseRevisionSource = Literal[
    "deterministic", "model", "knowledge_fallback", "affectations", "final"
]


def revision(value: object) -> str:
    """Return a stable, non-reversible revision for traceable input/context."""

    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _word_count(text: str | None) -> int:
    return len((text or "").split())


class ForegroundTurnStateMachine:
    """Ordered mutable builder whose externally visible snapshots are immutable."""

    def __init__(self, utterance: Utterance) -> None:
        self.trace_id = uuid4().hex
        self.utterance_id = utterance.id
        self.input_revision = revision(
            {
                "utteranceId": utterance.id,
                "satelliteId": utterance.satellite_id,
                "roomId": utterance.room_id,
                "startedAt": utterance.started_at,
                "endedAt": utterance.ended_at,
                "transcript": utterance.transcript,
                "confidence": utterance.transcript_confidence,
                "wakeDetected": utterance.wake_detected,
            }
        )
        self.context_revision: str | None = None
        self._started = time.perf_counter()
        self._stages: list[TurnStageRecord] = []
        self._policy: TurnPolicyDecision | None = None
        self._tool_journal: list[TurnToolJournalEntry] = []
        self._verification: list[TurnVerificationEvidence] = []
        self._response_revisions: list[TurnResponseRevision] = []
        self._cancellations: list[TurnCancellationRecord] = []
        self._terminal_status = TurnTerminalStatus.IN_PROGRESS
        self._terminal_reason: str | None = None

    @property
    def next_stage(self) -> TurnStage | None:
        return _STAGE_ORDER[len(self._stages)] if len(self._stages) < len(_STAGE_ORDER) else None

    def advance(
        self,
        stage: TurnStage,
        *,
        elapsed_ms: float = 0,
        status: TurnStageStatus = TurnStageStatus.COMPLETED,
        detail: str | None = None,
    ) -> None:
        expected = self.next_stage
        if stage != expected:
            raise RuntimeError(f"turn stage out of order: expected {expected}, received {stage}")
        self._stages.append(
            TurnStageRecord(
                stage=stage,
                status=status,
                elapsed_ms=max(0.0, float(elapsed_ms)),
                detail=detail,
            )
        )

    def set_context(self, value: object) -> None:
        self.context_revision = revision(value)

    def skip_until(self, stage: TurnStage, detail: str) -> None:
        """Advance compatibility adapters that do not yet emit service stages."""

        while self.next_stage is not None and self.next_stage != stage:
            self.advance(
                self.next_stage,
                status=TurnStageStatus.SKIPPED,
                detail=detail,
            )

    def set_policy(self, *, execute: bool, shadowed: bool, reason: str) -> None:
        self._policy = TurnPolicyDecision(execute=execute, shadowed=shadowed, reason=reason)

    def record_tools(self, actions: list[PlannedAction], results: list[ToolResult]) -> None:
        by_id = {result.action_id: result for result in results}
        self._tool_journal = []
        for action in sorted(actions, key=lambda item: item.order):
            result = by_id.get(action.id)
            if result is None:
                status: Literal["planned", "completed", "failed", "cancelled", "blocked"] = (
                    "planned"
                )
                code = None
            elif result.code == "cancelled":
                status = "cancelled"
                code = result.code
            elif result.code in {"blocked", "invalid", "shadowed"}:
                status = "blocked"
                code = result.code
            elif result.ok:
                status = "completed"
                code = result.code
            else:
                status = "failed"
                code = result.code
            self._tool_journal.append(
                TurnToolJournalEntry(
                    action_id=action.id,
                    provider=action.call.provider,
                    tool=action.call.tool,
                    order=action.order,
                    status=status,
                    result_code=code,
                )
            )

    def record_verification(self, results: list[ToolResult]) -> None:
        self._verification = [
            TurnVerificationEvidence(
                action_id=result.action_id,
                ok=result.ok,
                code=result.code,
                target=result.target,
                observed_revision=(
                    revision(result.observed) if result.observed is not None else None
                ),
            )
            for result in results
        ]

    def record_response(
        self,
        source: ResponseRevisionSource,
        text: str | None,
        *,
        force: bool = False,
    ) -> None:
        if text is None:
            return
        text_revision = revision(text)
        if (
            not force
            and self._response_revisions
            and self._response_revisions[-1].text_revision == text_revision
        ):
            return
        self._response_revisions.append(
            TurnResponseRevision(
                revision=len(self._response_revisions) + 1,
                source=source,
                text_revision=text_revision,
                word_count=_word_count(text),
            )
        )

    def record_cancellation(self, record: TurnCancellationRecord) -> None:
        self._cancellations.append(record)

    def finish(self, status: TurnTerminalStatus, reason: str | None = None) -> None:
        if self.next_stage is not None:
            raise RuntimeError(f"cannot finish turn before {self.next_stage}")
        self._terminal_status = status
        self._terminal_reason = reason

    def fail(self, reason: str) -> None:
        if self.next_stage is not None:
            self.advance(
                self.next_stage,
                status=TurnStageStatus.FAILED,
                detail=reason,
            )
        while self.next_stage is not None:
            self.advance(
                self.next_stage,
                status=TurnStageStatus.SKIPPED,
                detail="prior stage failed",
            )
        self.finish(TurnTerminalStatus.FAILED, reason)

    def snapshot(self) -> TurnTrace:
        return TurnTrace(
            trace_id=self.trace_id,
            utterance_id=self.utterance_id,
            input_revision=self.input_revision,
            context_revision=self.context_revision,
            stages=tuple(self._stages),
            policy=self._policy,
            tool_journal=tuple(self._tool_journal),
            verification=tuple(self._verification),
            response_revisions=tuple(self._response_revisions),
            cancellations=tuple(self._cancellations),
            total_ms=round((time.perf_counter() - self._started) * 1000, 3),
            terminal_status=self._terminal_status,
            terminal_reason=self._terminal_reason,
        )


CancellationSafety = Literal["never", "before_side_effects", "anytime"]


@dataclass(frozen=True)
class TaskCancellationDecision:
    active: bool
    accepted: bool
    phase: str
    reason: str

    def trace_record(self) -> TurnCancellationRecord:
        return TurnCancellationRecord(
            kind="task",
            accepted=self.accepted,
            phase=self.phase,
            reason=self.reason,
        )


@dataclass
class TurnCancellationController:
    """Separates provider-task cancellation from response playback cancellation."""

    requested: asyncio.Event = field(default_factory=asyncio.Event)
    phase: str = "before_side_effects"
    action_id: str | None = None
    safety: CancellationSafety = "before_side_effects"
    provider_tasks: dict[asyncio.Task[ToolResult], CancellationSafety] = field(
        default_factory=dict
    )
    decisions: list[TaskCancellationDecision] = field(default_factory=list)

    def request(self) -> TaskCancellationDecision:
        self.requested.set()
        if self.phase == "before_side_effects":
            decision = TaskCancellationDecision(
                True,
                True,
                self.phase,
                "cancelled before side effects",
            )
        elif self.phase == "provider_call" and self.provider_tasks and all(
            safety == "anytime" for safety in self.provider_tasks.values()
        ):
            for task in self.provider_tasks:
                task.cancel()
            decision = TaskCancellationDecision(
                True,
                True,
                self.phase,
                "providers permit in-flight cancellation",
            )
        elif self.phase == "provider_call":
            decision = TaskCancellationDecision(
                True,
                False,
                self.phase,
                "provider side effects may have started; completion and verification required",
            )
        else:
            decision = TaskCancellationDecision(
                True,
                False,
                self.phase,
                "side effects completed; cancellation cannot rewrite the result",
            )
        self.decisions.append(decision)
        return decision

    def bind_provider(
        self,
        action_id: str,
        safety: CancellationSafety,
        task: asyncio.Task[ToolResult],
    ) -> None:
        self.phase = "provider_call"
        self.action_id = action_id
        self.safety = safety
        self.provider_tasks[task] = safety

    def begin_non_cancellable_side_effect(self, action_id: str) -> None:
        self.phase = "provider_call"
        self.action_id = action_id
        self.safety = "never"

    def non_cancellable_side_effect_finished(self) -> None:
        self.phase = "after_side_effects"
        self.action_id = None

    def provider_finished(self, task: asyncio.Task[ToolResult]) -> None:
        self.provider_tasks.pop(task, None)
        if not self.provider_tasks:
            self.phase = "after_side_effects"

    def close(self) -> None:
        self.phase = "complete"
        self.provider_tasks.clear()
