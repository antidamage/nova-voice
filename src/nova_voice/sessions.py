from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from nova_voice.domain import (
    ActiveGoal,
    Decision,
    GoalStatus,
    Interpretation,
    ToolResult,
    Utterance,
)


@dataclass
class RoomSession:
    goal: ActiveGoal
    updated_at: datetime


class SessionManager:
    def __init__(
        self,
        follow_up_seconds: float = 60,
        *,
        key_fn: Callable[[str], str] | None = None,
    ) -> None:
        self.follow_up = timedelta(seconds=follow_up_seconds)
        # Same shared-air collapsing as the conversation window: a goal opened
        # via one satellite must stay active for a follow-up elected on another.
        self._key = key_fn if key_fn is not None else lambda room_id: room_id
        self._rooms: dict[str, RoomSession] = {}

    def set_follow_up_seconds(self, follow_up_seconds: float) -> None:
        """Track the conversation window live; goals follow the same clock."""

        self.follow_up = timedelta(seconds=max(1.0, float(follow_up_seconds)))

    def active_goal(self, room_id: str, now: datetime | None = None) -> ActiveGoal | None:
        room_id = self._key(room_id)
        session = self._rooms.get(room_id)
        if not session:
            return None
        current = now or datetime.now(UTC)
        if current - session.updated_at > self.follow_up:
            self._rooms.pop(room_id, None)
            return None
        return session.goal

    def is_active(self, room_id: str) -> bool:
        return self.active_goal(room_id) is not None

    def end(self, room_id: str) -> None:
        self._rooms.pop(self._key(room_id), None)

    def update(
        self,
        utterance: Utterance,
        interpretation: Interpretation,
        results: list[ToolResult],
        *,
        executed: bool = True,
    ) -> ActiveGoal | None:
        room_key = self._key(utterance.room_id)
        goal = interpretation.active_goal
        if not executed and interpretation.decision == Decision.EXECUTE:
            # A model cannot close a goal merely because policy shadowed or
            # denied its actions. Preserve an existing goal for a follow-up,
            # but never create a satisfied goal from an unexecuted plan.
            existing = self.active_goal(utterance.room_id, utterance.ended_at)
            if existing is not None:
                self._rooms[room_key] = RoomSession(
                    goal=existing, updated_at=utterance.ended_at
                )
            return existing
        if goal.status == GoalStatus.ABANDONED:
            self._rooms.pop(room_key, None)
            return None
        if interpretation.decision == Decision.IGNORE:
            return self.active_goal(utterance.room_id, utterance.ended_at)

        if interpretation.decision == Decision.EXECUTE and results:
            failed = [result for result in results if not result.ok]
            if failed:
                goal = goal.model_copy(
                    update={
                        "status": GoalStatus.IN_PROGRESS,
                        "pending": [result.target or result.action_id for result in failed],
                    }
                )
            else:
                goal = goal.model_copy(update={"status": GoalStatus.SATISFIED, "pending": []})

        if goal.status == GoalStatus.SATISFIED:
            self._rooms.pop(room_key, None)
            return None

        if goal.status in {GoalStatus.NEW, GoalStatus.IN_PROGRESS, GoalStatus.NEEDS_CLARIFICATION}:
            self._rooms[room_key] = RoomSession(goal=goal, updated_at=utterance.ended_at)
            return goal

        self._rooms.pop(room_key, None)
        return None
