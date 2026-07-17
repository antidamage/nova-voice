from __future__ import annotations

from conftest import interpretation

from nova_voice.domain import Decision, GoalStatus, ToolResult
from nova_voice.interpretation.speech_cues import enforce_speech_cues
from nova_voice.sessions import SessionManager


def test_verified_goal_closes(utterance) -> None:
    manager = SessionManager()
    value = interpretation(decision=Decision.EXECUTE)
    value.active_goal.status = GoalStatus.IN_PROGRESS
    manager.update(
        utterance,
        value,
        [ToolResult(action_id="a1", ok=True, code="ok", message="done")],
    )
    assert manager.active_goal(utterance.room_id) is None


def test_failed_goal_stays_open(utterance) -> None:
    manager = SessionManager()
    value = interpretation(decision=Decision.EXECUTE)
    value.active_goal.status = GoalStatus.IN_PROGRESS
    manager.update(
        utterance,
        value,
        [ToolResult(action_id="a1", ok=False, code="unverified", message="failed")],
    )
    goal = manager.active_goal(utterance.room_id)
    assert goal is not None
    assert goal.status == GoalStatus.IN_PROGRESS


def test_ignored_ambient_speech_does_not_open_session(utterance) -> None:
    manager = SessionManager()

    manager.update(utterance, interpretation(decision=Decision.IGNORE), [])

    assert manager.active_goal(utterance.room_id) is None


def test_explicit_abandonment_closes_existing_goal(utterance) -> None:
    manager = SessionManager()
    existing = interpretation(decision=Decision.EXECUTE)
    existing.active_goal.status = GoalStatus.IN_PROGRESS
    manager.update(
        utterance,
        existing,
        [ToolResult(action_id="a1", ok=False, code="unverified", message="pending")],
    )
    abandoned = enforce_speech_cues("never mind", interpretation(decision=Decision.IGNORE))

    manager.update(utterance, abandoned, [])

    assert manager.active_goal(utterance.room_id) is None


def test_unexecuted_satisfied_plan_does_not_close_or_create_goal(utterance) -> None:
    manager = SessionManager()
    value = interpretation(decision=Decision.EXECUTE)
    value.active_goal.status = GoalStatus.SATISFIED

    manager.update(utterance, value, [], executed=False)

    assert manager.active_goal(utterance.room_id) is None


def test_follow_up_window_is_live_tunable(utterance) -> None:
    from datetime import timedelta

    manager = SessionManager(follow_up_seconds=20)
    value = interpretation(decision=Decision.EXECUTE)
    value.active_goal.status = GoalStatus.IN_PROGRESS
    manager.update(
        utterance,
        value,
        [ToolResult(action_id="a1", ok=False, code="unverified", message="failed")],
    )

    manager.set_follow_up_seconds(60)

    later = utterance.ended_at + timedelta(seconds=45)
    assert manager.active_goal(utterance.room_id, later) is not None
    expired = utterance.ended_at + timedelta(seconds=61)
    assert manager.active_goal(utterance.room_id, expired) is None


def test_household_key_shares_goals_across_rooms(utterance) -> None:
    manager = SessionManager(key_fn=lambda room_id: "household")
    value = interpretation(decision=Decision.EXECUTE)
    value.active_goal.status = GoalStatus.IN_PROGRESS
    manager.update(
        utterance,
        value,
        [ToolResult(action_id="a1", ok=False, code="unverified", message="failed")],
    )

    assert manager.active_goal("some-other-room", utterance.ended_at) is not None
