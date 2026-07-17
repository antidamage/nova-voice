from __future__ import annotations

from datetime import UTC, datetime

import pytest

from nova_voice.domain import (
    ActiveGoal,
    Decision,
    Emotion,
    EmotionLabel,
    GoalStatus,
    Interpretation,
    ResponsePlan,
    SpeechAct,
    Utterance,
)


@pytest.fixture
def utterance() -> Utterance:
    now = datetime.now(UTC)
    return Utterance(
        id="utterance-1",
        satellite_id="test",
        room_id="lounge",
        started_at=now,
        ended_at=now,
        transcript="Turn the lounge light on",
        wake_detected=False,
    )


def interpretation(
    *,
    speech_act: SpeechAct = SpeechAct.DIRECTIVE,
    decision: Decision = Decision.IGNORE,
    addressed: float = 0.99,
    confidence: float = 0.95,
    actions: list | None = None,
) -> Interpretation:
    return Interpretation(
        emotion=Emotion(label=EmotionLabel.NEUTRAL, confidence=0.8, intensity=0.2),
        speech_act=speech_act,
        addressed_probability=addressed,
        decision=decision,
        confidence=confidence,
        active_goal=ActiveGoal(summary="test", status=GoalStatus.NEW),
        actions=actions or [],
        response_plan=ResponsePlan(),
    )
