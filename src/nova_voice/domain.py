from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class EmotionLabel(StrEnum):
    NEUTRAL = "neutral"
    CALM = "calm"
    GRUMPY = "grumpy"
    ANGRY = "angry"
    EXCITED = "excited"
    BORED = "bored"
    SAD = "sad"
    ANXIOUS = "anxious"


class SpeechAct(StrEnum):
    DIRECTIVE = "directive"
    DESIRED_STATE = "desired_state"
    SELF_INTENTION = "self_intention"
    OBSERVATION = "observation"
    QUESTION = "question"
    THIRD_PARTY = "third_party"
    QUOTED_OR_MEDIA = "quoted_or_media"
    SOCIAL = "social"
    UNCLEAR = "unclear"


class Decision(StrEnum):
    EXECUTE = "execute"
    REPLY = "reply"
    CLARIFY = "clarify"
    IGNORE = "ignore"


class GoalStatus(StrEnum):
    NEW = "new"
    IN_PROGRESS = "in_progress"
    NEEDS_CLARIFICATION = "needs_clarification"
    SATISFIED = "satisfied"
    ABANDONED = "abandoned"


class Emotion(StrictModel):
    label: EmotionLabel = EmotionLabel.NEUTRAL
    confidence: float = Field(ge=0, le=1)
    intensity: float = Field(ge=0, le=1)
    evidence: list[Literal["lexical", "energy", "pitch", "rate", "context"]] = []


class AcousticFeatures(StrictModel):
    duration_ms: int = Field(default=0, ge=0)
    rms_db: float = -120
    peak_db: float = -120
    snr_db: float | None = None
    pitch_median_relative: float | None = None
    pitch_range: float | None = None
    syllables_per_second: float | None = None
    pause_ratio: float | None = Field(default=None, ge=0, le=1)


class Utterance(StrictModel):
    id: str
    satellite_id: str
    room_id: str
    started_at: datetime
    ended_at: datetime
    transcript: str = Field(min_length=1)
    transcript_confidence: float = Field(default=1, ge=0, le=1)
    wake_detected: bool = False
    wake_score: float | None = Field(default=None, ge=0, le=1)
    # True while a wake-word-opened conversation window is active in the room;
    # follow-up turns are treated as addressed without repeating the wake word.
    conversation_active: bool = False
    dashboard_foreground: bool | None = None
    acoustic: AcousticFeatures = Field(default_factory=AcousticFeatures)

    @classmethod
    def text(
        cls,
        transcript: str,
        *,
        room_id: str = "unknown",
        satellite_id: str = "cli",
        wake_detected: bool = False,
    ) -> Utterance:
        import uuid

        now = datetime.now(UTC)
        return cls(
            id=str(uuid.uuid4()),
            satellite_id=satellite_id,
            room_id=room_id,
            started_at=now,
            ended_at=now,
            transcript=transcript,
            wake_detected=wake_detected,
        )


class CapabilityToolCall(StrictModel):
    provider: str
    tool: str
    arguments: dict[str, Any]


class PlannedAction(StrictModel):
    id: str
    order: int = Field(ge=0)
    depends_on: list[str] = []
    call: CapabilityToolCall


class ActiveGoal(StrictModel):
    summary: str = ""
    status: GoalStatus = GoalStatus.NEW
    pending: list[str] = []


class ResponsePlan(StrictModel):
    acknowledgement_style: str = "concise"
    pre_action_speech: str | None = None
    requires_post_tool_rendering: bool = False


class Interpretation(StrictModel):
    emotion: Emotion
    speech_act: SpeechAct
    addressed_probability: float = Field(ge=0, le=1)
    decision: Decision
    confidence: float = Field(ge=0, le=1)
    active_goal: ActiveGoal
    actions: list[PlannedAction] = Field(default_factory=list, max_length=4)
    response_plan: ResponsePlan = Field(default_factory=ResponsePlan)

    @model_validator(mode="after")
    def validate_plan(self) -> Interpretation:
        ids: set[str] = set()
        previous_order = -1
        for action in sorted(self.actions, key=lambda candidate: candidate.order):
            if action.id in ids:
                raise ValueError(f"duplicate action id: {action.id}")
            if action.order < previous_order:
                raise ValueError("action order must be monotonic")
            unknown = set(action.depends_on) - ids
            if unknown:
                raise ValueError(f"dependencies must refer to earlier actions: {sorted(unknown)}")
            ids.add(action.id)
            previous_order = action.order
        if self.decision != Decision.EXECUTE and self.actions:
            raise ValueError("non-execute decisions cannot contain actions")
        return self


ToolResultCode = Literal[
    "ok",
    "ambiguous",
    "not_found",
    "blocked",
    "invalid",
    "timeout",
    "unverified",
    "backend_error",
    "shadowed",
]


class ToolResult(StrictModel):
    action_id: str
    ok: bool
    code: ToolResultCode
    target: str | None = None
    requested: dict[str, Any] | None = None
    observed: dict[str, Any] | None = None
    candidates: list[str] = []
    message: str


class HandleResult(StrictModel):
    utterance_id: str
    interpretation: Interpretation
    executed: bool
    shadowed: bool
    policy_reason: str
    results: list[ToolResult] = []
    response_text: str | None = None
    response_tone_instruction: str | None = None
    timings_ms: dict[str, float] = Field(default_factory=dict)
