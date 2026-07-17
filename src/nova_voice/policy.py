from __future__ import annotations

from dataclasses import dataclass

from nova_voice.config import Settings
from nova_voice.domain import Decision, Interpretation, SpeechAct, Utterance
from nova_voice.interpretation.speech_cues import has_explicit_self_intention

EXECUTABLE_SPEECH_ACTS = {SpeechAct.DIRECTIVE, SpeechAct.DESIRED_STATE}
NON_EXECUTABLE_SPEECH_ACTS = {
    SpeechAct.SELF_INTENTION,
    SpeechAct.OBSERVATION,
    SpeechAct.THIRD_PARTY,
    SpeechAct.QUOTED_OR_MEDIA,
    SpeechAct.UNCLEAR,
}


@dataclass(frozen=True)
class PolicyOutcome:
    execute: bool
    shadowed: bool
    reason: str


class ExecutionPolicy:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def evaluate(
        self, utterance: Utterance, interpretation: Interpretation, *, session_active: bool
    ) -> PolicyOutcome:
        if has_explicit_self_intention(utterance.transcript):
            return PolicyOutcome(False, False, "utterance explicitly assigns the action to speaker")
        if interpretation.decision != Decision.EXECUTE or not interpretation.actions:
            return PolicyOutcome(False, False, "interpretation did not request execution")
        if interpretation.speech_act in NON_EXECUTABLE_SPEECH_ACTS:
            return PolicyOutcome(False, False, f"speech act is {interpretation.speech_act}")
        if interpretation.speech_act not in EXECUTABLE_SPEECH_ACTS:
            return PolicyOutcome(False, False, "speech act is not executable")

        active = utterance.wake_detected or utterance.conversation_active or session_active
        threshold = (
            self.settings.active_addressed_threshold
            if active
            else self.settings.passive_addressed_threshold
        )
        if interpretation.addressed_probability < threshold:
            return PolicyOutcome(False, False, "addressing probability is below policy threshold")
        confidence_threshold = (
            self.settings.active_interpretation_threshold
            if active
            else self.settings.passive_interpretation_threshold
        )
        if interpretation.confidence < confidence_threshold:
            return PolicyOutcome(
                False, False, "interpretation confidence is below policy threshold"
            )

        if self.settings.shadow_mode:
            return PolicyOutcome(False, True, "shadow mode")
        if not active and not self.settings.passive_execution_enabled:
            return PolicyOutcome(False, False, "passive execution is disabled")
        return PolicyOutcome(True, False, "allowed")
