from __future__ import annotations

from dataclasses import dataclass

from nova_voice.authority import HouseholdAuthority
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
    identity_role: str | None = None
    grant_ids: tuple[str, ...] = ()


class ExecutionPolicy:
    def __init__(self, settings: Settings, authority: HouseholdAuthority | None = None) -> None:
        self.settings = settings
        self.authority = authority

    def evaluate(
        self, utterance: Utterance, interpretation: Interpretation, *, session_active: bool
    ) -> PolicyOutcome:
        if has_explicit_self_intention(utterance.transcript):
            return PolicyOutcome(False, False, "utterance explicitly assigns the action to speaker")
        if interpretation.decision != Decision.EXECUTE or not interpretation.actions:
            return PolicyOutcome(False, False, "interpretation did not request execution")
        if interpretation.speech_act in NON_EXECUTABLE_SPEECH_ACTS:
            return PolicyOutcome(False, False, f"speech act is {interpretation.speech_act}")
        # A plan made only of read-only web lookups mutates no household state, so
        # a directly-addressed question ("who won?") may drive it even though a
        # question is not a household directive. Household mutations keep the
        # strict directive/desired-state gate.
        web_only = all(action.call.provider == "web" for action in interpretation.actions)
        executable_speech_acts = (
            EXECUTABLE_SPEECH_ACTS | {SpeechAct.QUESTION} if web_only else EXECUTABLE_SPEECH_ACTS
        )
        if interpretation.speech_act not in executable_speech_acts:
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

        authority_outcome = (
            self.authority.authorize(utterance.speaker, interpretation.actions)
            if self.authority is not None
            else None
        )
        if authority_outcome is not None and not authority_outcome.allowed:
            return PolicyOutcome(
                False,
                False,
                authority_outcome.reason,
                authority_outcome.role.value,
            )

        if self.settings.shadow_mode:
            return PolicyOutcome(
                False,
                True,
                "shadow mode",
                authority_outcome.role.value if authority_outcome else None,
                authority_outcome.grant_ids if authority_outcome else (),
            )
        # A web lookup sends the query off the local network, so it only ever
        # runs on a directly-addressed turn — never from passive ambient speech,
        # regardless of the passive-execution setting used for household control.
        if web_only and not active:
            return PolicyOutcome(False, False, "web lookup requires an addressed turn")
        if not web_only and not active and not self.settings.passive_execution_enabled:
            return PolicyOutcome(False, False, "passive execution is disabled")
        return PolicyOutcome(
            True,
            False,
            "allowed",
            authority_outcome.role.value if authority_outcome else None,
            authority_outcome.grant_ids if authority_outcome else (),
        )
