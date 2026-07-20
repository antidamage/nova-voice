from __future__ import annotations

import re

from nova_voice.domain import Decision, GoalStatus, Interpretation, SpeechAct, Utterance

SELF_INTENTION = re.compile(
    r"""
    \b(?:i|we)\s+
    (?:
        (?:have|need|want|got)\s+to
        |(?:have|got)\s+got\s+to
        |gotta
        |(?:am|are)\s+(?:going|planning)\s+to
        |(?:plan|intend)\s+to
        |(?:will|shall|should|must)
    )
    \s+(?:go\s+(?:and\s+)?|come\s+(?:and\s+)?)?
    (?:turn|switch|set|change|open|close|add|remove|start|stop|wake|sleep|fix|check)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)
# Complete utterances (after courtesy/wake stripping) that end the current
# conversation.  Matching the whole utterance is deliberate: "that's all the
# lights" or "stop that music" is a request, not an abandonment.
ABANDONMENT_PHRASES = frozenset(
    {
        "never mind",
        "nevermind",
        "forget it",
        "forget that",
        "cancel that",
        "cancel it",
        "stop that",
        "that's all",
        "that is all",
        "that's all for now",
        "that is all for now",
        "that will be all",
        "that'll be all",
        "we're done",
        "we are done",
        # Direct dismissals that must close the conversation, not just barge in
        # on the current reply.  "shut up"/"be quiet"/"stop talking" also match
        # SPEECH_INTERRUPT (playback barge-in); listing them here additionally
        # ends the conversation on the turn-handling path, including when a wake
        # word opened it in the same breath ("beemo, be quiet").
        "be quiet",
        "quiet",
        "shut up",
        "shut it",
        "stop talking",
        "goodbye",
        "good bye",
        "bye",
        "bye bye",
        "goodnight",
        "good night",
    }
)
# Greeting/courtesy tokens that may wrap an abandonment without changing it.
_ABANDONMENT_LEADING = frozenset({"hey", "ok", "okay", "hi", "hello", "yo", "oi", "please"})
_ABANDONMENT_TRAILING = frozenset({"please", "thanks", "thank", "you", "now"})
_ABANDONMENT_TOKENS = re.compile(r"[a-z']+")
SPEECH_INTERRUPT = re.compile(
    r"^\s*(?:(?:hey|ok(?:ay)?|please)\s*[,;:]?\s+)*"
    r"(?:(?:beemo|nova|bandit)\s*[,;:]?\s+)?"
    r"(?:shut\s+up|be\s+quiet|stop\s+talking)\s*[.!?]*\s*$",
    re.IGNORECASE,
)
EXPLICIT_NEGATED_COMMAND = re.compile(
    r"\b(?:do\s+not|don't|dont|never)\s+(?:please\s+)?"
    r"(?:turn|switch|set|change|open|close|start|stop|wake|sleep)\b",
    re.IGNORECASE,
)


def has_explicit_self_intention(transcript: str) -> bool:
    """Detect explicit self-agency, not a desired state or a request to 'you'."""

    return SELF_INTENTION.search(transcript) is not None


def has_abandonment(
    transcript: str,
    wake_words: tuple[str, ...] | list[str] | None = None,
) -> bool:
    """Return true when the whole utterance is an explicit abandonment.

    A substring search here ended conversations on sentences that merely
    contained a cue ("that's all the lights are for"); the phrase must be the
    entire utterance apart from courtesy tokens and the wake word.
    """

    tokens = _ABANDONMENT_TOKENS.findall(transcript.casefold())
    skip_leading = _ABANDONMENT_LEADING | {
        word.strip().casefold() for word in (wake_words or ()) if word.strip()
    }
    start = 0
    while start < len(tokens) and tokens[start] in skip_leading:
        start += 1
    end = len(tokens)
    while end > start:
        if " ".join(tokens[start:end]) in ABANDONMENT_PHRASES:
            return True
        if tokens[end - 1] not in _ABANDONMENT_TRAILING:
            return False
        end -= 1
    return False


def has_speech_interrupt(
    transcript: str,
    wake_words: tuple[str, ...] | list[str] | None = None,
) -> bool:
    """Match only a direct room-scoped request to stop the current response."""

    if wake_words is None:
        return SPEECH_INTERRUPT.fullmatch(transcript) is not None
    wake_expression = "|".join(re.escape(word) for word in wake_words if word)
    optional_wake = (
        rf"(?:(?:{wake_expression})\s*[,;:]?\s+)?"
        if wake_expression
        else ""
    )
    pattern = re.compile(
        r"^\s*(?:(?:hey|ok(?:ay)?|please)\s*[,;:]?\s+)*"
        + optional_wake
        + r"(?:shut\s+up|be\s+quiet|stop\s+talking)\s*[.!?]*\s*$",
        re.IGNORECASE,
    )
    return pattern.fullmatch(transcript) is not None


def has_explicit_negated_command(transcript: str) -> bool:
    """Detect a direct prohibition so a malformed model result cannot act."""

    return EXPLICIT_NEGATED_COMMAND.search(transcript) is not None


def enforce_speech_cues(
    transcript: str,
    interpretation: Interpretation,
    wake_words: tuple[str, ...] | list[str] | None = None,
) -> Interpretation:
    if has_abandonment(transcript, wake_words):
        return interpretation.model_copy(
            update={
                "decision": Decision.IGNORE,
                "actions": [],
                "active_goal": interpretation.active_goal.model_copy(
                    update={"status": GoalStatus.ABANDONED, "pending": []}
                ),
            }
        )
    if has_explicit_negated_command(transcript):
        return interpretation.model_copy(update={"decision": Decision.IGNORE, "actions": []})
    if not has_explicit_self_intention(transcript):
        return interpretation
    return interpretation.model_copy(
        update={
            "speech_act": SpeechAct.SELF_INTENTION,
            "decision": Decision.IGNORE,
            "actions": [],
        }
    )


def enforce_decision_consistency(
    utterance: Utterance,
    interpretation: Interpretation,
    *,
    addressed_threshold: float,
    wake_words: tuple[str, ...] | list[str] | None = None,
) -> Interpretation:
    """Repair safe but contradictory model decisions for addressed speech.

    The compact local model can correctly identify an addressed question or social
    turn and still select ``ignore`` because no semantic household tool applies. It
    can also select ``execute`` for a non-tool request while returning no actions.
    These repairs never create or execute an action; they only make the response
    decision agree with the already-classified speech act.
    """

    if utterance.conversation_active and interpretation.addressed_probability < 1:
        interpretation = interpretation.model_copy(update={"addressed_probability": 1.0})

    # An executable direct command is itself proof of address. The structured
    # early pass has already classified the speech act, selected ``execute``,
    # and produced a bounded action plan. Letting a second model-authored score
    # veto that result made clear passive commands fail unless a wake word first
    # opened a conversation. Negative/quoted/self-intention speech is repaired
    # before this function or rejected by deterministic execution policy.
    if (
        interpretation.decision == Decision.EXECUTE
        and interpretation.actions
        and interpretation.speech_act in {SpeechAct.DIRECTIVE, SpeechAct.DESIRED_STATE}
        and interpretation.addressed_probability < 1
    ):
        interpretation = interpretation.model_copy(update={"addressed_probability": 1.0})

    # The compact model sometimes uses an abandoned goal to mean "this social
    # turn has no household goal".  Only a deterministic user phrase may end
    # the room conversation; a model-selected goal status must not silence an
    # otherwise addressed greeting or follow-up.
    if has_abandonment(utterance.transcript, wake_words):
        return interpretation

    addressed = bool(
        utterance.wake_detected
        or utterance.conversation_active
        or interpretation.addressed_probability >= addressed_threshold
    )
    if not addressed:
        return interpretation

    if interpretation.decision == Decision.EXECUTE and not interpretation.actions:
        return interpretation.model_copy(update={"decision": Decision.REPLY})

    if interpretation.decision != Decision.IGNORE:
        return interpretation

    # Passive quoted/third-party/self-intention speech remains ambient. Once a
    # wake-opened conversation exists, every accepted user utterance is for the
    # assistant, but these speech acts still cannot become executable actions.
    if interpretation.speech_act in {
        SpeechAct.SELF_INTENTION,
        SpeechAct.THIRD_PARTY,
        SpeechAct.QUOTED_OR_MEDIA,
    } and not utterance.conversation_active:
        return interpretation

    if utterance.conversation_active and interpretation.speech_act in {
        SpeechAct.SELF_INTENTION,
        SpeechAct.THIRD_PARTY,
        SpeechAct.QUOTED_OR_MEDIA,
    }:
        return interpretation.model_copy(update={"decision": Decision.REPLY, "actions": []})

    if interpretation.speech_act in {
        SpeechAct.DIRECTIVE,
        SpeechAct.DESIRED_STATE,
        SpeechAct.UNCLEAR,
    }:
        return interpretation.model_copy(update={"decision": Decision.CLARIFY})

    return interpretation.model_copy(update={"decision": Decision.REPLY})
