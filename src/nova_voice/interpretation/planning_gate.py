"""Decide from the words alone whether a turn may plan actions at all.

LFM3-1.7B classifies almost every utterance as a directive and pads a reply
such as "are you there?" with a multi-action plan (lights on, aircon to 21, a
relay message). Writing that plan costs several seconds of decoding, and the
actions it invents are ones nobody asked for.

The gate withholds planning only for small talk it recognises outright: every
clause of the transcript is a greeting, a presence check, "how are you",
thanks, a request for a joke or a story, or a question about the assistant
itself. Anything else plans exactly as before. A closed list of action words
was tried first and dropped real commands ("I'm cold", "louder", "next"), so
the default is to plan and the exception is narrow.

Even a recognised chat clause still plans when the turn carries any other
signal: household or tool vocabulary, a relay to someone else, a number, a
word from a target, room or mode name in the snapshot, the web cue, or an open
goal.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from nova_voice.domain import ActiveGoal, GoalStatus

_FILLER = r"(?:\s+(?:please|mate|pal|then|again|today|now|there))*"
_CHAT_CLAUSE = re.compile(
    r"^(?:"
    # Greetings. "Good night" is left out: it can start a bedtime routine.
    r"(?:hi|hello|hey|hiya|yo|howdy|g'?day|kia\s+ora|"
    r"good\s+(?:morning|afternoon|evening)|morning|evening)"
    # Presence checks.
    r"|(?:are\s+)?you\s+(?:there|awake|listening|around|up)"
    r"|can\s+you\s+hear\s+me|testing(?:\s+testing)*|test"
    # How are you.
    r"|how\s+(?:are|r)\s+(?:you|ya)(?:\s+(?:doing|going|feeling))?"
    r"|how(?:'s|\s+is)\s+it\s+going|how\s+have\s+you\s+been|what'?s\s+up|sup"
    # Thanks and praise.
    r"|thanks|thank\s+you|cheers|ta|nice\s+one|good\s+(?:job|work)|well\s+done|"
    r"you'?re\s+(?:the\s+best|great|funny|hilarious)"
    # Jokes, stories, small talk.
    r"|tell\s+(?:me|us)\s+(?:a|another)\s+(?:joke|story|fun\s+fact)"
    r"|tell\s+(?:me|us)\s+something\s+(?:funny|interesting|random)"
    r"|(?:know|got)\s+any\s+(?:good\s+)?jokes"
    # The assistant itself.
    r"|who\s+are\s+you|what\s+are\s+you|what(?:'s|\s+is)\s+your\s+name"
    r"|how\s+old\s+are\s+you|do\s+you\s+have\s+feelings|are\s+you\s+(?:real|alive|happy)"
    r")" + _FILLER + r"$",
    re.IGNORECASE,
)

# Anything that could still be a household or tool request inside a chat-shaped
# line ("hey, turn the lights on", "thanks, now the heater").
_ACTION_SIGNAL = re.compile(
    r"\b(?:"
    r"turn|switch|set|put|make|start|stop|open|close|lock|unlock|raise|lower|"
    r"lights?|lamps?|heater|heat|aircon|air\s*con|fan|temperature|warm|cool|hot|cold|"
    r"bright|dim|dark|music|play|volume|loud|quiet|next|skip|tv|"
    r"remind|reminder|timer|alarm|task|list|note|message|call|text|email|send|"
    r"weather|news|search|look|find|where|mode|party|home|leaving|bed|night"
    r")\b",
    re.IGNORECASE,
)
_RELAY = re.compile(r"\b(?:tell|ask|let)\s+(?!me\b|us\b)\w+", re.IGNORECASE)
_NUMBER = re.compile(
    r"\d|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|twenty|thirty|half)\b",
    re.IGNORECASE,
)
_GENERIC_NAME_WORDS = frozenset({"the", "and", "of", "a", "an"})


def _target_words(state: dict[str, Any]) -> set[str]:
    """Every word of every target, room and mode name in the snapshot."""

    names: list[str] = []
    for value in state.values():
        if isinstance(value, list):
            for entry in value:
                if isinstance(entry, dict):
                    names.extend(str(entry[key]) for key in ("name", "room") if entry.get(key))
                elif isinstance(entry, str):
                    names.append(entry)
    modes = state.get("modes")
    if isinstance(modes, dict):
        names.extend(str(key).replace("_", " ") for key in modes)
    words: set[str] = set()
    for name in names:
        for word in re.findall(r"[a-z0-9]+", name.casefold()):
            if len(word) >= 2 and word not in _GENERIC_NAME_WORDS:
                words.add(word)
    return words


def _is_small_talk(transcript: str, address_words: Iterable[str]) -> bool:
    """Whether every clause is recognised small talk, once address words are gone."""

    text = transcript or ""
    for word in sorted({w for w in address_words if w}, key=len, reverse=True):
        text = re.sub(rf"\b{re.escape(word)}\b", " ", text, flags=re.IGNORECASE)
    clauses = [
        " ".join(clause.split()) for clause in re.split(r"[.,!?;:—-]+", text) if clause.strip()
    ]
    return bool(clauses) and all(_CHAT_CLAUSE.match(clause) for clause in clauses)


def planning_offered(
    transcript: str,
    *,
    household_state: dict[str, Any],
    active_goal: ActiveGoal | None,
    web_lookup_requested: bool = False,
    address_words: Iterable[str] = (),
) -> bool:
    """Whether this turn may decode an execute decision with actions."""

    if active_goal is not None and (
        active_goal.pending
        or active_goal.status in (GoalStatus.IN_PROGRESS, GoalStatus.NEEDS_CLARIFICATION)
    ):
        return True
    if web_lookup_requested or not _is_small_talk(transcript, address_words):
        return True
    text = transcript or ""
    if _ACTION_SIGNAL.search(text) or _RELAY.search(text) or _NUMBER.search(text):
        return True
    spoken = set(re.findall(r"[a-z0-9]+", text.casefold()))
    return bool(spoken & _target_words(household_state))
