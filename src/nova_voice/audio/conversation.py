from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal
from uuid import uuid4

_CHARS_PER_TOKEN_ESTIMATE = 4


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (no tokenizer round-trip) for a soft context budget."""

    return max(1, len(text) // _CHARS_PER_TOKEN_ESTIMATE)


def _estimate_value_tokens(value: Any) -> int:
    """Estimate the prompt cost of an arbitrary JSON-serialisable value."""

    if isinstance(value, str):
        return _estimate_tokens(value)
    try:
        return _estimate_tokens(json.dumps(value, separators=(",", ":"), default=str))
    except (TypeError, ValueError):
        return _estimate_tokens(str(value))


# Every bulky prompt input is bounded by a drop-oldest token budget, because a
# prompt that fills the interpretation model's whole context window leaves no
# room for the model's own structured JSON reply — it gets truncated mid-object
# and the turn fails outright (silent assistant). The frozen conversation-open
# snapshot below is the largest such input and, unlike conversation history, it
# grows with the household rather than with the conversation.
#
# These budgets are ceilings, not routine trimming: they are set well above a
# normal household snapshot so they bite only in the pathological case that
# would otherwise take the whole turn down. Nothing here can touch the system
# prompt or the callable tool schemas — dropping those would change what the
# assistant is and what it can do, which is never an acceptable way to save
# tokens. Only data entries are shed, oldest/first-listed first.
INITIAL_STATE_TOKEN_BUDGET = 5000
INITIAL_MEMORY_TOKEN_BUDGET = 800


def compact_memory_to_budget(
    memory: list[Any], budget: int = INITIAL_MEMORY_TOKEN_BUDGET
) -> list[Any]:
    """Drop selected memories from the start until the estimate fits ``budget``.

    Retrieval returns memories with the most relevant last, so shedding from the
    front sacrifices the weakest matches first. At least one memory always
    survives: a single oversized memory is still better context than none.
    """

    trimmed = list(memory)
    total = sum(_estimate_value_tokens(entry) for entry in trimmed)
    while total > budget and len(trimmed) > 1:
        total -= _estimate_value_tokens(trimmed.pop(0))
    return trimmed


def compact_state_to_budget(
    state: dict[str, Any], budget: int = INITIAL_STATE_TOKEN_BUDGET
) -> dict[str, Any]:
    """Bound a household snapshot by trimming its list fields from the start.

    Keys are never dropped and scalars are never touched: the system prompt
    refers to specific fields (``indoorTemperatureC``, ``climateControls`` and
    friends) by name, so a missing key breaks the prompt contract, whereas a
    shorter list is merely less complete. Entries are shed from the front of
    whichever list is currently largest, so one runaway collection cannot
    starve every other field, and each list keeps at least one entry so the
    model can still see that the category exists.

    Trimmed fields are reported in ``truncatedFields`` so the model knows a
    listing is partial rather than authoritative.
    """

    compacted = {key: (list(value) if isinstance(value, list) else value)
                 for key, value in state.items()}
    total = sum(_estimate_value_tokens(value) for value in compacted.values())
    if total <= budget:
        return compacted
    trimmed_counts: dict[str, int] = {}
    while total > budget:
        candidates = [
            key
            for key, value in compacted.items()
            if isinstance(value, list) and len(value) > 1
        ]
        if not candidates:
            break
        largest = max(candidates, key=lambda key: _estimate_value_tokens(compacted[key]))
        removed = compacted[largest].pop(0)
        total -= _estimate_value_tokens(removed)
        trimmed_counts[largest] = trimmed_counts.get(largest, 0) + 1
    if trimmed_counts:
        compacted["truncatedFields"] = {
            key: f"{count} older entries omitted to fit the context window"
            for key, count in sorted(trimmed_counts.items())
        }
    return compacted


@dataclass(frozen=True)
class ConversationMessage:
    role: Literal["user", "assistant"]
    content: str
    speaker_name: str | None = None
    speaker_pronouns: str | None = None


@dataclass(frozen=True)
class ConversationSnapshot:
    id: str
    room_id: str
    initial_environment: dict[str, Any] | None
    personality: str
    persona_prompt: str
    messages: tuple[ConversationMessage, ...]
    # Compact one-line summaries of dashboard API responses the assistant has
    # retrieved this conversation, oldest first. Injected into the model's
    # system context so follow-up turns can reason over data an earlier tool
    # call returned, rather than losing it after the reply is spoken.
    observations: tuple[str, ...] = ()
    # Household state and selected memory captured ONCE when the conversation
    # opened. These are the bulky, largely-static prompt inputs; freezing them
    # here (rather than re-injecting live state every turn) keeps each turn's
    # payload to the new utterance and lets the interpreter place them in a
    # stable, cacheable prompt prefix. They intentionally do not refresh mid
    # conversation — a new snapshot is taken only when the window reopens.
    initial_state: dict[str, Any] | None = None
    initial_memory: tuple[Any, ...] = ()


@dataclass
class _RoomConversation:
    id: str
    room_id: str
    started_monotonic: float
    last_turn_monotonic: float
    initial_environment: dict[str, Any] | None = None
    personality: str = ""
    persona_prompt: str = ""
    messages: list[ConversationMessage] | None = None
    observations: list[str] | None = None
    speaker_template_id: str | None = None
    # Frozen at conversation open alongside initial_environment; see
    # ConversationSnapshot.initial_state.
    initial_state: dict[str, Any] | None = None
    initial_memory: list[Any] | None = None


class ConversationTracker:
    """Wake-word initiated conversation windows, per room.

    A conversation starts when the wake word opens a turn and stays open while
    real (usable) input keeps arriving.  While open, follow-up utterances are
    treated as addressed — the wide vocabulary applies and conversational
    replies are allowed without repeating the wake word.  The window closes
    after ``idle_seconds`` without usable input, after ``max_seconds`` overall,
    or immediately on an explicit end (abandonment).
    """

    def __init__(
        self,
        *,
        idle_seconds: float = 60.0,
        max_seconds: float | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        key_fn: Callable[[str], str] | None = None,
    ) -> None:
        self.idle_seconds = idle_seconds
        self.max_seconds = max_seconds
        self._monotonic = monotonic
        # Rooms that share the same air share the conversation: a follow-up
        # elected on the other satellite must refresh this window, not start
        # a narrow-mode stranger.  The key function collapses room ids into
        # that shared scope (identity for isolated installs).
        self._key = key_fn if key_fn is not None else lambda room_id: room_id
        self._rooms: dict[str, _RoomConversation] = {}

    def set_idle_seconds(self, idle_seconds: float) -> None:
        """Apply a new follow-up window live; existing sessions adopt it."""

        self.idle_seconds = max(1.0, float(idle_seconds))

    def start(self, room_id: str) -> bool:
        """Open or refresh a room conversation; return true only for a new one."""

        room_id = self._key(room_id)
        now = self._monotonic()
        self._expire(room_id, now)
        existing = self._rooms.get(room_id)
        if existing is not None:
            existing.last_turn_monotonic = now
            return False
        self._rooms[room_id] = _RoomConversation(
            id=uuid4().hex,
            room_id=room_id,
            started_monotonic=now,
            last_turn_monotonic=now,
            messages=[],
        )
        return True

    def initialize_prompt(
        self,
        room_id: str,
        *,
        environment: dict[str, Any],
        personality: str,
        persona_prompt: str,
        state: dict[str, Any] | None = None,
        memory: list[Any] | None = None,
    ) -> ConversationSnapshot | None:
        """Snapshot the prompt inputs once, when the conversation begins.

        ``state`` (live household state) and ``memory`` (selected durable
        memories) are the bulky, largely-static inputs. Capturing them here,
        once, lets follow-up turns reuse a frozen, cacheable snapshot instead of
        re-injecting live state on every turn.
        """

        session = self._rooms.get(self._key(room_id))
        if session is None:
            return None
        if session.initial_environment is None:
            session.initial_environment = dict(environment)
            session.personality = personality
            session.persona_prompt = persona_prompt
            # Bound the snapshot as it is frozen. This is the one choke point
            # both prompt passes read from (interpretation and render_response),
            # so capping here caps every downstream use of it.
            session.initial_state = (
                compact_state_to_budget(state) if state is not None else None
            )
            session.initial_memory = (
                compact_memory_to_budget(memory) if memory is not None else None
            )
        return self._snapshot(session)

    def snapshot(self, room_id: str) -> ConversationSnapshot | None:
        session = self._active_session(room_id)
        return self._snapshot(session) if session is not None else None

    # Recent-turn window kept in the prompt. A conversation that stays open a
    # long time must not accrete unbounded "old, old" history; only the most
    # recent turns are retained (older ones age out of context).
    MESSAGE_HISTORY_LIMIT = 16
    # Soft, token-aware cap layered on top of the count cap above. The message
    # COUNT cap alone is not enough: a handful of unusually verbose replies can
    # still balloon the prompt well past the interpretation model's context
    # window (a small on-device model with a modest context size), crowding
    # out the room the model needs to finish its own structured JSON response
    # and producing truncated, invalid completions. Estimated with a
    # conservative chars-per-token heuristic, deliberately tight so there is
    # always headroom left for the system prompt, tool schemas, and output.
    MESSAGE_HISTORY_TOKEN_BUDGET = 700

    def append_turn(
        self,
        room_id: str,
        user: str,
        assistant: str | None,
        *,
        speaker_name: str | None = None,
        speaker_pronouns: str | None = None,
    ) -> None:
        session = self._rooms.get(self._key(room_id))
        if session is None:
            return
        if session.messages is None:
            session.messages = []
        session.messages.append(
            ConversationMessage("user", user, speaker_name, speaker_pronouns)
        )
        if assistant:
            session.messages.append(ConversationMessage("assistant", assistant))
        if len(session.messages) > self.MESSAGE_HISTORY_LIMIT:
            del session.messages[: -self.MESSAGE_HISTORY_LIMIT]
        self._compact_to_token_budget(session)

    def _compact_to_token_budget(self, session: _RoomConversation) -> None:
        """Drop the oldest messages while the estimated total exceeds budget.

        Always keeps at least the most recent exchange, however large, so a
        single verbose reply can never erase all context for its own
        follow-up turn.
        """

        messages = session.messages
        if not messages:
            return
        total = sum(_estimate_tokens(message.content) for message in messages)
        while total > self.MESSAGE_HISTORY_TOKEN_BUDGET and len(messages) > 2:
            removed = messages.pop(0)
            total -= _estimate_tokens(removed.content)

    # Soft token-aware cap on retained dashboard observations, mirroring
    # MESSAGE_HISTORY_TOKEN_BUDGET above. The count cap alone is not enough: a
    # single nova.query result can carry a large observed-state payload, and a
    # handful of those can balloon the injected context past the interpretation
    # model's whole context window on their own.
    OBSERVATIONS_TOKEN_BUDGET = 400

    def record_observations(
        self, room_id: str, entries: list[str], *, limit: int = 8
    ) -> None:
        """Retain dashboard API responses for later turns of this conversation.

        Newest entries win once ``limit`` (count) or ``OBSERVATIONS_TOKEN_BUDGET``
        (size) is exceeded so the injected context stays bounded; blank entries
        and consecutive duplicates are skipped.
        """

        session = self._rooms.get(self._key(room_id))
        if session is None:
            return
        if session.observations is None:
            session.observations = []
        for entry in entries:
            text = entry.strip()
            if not text or (session.observations and session.observations[-1] == text):
                continue
            session.observations.append(text)
        if len(session.observations) > limit:
            del session.observations[:-limit]
        total = sum(_estimate_tokens(entry) for entry in session.observations)
        while total > self.OBSERVATIONS_TOKEN_BUDGET and len(session.observations) > 1:
            removed = session.observations.pop(0)
            total -= _estimate_tokens(removed)

    def refresh(self, room_id: str) -> None:
        # A known in-flight turn may take longer than the idle window to render
        # and play.  Refreshing that turn must not expire it before the user has
        # had their full follow-up window.
        session = self._rooms.get(self._key(room_id))
        if session is not None:
            session.last_turn_monotonic = self._monotonic()

    def speaker_template(self, room_id: str) -> str | None:
        """Return the voice template currently bound to this live conversation."""

        session = self._active_session(room_id)
        return session.speaker_template_id if session is not None else None

    def bind_speaker_template(self, room_id: str, template_id: str) -> None:
        """Treat later turns as this voice until a genuinely different speaker appears."""

        session = self._active_session(room_id)
        if session is not None:
            session.speaker_template_id = template_id

    def end(self, room_id: str) -> None:
        self._rooms.pop(self._key(room_id), None)

    def clear(self) -> None:
        """Close every open conversation at once (e.g. the voice killswitch)."""

        self._rooms.clear()

    def open_room_count(self) -> int:
        """Number of conversation windows currently open, for reporting."""

        return len(self._rooms)

    def active(self, room_id: str) -> bool:
        return self._active_session(room_id) is not None

    def _active_session(self, room_id: str) -> _RoomConversation | None:
        room_id = self._key(room_id)
        now = self._monotonic()
        self._expire(room_id, now)
        return self._rooms.get(room_id)

    def _expire(self, room_id: str, now: float) -> None:
        session = self._rooms.get(room_id)
        if session is None:
            return
        idle_expired = now - session.last_turn_monotonic >= self.idle_seconds
        maximum_expired = (
            self.max_seconds is not None
            and now - session.started_monotonic >= self.max_seconds
        )
        if idle_expired or maximum_expired:
            self._rooms.pop(room_id, None)

    @staticmethod
    def _snapshot(session: _RoomConversation) -> ConversationSnapshot:
        return ConversationSnapshot(
            id=session.id,
            room_id=session.room_id,
            initial_environment=(
                dict(session.initial_environment)
                if session.initial_environment is not None
                else None
            ),
            personality=session.personality,
            persona_prompt=session.persona_prompt,
            messages=tuple(session.messages or ()),
            observations=tuple(session.observations or ()),
            initial_state=(
                dict(session.initial_state)
                if session.initial_state is not None
                else None
            ),
            initial_memory=tuple(session.initial_memory or ()),
        )
