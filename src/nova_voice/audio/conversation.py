from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal
from uuid import uuid4

_CHARS_PER_TOKEN_ESTIMATE = 4


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (no tokenizer round-trip) for a soft context budget."""

    return max(1, len(text) // _CHARS_PER_TOKEN_ESTIMATE)


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
    ) -> ConversationSnapshot | None:
        """Snapshot the prompt inputs once, when the conversation begins."""

        session = self._rooms.get(self._key(room_id))
        if session is None:
            return None
        if session.initial_environment is None:
            session.initial_environment = dict(environment)
            session.personality = personality
            session.persona_prompt = persona_prompt
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
        )
