from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal
from uuid import uuid4


@dataclass(frozen=True)
class ConversationMessage:
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True)
class ConversationSnapshot:
    id: str
    room_id: str
    initial_environment: dict[str, Any] | None
    personality: str
    persona_prompt: str
    messages: tuple[ConversationMessage, ...]


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

    def append_turn(self, room_id: str, user: str, assistant: str | None) -> None:
        session = self._rooms.get(self._key(room_id))
        if session is None:
            return
        if session.messages is None:
            session.messages = []
        session.messages.append(ConversationMessage("user", user))
        if assistant:
            session.messages.append(ConversationMessage("assistant", assistant))

    def refresh(self, room_id: str) -> None:
        # A known in-flight turn may take longer than the idle window to render
        # and play.  Refreshing that turn must not expire it before the user has
        # had their full follow-up window.
        session = self._rooms.get(self._key(room_id))
        if session is not None:
            session.last_turn_monotonic = self._monotonic()

    def end(self, room_id: str) -> None:
        self._rooms.pop(self._key(room_id), None)

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
        )
