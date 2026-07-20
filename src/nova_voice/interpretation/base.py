from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from nova_voice.audio.conversation import ConversationSnapshot
from nova_voice.domain import (
    ActiveGoal,
    Interpretation,
    SelfProfileUpdate,
    ToolResult,
    Utterance,
)


class Interpreter(ABC):
    async def extract_self_profile_update(
        self, utterance: Utterance
    ) -> SelfProfileUpdate | None:
        """Extract an explicit current-speaker name/pronoun disclosure.

        Backends may implement this as a small independent model pass. Keeping
        the default empty lets deterministic/test interpreters opt out without
        coupling identity persistence to the general interpretation schema.
        """

        return None

    @abstractmethod
    async def interpret(
        self,
        utterance: Utterance,
        *,
        active_goal: ActiveGoal | None,
        relevant_state: dict[str, Any],
        tools: list[dict],
        conversation: ConversationSnapshot | None = None,
    ) -> Interpretation: ...

    async def render_response(
        self,
        utterance: Utterance,
        interpretation: Interpretation,
        results: list[ToolResult],
        *,
        persona: str,
        environment: dict[str, Any] | None = None,
        relevant_state: dict[str, Any] | None = None,
        conversation: ConversationSnapshot | None = None,
        temperature: float | None = None,
        command_max_words: int | None = None,
    ) -> str | None:
        return None

    async def health(self) -> dict:
        return {"ok": True}

    async def close(self) -> None:
        return None
