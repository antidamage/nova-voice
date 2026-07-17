from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from nova_voice.audio.conversation import ConversationSnapshot
from nova_voice.domain import ActiveGoal, Interpretation, ToolResult, Utterance


class Interpreter(ABC):
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
        conversation: ConversationSnapshot | None = None,
    ) -> str | None:
        return None

    async def health(self) -> dict:
        return {"ok": True}

    async def close(self) -> None:
        return None
