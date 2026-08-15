from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from nova_voice.audio.conversation import ConversationSnapshot
from nova_voice.domain import (
    ActiveGoal,
    Interpretation,
    SelfProfileUpdate,
    ToolResult,
    Utterance,
    VerificationVerdict,
)


@dataclass(frozen=True)
class RenderRequest:
    """One reply-rendering pass, assembled but not yet run.

    Exists so the prompt and the model are separable: the same request can be
    answered by this host's model or handed to a companion device. Keeping it
    as one object is what stops the two paths drifting — the instructions and
    the post-generation contract travel together, and a reply is held to the
    same word budget whichever machine wrote it.
    """

    # The chat-completions form, for a backend that takes messages.
    messages: list[dict]
    # The same content split up, for a backend that takes instructions and a
    # structured input separately — which is what the on-device model wants.
    system: str
    facts: dict
    history: list[dict] = field(default_factory=list)
    temperature: float = 0.0
    max_tokens: int = 80
    # The post-generation contract. Enforced by ``finalize_rendered`` against
    # whatever came back, from wherever.
    all_succeeded: bool = False
    command_max_words: int | None = None
    bare_wake_max_words: int | None = None
    long_form: bool = False
    requested_depth: str = "normal"


@dataclass(frozen=True)
class InterpretRequest:
    """One interpretation pass, assembled but not yet run.

    Same split as :class:`RenderRequest`, and the same reason: the prompt and
    the model that answers it are separable. This one carries more, because the
    planner needs the callable tools and the household snapshot as named blocks
    the system prompt refers to by name.
    """

    messages: list[dict]
    system: str
    # The conversation-open block: callable tools, household state, memory.
    opening_context: dict
    # This turn only: the utterance and the cues derived from it.
    turn_context: dict
    history: list[dict] = field(default_factory=list)


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

    async def classify_icon(self, name: str, icons: list[str]) -> str | None:
        """Pick the sigil that best represents a reminder's name.

        Used by the dashboard's reminder icon bar, which cannot reach the LLM
        itself: llama-server is bound to loopback and firewalled to localhost,
        so the orchestrator proxies. ``icons`` is a closed vocabulary and the
        return value must be one of it, or None when nothing fits. Default
        returns None so deterministic/test interpreters opt out cleanly.
        """

        return None

    async def confirm_objective(
        self,
        utterance: Utterance,
        pending: list[dict[str, Any]],
    ) -> VerificationVerdict | None:
        """Judge whether observed device state now satisfies each objective.

        Called from inside the bounded device-verification loop, never as a
        front-facing turn: it must return only the structured verdict, never
        speech. ``pending`` holds one entry per still-unconfirmed target:
        ``{"target": str, "objective": str, "observed": dict | None,
        "attempts": int}``. Backends may implement this as a small independent
        JSON-only pass. The default returns None so the loop falls back to
        purely deterministic verification.
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
        bare_wake_max_words: int | None = None,
    ) -> str | None:
        return None

    def build_render_request(
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
        bare_wake_max_words: int | None = None,
    ) -> RenderRequest | None:
        """The reply prompt, for a backend that can hand it somewhere else.

        Returning None means "this interpreter's reply pass cannot be routed" —
        deterministic and test interpreters have no prompt to send — and the
        caller runs ``render_response`` unchanged. Opting out is the default so
        a new backend is never routed by accident.
        """

        return None

    def build_interpret_request(
        self,
        utterance: Utterance,
        *,
        active_goal: ActiveGoal | None,
        relevant_state: dict[str, Any],
        tools: list[dict],
        conversation: ConversationSnapshot | None = None,
    ) -> InterpretRequest | None:
        """The planning prompt, for a backend that can hand it somewhere else.

        None means "not routable", and the caller runs ``interpret`` unchanged.
        """

        return None

    async def run_interpret_request(self, request: InterpretRequest) -> Interpretation:
        """Answer an already-assembled planning request on this host."""

        raise NotImplementedError

    async def run_render_request(self, request: RenderRequest) -> str | None:
        """Answer an already-assembled reply request on this host."""

        return None

    def finalize_rendered(self, request: RenderRequest, rendered: str) -> str | None:
        """Apply the post-generation contract to text produced elsewhere."""

        return rendered.strip() or None

    async def health(self) -> dict:
        return {"ok": True}

    async def close(self) -> None:
        return None
