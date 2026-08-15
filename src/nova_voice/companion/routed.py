"""An Interpreter that offers its passes to a companion before running them.

Wrapping rather than editing each caller is deliberate. ``render_response`` has
three call sites with different arguments, and the reply contract — word
budgets, the "never claim a device changed" rule, the canned acknowledgement
that replaces an overrun — is enforced *after* generation. Routing at the call
sites would have meant three copies of that, and three chances for them to
diverge.

The wrapper builds the request **once** and hands the same object to both
paths. That is not a tidiness point: the long-form branch is chosen by a coin
flip inside the builder, so rebuilding it for the fallback would answer a
different question from the one the phone was asked, and the word-budget check
would then be applied against the wrong contract.

Everything else delegates unchanged, including ``interpret``. The spoken turn's
planning pass is not routed here — see the roadmap's NPT-307 — because it is
the one pass whose failure mode is a wrong action in the house rather than a
slower or blander sentence.
"""

from __future__ import annotations

import logging
from typing import Any

from nova_voice.audio.conversation import (
    ConversationSnapshot,
    compact_memory_to_budget,
    compact_state_to_budget,
)
from nova_voice.companion.router import CompanionWorkloadRouter
from nova_voice.companion.workloads import parse_result
from nova_voice.domain import (
    ActiveGoal,
    Interpretation,
    SelfProfileUpdate,
    ToolResult,
    Utterance,
    VerificationVerdict,
)
from nova_voice.interpretation.base import Interpreter, RenderRequest

logger = logging.getLogger(__name__)

# Elastic parts of the reply prompt's structured input. Everything else in
# ``facts`` is either tiny or load-bearing — the response instruction, the
# decision, the tool results the reply must not contradict — and is sent whole.
_ELASTIC_FACTS = ("relevantState", "selectedMemory")


def companion_render_payload(request: RenderRequest, *, context_tokens: int) -> dict:
    """Shape a reply request for a device with a much smaller context window.

    Only the two genuinely elastic inputs are trimmed, with the same helpers
    Iridium uses, so the phone sheds context the same way rather than in some
    second, divergent manner. The instructions are never trimmed: they are what
    keep a remote writer inside the same rules as the local one, and a reply
    generated without them is not a cheaper reply, it is an off-contract one.
    """

    facts = dict(request.facts)
    # A quarter of the window for the household snapshot, an eighth for memory.
    # Deliberately conservative: this prompt also carries the instructions and
    # the turn's own results, and an over-full window costs the answer.
    if isinstance(facts.get("relevantState"), dict):
        facts["relevantState"] = compact_state_to_budget(
            facts["relevantState"], max(1, context_tokens // 4)
        )
    if isinstance(facts.get("selectedMemory"), list):
        facts["selectedMemory"] = compact_memory_to_budget(
            facts["selectedMemory"], max(1, context_tokens // 8)
        )
    return {
        "instructions": request.system,
        "facts": facts,
        "history": request.history,
        "maxTokens": request.max_tokens,
    }


class RoutedInterpreter(Interpreter):
    """Delegates to ``inner``, offering routable passes to the companion first."""

    def __init__(self, inner: Interpreter, router: CompanionWorkloadRouter) -> None:
        self.inner = inner
        self.router = router

    # -- routed ---------------------------------------------------------------

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
        request = self.inner.build_render_request(
            utterance,
            interpretation,
            results,
            persona=persona,
            environment=environment,
            relevant_state=relevant_state,
            conversation=conversation,
            temperature=temperature,
            command_max_words=command_max_words,
            bare_wake_max_words=bare_wake_max_words,
        )
        if request is None:
            # A backend with no routable prompt — deterministic and test
            # interpreters — behaves exactly as it did before.
            return await self.inner.render_response(
                utterance,
                interpretation,
                results,
                persona=persona,
                environment=environment,
                relevant_state=relevant_state,
                conversation=conversation,
                temperature=temperature,
                command_max_words=command_max_words,
                bare_wake_max_words=bare_wake_max_words,
            )

        snapshot = self.router.sessions.snapshot()
        context_tokens = snapshot.hot_context_tokens or 4096
        outcome = await self.router.run(
            "render_response",
            companion_render_payload(request, context_tokens=context_tokens),
            lambda: self.inner.run_render_request(request),
            parse=lambda payload: parse_result("render_response", payload),
        )
        logger.info(
            "render_response resolved source=%s reason=%s elapsed_ms=%s",
            outcome.source,
            outcome.reason,
            outcome.elapsed_ms,
        )
        if outcome.source != "companion":
            # The local path already applied the contract on its way out.
            return outcome.value
        rendered = getattr(outcome.value, "text", None)
        if not isinstance(rendered, str):
            return None
        # The same post-generation contract, applied to words written on a
        # phone. A companion is not trusted to have honoured its own word
        # budget any more than the local model is.
        return self.inner.finalize_rendered(request, rendered)

    # -- delegated ------------------------------------------------------------

    async def interpret(
        self,
        utterance: Utterance,
        *,
        active_goal: ActiveGoal | None,
        relevant_state: dict[str, Any],
        tools: list[dict],
        conversation: ConversationSnapshot | None = None,
    ) -> Interpretation:
        return await self.inner.interpret(
            utterance,
            active_goal=active_goal,
            relevant_state=relevant_state,
            tools=tools,
            conversation=conversation,
        )

    async def extract_self_profile_update(
        self, utterance: Utterance
    ) -> SelfProfileUpdate | None:
        return await self.inner.extract_self_profile_update(utterance)

    async def confirm_objective(
        self,
        utterance: Utterance,
        pending: list[dict[str, Any]],
    ) -> VerificationVerdict | None:
        return await self.inner.confirm_objective(utterance, pending)

    async def classify_icon(self, name: str, icons: list[str]) -> str | None:
        return await self.inner.classify_icon(name, icons)

    def build_render_request(self, *args, **kwargs) -> RenderRequest | None:
        return self.inner.build_render_request(*args, **kwargs)

    async def run_render_request(self, request: RenderRequest) -> str | None:
        return await self.inner.run_render_request(request)

    def finalize_rendered(self, request: RenderRequest, rendered: str) -> str | None:
        return self.inner.finalize_rendered(request, rendered)

    async def health(self) -> dict:
        return await self.inner.health()

    async def close(self) -> None:
        await self.inner.close()

    def __getattr__(self, name: str):
        """Forward anything this wrapper does not name itself.

        Backends carry configuration the rest of the service reads directly —
        ``agent_name``, ``web_access_enabled``, the personality text. Forwarding
        keeps the wrapper from silently hiding it, which would show up as a
        behaviour change far away from here.
        """

        return getattr(self.inner, name)
