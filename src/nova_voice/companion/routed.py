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

``interpret`` is routed here too, and it is the one that needed more than
schema validation. A plan whose shape is valid can still name a tool that does
not exist: the action would be planned, spoken about, and only refused at
execution — after the assistant had already said it was doing it. So every
action is checked against the exact catalogue the turn offered, and one bad
action discards the whole plan rather than being quietly dropped from it.

Both hot-path routes ship ``local``. The capability is built and switchable at
runtime; on measured evidence the on-device model is not yet good enough to
hold the spoken turn. See docs/evidence/companion-offload-live-20260815.md.
"""

from __future__ import annotations

import logging
from typing import Any

from nova_voice.audio.conversation import (
    ConversationSnapshot,
    compact_memory_to_budget,
    compact_state_to_budget,
)
from nova_voice.companion.compaction import DEFAULT_CONTEXT_TOKENS, compact_for_companion
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
from nova_voice.interpretation.base import Interpreter, InterpretRequest, RenderRequest

logger = logging.getLogger(__name__)

# Elastic parts of the reply prompt's structured input. Everything else in
# ``facts`` is either tiny or load-bearing — the response instruction, the
# decision, the tool results the reply must not contradict — and is sent whole.
_ELASTIC_FACTS = ("relevantState", "selectedMemory")


def _catalogue_names(offered_tools: list[dict]) -> frozenset[str]:
    """The names a turn actually showed the phone, in OpenAI tool-schema shape.

    One source for both checks — what may be called back for during reasoning,
    and what may appear in the returned plan — so the two can never disagree
    about what "offered" means.
    """

    return frozenset(
        name
        for tool in offered_tools
        if isinstance(tool, dict)
        and isinstance(tool.get("function"), dict)
        and (name := tool["function"].get("name"))
    )


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
        request = self.inner.build_interpret_request(
            utterance,
            active_goal=active_goal,
            relevant_state=relevant_state,
            tools=tools,
            conversation=conversation,
        )
        if request is None:
            return await self.inner.interpret(
                utterance,
                active_goal=active_goal,
                relevant_state=relevant_state,
                tools=tools,
                conversation=conversation,
            )

        snapshot = self.router.sessions.snapshot()
        context_tokens = snapshot.hot_context_tokens or DEFAULT_CONTEXT_TOKENS
        compacted = compact_for_companion(
            instructions=request.system,
            tools=request.opening_context.get("semanticTools") or [],
            state=request.opening_context.get("relevantState") or {},
            memory=request.opening_context.get("selectedMemory") or [],
            history=request.history,
            context_tokens=context_tokens,
        )
        payload = compacted.as_payload() | {"turn": request.turn_context}

        outcome = await self.router.run(
            "interpret",
            payload,
            lambda: self.inner.run_interpret_request(request),
            parse=lambda result: self._validated_interpretation(result, compacted.tools),
            # Exactly what this turn showed the phone, and nothing else. The
            # same set gates both directions: a tool it may *call back* for
            # while reasoning, and a tool it may name in the plan it returns.
            allowed_tools=_catalogue_names(compacted.tools),
        )
        logger.info(
            "interpret resolved source=%s reason=%s elapsed_ms=%s",
            outcome.source,
            outcome.reason,
            outcome.elapsed_ms,
        )
        if isinstance(outcome.value, Interpretation):
            return outcome.value
        # Reached only when the route is `disabled` or `companion_only`, which
        # return without running anything locally. Neither is a state a spoken
        # turn can proceed from — there is no "no interpretation" branch above
        # this — so the local pass runs regardless of what the route said.
        return await self.inner.run_interpret_request(request)

    @staticmethod
    def _validated_interpretation(result: object, offered_tools: list[dict]) -> Any:
        """Accept a plan from the phone only if it stays inside what it was shown.

        Schema validation is not enough on its own. It proves the shape, not
        that the plan is *actionable*: a model can invent a plausible tool name,
        and the resulting action would be planned, spoken about, and only then
        refused — after the assistant has already said it was doing it.

        So every action is checked against the exact catalogue this turn sent.
        One bad action discards the whole interpretation rather than being
        dropped from it, because a plan with a clause removed is not the plan
        the model made: "turn the lights off and lock the door" minus one
        clause still reads as success.
        """

        parsed = parse_result("interpret", result)
        if not isinstance(parsed, Interpretation):
            return None

        allowed = _catalogue_names(offered_tools)
        for action in parsed.actions:
            qualified = f"{action.call.provider}.{action.call.tool}"
            if qualified not in allowed and action.call.tool not in allowed:
                logger.warning(
                    "companion interpretation planned an unoffered tool: %s", qualified
                )
                return None
        return parsed

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

    def build_interpret_request(self, *args, **kwargs) -> InterpretRequest | None:
        return self.inner.build_interpret_request(*args, **kwargs)

    async def run_interpret_request(self, request: InterpretRequest) -> Interpretation:
        return await self.inner.run_interpret_request(request)

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
