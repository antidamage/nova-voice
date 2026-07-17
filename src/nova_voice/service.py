from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Iterable
from datetime import datetime, timedelta, tzinfo

from nova_voice.audio.conversation import ConversationTracker
from nova_voice.capabilities.registry import CapabilityRegistry
from nova_voice.config import Settings
from nova_voice.domain import (
    Decision,
    HandleResult,
    PlannedAction,
    SpeechAct,
    ToolResult,
    Utterance,
)
from nova_voice.interpretation.base import Interpreter
from nova_voice.interpretation.speech_cues import (
    enforce_decision_consistency,
    enforce_speech_cues,
    has_abandonment,
)
from nova_voice.persistence import TranscriptStore
from nova_voice.persona import Persona
from nova_voice.policy import ExecutionPolicy
from nova_voice.providers.nova.client import NovaDashboardError
from nova_voice.providers.nova.provider import NovaProvider
from nova_voice.sessions import SessionManager
from nova_voice.voice_settings import VoiceSettings

logger = logging.getLogger(__name__)


def current_clock_context(
    timezone: tzinfo | None = None,
    *,
    now: datetime | None = None,
) -> dict[str, str]:
    """Rounded household-local date/time for the conversation-start prompt."""

    current = now or datetime.now(timezone)
    if current.tzinfo is None:
        current = current.astimezone()
    elif timezone is not None:
        current = current.astimezone(timezone)
    rounded = (current + timedelta(seconds=30)).replace(second=0, microsecond=0)
    return {
        "iso": rounded.isoformat(timespec="minutes"),
        "date": rounded.strftime("%A %d %B %Y"),
        "time": rounded.strftime("%I:%M %p").lstrip("0").lower(),
    }


class NovaVoiceService:
    def __init__(
        self,
        settings: Settings,
        interpreter: Interpreter,
        registry: CapabilityRegistry,
        nova_provider: NovaProvider,
        store: TranscriptStore,
        persona: Persona,
        sessions: SessionManager | None = None,
        conversations: ConversationTracker | None = None,
    ) -> None:
        self.settings = settings
        self.interpreter = interpreter
        self.registry = registry
        self.nova_provider = nova_provider
        self.store = store
        self.persona = persona
        # Satellites within earshot share one conversation/goal scope so a
        # follow-up elected on another microphone continues the same exchange.
        scope_key = (
            (lambda room_id: "household")
            if settings.arbitration_scope == "household"
            else None
        )
        self.sessions = sessions or SessionManager(
            follow_up_seconds=settings.conversation_idle_seconds,
            key_fn=scope_key,
        )
        self.conversations = conversations or ConversationTracker(
            idle_seconds=settings.conversation_idle_seconds,
            max_seconds=settings.conversation_max_seconds,
            key_fn=scope_key,
        )
        self._scope_key = scope_key if scope_key is not None else lambda room_id: room_id
        self._turn_locks: dict[str, asyncio.Lock] = {}
        self.policy = ExecutionPolicy(settings)
        self.voice_settings: VoiceSettings | None = None

    def apply_voice_settings(self, settings: VoiceSettings) -> None:
        self.persona = self.persona.with_voice_settings(settings)
        self.voice_settings = settings
        self.nova_provider.agent_name = settings.agent_name
        # The conversation window is dashboard-tunable and applies live: both
        # the wide-vocabulary follow-up window and the goal-session clock must
        # follow the same value or turns die on whichever timer is shorter.
        self.conversations.set_idle_seconds(settings.conversation_idle_seconds)
        self.sessions.set_follow_up_seconds(settings.conversation_idle_seconds)
        # The renderer temperature, agent identity, and wake words are
        # interpreter knobs rather
        # than persona state; not every interpreter implementation has them.
        if hasattr(self.interpreter, "render_temperature"):
            self.interpreter.render_temperature = settings.temperature
        if hasattr(self.interpreter, "agent_name"):
            self.interpreter.agent_name = settings.agent_name
        if hasattr(self.interpreter, "wake_words"):
            self.interpreter.wake_words = list(settings.wake_words)
        if hasattr(self.interpreter, "personality"):
            self.interpreter.personality = settings.personality

    def end_conversation(self, room_id: str) -> None:
        self.conversations.end(room_id)
        self.sessions.end(room_id)

    async def initialize(self) -> None:
        await self.store.initialize()
        await self.store.delete_expired()
        # Nova is an optional capability from the core service's point of view.
        # Keep the interpreter, retention janitor, and satellite health endpoint
        # available when the dashboard is restarting; a later health/refresh
        # call will make the provider available again without a process restart.
        try:
            await self.nova_provider.refresh(force=True)
        except NovaDashboardError:
            logger.warning("Nova provider unavailable during startup", exc_info=True)

    async def handle(self, utterance: Utterance) -> HandleResult:
        lock = self._turn_locks.setdefault(self._scope_key(utterance.room_id), asyncio.Lock())
        async with lock:
            return await self._handle(utterance)

    async def _handle(self, utterance: Utterance) -> HandleResult:
        started = time.perf_counter()
        timings_ms: dict[str, float] = {}
        if utterance.wake_detected:
            self.conversations.start(utterance.room_id)
        elif utterance.conversation_active:
            self.conversations.refresh(utterance.room_id)
        conversation = (
            self.conversations.snapshot(utterance.room_id)
            if utterance.wake_detected or utterance.conversation_active
            else None
        )
        goal = self.sessions.active_goal(utterance.room_id, utterance.ended_at)
        context_started = time.perf_counter()
        try:
            relevant_state = await asyncio.wait_for(
                self.nova_provider.prompt_context(utterance.room_id),
                timeout=self.settings.provider_context_timeout_seconds,
            )
        except (NovaDashboardError, TimeoutError):
            # Ambient speech must still be transcribed/classified while Nova is
            # offline or DNS is slow, but a missing dashboard must never add
            # seconds to the spoken turn. No stale household state is supplied.
            relevant_state = {"room": utterance.room_id, "zones": [], "nearbyTargets": []}
        # Date/time and weather are a conversation-start snapshot.  Household
        # target state remains live on every turn, but these ambient prompt
        # injections are never appended again during the same conversation.
        weather = relevant_state.pop("weather", None)
        if conversation is not None and conversation.initial_environment is None:
            conversation = self.conversations.initialize_prompt(
                utterance.room_id,
                environment={
                    "now": current_clock_context(self.settings.household_tzinfo()),
                    "weather": weather,
                },
                personality=getattr(self.interpreter, "personality", ""),
                persona_prompt=self.persona.response_prompt,
            )
        timings_ms["providerContext"] = round((time.perf_counter() - context_started) * 1000, 3)

        interpretation_started = time.perf_counter()
        interpretation = await self.interpreter.interpret(
            utterance,
            active_goal=goal,
            relevant_state=relevant_state,
            tools=self.registry.tool_catalog(),
            conversation=conversation,
        )
        timings_ms["interpretation"] = round(
            (time.perf_counter() - interpretation_started) * 1000, 3
        )
        interpretation = enforce_speech_cues(
            utterance.transcript,
            interpretation,
            wake_words=getattr(self.interpreter, "wake_words", None),
        )
        shortcut_planner = getattr(self.nova_provider, "deterministic_lighting_shortcut", None)
        shortcut = (
            shortcut_planner(
                utterance.transcript,
                wake_words=getattr(self.interpreter, "wake_words", ["beemo"]),
            )
            if callable(shortcut_planner)
            else None
        )
        if shortcut is not None:
            interpretation = interpretation.model_copy(
                update={
                    "speech_act": SpeechAct.DIRECTIVE,
                    "addressed_probability": 1.0,
                    "decision": Decision.EXECUTE,
                    "confidence": 1.0,
                    "actions": [shortcut],
                    "response_plan": interpretation.response_plan.model_copy(
                        update={"requires_post_tool_rendering": True}
                    ),
                }
            )
        interpretation = enforce_decision_consistency(
            utterance,
            interpretation,
            addressed_threshold=self.settings.active_addressed_threshold,
            wake_words=getattr(self.interpreter, "wake_words", None),
        )
        policy_started = time.perf_counter()
        outcome = self.policy.evaluate(
            utterance,
            interpretation,
            session_active=goal is not None,
        )
        timings_ms["policy"] = round((time.perf_counter() - policy_started) * 1000, 3)

        # Ambient speech must still be classified so the assistant can decide
        # whether to act, but its verbatim words are only retained when the
        # household actually addressed the assistant: a wake word, an open
        # follow-up conversation, or a directive confident enough to clear the
        # passive-execution policy (a genuine dashboard command, executed or
        # shadowed). Background chatter, television, and third-party talk are
        # discarded once interpreted rather than written to the transcript store.
        retention_started = time.perf_counter()
        addressed_for_retention = (
            utterance.wake_detected
            or utterance.conversation_active
            or outcome.execute
            or outcome.shadowed
        )
        if addressed_for_retention:
            await self.store.add(utterance, interpretation)
        timings_ms["retention"] = round((time.perf_counter() - retention_started) * 1000, 3)

        results: list[ToolResult] = []
        execution_started = time.perf_counter()
        if outcome.shadowed:
            results = self._shadow_plan(interpretation.actions)
        elif outcome.execute:
            results = await self._execute_plan(interpretation.actions)
        timings_ms["execution"] = round((time.perf_counter() - execution_started) * 1000, 3)

        # A waked/conversation turn whose planned actions were blocked by
        # policy must still answer the user — silence is only correct for
        # unaddressed ambient speech.  Downgrade to a conversational reply so
        # the model renders an answer (it cannot execute anything from here).
        addressed = utterance.wake_detected or utterance.conversation_active
        if (
            addressed
            and not outcome.execute
            and not outcome.shadowed
            and interpretation.decision == Decision.EXECUTE
            and not results
        ):
            interpretation = interpretation.model_copy(
                update={"decision": Decision.REPLY, "actions": []}
            )

        session_started = time.perf_counter()
        self.sessions.update(
            utterance,
            interpretation,
            results if outcome.execute else [],
            executed=outcome.execute,
        )
        timings_ms["session"] = round((time.perf_counter() - session_started) * 1000, 3)
        response_started = time.perf_counter()
        response_text = self.persona.render(
            utterance,
            interpretation.decision,
            results,
            shadowed=outcome.shadowed,
        )
        # A nonzero renderer temperature opts successful command turns into
        # the model renderer too — otherwise confirmations are a fixed
        # template and the dashboard temperature control is inaudible on the
        # most common turn type.  Zero keeps confirmations instant and
        # TTS-cacheable.
        wants_varied_wording = (
            getattr(self.interpreter, "render_temperature", 0.0) > 0
            and response_text is not None
        )
        verified_dashboard_command = bool(
            outcome.execute
            and results
            and all(result.ok for result in results)
            and any(
                action.call.provider == "nova"
                and action.call.tool in {"nova.control", "nova.lighting_shortcut"}
                for action in interpretation.actions
            )
        )
        if not outcome.shadowed and (
            self._needs_model_render(interpretation, results)
            or wants_varied_wording
            or verified_dashboard_command
        ):
            rendered = await self.interpreter.render_response(
                utterance,
                interpretation,
                results,
                persona=self.persona.response_prompt,
                environment=None,
                conversation=conversation,
            )
            response_text = rendered or response_text
        if conversation is not None:
            if has_abandonment(
                utterance.transcript, getattr(self.interpreter, "wake_words", None)
            ):
                self.conversations.end(utterance.room_id)
            else:
                self.conversations.append_turn(
                    utterance.room_id,
                    utterance.transcript,
                    response_text,
                )
        timings_ms["response"] = round((time.perf_counter() - response_started) * 1000, 3)
        timings_ms["total"] = round((time.perf_counter() - started) * 1000, 3)
        logger.info(
            "utterance handled id=%s room=%s decision=%s actions=%d executed=%s shadowed=%s",
            utterance.id,
            utterance.room_id,
            interpretation.decision,
            len(interpretation.actions),
            outcome.execute,
            outcome.shadowed,
        )
        return HandleResult(
            utterance_id=utterance.id,
            interpretation=interpretation,
            executed=outcome.execute,
            shadowed=outcome.shadowed,
            policy_reason=outcome.reason,
            results=results,
            response_text=response_text,
            response_tone_instruction=self.persona.tone_instruction(interpretation.emotion),
            timings_ms=timings_ms,
        )

    async def _execute_plan(self, actions: Iterable[PlannedAction]) -> list[ToolResult]:
        """Execute a bounded action DAG with provider-declared parallelism.

        Actions are still considered in their declared order.  Independent
        actions may share a wave only when their provider metadata explicitly
        marks the tool ``parallel_safe`` (and low-risk).  A serial action forms
        a barrier, preserving ordered plans such as "turn on, then set level".
        Dependency failures are materialized as blocked results so downstream
        actions never run and the caller can report partial failure.
        """
        ordered = sorted(actions, key=lambda candidate: candidate.order)
        canonical_actions: dict[str, PlannedAction] = {}
        invalid_actions: set[str] = set()
        duplicate_of: dict[str, str] = {}
        seen_idempotent: dict[str, str] = {}
        for action in ordered:
            try:
                canonical = self.registry.validate_action(action)
                canonical_actions[action.id] = canonical
                policy = self.registry.policy_for(canonical.call.provider, canonical.call.tool)
                idempotent = bool(policy is not None and getattr(policy, "idempotent", False))
                if idempotent and not canonical.call.arguments.get("nonIdempotent", False):
                    signature = json.dumps(
                        [
                            canonical.call.provider,
                            canonical.call.tool,
                            canonical.call.arguments,
                        ],
                        sort_keys=True,
                        separators=(",", ":"),
                        default=str,
                    )
                    previous = seen_idempotent.get(signature)
                    if previous is None:
                        seen_idempotent[signature] = action.id
                    else:
                        duplicate_of[action.id] = previous
            except (KeyError, ValueError):
                invalid_actions.add(action.id)
        if invalid_actions:
            # Do not partially mutate the household when the model mixed a
            # supported request with an unknown/invalid tool call.
            return [
                ToolResult(
                    action_id=action.id,
                    ok=False,
                    code="invalid" if action.id in invalid_actions else "blocked",
                    requested=action.call.arguments,
                    message=(
                        "Action did not match an allowlisted semantic tool schema"
                        if action.id in invalid_actions
                        else "Plan contains an unsupported action; nothing was executed"
                    ),
                )
                for action in ordered
            ]
        pending = {action.id: action for action in ordered}
        results: list[ToolResult] = []
        by_id: dict[str, ToolResult] = {}

        def append_result(action: PlannedAction, result: ToolResult) -> None:
            pending.pop(action.id, None)
            by_id[action.id] = result
            results.append(result)

        async def invoke(action: PlannedAction, provider) -> ToolResult:
            try:
                return await provider.execute(action)
            except Exception:  # providers must not take down sibling actions
                logger.exception(
                    "capability execution failed provider=%s tool=%s",
                    action.call.provider,
                    action.call.tool,
                )
                return ToolResult(
                    action_id=action.id,
                    ok=False,
                    code="backend_error",
                    requested=action.call.arguments,
                    message="Capability provider failed while executing the action",
                )

        while pending:
            # Dependencies are constrained to earlier actions by Interpretation,
            # but retain a safe guard here for callers that invoke this method
            # directly or provide a malformed plan.
            ready = [
                action
                for action in ordered
                if action.id in pending
                and all(dependency in by_id for dependency in action.depends_on)
            ]
            if not ready:
                for action in sorted(pending.values(), key=lambda candidate: candidate.order):
                    append_result(
                        action,
                        ToolResult(
                            action_id=action.id,
                            ok=False,
                            code="invalid",
                            requested=action.call.arguments,
                            message="Action dependency graph could not be resolved",
                        ),
                    )
                break

            runnable: list[tuple[PlannedAction, PlannedAction, object, object]] = []
            # Resolve dependency/policy/schema decisions before launching a
            # wave.  Invalid or blocked actions are terminal and can unlock a
            # dependent blocked result on the next loop.
            for action in ready:
                duplicate_source = duplicate_of.get(action.id)
                if duplicate_source is not None and duplicate_source not in by_id:
                    # Wait until the canonical action has completed so a
                    # failed source cannot be reported as a successful duplicate.
                    continue
                failed_dependencies = [
                    dependency for dependency in action.depends_on if not by_id[dependency].ok
                ]
                if failed_dependencies:
                    append_result(
                        action,
                        ToolResult(
                            action_id=action.id,
                            ok=False,
                            code="blocked",
                            requested=action.call.arguments,
                            message="Action dependency failed",
                        ),
                    )
                    continue
                if duplicate_source is not None:
                    source = by_id[duplicate_source]
                    append_result(
                        action,
                        ToolResult(
                            action_id=action.id,
                            ok=source.ok,
                            code=source.code,
                            target=source.target,
                            observed=source.observed,
                            message="Duplicate idempotent action coalesced",
                        ),
                    )
                    continue
                try:
                    canonical_action = canonical_actions[action.id]
                    policy = self.registry.policy_for(
                        canonical_action.call.provider, canonical_action.call.tool
                    )
                    if policy is None:
                        append_result(
                            action,
                            ToolResult(
                                action_id=action.id,
                                ok=False,
                                code="blocked",
                                requested=action.call.arguments,
                                message="Capability has no execution policy",
                            ),
                        )
                        continue
                    if policy.risk == "blocked":
                        append_result(
                            action,
                            ToolResult(
                                action_id=action.id,
                                ok=False,
                                code="blocked",
                                requested=action.call.arguments,
                                message="Capability policy blocks this operation",
                            ),
                        )
                        continue
                    if policy.risk == "confirmation" or policy.requires_confirmation:
                        append_result(
                            action,
                            ToolResult(
                                action_id=action.id,
                                ok=False,
                                code="blocked",
                                requested=action.call.arguments,
                                message="Capability requires an explicit confirmation turn",
                            ),
                        )
                        continue
                    provider = self.registry.provider(canonical_action.call.provider)
                    runnable.append((action, canonical_action, policy, provider))
                except (KeyError, ValueError):
                    append_result(
                        action,
                        ToolResult(
                            action_id=action.id,
                            ok=False,
                            code="invalid",
                            requested=action.call.arguments,
                            message="Action did not match an allowlisted semantic tool schema",
                        ),
                    )

            if not runnable:
                continue

            def is_parallel_safe(item: tuple[PlannedAction, PlannedAction, object, object]) -> bool:
                policy = item[2]
                return bool(
                    policy is not None
                    and policy.risk == "low"
                    and policy.parallel_safe
                    and not policy.requires_confirmation
                )

            # A non-parallel-safe action is a barrier.  Complete any earlier
            # parallel-safe actions first, then execute the barrier by itself;
            # later actions wait for the next wave.
            first_serial = next(
                (index for index, item in enumerate(runnable) if not is_parallel_safe(item)),
                None,
            )
            if first_serial is not None and first_serial == 0:
                action, canonical_action, _policy, provider = runnable[0]
                append_result(action, await invoke(canonical_action, provider))
                continue

            if first_serial is not None:
                runnable = runnable[:first_serial]

            # Every ready action in this wave has opted into safe concurrency.
            # gather preserves input ordering, so response/result ordering stays
            # deterministic even when provider calls complete out of order.
            outputs = await asyncio.gather(
                *(
                    invoke(canonical_action, provider)
                    for _action, canonical_action, _policy, provider in runnable
                )
            )
            for (action, _canonical_action, _policy, _provider), result in zip(
                runnable, outputs, strict=True
            ):
                append_result(action, result)

        # append_result follows waves; return in declared order regardless of
        # whether one wave completed faster than another.
        return sorted(
            results,
            key=lambda result: next(
                action.order for action in ordered if action.id == result.action_id
            ),
        )

    @staticmethod
    def _needs_model_render(interpretation, results: list[ToolResult]) -> bool:
        return bool(
            interpretation.decision.value in {"reply", "clarify"}
            or interpretation.response_plan.requires_post_tool_rendering
            or any(not result.ok for result in results)
        )

    def _shadow_plan(self, actions: Iterable[PlannedAction]) -> list[ToolResult]:
        results: list[ToolResult] = []
        for action in actions:
            try:
                self.registry.validate_action(action)
                code = "shadowed"
                message = "Action was not executed because shadow mode is active"
            except ValueError:
                code = "invalid"
                message = "Action did not match an allowlisted semantic tool schema"
            results.append(
                ToolResult(
                    action_id=action.id,
                    ok=False,
                    code=code,
                    target=str(action.call.arguments.get("target") or "") or None,
                    requested=action.call.arguments,
                    message=message,
                )
            )
        return results

    async def health(self) -> dict:
        provider_health = await self.nova_provider.health()
        llm_health = await self.interpreter.health()
        return {
            "ok": bool(provider_health.get("ok") and llm_health.get("ok")),
            "mode": self.settings.mode,
            "shadowMode": self.settings.shadow_mode,
            "passiveExecutionEnabled": self.settings.passive_execution_enabled,
            "provider": provider_health,
            "llm": llm_health,
            "retainedTranscripts": await self.store.count(),
            "voiceSettings": (
                self.voice_settings.model_dump(mode="json", by_alias=True)
                if self.voice_settings is not None
                else None
            ),
        }

    async def close(self) -> None:
        self.store.stop()
        await self.interpreter.close()
        await self.registry.close()
