from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
from collections.abc import Iterable
from datetime import datetime, timedelta, tzinfo

from nova_voice.affectations import apply_affectations
from nova_voice.agent_settings import AgentSettings
from nova_voice.audio.conversation import ConversationTracker
from nova_voice.capabilities.registry import CapabilityRegistry
from nova_voice.config import Settings
from nova_voice.domain import (
    ActiveGoal,
    Decision,
    Emotion,
    GoalStatus,
    HandleResult,
    Interpretation,
    PlannedAction,
    SelfProfileUpdate,
    SpeechAct,
    ToolResult,
    Utterance,
)
from nova_voice.interpretation.base import Interpreter
from nova_voice.interpretation.response_length import (
    command_acknowledgement,
    spoken_word_count,
)
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
from nova_voice.speaker_profiles import SpeakerProfileStore
from nova_voice.voice_settings import VoiceSettings

logger = logging.getLogger(__name__)

# Speech acts that are never the user engaging the assistant, even inside an
# open conversation. Turns classified this way must not extend the conversation
# window or be written into the history the model sees on later turns, so
# background talk and television can't keep a conversation alive or pollute it.
_AMBIENT_SPEECH_ACTS = frozenset(
    {SpeechAct.THIRD_PARTY, SpeechAct.QUOTED_OR_MEDIA, SpeechAct.SELF_INTENTION}
)

_SPOKEN_WORD_RE = re.compile(r"\b\w+(?:['\N{RIGHT SINGLE QUOTATION MARK}]\w+)?\b", re.UNICODE)
_PROFILE_FILLERS_RE = re.compile(
    r"^(?:(?:okay|ok|actually|well|please|hi|hello)\b[\s,;:\-\N{EM DASH}]*)+",
    re.IGNORECASE,
)
_PROFILE_NAME_WORD = r"[^\W\d_]+(?:['\N{RIGHT SINGLE QUOTATION MARK}.\-][^\W\d_]+)*"
_PROFILE_NAME_RE = re.compile(
    rf"^(?:my\s+name\s+is|(?:you\s+can\s+)?call\s+me|i\s+go\s+by|this\s+is)\s+"
    rf"(?P<name>{_PROFILE_NAME_WORD}(?:\s+{_PROFILE_NAME_WORD}){{0,3}}?)"
    r"(?:\s+speaking)?(?=\s+(?:and|but)\b|[,.;!?]|$)",
    re.IGNORECASE,
)
_PROFILE_PRONOUN_WORD = r"[^\W_]+(?:['\N{RIGHT SINGLE QUOTATION MARK}.\-][^\W_]+)*"
_PROFILE_PRONOUN_PATTERN = (
    rf"(?:my\s+pronouns\s+are|i\s+use)\s+"
    rf"(?P<pronouns>{_PROFILE_PRONOUN_WORD}(?:\s*/\s*{_PROFILE_PRONOUN_WORD}){{1,2}})"
    r"(?:\s+pronouns)?"
)
_PROFILE_PRONOUN_AT_START_RE = re.compile(rf"^{_PROFILE_PRONOUN_PATTERN}", re.IGNORECASE)
_PROFILE_PRONOUN_RE = re.compile(rf"\b{_PROFILE_PRONOUN_PATTERN}", re.IGNORECASE)


def command_word_count(transcript: str, wake_words: Iterable[str]) -> int:
    """Count command words after one leading wake phrase, if present."""

    words = _SPOKEN_WORD_RE.findall(transcript.casefold())
    wake_sequences = sorted(
        (
            _SPOKEN_WORD_RE.findall(wake_word.casefold())
            for wake_word in wake_words
            if wake_word.strip()
        ),
        key=len,
        reverse=True,
    )
    for wake in wake_sequences:
        if wake and words[: len(wake)] == wake:
            words = words[len(wake) :]
            break
    return len(words)


def explicit_self_profile_update(
    transcript: str,
    address_words: Iterable[str],
) -> SelfProfileUpdate | None:
    """Extract only clear, current-speaker identity statements.

    This deterministic fallback covers the small set of phrases Nova explicitly
    tells people to use. It is deliberately anchored after an optional wake
    phrase and harmless conversational fillers, so quoted or third-party text
    containing "my name is" does not become a biometric label.
    """

    text = transcript.strip()
    for address in sorted(
        (value.strip() for value in address_words if value.strip()),
        key=len,
        reverse=True,
    ):
        escaped = re.escape(address).replace(r"\ ", r"\s+")
        stripped, count = re.subn(
            rf"^(?:hey\s+)?{escaped}\b[\s,;:\-\N{{EM DASH}}]*",
            "",
            text,
            count=1,
            flags=re.IGNORECASE,
        )
        if count:
            text = stripped
            break
    text = _PROFILE_FILLERS_RE.sub("", text).strip()
    name_match = _PROFILE_NAME_RE.match(text)
    pronoun_match = (
        _PROFILE_PRONOUN_RE.search(text)
        if name_match is not None
        else _PROFILE_PRONOUN_AT_START_RE.match(text)
    )
    name = name_match.group("name").strip() if name_match is not None else None
    pronouns = (
        re.sub(r"\s*/\s*", "/", pronoun_match.group("pronouns")).strip()
        if pronoun_match is not None
        else None
    )
    if name is None and pronouns is None:
        return None
    return SelfProfileUpdate(name=name, pronouns=pronouns, evidence=text[:200])


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


def _dashboard_observations(
    interpretation: Interpretation, results: list[ToolResult]
) -> list[str]:
    """Compact one-line summaries of this turn's dashboard API responses.

    Only Nova (dashboard) results carrying observed state or a message are
    retained. The summaries seed the conversation context so later turns can
    reason over data an earlier tool call returned instead of losing it once the
    reply is spoken.
    """

    actions = {action.id: action for action in interpretation.actions}
    entries: list[str] = []
    for result in results:
        action = actions.get(result.action_id)
        if action is None or action.call.provider != "nova":
            continue
        label = result.target or action.call.tool
        if result.observed:
            detail = json.dumps(result.observed, separators=(",", ":"))
        elif result.message:
            detail = result.message.strip()
        else:
            continue
        suffix = "" if result.ok else " (failed)"
        entries.append(f"{label}{suffix}: {detail}")
    return entries


class NovaVoiceService:
    def __init__(
        self,
        settings: Settings,
        interpreter: Interpreter,
        registry: CapabilityRegistry,
        nova_provider: NovaProvider,
        store: TranscriptStore,
        persona: Persona,
        speaker_profiles: SpeakerProfileStore | None = None,
        sessions: SessionManager | None = None,
        conversations: ConversationTracker | None = None,
    ) -> None:
        self.settings = settings
        self.interpreter = interpreter
        self.registry = registry
        self.nova_provider = nova_provider
        self.store = store
        self.speaker_profiles = speaker_profiles
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
        self.agent_settings = AgentSettings()
        # Conversation scopes whose spoken replies are pinned to temperature 0
        # after a failed dashboard command, until that conversation ends. Keeps
        # failure-recovery wording deterministic regardless of the dashboard's
        # configured render temperature.
        self._zero_render_temperature_scopes: set[str] = set()

    def apply_voice_settings(self, settings: VoiceSettings) -> None:
        self.persona = self.persona.with_voice_settings(settings)
        self.voice_settings = settings
        self.nova_provider.agent_name = settings.spoken_name
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
        if hasattr(self.interpreter, "long_response_probability"):
            self.interpreter.long_response_probability = settings.long_response_probability
        if hasattr(self.interpreter, "agent_name"):
            self.interpreter.agent_name = settings.spoken_name
        if hasattr(self.interpreter, "wake_words"):
            self.interpreter.wake_words = list(settings.wake_words)
        if hasattr(self.interpreter, "personality"):
            self.interpreter.personality = settings.personality
        if hasattr(self.interpreter, "pronoun_instruction"):
            self.interpreter.pronoun_instruction = settings.pronoun_instruction()

    def apply_agent_settings(self, settings: AgentSettings) -> None:
        """Apply global execution-loop settings without touching voice/personality."""

        self.agent_settings = settings
        self.nova_provider.configure_verification_loop(
            enabled=settings.ralph_loop_enabled,
            max_iterations=settings.ralph_loop_max_iterations,
            sleep_seconds=settings.ralph_loop_sleep_ms / 1000,
            failure_seconds=settings.ralph_loop_failure_seconds,
        )

    def _apply_affectations(self, text: str | None) -> str | None:
        """Deterministic speech quirks on a finished reply string."""

        if not text or self.voice_settings is None:
            return text
        return apply_affectations(text, self.voice_settings.affectations)

    def _address_words(self) -> list[str]:
        """Wake words plus the agent display name, for cue stripping.

        Transcripts reach this service with the spoken wake phrase already
        rewritten to the agent's display name, so deterministic cue checks
        must skip both the raw codewords and the name.
        """

        words = [str(word) for word in getattr(self.interpreter, "wake_words", None) or []]
        agent = str(getattr(self.interpreter, "agent_name", "") or "").strip()
        if agent:
            words.append(agent)
        return words

    def end_conversation(self, room_id: str) -> None:
        self.conversations.end(room_id)
        self.sessions.end(room_id)
        self._zero_render_temperature_scopes.discard(self._scope_key(room_id))

    # Light, open-ended prompts for the dashboard's voice preview. Asking the
    # live LLM one of these — rather than speaking a fixed line — exercises the
    # render temperature, personality description, pronouns, and long-response
    # chance so an audition matches how the agent actually talks.
    _PREVIEW_QUESTIONS: tuple[str, ...] = (
        "Say a friendly hello and tell me one thing you like.",
        "In one sentence, how are you doing right now?",
        "Introduce yourself in one short, natural sentence.",
        "Tell me something small and cheerful.",
        "What's your favourite kind of weather?",
        "If you could be anywhere right now, where would it be?",
        "Give me a quick, upbeat thought for the day.",
    )

    async def render_preview_reply(self, question: str | None = None) -> str | None:
        """Generate a spoken-style reply for the dashboard voice preview.

        Runs only the response renderer with the live language-model knobs
        (temperature, personality description, pronouns, long-response chance)
        so an audition reflects how the agent actually speaks — not just how the
        voice sounds. The prompt is never interpreted as a command and nothing
        in the household is read or touched.
        """

        prompt = (question or random.choice(self._PREVIEW_QUESTIONS)).strip()
        if not prompt:
            return None
        utterance = Utterance.text(
            prompt, room_id="preview", satellite_id="dashboard-preview"
        )
        interpretation = Interpretation(
            emotion=Emotion(confidence=1.0, intensity=0.0),
            speech_act=SpeechAct.QUESTION,
            addressed_probability=1.0,
            decision=Decision.REPLY,
            confidence=1.0,
            active_goal=ActiveGoal(status=GoalStatus.SATISFIED),
        )
        rendered = await self.interpreter.render_response(
            utterance,
            interpretation,
            [],
            persona=self.persona.response_prompt,
            environment=None,
            conversation=None,
        )
        return self._apply_affectations(rendered)

    async def initialize(self) -> None:
        await self.store.initialize()
        await self.store.delete_expired()
        if self.speaker_profiles is not None:
            await self.speaker_profiles.initialize()
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
            started_new_conversation = self.conversations.start(utterance.room_id)
            if started_new_conversation:
                # A brand-new conversation (the previous one ended or timed out)
                # starts clean: no inherited goal, no lingering temperature lock.
                self.sessions.end(utterance.room_id)
                self._zero_render_temperature_scopes.discard(self._scope_key(utterance.room_id))
        # A live conversation's window is extended only by genuinely engaged turns
        # at the end of this method, so ambient speech can't keep it open forever.
        conversation = (
            self.conversations.snapshot(utterance.room_id)
            if utterance.wake_detected or utterance.conversation_active
            else None
        )
        scope = self._scope_key(utterance.room_id)
        # A forced-zero render temperature only lasts while its conversation is
        # open; once that conversation has expired, restore normal wording.
        if conversation is None:
            self._zero_render_temperature_scopes.discard(scope)
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
            wake_words=self._address_words(),
        )
        shortcut_planner = getattr(self.nova_provider, "deterministic_lighting_shortcut", None)
        shortcut = (
            shortcut_planner(
                utterance.transcript,
                wake_words=self._address_words() or ["beemo"],
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
            wake_words=self._address_words(),
        )
        # Identity claims are bound only to the biometric template extracted
        # from this exact addressed utterance. Never inherit a prior speaker
        # merely because the household-wide conversation remains open.
        profile_update = interpretation.self_profile_update
        if profile_update is None:
            profile_update = explicit_self_profile_update(
                utterance.transcript, self._address_words()
            )
            if profile_update is not None:
                interpretation = interpretation.model_copy(
                    update={"self_profile_update": profile_update}
                )
        if (
            self.speaker_profiles is not None
            and profile_update is not None
            and (utterance.wake_detected or utterance.conversation_active)
            and interpretation.speech_act not in _AMBIENT_SPEECH_ACTS
        ):
            updated_speaker = await self.speaker_profiles.apply_disclosure(
                utterance.speaker,
                profile_update,
                utterance.transcript,
            )
            utterance = utterance.model_copy(update={"speaker": updated_speaker})
        speaker_recognition_active = (
            self.settings.speaker_recognition_enabled
            and self.speaker_profiles is not None
            and (
                self.voice_settings is None
                or self.voice_settings.speaker_recognition_enabled
            )
        )
        if (
            interpretation.decision == Decision.EXECUTE
            and interpretation.actions
            and self.settings.speaker_short_execute_requires_recognition
            and speaker_recognition_active
            and utterance.speaker.status != "recognized"
            and command_word_count(utterance.transcript, self._address_words())
            <= self.settings.speaker_short_execute_max_words
        ):
            logger.info(
                "ignored short execute from unrecognized speaker: utterance=%s words=%d",
                utterance.id,
                command_word_count(utterance.transcript, self._address_words()),
            )
            interpretation = interpretation.model_copy(
                update={"decision": Decision.IGNORE, "actions": []}
            )
        # A schema-valid tool call is not enough to make ambient speech a
        # dashboard command. Resolve every proposed Nova device mutation
        # against a freshly-read dashboard device tree before policy can grant
        # execution. This stays domain-agnostic: any current or future entity
        # exposed by the dashboard is eligible, while nonexistent targets from
        # TV/media dialogue are discarded before they affect the household.
        try:
            invalid_device_actions = await self.nova_provider.invalid_device_action_ids(
                interpretation.actions
            )
        except NovaDashboardError:
            # If the tree cannot be read, it cannot establish that a household
            # device would be affected. Fail closed without taking down speech
            # interpretation or turning an outage into a voice-service error.
            invalid_device_actions = {
                action.id
                for action in interpretation.actions
                if action.call.provider == "nova"
                and action.call.tool in {"nova.control", "nova.lighting_shortcut"}
            }
        if invalid_device_actions:
            still_valid_actions = [
                action
                for action in interpretation.actions
                if action.id not in invalid_device_actions
            ]
            if not still_valid_actions:
                interpretation = interpretation.model_copy(
                    update={
                        "decision": (
                            Decision.REPLY
                            if utterance.wake_detected or utterance.conversation_active
                            else Decision.IGNORE
                        ),
                        "actions": [],
                    }
                )
            else:
                interpretation = interpretation.model_copy(update={"actions": still_valid_actions})
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

        # A failed dashboard command opens a conversation (if one is not already
        # open) so the user can respond without repeating the wake word, and
        # pins spoken replies to temperature 0 until that conversation ends so
        # failure recovery stays deterministic regardless of the configured
        # render temperature.
        nova_action_ids = {
            action.id for action in interpretation.actions if action.call.provider == "nova"
        }
        dashboard_task_failed = outcome.execute and any(
            (not result.ok) and result.action_id in nova_action_ids for result in results
        )
        if dashboard_task_failed:
            if conversation is None:
                self.conversations.start(utterance.room_id)
                conversation = self.conversations.snapshot(utterance.room_id)
            self._zero_render_temperature_scopes.add(scope)

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
        # TTS-cacheable.  A failure-recovery conversation forces zero here.
        force_zero_temperature = scope in self._zero_render_temperature_scopes
        effective_render_temperature = (
            0.0 if force_zero_temperature else getattr(self.interpreter, "render_temperature", 0.0)
        )
        wants_varied_wording = effective_render_temperature > 0 and response_text is not None
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
        # A verified command's spoken acknowledgement length is rolled fresh per
        # reply in [0, commandReplyMaxWords]; zero means a silent acknowledgement
        # (the command still ran, nothing is spoken).
        command_max_words: int | None = None
        if verified_dashboard_command and self.voice_settings is not None:
            command_max_words = random.randint(0, self.voice_settings.command_reply_max_words)

        if verified_dashboard_command and command_max_words == 0:
            response_text = None
        elif not outcome.shadowed and (
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
                relevant_state=relevant_state,
                conversation=conversation,
                temperature=effective_render_temperature,
                command_max_words=command_max_words,
            )
            response_text = rendered or response_text
        # Affectations run on the finished reply — template or model alike — so
        # the quirk is consistent everywhere the text goes: TTS, transcript,
        # and the conversation history the model sees on later turns.
        response_text = self._apply_affectations(response_text)
        if (
            verified_dashboard_command
            and command_max_words is not None
            and command_max_words > 0
            and response_text is not None
            and spoken_word_count(response_text) != command_max_words
        ):
            # Affectations run after the renderer and may remove words. Keep the
            # user's sampled command length exact even after those transforms.
            response_text = command_acknowledgement(command_max_words)
        engaged = (
            utterance.wake_detected or interpretation.speech_act not in _AMBIENT_SPEECH_ACTS
        )
        if conversation is not None:
            if has_abandonment(utterance.transcript, self._address_words()):
                # The user ended the conversation: clear every scrap of its
                # context so nothing carries into the next one.
                self.conversations.end(utterance.room_id)
                self.sessions.end(utterance.room_id)
                self._zero_render_temperature_scopes.discard(scope)
            elif engaged:
                # Only a genuinely engaged turn keeps the window open and is
                # remembered; ambient/third-party/media speech neither extends
                # the conversation nor pollutes the history seen on later turns.
                self.conversations.refresh(utterance.room_id)
                self.conversations.append_turn(
                    utterance.room_id,
                    utterance.transcript,
                    response_text,
                    speaker_name=(
                        utterance.speaker.display_name
                        if utterance.speaker.status == "recognized"
                        else None
                    ),
                )
                observations = _dashboard_observations(interpretation, results)
                if observations:
                    self.conversations.record_observations(utterance.room_id, observations)
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
            "speakerShortExecuteGate": {
                "enabled": self.settings.speaker_short_execute_requires_recognition,
                "maxWords": self.settings.speaker_short_execute_max_words,
            },
            "provider": provider_health,
            "llm": llm_health,
            "retainedTranscripts": await self.store.count(),
            "speakerProfiles": (
                await self.speaker_profiles.health()
                if self.speaker_profiles is not None
                else {"ok": True, "enabled": False}
            ),
            "agentSettings": self.agent_settings.model_dump(mode="json", by_alias=True),
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
