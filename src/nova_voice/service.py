from __future__ import annotations

import asyncio
import copy
import json
import logging
import random
import re
import time
from collections.abc import Awaitable, Callable, Iterable
from datetime import UTC, datetime, timedelta, tzinfo
from typing import Any

from nova_voice.affectations import apply_affectations
from nova_voice.agent_settings import AgentSettings
from nova_voice.audio.conversation import ConversationSnapshot, ConversationTracker
from nova_voice.audio.prefetch import ForegroundPrefetch, likely_tools
from nova_voice.authority import HouseholdAuthority
from nova_voice.automation import AutomationManager
from nova_voice.briefings import BriefingManager
from nova_voice.capabilities.registry import CapabilityRegistry
from nova_voice.commitments import CommitmentManager
from nova_voice.communications import CommunicationManager
from nova_voice.companion.protocol import ToolResultMessage
from nova_voice.companion.router import CompanionWorkloadRouter, turn_route_log
from nova_voice.companion.session import CompanionSessionManager
from nova_voice.companion.workloads import SelfProfileResult
from nova_voice.companion.workloads import parse_result as parse_companion_result
from nova_voice.config import Settings
from nova_voice.continuity import ConversationContinuityManager
from nova_voice.dialogue import MultiPartyDialogueManager, detect_dialogue_routing
from nova_voice.domain import (
    ActiveGoal,
    CapabilityToolCall,
    Decision,
    Emotion,
    GoalStatus,
    HandleResult,
    Interpretation,
    PlannedAction,
    SelfProfileUpdate,
    SpeakerIdentity,
    SpeechAct,
    ToolResult,
    TurnStage,
    TurnStageStatus,
    TurnTerminalStatus,
    Utterance,
    VerificationVerdict,
)
from nova_voice.dry_run import begin_dry_run, current_dry_run, end_dry_run
from nova_voice.durable.store import DurableAgentStore
from nova_voice.events import HouseholdEventConsumer
from nova_voice.interpretation.base import Interpreter
from nova_voice.interpretation.llama_cpp import household_state_is_relevant
from nova_voice.interpretation.response_length import (
    command_acknowledgement,
    spoken_word_count,
)
from nova_voice.interpretation.speech_cues import (
    enforce_decision_consistency,
    enforce_speech_cues,
    has_abandonment,
)
from nova_voice.memory import (
    MemoryAccessContext,
    MemoryIntentKind,
    MemoryOperation,
    MemoryRecord,
    MemPalaceClient,
    classify_memory_intent,
    salient_memory_candidate,
)
from nova_voice.offline_optimizers import OfflineOptimizerPool
from nova_voice.persistence import TranscriptStore
from nova_voice.persona import Persona
from nova_voice.policy import ExecutionPolicy, PolicyOutcome
from nova_voice.proactive import ProactiveInterventionEngine
from nova_voice.providers.nova import verify_loop
from nova_voice.providers.nova.client import NovaDashboardError
from nova_voice.providers.nova.provider import NovaProvider
from nova_voice.providers.web.provider import WebProvider
from nova_voice.research import ResearchManager
from nova_voice.sessions import SessionManager
from nova_voice.speaker_profiles import (
    SpeakerProfileStore,
    validated_self_profile_update,
)
from nova_voice.transactions import TransactionManager
from nova_voice.turns import (
    ForegroundTurnStateMachine,
    TaskCancellationDecision,
    TurnCancellationController,
)
from nova_voice.voice_settings import VoiceSettings

logger = logging.getLogger(__name__)

# Speech acts that are never the user engaging the assistant, even inside an
# open conversation. Turns classified this way must not extend the conversation
# window or be written into the history the model sees on later turns, so
# background talk and television can't keep a conversation alive or pollute it.
_AMBIENT_SPEECH_ACTS = frozenset(
    {SpeechAct.THIRD_PARTY, SpeechAct.QUOTED_OR_MEDIA, SpeechAct.SELF_INTENTION}
)

# Speech acts that ask the household to change. A turn classified this way is a
# command: it either acts or it says nothing.
_COMMAND_SPEECH_ACTS = frozenset({SpeechAct.DIRECTIVE, SpeechAct.DESIRED_STATE})

# Household-snapshot fields that must never go stale inside a conversation.
# These are what a command is resolved against — which devices exist, what they
# are called, and what they are currently doing — so a frozen copy silently
# limits the assistant to the house as it was when the conversation opened.
# Everything else in the snapshot stays frozen for prompt-cache stability.
_LIVE_MANIFEST_KEYS = frozenset(
    {
        "roomDevices",
        "nearbyTargets",
        "deviceStates",
        "climateControls",
        "indoorRooms",
        "modes",
    }
)


def turn_extends_conversation(
    *,
    wake_detected: bool,
    interpretation: Interpretation,
    addressed_threshold: float,
) -> bool:
    """Decide whether this turn keeps the follow-up window open.

    Excluding the three ambient speech acts is not enough on its own. A turn the
    model could not make sense of (``unclear``) or deliberately declined to act
    on (``ignore``), and any turn it did not believe was addressed to the
    assistant, is not the user engaging — and a conversation that such turns can
    refresh never closes while there is any noise in the room. Only a turn that
    was addressed *and* understood extends the window; everything else lets it
    time out normally.

    The audio runtime refreshes the same window again after playback, so this
    rule has to be one function rather than two copies that can drift apart.
    """

    if wake_detected:
        return True
    if interpretation.speech_act in _AMBIENT_SPEECH_ACTS:
        return False
    if interpretation.speech_act == SpeechAct.UNCLEAR:
        return False
    if interpretation.decision == Decision.IGNORE:
        return False
    return interpretation.addressed_probability >= addressed_threshold

_SPOKEN_WORD_RE = re.compile(r"\b\w+(?:['\N{RIGHT SINGLE QUOTATION MARK}]\w+)?\b", re.UNICODE)
_PROFILE_FILLERS_RE = re.compile(
    r"^(?:(?:okay|ok|actually|well|please|hi|hello|by\s+the\s+way)\b"
    r"[\s,;:\-\N{EM DASH}]*)+",
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
_KNOWLEDGE_FAILURE_RE = re.compile(
    r"\b(?:"
    r"i\s+(?:do\s+not|don'?t)\s+know|"
    r"i(?:\s+am|['’]m)\s+not\s+(?:sure|certain)|"
    r"i\s+(?:do\s+not|don'?t)\s+have\s+(?:access\s+to\s+)?"
    r"(?:enough|any|that|the)\s+(?:information|context|answer)|"
    r"i\s+(?:have\s+no|lack|have\s+insufficient)\s+(?:the\s+)?"
    r"(?:information|context|answer)|"
    r"i\s+(?:can\s*not|can'?t|am\s+unable\s+to)\s+"
    r"(?:answer|provide|help\s+with|find|look\s+up|access|tell|do\s+that)"
    r")\b",
    re.IGNORECASE,
)
_KNOWLEDGE_REQUEST_RE = re.compile(
    r"(?:\b(?:who|what|when|where|why|how|which)\b|"
    r"\b(?:tell\s+me|give\s+me\s+(?:information|details)|"
    r"explain|describe|define|name|do\s+you\s+know|find\s+out)\b)",
    re.IGNORECASE,
)
_OPERATIONAL_REQUEST_RE = re.compile(
    r"\b(?:turn|switch|set|dim|brighten|open|close|lock|unlock|start|stop|play|pause|"
    r"send|message|email|buy|book|order|call)\b",
    re.IGNORECASE,
)
_CAPABILITY_REQUEST_RE = re.compile(r"\b(?:can|could|will|would)\s+you\b", re.IGNORECASE)
_LOCAL_KNOWLEDGE_RE = re.compile(
    r"\b(?:in\s+here|this\s+room|at\s+home|my\s+(?:light|lights|switch|device|thermostat)|"
    r"what(?:'s|\s+is)\s+the\s+(?:temperature|status|state)|"
    r"how\s+(?:warm|cold)\s+is\s+it|remember|memory|what(?:'s|\s+is)\s+my\s+name)\b",
    re.IGNORECASE,
)
_CREATIVE_REQUEST_RE = re.compile(
    r"\b(?:joke|story|poem|song|roleplay|improvise|pretend)\b", re.IGNORECASE
)
_FOLLOW_UP_REFERENCE_RE = re.compile(
    r"\b(?:he|him|his|she|her|hers|they|them|their|theirs|it|its|that|those|there)\b",
    re.IGNORECASE,
)


# Said when the household addressed the assistant but the turn produced no
# words of its own. Kept deliberately vague about the cause: from the room, a
# misheard phrase and a failed rendering are the same event, and inviting a
# repeat is the useful response to both.
UNPARSED_REQUEST_REPLY = "Sorry, I didn't catch that."


def addressed_directive_needs_an_answer(
    utterance: Utterance,
    interpretation: Interpretation,
    outcome: PolicyOutcome,
) -> bool:
    """Was this a turn the household expects an audible answer to?

    Narrow on purpose. Ambient speech must stay silent — that is the whole
    point of the addressing model — so this fires only when the assistant was
    unambiguously spoken to (a wake word, or an open conversation), it did not
    act, and nothing else produced words.

    A shadowed turn is excluded: it is deliberately not acting, and announcing
    that would be noise rather than help.
    """

    if outcome.shadowed or outcome.execute or interpretation.actions:
        return False
    if not (utterance.wake_detected or utterance.conversation_active):
        return False
    return interpretation.decision in {Decision.REPLY, Decision.CLARIFY}


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


def response_has_knowledge_failure(response_text: str | None) -> bool:
    """Detect a drafted reply that admits an epistemic knowledge gap."""

    return bool(response_text and _KNOWLEDGE_FAILURE_RE.search(response_text))


def knowledge_web_fallback_relevant(
    utterance: Utterance,
    interpretation: Interpretation,
    response_text: str | None,
) -> bool:
    """Conservatively distinguish knowledge gaps from operational failures."""

    if not (utterance.wake_detected or utterance.conversation_active):
        return False
    if interpretation.decision != Decision.REPLY or interpretation.actions:
        return False
    if not response_has_knowledge_failure(response_text):
        return False

    transcript = utterance.transcript
    if _LOCAL_KNOWLEDGE_RE.search(transcript) or _CREATIVE_REQUEST_RE.search(transcript):
        return False
    if _OPERATIONAL_REQUEST_RE.search(transcript):
        return False
    if _CAPABILITY_REQUEST_RE.search(transcript) and not _KNOWLEDGE_REQUEST_RE.search(transcript):
        return False
    return bool(
        interpretation.speech_act == SpeechAct.QUESTION or _KNOWLEDGE_REQUEST_RE.search(transcript)
    )


def knowledge_fallback_query(
    transcript: str,
    conversation: ConversationSnapshot | None,
    *,
    max_length: int = 400,
) -> str:
    """Build one bounded search query, adding prior user context for pronouns."""

    current = " ".join(transcript.split())
    if conversation is None or not _FOLLOW_UP_REFERENCE_RE.search(current):
        return current[:max_length]

    prior_user_turns = [
        " ".join(message.content.split()).rstrip(" .?!")
        for message in conversation.messages
        if message.role == "user" and message.content.strip()
    ][-2:]
    if not prior_user_turns:
        return current[:max_length]
    contextual = f"{'. '.join(prior_user_turns)}. Follow-up: {current}"
    return contextual[:max_length]


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


def _dashboard_observations(interpretation: Interpretation, results: list[ToolResult]) -> list[str]:
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
        web_provider: WebProvider | None = None,
        durable_store: DurableAgentStore | None = None,
        event_consumer: HouseholdEventConsumer | None = None,
        authority: HouseholdAuthority | None = None,
        automations: AutomationManager | None = None,
        proactive: ProactiveInterventionEngine | None = None,
        memory: MemPalaceClient | None = None,
        communications: CommunicationManager | None = None,
        transactions: TransactionManager | None = None,
        commitments: CommitmentManager | None = None,
        research: ResearchManager | None = None,
        briefings: BriefingManager | None = None,
        continuity: ConversationContinuityManager | None = None,
        dialogue: MultiPartyDialogueManager | None = None,
        optimizers: OfflineOptimizerPool | None = None,
        companion_sessions: CompanionSessionManager | None = None,
        companion_router: CompanionWorkloadRouter | None = None,
    ) -> None:
        self.settings = settings
        self.interpreter = interpreter
        self.registry = registry
        self.nova_provider = nova_provider
        self.web_provider = web_provider
        self.store = store
        self.durable_store = durable_store
        self.event_consumer = event_consumer
        self.authority = authority
        self.automations = automations
        self.proactive = proactive
        self.memory = memory
        self.communications = communications
        self.transactions = transactions
        self.commitments = commitments
        self.research = research
        self.briefings = briefings
        self.continuity = continuity
        self.dialogue = dialogue
        self.optimizers = optimizers
        # The companion is optional in the strongest sense: absent these, every
        # routing decision resolves to the local path and nothing else in the
        # service behaves differently.
        self.companion_sessions = companion_sessions
        self.companion_router = companion_router
        if companion_sessions is not None:
            companion_sessions.bind_tool_executor(self.execute_companion_tool_call)
        self.speaker_profiles = speaker_profiles
        self.persona = persona
        # Satellites within earshot share one conversation/goal scope so a
        # follow-up elected on another microphone continues the same exchange.
        scope_key = (
            (lambda room_id: "household") if settings.arbitration_scope == "household" else None
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
        self._active_task_cancellations: dict[str, TurnCancellationController] = {}
        self.policy = ExecutionPolicy(settings, authority)
        self.voice_settings: VoiceSettings | None = None
        self.agent_settings = AgentSettings()
        # Conversation scopes whose spoken replies are pinned to temperature 0
        # after a failed dashboard command, until that conversation ends. Keeps
        # failure-recovery wording deterministic regardless of the dashboard's
        # configured render temperature.
        self._zero_render_temperature_scopes: set[str] = set()
        # A bare wake word ("<name>" or "<address prefix> <name>", nothing
        # else) is rendered at a fixed high temperature regardless of the
        # dashboard's configured render temperature, so the one-or-two-word
        # acknowledgement varies turn to turn instead of reading as a canned
        # template.
        self._bare_wake_word_temperature: float = 10.0

    def apply_voice_settings(self, settings: VoiceSettings) -> None:
        self.persona = self.persona.with_voice_settings(settings)
        self.voice_settings = settings
        self._apply_companion_switches(settings)
        self._apply_companion_routes(settings.companion_routes)
        self.nova_provider.agent_name = settings.spoken_name
        # The conversation window is dashboard-tunable and applies live: both
        # the wide-vocabulary follow-up window and the goal-session clock must
        # follow the same value or turns die on whichever timer is shorter.
        self.conversations.set_idle_seconds(settings.conversation_idle_seconds)
        self.sessions.set_follow_up_seconds(settings.conversation_idle_seconds)
        # The absolute ceiling is the backstop the idle window cannot provide.
        self.conversations.set_max_seconds(settings.conversation_max_seconds)
        self.sessions.set_max_seconds(settings.conversation_max_seconds)
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
        # Web access: the interpreter needs the enabled flag (to offer/omit the
        # tool and know when a lookup is permitted) and the sentence budget (to
        # bound the spoken web answer); the provider needs the backend choice and
        # the same budget (to shape the cloud model's answer length).
        if hasattr(self.interpreter, "web_access_enabled"):
            self.interpreter.web_access_enabled = settings.web_access_enabled
        if hasattr(self.interpreter, "web_answer_max_sentences"):
            self.interpreter.web_answer_max_sentences = settings.web_answer_max_sentences
        if self.web_provider is not None:
            self.web_provider.configure(
                backend=settings.web_backend,
                answer_max_sentences=settings.web_answer_max_sentences,
            )

    def apply_agent_settings(self, settings: AgentSettings) -> None:
        """Apply global execution-loop settings without touching voice/personality."""

        self.agent_settings = settings
        self.nova_provider.configure_verification_loop(
            enabled=settings.ralph_loop_enabled,
            max_iterations=settings.ralph_loop_max_iterations,
            sleep_seconds=settings.ralph_loop_sleep_ms / 1000,
            failure_seconds=settings.ralph_loop_failure_seconds,
            thinking_threshold_seconds=settings.ralph_loop_thinking_threshold_ms / 1000,
            llm_verify_enabled=settings.ralph_loop_llm_verify_enabled,
            llm_verify_min_interval_seconds=settings.ralph_loop_llm_verify_min_interval_ms / 1000,
            llm_confirm_timeout_seconds=settings.ralph_loop_llm_confirm_timeout_seconds,
        )

    def _apply_affectations(self, text: str | None) -> str | None:
        """Deterministic speech quirks on a finished reply string."""

        if not text or self.voice_settings is None:
            return text
        return apply_affectations(text, self.voice_settings.affectations)

    def _available_tools(self) -> list[dict]:
        """The semantic tools offered to the planner this turn.

        Web access is opt-in: when the dashboard switch is off the ``web.*``
        tools are hidden entirely, so the model cannot plan a lookup at all
        (rather than planning one that is later refused).
        """

        catalog = self.registry.tool_catalog()
        web_enabled = self.voice_settings is not None and self.voice_settings.web_access_enabled
        if web_enabled:
            return catalog
        return [
            tool
            for tool in catalog
            if not str(tool.get("function", {}).get("name", "")).startswith("web.")
        ]

    def _apply_companion_switches(self, settings) -> None:
        """Apply the two kill switches, if the dashboard has an opinion.

        `None` means "leave it as configured", which is what stops a dashboard
        that has never shown these controls from silently switching the feature
        on or off for a deployment that set it in configuration.

        Applied *before* the routes, so an operator who switches the feature
        off and moves a pass in the same save does not get one turn where the
        pass has moved and the feature is still live.
        """

        router = self.companion_router
        if router is None:
            return
        if settings.companion_enabled is not None:
            router.set_enabled(settings.companion_enabled)
        if settings.companion_force_local is not None:
            router.set_force_local(settings.companion_force_local)

    def _apply_companion_routes(self, choices: dict[str, str]) -> None:
        """Put each pass where the dashboard says it should run.

        The three operator-facing words map onto the router's modes rather than
        being stored as modes, because an operator is choosing a *machine*:
        "local" is this host, "companion" is the phone, and "both" runs the
        pass on both and keeps both answers.

        "Both" is the measurement setting, not the fast one. It occupies this
        host's single LLM slot *and* the phone, so it gives up the latency
        benefit of offloading entirely — in exchange for being able to see the
        two answers side by side, which is the only way to judge whether a pass
        should move for real.

        A workload absent from the map keeps its code default. That is what
        makes an empty map mean "as shipped" rather than "route nothing", so a
        deployment that has never touched these controls behaves exactly as it
        did before they existed.
        """

        router = self.companion_router
        if router is None or not choices:
            return
        # "both" means both, literally: the pass runs on the phone *and* here,
        # and both answers are kept. It used to map to `companion_preferred`,
        # which tries the phone and falls back — so the control said "Both"
        # while only ever one stack did the work.
        # "companion" is `companion_preferred`, not `companion_only`. An
        # operator choosing "Companion" is saying which machine should do the
        # work, not asking for the pass to fail when the phone is asleep —
        # and `companion_only` returns nothing at all in that case.
        modes = {"local": "local", "companion": "companion_preferred", "both": "both"}
        known = router.routes()
        for workload, choice in choices.items():
            mode = modes.get(choice)
            if mode is None:
                continue
            if workload not in known:
                # A workload this build does not have. Checked rather than
                # caught, because `override` would happily *invent* the route
                # and it would then show up in the status table as though it
                # were real. Ignored so a stale preferences file cannot stop
                # the settings pull that applies every other setting.
                logger.warning("ignoring companion route for unknown workload %r", workload)
                continue
            router.override(workload, mode=mode)

    async def _routed_self_profile_update(self, utterance: Utterance) -> SelfProfileUpdate | None:
        """Identity extraction, on the companion when one is eligible.

        Off the spoken path already — it runs as a side task next to the main
        interpretation — so the win here is not this call's own latency. It is
        that this pass and `interpret` were competing for llama.cpp's single
        slot on the same turn, which is what turned two cheap passes into two
        serialised ones.

        The utterance is `personal`: the transcript is what someone said about
        themselves. That is permitted to leave Iridium under the amended plan,
        which is exactly why `sensitivity` keeps the value out of the logs
        rather than trusting anyone to remember not to print it.
        """

        local = self.interpreter.extract_self_profile_update
        router = self.companion_router
        if router is None:
            return await local(utterance)

        outcome = await router.run(
            "extract_self_profile_update",
            {"transcript": utterance.transcript},
            lambda: local(utterance),
            parse=lambda payload: parse_companion_result("extract_self_profile_update", payload),
        )
        logger.info(
            "extract_self_profile_update resolved source=%s reason=%s elapsed_ms=%s",
            outcome.source,
            outcome.reason,
            outcome.elapsed_ms,
        )
        value = outcome.value
        # The companion answers with the wrapper that can say "no disclosure";
        # the local path answers with the bare model or None. Both reduce here
        # so callers keep seeing exactly one shape.
        if isinstance(value, SelfProfileResult):
            return value.update
        return value

    async def _routed_confirm_objective(
        self,
        utterance: Utterance,
        pending: list[dict[str, Any]],
    ) -> VerificationVerdict | None:
        """The verification loop's judgement pass, on the companion when eligible.

        This one is worth moving even though it is not on the spoken path,
        because of *when* it runs: the loop fires it repeatedly while a
        multi-device command settles, so every one of its calls lands on
        Iridium precisely when the household is busiest. On the phone they
        cost nothing local at all.

        `pending` is already narrow by construction — the loop passes each
        still-unsettled target's own observed state, not the household
        snapshot — so there is nothing further to compact here.
        """

        local = self.interpreter.confirm_objective
        router = self.companion_router
        if router is None:
            return await local(utterance, pending)

        outcome = await router.run(
            "confirm_objective",
            {"transcript": utterance.transcript, "pending": pending},
            lambda: local(utterance, pending),
            parse=lambda payload: parse_companion_result("confirm_objective", payload),
        )
        logger.info(
            "confirm_objective resolved source=%s reason=%s elapsed_ms=%s pending=%d",
            outcome.source,
            outcome.reason,
            outcome.elapsed_ms,
            len(pending),
        )
        return outcome.value

    async def _recover_knowledge_failure(
        self,
        utterance: Utterance,
        interpretation: Interpretation,
        results: list[ToolResult],
        response_text: str | None,
        *,
        conversation: ConversationSnapshot | None,
        relevant_state: dict[str, Any],
        temperature: float,
        session_active: bool,
        cancellation: TurnCancellationController | None = None,
    ) -> tuple[Interpretation, list[ToolResult], PolicyOutcome, str | None] | None:
        """Run one policy-checked web lookup after an epistemic reply failure."""

        web_enabled = bool(
            self.web_provider is not None
            and self.voice_settings is not None
            and self.voice_settings.web_access_enabled
        )
        if (
            not web_enabled
            or results
            or not knowledge_web_fallback_relevant(utterance, interpretation, response_text)
        ):
            return None

        query = knowledge_fallback_query(utterance.transcript, conversation)
        if not query:
            return None
        action = PlannedAction(
            id=f"{utterance.id}-web-fallback",
            order=0,
            call=CapabilityToolCall(
                provider="web",
                tool="web.ask",
                arguments={"query": query},
            ),
        )
        fallback_interpretation = interpretation.model_copy(
            deep=True,
            update={
                "speech_act": SpeechAct.QUESTION,
                "addressed_probability": 1.0,
                "decision": Decision.EXECUTE,
                "confidence": 1.0,
                "actions": [action],
                "response_plan": interpretation.response_plan.model_copy(
                    update={"requires_post_tool_rendering": True}
                ),
            },
        )
        fallback_outcome = self.policy.evaluate(
            utterance,
            fallback_interpretation,
            session_active=session_active,
        )
        if not fallback_outcome.execute:
            logger.info(
                "knowledge web fallback blocked utterance=%s reason=%s",
                utterance.id,
                fallback_outcome.reason,
            )
            return None

        fallback_results = await self._execute_plan(
            [action],
            cancellation=cancellation,
        )
        deterministic = self.persona.render(
            utterance,
            fallback_interpretation.decision,
            fallback_results,
            shadowed=False,
        )
        rendered = await self.interpreter.render_response(
            utterance,
            fallback_interpretation,
            fallback_results,
            persona=self.persona.response_prompt,
            environment=None,
            relevant_state=relevant_state,
            conversation=conversation,
            temperature=temperature,
            command_max_words=None,
        )
        final_text = rendered or deterministic or response_text
        logger.info(
            "knowledge web fallback completed utterance=%s query=%r ok=%s",
            utterance.id,
            query,
            bool(fallback_results and all(result.ok for result in fallback_results)),
        )
        return fallback_interpretation, fallback_results, fallback_outcome, final_text

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
        utterance = Utterance.text(prompt, room_id="preview", satellite_id="dashboard-preview")
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
        if self.durable_store is not None:
            await self.durable_store.initialize()
            await self.durable_store.prune_expired()
        if self.authority is not None:
            await self.authority.initialize()
        if self.commitments is not None:
            await self.commitments.poll()
            self.commitments.start()
        if self.research is not None:
            await self.research.start()
        if self.briefings is not None:
            await self.briefings.poll()
            self.briefings.start()
        if self.event_consumer is not None:
            await self.event_consumer.initialize()
            self.event_consumer.start()
        if self.speaker_profiles is not None:
            await self.speaker_profiles.initialize()
        if self.optimizers is not None:
            self.optimizers.start()
        # Nova is an optional capability from the core service's point of view.
        # Keep the interpreter, retention janitor, and satellite health endpoint
        # available when the dashboard is restarting; a later health/refresh
        # call will make the provider available again without a process restart.
        try:
            await self.nova_provider.refresh(force=True)
        except NovaDashboardError:
            logger.warning("Nova provider unavailable during startup", exc_info=True)

    async def prefetch_foreground(
        self,
        room_id: str,
        stable_text: str,
    ) -> ForegroundPrefetch:
        """Warm only read-only state for a stable, non-final transcript.

        This method intentionally has no access to policy execution, transcript
        persistence, response rendering, or TTS. The final turn must still pass
        every normal foreground stage before any externally visible effect.
        """

        try:
            context = await asyncio.wait_for(
                self.nova_provider.prompt_context(room_id),
                timeout=self.settings.provider_context_timeout_seconds,
            )
        except (NovaDashboardError, TimeoutError):
            context = {"room": room_id, "zones": [], "nearbyTargets": []}
        selected_tools = likely_tools(stable_text, self._available_tools())
        return ForegroundPrefetch.create(
            stable_text,
            copy.deepcopy(context),
            selected_tools,
        )

    async def handle(
        self,
        utterance: Utterance,
        *,
        on_thinking: Callable[[], Awaitable[None]] | None = None,
        turn_machine: ForegroundTurnStateMachine | None = None,
        prefetch: ForegroundPrefetch | None = None,
    ) -> HandleResult:
        scope = self._scope_key(utterance.room_id)
        lock = self._turn_locks.setdefault(scope, asyncio.Lock())
        async with lock:
            owns_machine = turn_machine is None
            machine = turn_machine or ForegroundTurnStateMachine(utterance)
            if owns_machine:
                machine.advance(
                    TurnStage.CAPTURE,
                    status=TurnStageStatus.SKIPPED,
                    detail="text/already-captured entry",
                )
                machine.advance(
                    TurnStage.ENDPOINT,
                    status=TurnStageStatus.SKIPPED,
                    detail="text/already-finalized entry",
                )
            cancellation = TurnCancellationController()
            self._active_task_cancellations[scope] = cancellation
            # Arm the dry run around the whole turn rather than just the
            # execution step, so no path — recovery fallback, verification,
            # late side effect — can reach the household behind its back.
            dry_run_token = begin_dry_run() if utterance.dry_run else None
            try:
                # Collect where each reasoning pass of this turn actually ran.
                # Wrapping the whole turn rather than a single call means a pass
                # invoked from anywhere inside it is attributed correctly — and
                # a pass that ran on *both* stacks appears twice, which is how
                # a doubled turn becomes visible instead of being inferred.
                with turn_route_log() as route_chain:
                    result = await self._handle(
                        utterance,
                        on_thinking=on_thinking,
                        turn_machine=machine,
                        cancellation=cancellation,
                        prefetch=prefetch,
                    )
                if route_chain:
                    result = result.model_copy(update={"route_chain": list(route_chain)})
                if dry_run_token is not None:
                    recorder = current_dry_run()
                    result = result.model_copy(
                        update={
                            "dry_run": True,
                            "dry_run_requests": list(recorder.requests) if recorder else [],
                        }
                    )
                for decision in cancellation.decisions:
                    machine.record_cancellation(decision.trace_record())
                if owns_machine:
                    machine.advance(
                        TurnStage.SPEAK,
                        status=TurnStageStatus.SKIPPED,
                        detail="no audio runtime",
                    )
                    machine.advance(TurnStage.COMMIT)
                    terminal = (
                        TurnTerminalStatus.CANCELLED
                        if any(item.code == "cancelled" for item in result.results)
                        else TurnTerminalStatus.IGNORED
                        if result.interpretation.decision == Decision.IGNORE
                        and result.response_text is None
                        else TurnTerminalStatus.COMPLETED
                    )
                    machine.finish(terminal)
                return result.model_copy(update={"turn_trace": machine.snapshot()})
            except BaseException as error:
                if machine.snapshot().terminal_status == TurnTerminalStatus.IN_PROGRESS:
                    machine.fail(type(error).__name__)
                raise
            finally:
                if dry_run_token is not None:
                    end_dry_run(dry_run_token)
                cancellation.close()
                if self._active_task_cancellations.get(scope) is cancellation:
                    self._active_task_cancellations.pop(scope, None)

    def request_task_cancellation(self, room_id: str) -> TaskCancellationDecision:
        """Request cancellation without conflating it with response playback."""

        controller = self._active_task_cancellations.get(self._scope_key(room_id))
        if controller is None:
            return TaskCancellationDecision(
                active=False,
                accepted=False,
                phase="idle",
                reason="no foreground task is active",
            )
        return controller.request()

    async def _handle(
        self,
        utterance: Utterance,
        *,
        on_thinking: Callable[[], Awaitable[None]] | None = None,
        turn_machine: ForegroundTurnStateMachine,
        cancellation: TurnCancellationController,
        prefetch: ForegroundPrefetch | None = None,
    ) -> HandleResult:
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
        memory_control_reply: str | None = None
        memory_access = MemoryAccessContext(
            actor_id=utterance.speaker.person_id,
            recognized=utterance.speaker.status == "recognized",
        )

        async def mutate_memory(
            method: str,
            path: str,
            payload: dict[str, Any] | None = None,
        ) -> dict[str, Any] | None:
            memory = self.memory
            if memory is None:
                return None
            cancellation.begin_non_cancellable_side_effect("memory")
            try:
                body = dict(payload or {})
                if method.upper() == "DELETE":
                    separator = "&" if "?" in path else "?"
                    path = (
                        f"{path}{separator}actor_id={memory_access.actor_id or ''}"
                        f"&recognized={str(memory_access.recognized).lower()}"
                    )
                else:
                    body["_access"] = memory_access.model_payload()
                return await memory.request(method, path, body or None)
            finally:
                cancellation.non_cancellable_side_effect_finished()

        context_started = time.perf_counter()
        if prefetch is not None and prefetch.compatible_with(utterance.transcript):
            relevant_state = copy.deepcopy(prefetch.context)
            timings_ms["prefetchHit"] = 1.0
        else:
            try:
                relevant_state = await asyncio.wait_for(
                    self.nova_provider.prompt_context(utterance.room_id),
                    timeout=self.settings.provider_context_timeout_seconds,
                )
            except (NovaDashboardError, TimeoutError) as error:
                # Ambient speech must still be transcribed/classified while Nova is
                # offline or DNS is slow, but a missing dashboard must never add
                # seconds to the spoken turn. No stale household state is supplied.
                #
                # This is loud on purpose. An empty zone list makes the model
                # correctly report that it knows of no lights, so a turn that
                # silently landed here is indistinguishable — to a listener —
                # from the assistant being broken. It must always leave a trace.
                elapsed_ms = round((time.perf_counter() - context_started) * 1000, 1)
                logger.warning(
                    "household context unavailable; replying with no household state "
                    "room=%s reason=%s elapsed_ms=%s budget_ms=%s",
                    utterance.room_id,
                    type(error).__name__,
                    elapsed_ms,
                    round(self.settings.provider_context_timeout_seconds * 1000, 1),
                )
                timings_ms["providerContextFailed"] = elapsed_ms
                relevant_state = {
                    "room": utterance.room_id,
                    "zones": [],
                    "nearbyTargets": [],
                }
        # Memory is a best-effort private service.  Retrieval is scoped to an
        # enrolled household member; unknown/provisional voices never receive
        # personal context, and a service fault can never delay a normal turn.
        if self.memory is not None and utterance.speaker.status == "recognized":
            memory_text = utterance.transcript.strip()
            memory_intent = classify_memory_intent(memory_text)
            if memory_intent.kind == MemoryIntentKind.QUERY_ALL:
                selected_memories = await self.memory.list(
                    access=memory_access, operation=MemoryOperation.CALLBACK
                )
            elif memory_intent.kind == MemoryIntentKind.QUERY_TOPIC:
                selected_memories = await self.memory.search(
                    memory_intent.query or memory_text,
                    access=memory_access,
                    operation=MemoryOperation.CALLBACK,
                )
            elif memory_intent.kind == MemoryIntentKind.CONTROL:
                selected_memories = await self.memory.search(
                    memory_text, access=memory_access, operation=MemoryOperation.CORRECT
                )
            else:
                selected_memories = []
            # Direct spoken pin/forget requests act only on one unambiguous
            # personal match. Ambiguity is left to the dashboard review screen.
            memory_control = re.match(r"^(?:please )?(pin|forget)\s+(.+)$", memory_text, re.I)
            correction = re.match(r"^(?:please )?correct memory (.+?) to (.+)$", memory_text, re.I)
            expiry = re.match(
                r"^(?:please )?expire memory (.+?) in (\d{1,3}) days?$", memory_text, re.I
            )
            if correction:
                selected_memories = await self.memory.search(
                    correction.group(1), access=memory_access, operation=MemoryOperation.CORRECT
                )
                if len(selected_memories) == 1:
                    result = await mutate_memory(
                        "PATCH",
                        f"/v1/memories/{selected_memories[0].id}",
                        {"text": correction.group(2).strip()},
                    )
                    memory_control_reply = "Corrected." if result is not None else None
            elif expiry:
                selected_memories = await self.memory.search(
                    expiry.group(1), access=memory_access, operation=MemoryOperation.CORRECT
                )
                if len(selected_memories) == 1:
                    expires_at = datetime.now(UTC) + timedelta(days=int(expiry.group(2)))
                    result = await mutate_memory(
                        "PATCH",
                        f"/v1/memories/{selected_memories[0].id}",
                        {"expires_at": expires_at.isoformat()},
                    )
                    memory_control_reply = "Expiry set." if result is not None else None
            elif memory_control and len(selected_memories) == 1:
                action, _ = memory_control.groups()
                memory_id = selected_memories[0].id
                if action.casefold() == "forget":
                    result = await mutate_memory("DELETE", f"/v1/memories/{memory_id}")
                    if result is not None:
                        selected_memories = []
                        memory_control_reply = "Forgot it."
                else:
                    result = await mutate_memory(
                        "PATCH", f"/v1/memories/{memory_id}", {"pinned": True}
                    )
                    memory_control_reply = "Pinned." if result is not None else None
            if selected_memories:
                relevant_state["selectedMemory"] = [
                    {
                        "id": memory.id,
                        "type": memory.memory_type.value,
                        "text": memory.text,
                        "confidence": memory.confidence,
                    }
                    for memory in selected_memories
                ]
            elif memory_intent.kind in {
                MemoryIntentKind.QUERY_ALL,
                MemoryIntentKind.QUERY_TOPIC,
            }:
                memory_control_reply = "I don't have a saved memory about that."
        if self.continuity is not None and utterance.speaker.status == "recognized":
            try:
                continuity_context = await self.continuity.context_for(
                    utterance.speaker.person_id or "", utterance.transcript
                )
            except Exception:
                continuity_context = {}
                logger.warning("relationship continuity retrieval failed", exc_info=True)
            if continuity_context:
                relevant_state["conversationContinuity"] = continuity_context
        if self.continuity is not None and conversation is not None:
            try:
                relevant_state["discussionMode"] = await self.continuity.discussion_mode_for(
                    conversation.id, utterance.transcript
                )
            except Exception:
                logger.warning("discussion mode retrieval failed", exc_info=True)
        dialogue_routing = detect_dialogue_routing(
            utterance.transcript,
            agent_names=tuple(self._address_words()),
            participant_names=tuple(
                dict.fromkeys(
                    message.speaker_name
                    for message in (conversation.messages if conversation else ())
                    if message.speaker_name
                )
            ),
        )
        if dialogue_routing.addressee != "unspecified":
            relevant_state["dialogueRouting"] = {
                "addressee": dialogue_routing.addressee,
                "targetName": dialogue_routing.target_name,
                "relayAct": dialogue_routing.relay_act,
            }
        if self.dialogue is not None and utterance.speaker.status == "recognized":
            pending_messages = await self.dialogue.pending_for(
                person_id=utterance.speaker.person_id or "",
                display_name=utterance.speaker.display_name,
            )
            if pending_messages:
                relevant_state["pendingDialogueMessages"] = [
                    {
                        "id": item.id,
                        "senderId": item.sender_id,
                        "speechAct": item.speech_act,
                        "content": item.content,
                    }
                    for item in pending_messages[:5]
                ]
        # Date/time and weather are a conversation-start snapshot.  Household
        # target state remains live on every turn, but these ambient prompt
        # injections are never appended again during the same conversation.
        # Whether either fact actually reaches the model on a given turn is
        # decided later, per-turn, by select_environment_context — this only
        # captures the snapshot once.
        weather = relevant_state.pop("weather", None)
        if conversation is not None and conversation.initial_environment is None:
            # Freeze the bulky prompt inputs (household state + selected memory)
            # at conversation open. Follow-up turns reuse this snapshot from a
            # stable, cacheable prompt prefix instead of re-sending live state
            # every turn; it refreshes only when the window reopens.
            conversation = self.conversations.initialize_prompt(
                utterance.room_id,
                environment={
                    "now": current_clock_context(self.settings.household_tzinfo()),
                    "weather": weather,
                },
                personality=getattr(self.interpreter, "personality", ""),
                persona_prompt=self.persona.response_prompt,
                state={
                    key: value
                    for key, value in relevant_state.items()
                    if key != "selectedMemory"
                },
                memory=relevant_state.get("selectedMemory", []),
            )
        elif conversation is not None:
            # The weather snapshot stays frozen — it is bulky and its own
            # freshness is the dashboard's concern — but the clock must not be.
            # A conversation can run for minutes, and "what time is it" has to
            # answer with now, not with when the conversation opened.
            self.conversations.set_environment_value(
                utterance.room_id,
                "now",
                current_clock_context(self.settings.household_tzinfo()),
            )
            # Nor may the device manifest be frozen. It is what decides whether
            # an utterance is a command, so a conversation that opened before a
            # device — or before a deployment that changed the manifest's shape
            # — would otherwise be unable to act on it for its whole lifetime.
            self.conversations.refresh_state_values(
                utterance.room_id,
                {
                    key: value
                    for key, value in relevant_state.items()
                    if key in _LIVE_MANIFEST_KEYS
                },
            )
            conversation = self.conversations.snapshot(utterance.room_id) or conversation
        timings_ms["providerContext"] = round((time.perf_counter() - context_started) * 1000, 3)
        turn_machine.set_context(
            {
                "relevantState": relevant_state,
                "conversation": conversation,
                "activeGoal": goal,
            }
        )
        turn_machine.advance(
            TurnStage.CONTEXTUALIZE,
            elapsed_ms=timings_ms["providerContext"],
            detail=(
                f"prefetch {prefetch.llm_state_revision} reused"
                if prefetch is not None and prefetch.compatible_with(utterance.transcript)
                else None
            ),
        )

        addressed_identity_turn = bool(
            self.speaker_profiles is not None
            and utterance.speaker.template_id is not None
            and (utterance.wake_detected or utterance.conversation_active)
        )
        # Identity extraction is deliberately separate from general intent and
        # speech-act classification. Start the tiny context-free pass first so
        # the local model can batch/run it alongside the normal interpretation.
        profile_task = (
            asyncio.create_task(self._routed_self_profile_update(utterance))
            if addressed_identity_turn
            else None
        )
        interpretation_started = time.perf_counter()
        try:
            interpretation = await self.interpreter.interpret(
                utterance,
                active_goal=goal,
                relevant_state=relevant_state,
                tools=self._available_tools(),
                conversation=conversation,
            )
        except BaseException:
            if profile_task is not None:
                profile_task.cancel()
            raise
        timings_ms["interpretation"] = round(
            (time.perf_counter() - interpretation_started) * 1000, 3
        )
        # The model's own verdict, before any deterministic repair. Everything
        # downstream can only narrow this, so a command that never appears here
        # was never planned — which separates "the model did not understand" from
        # "something later threw the plan away", and those have opposite fixes.
        logger.info(
            "model verdict: utterance=%s decision=%s speech_act=%s addressed=%.2f conf=%.2f "
            "tools=%s conversation=%s tool_count=%d",
            utterance.id,
            interpretation.decision.value,
            interpretation.speech_act.value,
            interpretation.addressed_probability,
            interpretation.confidence,
            [action.call.tool for action in interpretation.actions],
            conversation.id if conversation is not None else None,
            len(self._available_tools()),
        )
        if interpretation.decision == Decision.EXECUTE and not interpretation.actions:
            # "Execute, but here is nothing to run" is the model agreeing it was
            # given a command and then failing to emit the call. Nothing
            # downstream can recover it, and the generic empty-plan repair turns
            # it into a reply, so the evidence has to be captured here or the
            # turn is indistinguishable from ordinary conversation.
            frozen = (
                conversation.initial_state
                if conversation is not None and conversation.initial_state is not None
                else None
            )
            # Neutral field names: the development redaction filter blanks the
            # arguments of anything that looks like it carries speech, and a
            # redacted diagnostic is no diagnostic at all.
            logger.warning(
                "empty execute plan: utterance=%s words=%r liveKeys=%s frozenKeys=%s "
                "roomDeviceCount=%d truncated=%s",
                utterance.id,
                utterance.transcript,
                sorted(relevant_state),
                sorted(frozen) if frozen is not None else None,
                len((frozen or relevant_state).get("roomDevices") or []),
                (frozen or {}).get("truncatedFields"),
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
        elif dialogue_routing.relay_act is not None and dialogue_routing.relay_content:
            interpretation = interpretation.model_copy(
                update={
                    "speech_act": SpeechAct.DIRECTIVE,
                    "addressed_probability": 1.0,
                    "decision": Decision.EXECUTE,
                    "confidence": 1.0,
                    "actions": [
                        PlannedAction(
                            id=f"dialogue-relay:{utterance.id}",
                            order=0,
                            call=CapabilityToolCall(
                                provider="dialogue",
                                tool="dialogue.relay",
                                arguments={
                                    "senderId": utterance.speaker.person_id or "unknown",
                                    "recipientScope": dialogue_routing.addressee,
                                    "recipientName": dialogue_routing.target_name,
                                    "speechAct": dialogue_routing.relay_act,
                                    "content": dialogue_routing.relay_content,
                                    "conversationId": conversation.id if conversation else None,
                                },
                            ),
                        )
                    ],
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
        if dialogue_routing.addressee == "person" and dialogue_routing.relay_act is None:
            interpretation = interpretation.model_copy(
                update={
                    "decision": Decision.IGNORE,
                    "addressed_probability": 0.0,
                    "actions": [],
                }
            )
        # Identity claims are bound only to the biometric template extracted
        # from this exact addressed utterance. Never inherit a prior speaker
        # merely because the household-wide conversation remains open.
        profile_update = await profile_task if profile_task is not None else None
        if profile_update is None:
            profile_update = explicit_self_profile_update(
                utterance.transcript, self._address_words()
            )
        if profile_update is not None:
            profile_update = validated_self_profile_update(
                profile_update,
                utterance.transcript,
            )
        # The general interpretation model no longer owns this field. Replace
        # any value it emitted with the independent current-turn result.
        interpretation = interpretation.model_copy(update={"self_profile_update": profile_update})
        if addressed_identity_turn and profile_update is not None:
            cancellation.begin_non_cancellable_side_effect("speaker_profile")
            try:
                updated_speaker = await self.speaker_profiles.apply_disclosure(
                    utterance.speaker,
                    profile_update,
                    utterance.transcript,
                )
            finally:
                cancellation.non_cancellable_side_effect_finished()
            utterance = utterance.model_copy(update={"speaker": updated_speaker})
        speaker_recognition_active = (
            self.settings.speaker_recognition_enabled
            and self.speaker_profiles is not None
            and (self.voice_settings is None or self.voice_settings.speaker_recognition_enabled)
        )
        # Voice training off means only recognized household voices are acted
        # on at all; on, an unrecognized voice may still command, and only very
        # short commands (where a misheard fragment is most likely) require
        # recognition. Both rules need an identity to gate on, so both are
        # skipped when speaker recognition is unavailable.
        training_enabled = self.voice_settings is None or self.voice_settings.voice_training_enabled
        unrecognized = speaker_recognition_active and utterance.speaker.status != "recognized"
        blocked_by_training = unrecognized and not training_enabled
        blocked_as_short_command = (
            unrecognized
            and self.settings.speaker_short_execute_requires_recognition
            and command_word_count(utterance.transcript, self._address_words())
            <= self.settings.speaker_short_execute_max_words
        )
        if (
            interpretation.decision == Decision.EXECUTE
            and interpretation.actions
            and (blocked_by_training or blocked_as_short_command)
        ):
            logger.info(
                "ignored execute from unrecognized speaker: utterance=%s words=%d reason=%s",
                utterance.id,
                command_word_count(utterance.transcript, self._address_words()),
                "training_off" if blocked_by_training else "short_command",
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
            #
            # Logged because failing closed here silently mutes every command
            # in the house while every other signal — interpretation, policy,
            # the dashboard's own health — still looks perfectly healthy.
            logger.warning(
                "device admission failed closed; dropping every household action: utterance=%s",
                utterance.id,
                exc_info=True,
            )
            invalid_device_actions = {
                action.id
                for action in interpretation.actions
                if action.call.provider == "nova"
                and action.call.tool in {"nova.control", "nova.lighting_shortcut"}
            }
        # Every planned household mutation resolved to a real device, so the
        # phrasing named something this house can actually do. Policy treats
        # that as harder evidence than the model's own confidence score.
        device_actions = [
            action
            for action in interpretation.actions
            if action.call.provider == "nova"
            and action.call.tool in {"nova.control", "nova.lighting_shortcut", "nova.mode"}
        ]
        targets_verified = bool(device_actions) and not invalid_device_actions
        if invalid_device_actions:
            # The other way a command dies quietly: the model named a target the
            # freshly-read device tree could not resolve. Naming it is the
            # difference between "the assistant is being difficult" and "the
            # alias index does not know that phrase".
            logger.info(
                "dropped unresolvable household actions: utterance=%s targets=%s",
                utterance.id,
                [
                    action.call.arguments.get("target")
                    for action in interpretation.actions
                    if action.id in invalid_device_actions
                ],
            )
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
        turn_machine.advance(
            TurnStage.INTERPRET,
            elapsed_ms=round((time.perf_counter() - interpretation_started) * 1000, 3),
        )
        policy_started = time.perf_counter()
        outcome = self.policy.evaluate(
            utterance,
            interpretation,
            session_active=goal is not None,
            targets_verified=targets_verified,
        )
        timings_ms["policy"] = round((time.perf_counter() - policy_started) * 1000, 3)
        turn_machine.set_policy(
            execute=outcome.execute,
            shadowed=outcome.shadowed,
            reason=outcome.reason,
        )
        turn_machine.advance(TurnStage.AUTHORIZE, elapsed_ms=timings_ms["policy"])

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

            async def confirm_pending(
                tasks: list[verify_loop.VerificationTask],
                _state: dict[str, Any],
                items: list[verify_loop.VerificationItemResult],
            ) -> VerificationVerdict | None:
                # The verification loop's LLM pass is JSON-only and never
                # front-facing: it judges each still-pending target from its
                # own observed state, not the whole household snapshot, so a
                # slow multi-device command stays a cheap, narrow call.
                by_id = {item.action_id: item for item in items}
                pending = [
                    {
                        "target": task.label,
                        "objective": task.objective,
                        "observed": by_id[task.action_id].observed,
                        "attempts": by_id[task.action_id].attempts,
                    }
                    for task in tasks
                ]
                return await self._routed_confirm_objective(utterance, pending)

            with verify_loop.turn_scope(thinking=on_thinking, llm_confirm=confirm_pending):
                results = await self._execute_plan(
                    interpretation.actions,
                    cancellation=cancellation,
                )
        if (
            outcome.execute
            and outcome.grant_ids
            and any(result.ok for result in results)
            and self.authority is not None
        ):
            await self.authority.record_use(
                outcome.grant_ids,
                interpretation.actions,
                actor_id=utterance.speaker.person_id or "recognized-speaker",
            )
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

        # A household directive that planned nothing is a command the model
        # declined to act on, not a conversation. Letting it render freely is
        # how the assistant ended up cheerfully explaining that it could not
        # find lights it can see and control — an answer that is both wrong and
        # worse than silence, because it reads as a considered refusal. A
        # command speaks when it succeeds; otherwise it stays quiet and the
        # transcript records that nothing happened.
        unplanned_command = bool(
            addressed
            and not outcome.execute
            and not outcome.shadowed
            and not results
            and interpretation.speech_act in _COMMAND_SPEECH_ACTS
            # A clarify is a deliberate, useful question ("which lamp?") and is
            # the one thing an unexecuted command may legitimately say.
            and interpretation.decision != Decision.CLARIFY
            # The speech act alone is too loose: "tell me a joke" and "how are
            # you" are routinely classified as directives, and silencing those
            # would make the assistant mute rather than accurate. Only a turn
            # that plausibly concerns the household is a household command.
            and household_state_is_relevant(utterance.transcript)
            # A bare wake word carries no command to have failed.
            and command_word_count(utterance.transcript, self._address_words()) > 0
        )
        if unplanned_command:
            logger.info(
                "silenced unplanned command: utterance=%s speech_act=%s decision=%s",
                utterance.id,
                interpretation.speech_act.value,
                interpretation.decision.value,
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
        turn_machine.record_response("deterministic", response_text)
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
        # reply in [commandReplyMinWords, commandReplyMaxWords]; zero at both
        # ends means a silent acknowledgement (the command still ran, nothing
        # is spoken). A dashboard update can momentarily leave min > max (the
        # two sliders commit independently); clamp rather than raise.
        command_max_words: int | None = None
        if verified_dashboard_command and self.voice_settings is not None:
            reply_min = min(
                self.voice_settings.command_reply_min_words,
                self.voice_settings.command_reply_max_words,
            )
            command_max_words = random.randint(
                reply_min, self.voice_settings.command_reply_max_words
            )

        # A bare wake word/address prefix ("<name>", "hey <name>") with no
        # command attached is a person just calling out, not a request. Give
        # it a short, varied verbal nudge instead of the normal conversational
        # reply, rendered at a fixed high temperature so it doesn't repeat.
        bare_wake_word_turn = bool(
            utterance.wake_detected
            and interpretation.decision == Decision.REPLY
            and not interpretation.actions
            and not results
            and command_word_count(utterance.transcript, self._address_words()) == 0
        )
        bare_wake_max_words = random.choice((1, 2)) if bare_wake_word_turn else None

        if unplanned_command:
            # No model render, and no deterministic persona line either: a
            # command that did not run has nothing true to say.
            response_text = None
            turn_machine.record_response("final", None, force=True)
        elif verified_dashboard_command and command_max_words == 0:
            response_text = None
        elif not outcome.shadowed and (
            self._needs_model_render(interpretation, results)
            or wants_varied_wording
            or verified_dashboard_command
            or bare_wake_word_turn
        ):
            rendered = await self.interpreter.render_response(
                utterance,
                interpretation,
                results,
                persona=self.persona.response_prompt,
                environment=None,
                relevant_state=relevant_state,
                conversation=conversation,
                temperature=(
                    self._bare_wake_word_temperature
                    if bare_wake_word_turn
                    else effective_render_temperature
                ),
                command_max_words=command_max_words,
                bare_wake_max_words=bare_wake_max_words,
            )
            response_text = rendered or response_text
            if rendered:
                turn_machine.record_response("model", response_text)

        if (
            response_text is None
            # An unplanned command's silence is deliberate and argued for above:
            # a command speaks when it succeeds, and explaining a failure it
            # does not understand reads as a considered refusal. That decision
            # is not this one's to overturn.
            and not unplanned_command
            and addressed_directive_needs_an_answer(utterance, interpretation, outcome)
        ):
            # What is left is a turn that fell silent for no deliberate reason —
            # most often because rendering failed and returned nothing. From the
            # room that is indistinguishable from the assistant being broken.
            #
            # Deliberately not a model call: the model is either unavailable or
            # has just failed to produce something usable, and this reply has to
            # work in exactly that case.
            response_text = UNPARSED_REQUEST_REPLY
            turn_machine.record_response("deterministic", response_text)
            logger.info(
                "addressed turn produced no reply; using the deterministic "
                "acknowledgement id=%s decision=%s actions=%d",
                utterance.id,
                interpretation.decision,
                len(interpretation.actions),
            )

        fallback_started = time.perf_counter()
        fallback = await self._recover_knowledge_failure(
            utterance,
            interpretation,
            results,
            response_text,
            conversation=conversation,
            relevant_state=relevant_state,
            temperature=effective_render_temperature,
            session_active=goal is not None,
            cancellation=cancellation,
        )
        if fallback is not None:
            interpretation, results, outcome, response_text = fallback
            # The normal session update ran before response drafting. Record the
            # late, read-only recovery action too so the final HandleResult and
            # goal lifecycle describe what actually happened this turn.
            self.sessions.update(
                utterance,
                interpretation,
                results,
                executed=outcome.execute,
            )
            # TranscriptStore uses INSERT OR REPLACE by utterance id, so this
            # updates the retained interpretation to include the late tool call
            # instead of leaving the pre-fallback reply classification behind.
            await self.store.add(utterance, interpretation)
            timings_ms["knowledgeFallback"] = round(
                (time.perf_counter() - fallback_started) * 1000,
                3,
            )
            turn_machine.record_response("knowledge_fallback", response_text)
        if memory_control_reply is not None:
            response_text = memory_control_reply
            turn_machine.record_response("deterministic", response_text)
        # Persist only explicit/salient non-routine memories for a recognized
        # speaker. Sensitive material is intentionally not auto-filed: the
        # owner must use the dashboard confirmation workflow once exposed.
        if self.memory is not None and utterance.speaker.status == "recognized":
            candidate = salient_memory_candidate(utterance.transcript)
            if candidate is not None:
                cancellation.begin_non_cancellable_side_effect("memory_create")
                try:
                    await self.memory.create(
                        MemoryRecord(
                            text=candidate.text,
                            memory_type=candidate.memory_type,
                            sensitivity=candidate.sensitivity,
                            needs_confirmation=candidate.needs_confirmation,
                            owner_id=utterance.speaker.person_id,
                            audience=[utterance.speaker.person_id]
                            if utterance.speaker.person_id
                            else [],
                            provenance="voice conversation salience rule",
                            source_turn_id=utterance.id,
                        ),
                        access=memory_access,
                    )
                finally:
                    cancellation.non_cancellable_side_effect_finished()
        turn_machine.set_policy(
            execute=outcome.execute,
            shadowed=outcome.shadowed,
            reason=outcome.reason,
        )
        turn_machine.record_tools(list(interpretation.actions), results)
        turn_machine.advance(
            TurnStage.EXECUTE,
            elapsed_ms=timings_ms["execution"] + timings_ms.get("knowledgeFallback", 0),
            status=(
                TurnStageStatus.COMPLETED
                if interpretation.actions or results
                else TurnStageStatus.SKIPPED
            ),
            detail=None if interpretation.actions or results else "no tool plan",
        )
        verify_started = time.perf_counter()
        turn_machine.record_verification(results)
        turn_machine.advance(
            TurnStage.VERIFY,
            elapsed_ms=round((time.perf_counter() - verify_started) * 1000, 3),
            status=TurnStageStatus.COMPLETED if results else TurnStageStatus.SKIPPED,
            detail=None if results else "no tool results",
        )
        # Affectations run on the finished reply — template or model alike — so
        # the quirk is consistent everywhere the text goes: TTS, transcript,
        # and the conversation history the model sees on later turns.
        before_affectations = response_text
        response_text = self._apply_affectations(response_text)
        if response_text != before_affectations:
            turn_machine.record_response("affectations", response_text)
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
            turn_machine.record_response("final", response_text)
        turn_machine.record_response("final", response_text, force=True)
        engaged = turn_extends_conversation(
            wake_detected=utterance.wake_detected,
            interpretation=interpretation,
            addressed_threshold=self.settings.active_addressed_threshold,
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
                if self.continuity is not None:
                    try:
                        await self.continuity.observe(
                            conversation_id=conversation.id,
                            room_id=conversation.room_id,
                            participant_id=(
                                utterance.speaker.person_id
                                if utterance.speaker.status == "recognized"
                                else None
                            ),
                            topic_summary=interpretation.active_goal.summary or None,
                            user_text=utterance.transcript,
                        )
                    except Exception:
                        logger.warning(
                            "conversation continuity update failed id=%s",
                            conversation.id,
                            exc_info=True,
                        )
                self.conversations.append_turn(
                    utterance.room_id,
                    utterance.transcript,
                    response_text,
                    speaker_name=(
                        utterance.speaker.display_name
                        if utterance.speaker.status == "recognized"
                        else None
                    ),
                    speaker_pronouns=(
                        utterance.speaker.pronouns
                        if utterance.speaker.status == "recognized"
                        else None
                    ),
                )
                observations = _dashboard_observations(interpretation, results)
                if observations:
                    self.conversations.record_observations(utterance.room_id, observations)
        timings_ms["response"] = round((time.perf_counter() - response_started) * 1000, 3)
        turn_machine.advance(TurnStage.RENDER, elapsed_ms=timings_ms["response"])
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
            speaker=utterance.speaker,
            executed=outcome.execute,
            shadowed=outcome.shadowed,
            policy_reason=outcome.reason,
            results=results,
            response_text=response_text,
            response_tone_instruction=self.persona.tone_instruction(interpretation.emotion),
            timings_ms=timings_ms,
            route_comparison=self._route_comparison(),
        )

    def _route_comparison(self) -> dict | None:
        """Both sides' replies, if the reply pass is being compared this turn.

        Only the reply is carried into the transcript. The other passes are
        compared too, and their answers are on the status endpoint — but a
        transcript is a record of what was *said*, and an interpretation or an
        icon id shown beside a spoken line would be noise in the place people
        read to understand a conversation.
        """

        router = self.companion_router
        if router is None or "render_response" not in router.comparing():
            return None
        return router.pop_comparison("render_response")

    async def execute_companion_tool_call(self, call) -> ToolResultMessage:
        """Run one tool a companion asked for, and answer on the same socket.

        The companion plans; it never executes. Its ``tool_call`` re-enters
        here so registry validation, tool policy, the household authority gate
        and the audit trail apply exactly as they would to a locally planned
        action — and so the companion never needs, or receives, dashboard
        credentials.
        """

        action = PlannedAction(
            id=f"companion-{call.call_id}",
            order=0,
            call=CapabilityToolCall(
                provider=call.provider, tool=call.tool, arguments=call.arguments
            ),
        )
        result = await self.execute_companion_action(action)
        return ToolResultMessage(
            jobId=call.job_id,
            attemptId=call.attempt_id,
            callId=call.call_id,
            ok=result.ok,
            code=result.code,
            message=result.message,
            observed=result.observed,
        )

    async def execute_companion_action(
        self,
        action: PlannedAction,
        on_behalf_of: SpeakerIdentity | None = None,
    ) -> ToolResult:
        """Apply the companion-specific gate, then the ordinary execution path.

        Two restrictions sit on top of the normal policy, because this call
        arrives without a live speaker in the room to answer for it:

        * A mutation needing confirmation is refused *here* and becomes a
          durable approval step instead. An approval is not an execution: the
          write happens afterwards, once, under the stored idempotency key.
          Until that machinery lands, refusing is the safe half of the
          contract — a companion can never silently perform a confirmable
          mutation.
        * A job with no originating speaker (a background job, not a turn) may
          only run low-risk actions. Holding a household certificate does not
          confer the household's authority.
        """

        try:
            canonical = self.registry.validate_action(action)
        except (KeyError, ValueError):
            return ToolResult(
                action_id=action.id,
                ok=False,
                code="invalid",
                requested=action.call.arguments,
                message="Action did not match an allowlisted semantic tool schema",
            )
        policy = self.registry.policy_for(canonical.call.provider, canonical.call.tool)
        if policy is None or policy.risk == "blocked":
            return ToolResult(
                action_id=action.id,
                ok=False,
                code="blocked",
                requested=action.call.arguments,
                message="This action is not available to a companion agent",
            )
        if policy.requires_confirmation or policy.risk == "confirmation":
            return ToolResult(
                action_id=action.id,
                ok=False,
                code="blocked",
                requested=action.call.arguments,
                message="This action needs approval before it can run",
            )
        if on_behalf_of is None:
            if policy.risk != "low":
                return ToolResult(
                    action_id=action.id,
                    ok=False,
                    code="blocked",
                    requested=action.call.arguments,
                    message="A background companion job may only run low-risk actions",
                )
        elif self.authority is not None:
            outcome = self.authority.authorize(on_behalf_of, [canonical])
            if not outcome.allowed:
                return ToolResult(
                    action_id=action.id,
                    ok=False,
                    code="blocked",
                    requested=action.call.arguments,
                    message=outcome.reason,
                )
        results = await self._execute_plan([canonical])
        if not results:
            return ToolResult(
                action_id=action.id,
                ok=False,
                code="backend_error",
                requested=action.call.arguments,
                message="The action produced no result",
            )
        return results[0]

    async def _execute_plan(
        self,
        actions: Iterable[PlannedAction],
        *,
        cancellation: TurnCancellationController | None = None,
    ) -> list[ToolResult]:
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

        async def invoke(action: PlannedAction, policy, provider) -> ToolResult:
            provider_task = asyncio.create_task(provider.execute(action))
            if cancellation is not None:
                cancellation.bind_provider(
                    action.id,
                    getattr(policy, "cancellation", "before_side_effects"),
                    provider_task,
                )
            try:
                return await provider_task
            except asyncio.CancelledError:
                if cancellation is None or not cancellation.requested.is_set():
                    raise
                return ToolResult(
                    action_id=action.id,
                    ok=False,
                    code="cancelled",
                    requested=action.call.arguments,
                    message="Capability execution was cancelled by request",
                )
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
            finally:
                if cancellation is not None:
                    cancellation.provider_finished(provider_task)

        while pending:
            if cancellation is not None and cancellation.requested.is_set():
                for action in sorted(pending.values(), key=lambda candidate: candidate.order):
                    append_result(
                        action,
                        ToolResult(
                            action_id=action.id,
                            ok=False,
                            code="cancelled",
                            requested=action.call.arguments,
                            message="Action was cancelled before provider side effects",
                        ),
                    )
                break
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
                action, canonical_action, policy, provider = runnable[0]
                append_result(action, await invoke(canonical_action, policy, provider))
                continue

            if first_serial is not None:
                runnable = runnable[:first_serial]

            # Every ready action in this wave has opted into safe concurrency.
            # gather preserves input ordering, so response/result ordering stays
            # deterministic even when provider calls complete out of order.
            outputs = await asyncio.gather(
                *(
                    invoke(canonical_action, policy, provider)
                    for _action, canonical_action, policy, provider in runnable
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
        # A successful web lookup must be spoken by the model (it relays the
        # retrieved answer); unlike a device command, its result is content, not
        # a confirmation, so the fixed template renderer cannot voice it.
        web_action_ids = {
            action.id for action in interpretation.actions if action.call.provider == "web"
        }
        return bool(
            interpretation.decision.value in {"reply", "clarify"}
            or interpretation.response_plan.requires_post_tool_rendering
            or any(not result.ok for result in results)
            or any(result.action_id in web_action_ids for result in results)
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
            "capabilityProviders": await self.registry.health(),
            "llm": llm_health,
            "retainedTranscripts": await self.store.count(),
            "durableAgent": (
                await self.durable_store.health()
                if self.durable_store is not None
                else {"ok": True, "enabled": False}
            ),
            "householdEvents": (
                self.event_consumer.health()
                if self.event_consumer is not None
                else {"ok": True, "enabled": False}
            ),
            "speakerProfiles": (
                await self.speaker_profiles.health()
                if self.speaker_profiles is not None
                else {"ok": True, "enabled": False}
            ),
            "memory": (
                await self.memory.health()
                if self.memory is not None
                else {"ok": True, "enabled": False}
            ),
            "offlineOptimizers": (
                self.optimizers.health()
                if self.optimizers is not None
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
        if self.event_consumer is not None:
            await self.event_consumer.close()
        if self.memory is not None:
            await self.memory.close()
        if self.commitments is not None:
            await self.commitments.close()
        if self.research is not None:
            await self.research.close()
        if self.briefings is not None:
            await self.briefings.close()
        if self.optimizers is not None:
            await self.optimizers.close()
        await self.registry.close()
        await self.interpreter.close()
