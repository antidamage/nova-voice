from __future__ import annotations

import json
import logging
import random
import re
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from nova_voice.audio.conversation import ConversationMessage, ConversationSnapshot
from nova_voice.domain import (
    ActiveGoal,
    Interpretation,
    SelfProfileUpdate,
    ToolResult,
    Utterance,
)
from nova_voice.interpretation.base import Interpreter
from nova_voice.interpretation.response_length import (
    bounded_long_reply,
    command_acknowledgement,
    spoken_word_count,
)
from nova_voice.interpretation.speech_cues import has_explicit_self_intention

logger = logging.getLogger(__name__)

_TIME_RELEVANCE = re.compile(
    r"\b(?:time\s+is\s+it|current\s+time|what(?:'s|\s+is)\s+the\s+time|"
    r"date|what\s+day|today|tonight|tomorrow|sunrise|sunset)\b",
    re.IGNORECASE,
)
_WEATHER_RELEVANCE = re.compile(
    r"\b(?:weather|forecast|temperature|degrees?|"
    r"rain(?:ing|y)?|sunny|cloud(?:y|s)?|wind(?:y)?|storm(?:y)?|"
    r"snow(?:ing|y)?|outside|outdoors|umbrella|coat|jacket|hot|cold|warm|cool|"
    r"what\s+should\s+i\s+wear|go\s+for\s+a\s+walk|hang\s+(?:the\s+)?washing)\b",
    re.IGNORECASE,
)


def environment_context_is_relevant(transcript: str) -> bool:
    """Gate time/weather facts to requests they can materially help answer."""

    return bool(_TIME_RELEVANCE.search(transcript) or _WEATHER_RELEVANCE.search(transcript))


# Retained dashboard data is only injected when the current utterance plausibly
# concerns household state, so it is never dumped into unrelated turns.
_HOUSEHOLD_STATE_RELEVANCE = re.compile(
    r"\b(?:status|state|already|still|"
    r"turn(?:ed|ing)?|switch(?:ed|ing)?|"
    r"temperature|degrees?|thermostat|set(?:ting)?|"
    r"aircon|air\s*con(?:ditioner)?|heater|heating|cooling|"
    r"lights?|lamp|fan|door|lock|blinds?|curtains?|plug|heat|"
    r"what'?s\s+on|what'?s\s+off|is\s+it\s+(?:on|off)|"
    r"how\s+(?:warm|cold|hot|cool))\b",
    re.IGNORECASE,
)


def household_state_is_relevant(transcript: str) -> bool:
    """Gate retained dashboard data to turns that plausibly concern devices."""

    return bool(_HOUSEHOLD_STATE_RELEVANCE.search(transcript))


def select_environment_context(
    transcript: str,
    environment: dict[str, Any],
) -> dict[str, Any] | None:
    """Return only the time/weather portion relevant to this specific turn."""

    selected: dict[str, Any] = {}
    if _TIME_RELEVANCE.search(transcript) and environment.get("now") is not None:
        selected["now"] = environment["now"]
    if _WEATHER_RELEVANCE.search(transcript) and environment.get("weather") is not None:
        selected["weather"] = environment["weather"]
    return selected or None


def _current_speaker_context(utterance: Utterance) -> str | None:
    speaker = utterance.speaker
    if speaker.status != "recognized" or not speaker.display_name:
        return None
    profile = {
        "name": speaker.display_name,
        "pronouns": speaker.pronouns,
    }
    return (
        "Authoritative current-speaker identity for this exact acoustic turn: "
        + json.dumps(profile, ensure_ascii=False, separators=(",", ":"))
        + ". Use this person's name and stated pronouns naturally when relevant. "
        "If earlier conversation messages name a different speaker, the user has changed; "
        "never carry the previous speaker's name or pronouns onto this turn."
    )


def _conversation_message_content(message: ConversationMessage) -> str:
    if message.role != "user" or not message.speaker_name:
        return message.content
    identity = f"Speaker: {message.speaker_name}"
    if message.speaker_pronouns:
        identity += f"; pronouns: {message.speaker_pronouns}"
    return f"[{identity}] {message.content}"

SYSTEM_PROMPT = """You are {agent_name}'s interpretation engine, not a general chat UI.
The assistant's accepted spoken wake words are {wake_words}; an utterance whose first
words include one of them is directed at the assistant.
For every final utterance, classify emotion and speech act, decide whether it is
directed to the assistant, maintain the room goal, and optionally create a bounded
semantic action plan.

Critical distinctions:
- Imperatives, polite requests, and explicit desired household states are directives.
- A speaker saying they plan, need, have, or have got to do something themselves is
  self_intention and must not execute.
- If deterministicCues.explicitSelfIntention is true, return self_intention, ignore,
  and zero actions even inside an active conversation. It is a safety constraint.
- Quoted, third-party, television, assistant playback, observations, and uncertain
  pronouns do not execute in passive mode.
- A wake word increases addressing confidence but is not required for a clear directive.
- A clear, unquoted directive or desired household state that requires tool actions is
  addressed by definition. When decision is execute, set addressedProbability to 1.0 even
  when no wake word was heard; use the speech-act and confidence fields to reject ambiguity.
- When utterance.conversationActive is true, the speech is addressed to the assistant even
  without another wake word. Do not ignore it merely because the wake word is absent.
- Persona/tone never changes the requested action.
- utterance.speaker describes only the current acoustic turn. A recognized profile may
  personalize the reply; an unknown speaker must never inherit identity from conversation history.
- Always set selfProfileUpdate to null. A separate current-turn identity extractor owns
  name/pronoun disclosure detection and securely binds accepted values to the voice template.
- You have a local speaker-profile capability that can remember a person's explicitly stated
  name and pronouns against their voice. If someone asks how to correct either value, explain
  that they can tell you directly, for example "call me Addie" or "I use she/her pronouns".
  The separate identity extractor handles any correction; keep selfProfileUpdate null here.
- Never invent provider names, tool names, entity IDs, services, rooms, or success.
- In each action, copy call.tool exactly from semanticTools.function.name, including namespace.
- Return zero to four actions. Dependencies refer only to earlier action IDs.
- Non-execute decisions have no actions.
- The response plan must not claim an action worked; execution happens later.

Decision mapping:
- execute only when one or more semantic tool actions are required and returned.
- reply to an addressed question, greeting, social/conversational turn, general-knowledge
  request, or request such as "tell me a joke" that needs no semantic tool.
- The conversation-start system context may carry a local date/time and outdoor-weather
  snapshot. A question about those facts is a reply, never a tool call. Mention them in
  other replies only when they are a meaningful addition.
- relevantState.indoorTemperatureC is the measured indoor temperature for this room in
  Celsius; null means no sensor is configured, so the indoor temperature is unknown. It is
  distinct from the outdoor-weather snapshot — never report one as the other, and say the
  indoor temperature is unknown rather than substituting the outdoor value.
- relevantState.indoorRooms contains only physical rooms inside the home. Outside weather
  is separate; Home, Climate, and Network are organisational zones, not rooms.
- relevantState.climateControls is the authoritative climate interface. Offer only power
  on/off and target temperature. Raw heat/cool/manual HVAC modes are implementation details,
  never separate controls. Use turn_on/turn_off for power and set_temperature for a target.
- clarify an addressed household request only when a required target or value is missing.
- ignore only ambient/unaddressed speech, quoted/media speech, third-party speech, explicit
  self-intention, or abandoned/negated requests. Never ignore an addressed social turn or
  question merely because it does not use a tool.

Return only the JSON object described by the response schema.
"""

IDENTITY_DISCLOSURE_PROMPT = """You are a narrow identity-disclosure extractor.
Inspect only the supplied current-turn transcript. Decide whether the current speaker
explicitly states or corrects their own name, their own pronouns, or both.

Rules:
- Accept first-person disclosures such as "my name is Adeline", "call me Addie",
  "I go by Adeline", "my pronouns are she her", or "I use she/her pronouns".
- Never infer identity from voice, grammar, stereotypes, prior turns, or assistant context.
- Reject names/pronouns said about another person, quoted text, examples, media dialogue,
  questions about how profiles work, and requests that do not themselves disclose a value.
- evidence must be one exact contiguous substring copied verbatim from the transcript.
- Normalize pronouns to slash form when the speaker says separators aloud: "she her" and
  "she and her" both become "she/her". Preserve explicitly supplied forms otherwise.
- If neither value is explicitly disclosed, return disclosed=false with all other fields null.
- If disclosed=true, include at least one of name or pronouns and include exact evidence.
Return only the JSON object described by the response schema."""


class InterpretationError(RuntimeError):
    pass


class RenderedResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=500)


class IdentityDisclosure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    disclosed: bool
    name: str | None
    pronouns: str | None
    evidence: str | None

    def profile_update(self) -> SelfProfileUpdate | None:
        if not self.disclosed or (self.name is None and self.pronouns is None):
            return None
        if not self.evidence:
            return None
        return SelfProfileUpdate(
            name=self.name,
            pronouns=self.pronouns,
            evidence=self.evidence,
        )


class LlamaCppInterpreter(Interpreter):
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        skills_text: str = "",
        timeout_seconds: float = 20,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.model = model
        self.skills_text = skills_text
        # Sampling temperature for the spoken-response renderer only; the
        # interpretation pass stays deterministic at zero. Live-tunable from
        # the dashboard's Voice Agent settings, as are the wake word and
        # personality description below.
        self.render_temperature: float = 0.0
        # Chance (0-1) that a single conversational reply is rendered
        # long-form (two to four sentences); rolled fresh for each response.
        self.long_response_probability: float = 0.0
        self.agent_name: str = "Nova"
        self.wake_words: list[str] = ["beemo", "bimo", "bemo", "beamo", "bmo"]
        self.personality: str = "You are a bright, bubbly helper!"
        # One-line pronoun instruction (subjective/objective/possessive) applied
        # to both the interpretation and response prompts. Dashboard-tunable and
        # part of a saved voice personality; empty disables it.
        self.pronoun_instruction: str = ""
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
            transport=transport,
        )

    async def extract_self_profile_update(
        self, utterance: Utterance
    ) -> SelfProfileUpdate | None:
        """Run a small context-free pass for this turn's identity disclosure."""

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": IDENTITY_DISCLOSURE_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {"transcript": utterance.transcript}, separators=(",", ":")
                    ),
                },
            ],
            "temperature": 0,
            "max_tokens": 120,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "nova_identity_disclosure",
                    "strict": True,
                    "schema": IdentityDisclosure.model_json_schema(),
                },
            },
            "chat_template_kwargs": {"enable_thinking": False},
        }
        try:
            response = await self._client.post("/chat/completions", json=payload)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise TypeError("completion content was not text")
            return IdentityDisclosure.model_validate_json(content).profile_update()
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as error:
            # Identity learning is best-effort and must never take down the
            # household's normal interpretation/reply path.
            logger.warning("identity disclosure extraction unavailable: %s", error)
            return None

    async def interpret(
        self,
        utterance: Utterance,
        *,
        active_goal: ActiveGoal | None,
        relevant_state: dict[str, Any],
        tools: list[dict],
        conversation: ConversationSnapshot | None = None,
    ) -> Interpretation:
        context = {
            "utterance": {
                "transcript": utterance.transcript,
                "confidence": utterance.transcript_confidence,
                "room": utterance.room_id,
                "wakeDetected": utterance.wake_detected,
                "conversationActive": utterance.conversation_active,
                "dashboardForeground": utterance.dashboard_foreground,
                "acoustic": utterance.acoustic.model_dump(mode="json"),
                "speaker": utterance.speaker.model_dump(mode="json"),
            },
            "activeGoal": active_goal.model_dump(mode="json") if active_goal else None,
            "deterministicCues": {
                "explicitSelfIntention": has_explicit_self_intention(utterance.transcript)
            },
            "relevantState": relevant_state,
            "semanticTools": tools,
        }
        # Transcripts arrive with the spoken wake phrase already rewritten to
        # the agent's display name, so the name itself is an accepted wake
        # word from the model's point of view.
        system = SYSTEM_PROMPT.format(
            agent_name=self.agent_name,
            wake_words=json.dumps(
                [*self.wake_words, self.agent_name], separators=(",", ":")
            ),
        )
        speaker_context = _current_speaker_context(utterance)
        if speaker_context:
            system += "\n\n" + speaker_context
        personality = (
            conversation.personality
            if conversation is not None and conversation.personality
            else self.personality
        )
        if personality:
            system += (
                "\n\nPersonality description (shapes tone and phrasing only, "
                "never decisions or actions):\n" + personality
            )
        if self.pronoun_instruction:
            system += "\n\n" + self.pronoun_instruction
        if self.skills_text:
            system += "\n\nCompact operating skills:\n" + self.skills_text
        if conversation is not None and conversation.initial_environment is not None:
            system += (
                "\n\nConversation-start local time and weather snapshot (do not assume it "
                "has refreshed during this conversation). Use it when it answers the request "
                "or is a meaningful addition; do not force it into unrelated replies:\n"
                + json.dumps(conversation.initial_environment, separators=(",", ":"))
            )
        room = (utterance.room_id or "").strip()
        if room and room != "preview":
            system += (
                f"\n\nThis satellite is in the {room}. 'Here', 'this room', and an "
                f"unqualified target refer to the {room}."
            )
        if (
            conversation is not None
            and conversation.observations
            and household_state_is_relevant(utterance.transcript)
        ):
            system += (
                "\n\nDashboard data already retrieved earlier in this conversation (may be "
                "stale; use it to answer follow-ups and maintain the goal, never invent "
                "beyond it):\n"
                + "\n".join(f"- {entry}" for entry in conversation.observations)
            )
        schema = Interpretation.model_json_schema()
        messages = [{"role": "system", "content": system}]
        if conversation is not None:
            messages.extend(
                {
                    "role": message.role,
                    "content": _conversation_message_content(message),
                }
                for message in conversation.messages
            )
        messages.append({"role": "user", "content": json.dumps(context, separators=(",", ":"))})
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 1000,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "nova_interpretation", "strict": True, "schema": schema},
            },
            "chat_template_kwargs": {"enable_thinking": False},
        }
        try:
            response = await self._client.post("/chat/completions", json=payload)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise TypeError("completion content was not text")
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            return Interpretation.model_validate_json(content)
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as error:
            raise InterpretationError("local LLM interpretation failed") from error

    async def health(self) -> dict:
        try:
            response = await self._client.get("/models", timeout=3)
            response.raise_for_status()
            return {"ok": True, "model": self.model}
        except httpx.HTTPError:
            return {"ok": False, "model": self.model}

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
        all_succeeded = bool(results) and all(result.ok for result in results)
        if all_succeeded:
            word_budget = 1 if command_max_words is None else command_max_words
            plural = "s" if word_budget != 1 else ""
            response_instruction = (
                "The requested dashboard action has just completed and been verified. "
                f"Acknowledge it in exactly {word_budget} spoken word{plural} in the configured "
                "personality (for example: Done). No device names, no values."
            )
        elif results:
            response_instruction = (
                "Report the supplied partial failure or failure briefly. Do not imply success "
                "for any result whose ok value is false."
            )
        else:
            response_instruction = "Answer the user's conversational request directly."
        conversation_environment = (
            select_environment_context(
                utterance.transcript,
                conversation.initial_environment,
            )
            if conversation is not None
            and conversation.initial_environment is not None
            else None
        )
        facts = {
            "utterance": utterance.transcript,
            "speaker": utterance.speaker.model_dump(mode="json"),
            "selfProfileUpdate": (
                interpretation.self_profile_update.model_dump(mode="json")
                if interpretation.self_profile_update is not None
                else None
            ),
            "speakerProfileUpdateApplied": (
                interpretation.self_profile_update is not None
                and utterance.speaker.status in {"pending", "recognized"}
            ),
            "decision": interpretation.decision,
            "goal": interpretation.active_goal.model_dump(mode="json"),
            "actions": [action.model_dump(mode="json") for action in interpretation.actions],
            "results": [result.model_dump(mode="json") for result in results],
            "environment": environment if conversation is None else conversation_environment,
            "relevantState": (
                relevant_state if household_state_is_relevant(utterance.transcript) else None
            ),
            "responseInstruction": response_instruction,
        }
        selected_persona = (
            conversation.persona_prompt
            if conversation is not None and conversation.persona_prompt
            else persona
        )
        conversational_reply = not results and interpretation.decision == "reply"
        long_form = (
            conversational_reply and random.random() < self.long_response_probability
        )
        length_instruction = (
            "This conversational reply may use up to three substantial sentences. Relate "
            "or create a story that is relevant to the user's subject and shaped by your "
            "personality, then guide the final sentence back to the user's topic."
            if long_form
            else "Conversational replies must be one sentence and at most 10 spoken words."
        )
        system = f"""You are {self.agent_name}, and you are speaking your own reply out loud
right now. Speak in the first person: call yourself "I", "me", and "my". Never refer to
yourself by name or in the third person — never "{self.agent_name} will", "she can", or
"the assistant did"; say "I will", "I can", "I did".
{selected_persona}
Be concise and natural. You may complain in at most one short sentence, but you must
still help. Never change a requested target or value. Never claim an action succeeded
unless its supplied result has ok=true. If a result failed, say so plainly. If the
decision is clarify, ask one concrete question. When decision=reply and there are no tool
results, answer the conversational request directly, but never imply that household state
changed. facts.relevantState is either null or the current authoritative smart-home state.
Its indoorRooms are inside, climateControls offer only on/off plus target temperature, and
outdoor weather is separate. Use measured room temperature only for that named indoor room.
facts.environment is either null or a vetted conversation-start time/weather
snapshot. When it is null, never mention the date, time, temperature, or weather. When it
is present, use it only to answer the request or make a materially useful inference. Never
use those facts as filler in a greeting, joke, acknowledgement, offer to help, or unrelated
answer. You can remember a speaker's explicitly stated name and pronouns locally. If asked
how to fix them, tell the speaker to state the correction directly, such as "call me Addie"
or "I use she/her pronouns". When facts.selfProfileUpdate is non-null and
facts.speakerProfileUpdateApplied is true, briefly acknowledge the accepted values. When it
is false, do not claim the correction was saved. {length_instruction} When the
responseInstruction asks for a single word,
return exactly one word. For a greeting, do not spend the complaint budget;
greet briefly and offer help. Return only the response JSON schema."""
        speaker_context = _current_speaker_context(utterance)
        if speaker_context:
            system += "\n" + speaker_context
        personality = (
            conversation.personality
            if conversation is not None and conversation.personality
            else self.personality
        )
        if personality:
            system += "\nPersonality description: " + personality
        if self.pronoun_instruction:
            system += "\n" + self.pronoun_instruction
        room = (utterance.room_id or "").strip()
        if room and room != "preview":
            system += f"\nYou are speaking in the {room}; 'here' and 'this room' mean the {room}."
        if (
            conversation is not None
            and conversation.observations
            and household_state_is_relevant(utterance.transcript)
        ):
            system += (
                "\nDashboard data retrieved earlier this conversation (may be stale; use it to "
                "answer follow-ups, never claim it changed or invent beyond it): "
                + " | ".join(conversation.observations)
            )
        messages = [{"role": "system", "content": system}]
        if conversation is not None:
            messages.extend(
                {
                    "role": message.role,
                    "content": _conversation_message_content(message),
                }
                for message in conversation.messages
            )
        messages.append({"role": "user", "content": json.dumps(facts, separators=(",", ":"))})
        payload = {
            "model": self.model,
            "messages": messages,
            # Zero keeps replies deterministic and TTS-cacheable; the dashboard
            # can raise it for more varied phrasing. The caller may override it
            # per turn (e.g. forcing zero while recovering from a failed command).
            "temperature": (
                temperature if temperature is not None else self.render_temperature
            ),
            "max_tokens": 240 if long_form else 80,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "nova_spoken_response",
                    "strict": True,
                    "schema": RenderedResponse.model_json_schema(),
                },
            },
            "chat_template_kwargs": {"enable_thinking": False},
        }
        try:
            response = await self._client.post("/chat/completions", json=payload)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            rendered = RenderedResponse.model_validate_json(content).text.strip()
            if all_succeeded and command_max_words is not None:
                if spoken_word_count(rendered) != command_max_words:
                    return command_acknowledgement(command_max_words)
            if long_form:
                return bounded_long_reply(rendered)
            return rendered
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError):
            return None

    async def close(self) -> None:
        await self._client.aclose()
