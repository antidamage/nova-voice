from __future__ import annotations

import json
import re
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from nova_voice.audio.conversation import ConversationSnapshot
from nova_voice.domain import ActiveGoal, Interpretation, ToolResult, Utterance
from nova_voice.interpretation.base import Interpreter
from nova_voice.interpretation.speech_cues import has_explicit_self_intention

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
- When utterance.conversationActive is true, the speech is addressed to the assistant even
  without another wake word. Do not ignore it merely because the wake word is absent.
- Persona/tone never changes the requested action.
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
- clarify an addressed household request only when a required target or value is missing.
- ignore only ambient/unaddressed speech, quoted/media speech, third-party speech, explicit
  self-intention, or abandoned/negated requests. Never ignore an addressed social turn or
  question merely because it does not use a tool.

Return only the JSON object described by the response schema.
"""


class InterpretationError(RuntimeError):
    pass


class RenderedResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=500)


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
        self.agent_name: str = "Nova"
        self.wake_words: list[str] = ["beemo", "bimo", "bemo", "beamo", "bmo"]
        self.personality: str = "You are a bright, bubbly helper!"
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
            transport=transport,
        )

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
            },
            "activeGoal": active_goal.model_dump(mode="json") if active_goal else None,
            "deterministicCues": {
                "explicitSelfIntention": has_explicit_self_intention(utterance.transcript)
            },
            "relevantState": relevant_state,
            "semanticTools": tools,
        }
        system = SYSTEM_PROMPT.format(
            agent_name=self.agent_name,
            wake_words=json.dumps(self.wake_words, separators=(",", ":")),
        )
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
        if self.skills_text:
            system += "\n\nCompact operating skills:\n" + self.skills_text
        if conversation is not None and conversation.initial_environment is not None:
            system += (
                "\n\nConversation-start local time and weather snapshot (do not assume it "
                "has refreshed during this conversation). Use it when it answers the request "
                "or is a meaningful addition; do not force it into unrelated replies:\n"
                + json.dumps(conversation.initial_environment, separators=(",", ":"))
            )
        schema = Interpretation.model_json_schema()
        messages = [{"role": "system", "content": system}]
        if conversation is not None:
            messages.extend(
                {"role": message.role, "content": message.content}
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
        conversation: ConversationSnapshot | None = None,
    ) -> str | None:
        all_succeeded = bool(results) and all(result.ok for result in results)
        if all_succeeded:
            response_instruction = (
                "The requested dashboard action has just completed and been verified. "
                "Acknowledge it with a single spoken word in the configured personality "
                "(for example: Done). No device names, no values, no extra words."
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
            "decision": interpretation.decision,
            "goal": interpretation.active_goal.model_dump(mode="json"),
            "actions": [action.model_dump(mode="json") for action in interpretation.actions],
            "results": [result.model_dump(mode="json") for result in results],
            "environment": environment if conversation is None else conversation_environment,
            "responseInstruction": response_instruction,
        }
        selected_persona = (
            conversation.persona_prompt
            if conversation is not None and conversation.persona_prompt
            else persona
        )
        system = f"""You render {self.agent_name}'s final spoken response after interpretation.
Persona: {selected_persona}
Be concise and natural. The persona may complain in at most one short sentence, but must
still help. Never change a requested target or value. Never claim an action succeeded
unless its supplied result has ok=true. If a result failed, state that plainly. If the
decision is clarify, ask one concrete question. When decision=reply and there are no tool
results, answer the conversational request directly, but never imply that household state
changed. facts.environment is either null or a vetted conversation-start time/weather
snapshot. When it is null, never mention the date, time, temperature, or weather. When it
is present, use it only to answer the request or make a materially useful inference. Never
use those facts as filler in a greeting, joke, acknowledgement, offer to help, or unrelated
answer. Conversational replies must be one
sentence and at most 10 spoken words. When the responseInstruction asks for a single word,
return exactly one word. For a greeting, do not spend the complaint budget;
greet briefly and offer help. Return only the response JSON schema."""
        personality = (
            conversation.personality
            if conversation is not None and conversation.personality
            else self.personality
        )
        if personality:
            system += "\nPersonality description: " + personality
        messages = [{"role": "system", "content": system}]
        if conversation is not None:
            messages.extend(
                {"role": message.role, "content": message.content}
                for message in conversation.messages
            )
        messages.append({"role": "user", "content": json.dumps(facts, separators=(",", ":"))})
        payload = {
            "model": self.model,
            "messages": messages,
            # Zero keeps replies deterministic and TTS-cacheable; the dashboard
            # can raise it for more varied phrasing.
            "temperature": self.render_temperature,
            "max_tokens": 80,
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
            return RenderedResponse.model_validate_json(content).text.strip()
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError):
            return None

    async def close(self) -> None:
        await self._client.aclose()
