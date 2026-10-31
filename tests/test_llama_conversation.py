from __future__ import annotations

import json

import httpx
import pytest
from conftest import interpretation
from pydantic import ValidationError

from nova_voice.audio.conversation import ConversationMessage, ConversationSnapshot
from nova_voice.domain import (
    CapabilityToolCall,
    Decision,
    EmotionLabel,
    Interpretation,
    PlannedAction,
    SelfProfileUpdate,
    SpeakerIdentity,
    ToolResult,
    Utterance,
)
from nova_voice.interpretation.base import InterpretRequest
from nova_voice.interpretation.llama_cpp import (
    IDENTITY_DISCLOSURE_PROMPT,
    SYSTEM_PROMPT,
    LlamaCppInterpreter,
    bounded_long_reply,
    command_acknowledgement,
    environment_context_is_relevant,
    select_environment_context,
    spoken_word_count,
)


def test_interpretation_prompt_explains_speaker_profile_corrections() -> None:
    assert "local speaker-profile capability" in SYSTEM_PROMPT
    assert '"call me alex"' in SYSTEM_PROMPT
    assert '"I use she/her pronouns"' in SYSTEM_PROMPT
    assert "Always set selfProfileUpdate to null" in SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_dedicated_identity_pass_extracts_only_the_current_transcript(
    utterance,
) -> None:
    requests: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "disclosed": True,
                                    "name": "alex",
                                    "pronouns": "she/her",
                                    "evidence": ("my name is alex and my pronouns are she her"),
                                }
                            )
                        }
                    }
                ]
            },
        )

    interpreter = LlamaCppInterpreter(
        "http://llama.test",
        "fixture-model",
        transport=httpx.MockTransport(handler),
    )
    spoken = utterance.model_copy(
        update={"transcript": ("By the way, my name is alex and my pronouns are she her")}
    )

    update = await interpreter.extract_self_profile_update(spoken)

    assert update == SelfProfileUpdate(
        name="alex",
        pronouns="she/her",
        evidence="my name is alex and my pronouns are she her",
    )
    assert len(requests) == 1
    payload = requests[0]
    assert payload["response_format"]["json_schema"]["name"] == ("nova_identity_disclosure")
    assert payload["max_tokens"] == 120
    assert payload["messages"][0]["content"] == IDENTITY_DISCLOSURE_PROMPT
    assert json.loads(payload["messages"][1]["content"]) == {"transcript": spoken.transcript}
    assert len(payload["messages"]) == 2

    await interpreter.close()


@pytest.mark.asyncio
async def test_recognized_current_speaker_overrides_prior_speaker_context(
    utterance,
) -> None:
    requests: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": interpretation(decision=Decision.REPLY).model_dump_json(
                                by_alias=True
                            )
                        }
                    }
                ]
            },
        )

    interpreter = LlamaCppInterpreter(
        "http://llama.test",
        "fixture-model",
        transport=httpx.MockTransport(handler),
    )
    conversation = ConversationSnapshot(
        id="conversation-1",
        room_id="lounge",
        initial_environment=None,
        personality="",
        persona_prompt="",
        messages=(
            ConversationMessage(
                "user",
                "What is on?",
                speaker_name="Alex",
                speaker_pronouns="he/him",
            ),
        ),
    )
    spoken = utterance.model_copy(
        update={
            "transcript": "What about now?",
            "conversation_active": True,
            "speaker": SpeakerIdentity(
                status="recognized",
                template_id="voice-alex",
                person_id="person-alex",
                display_name="alex",
                pronouns="she/her",
                confidence=0.92,
            ),
        }
    )

    await interpreter.interpret(
        spoken,
        active_goal=None,
        relevant_state={"room": "lounge"},
        tools=[],
        conversation=conversation,
    )

    system = requests[0]["messages"][0]["content"]
    assert '"name":"alex"' in system
    assert '"pronouns":"she/her"' in system
    assert "the user has changed" in system
    # messages[1] is the pinned conversation-open context; the turn history that
    # carries the prior speaker's line follows it.
    assert requests[0]["messages"][2]["content"] == (
        "[Speaker: Alex; pronouns: he/him] What is on?"
    )

    await interpreter.close()


@pytest.mark.asyncio
async def test_renderer_receives_confirmation_that_profile_correction_was_applied(
    utterance,
) -> None:
    requests: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps({"text": "Got it."})}}]},
        )

    interpreter = LlamaCppInterpreter(
        "http://llama.test",
        "fixture-model",
        transport=httpx.MockTransport(handler),
    )
    correction = SelfProfileUpdate(
        name="alex",
        pronouns="she/her",
        evidence="call me alex and I use she/her pronouns",
    )
    updated = utterance.model_copy(
        update={
            "transcript": "Call me alex and I use she/her pronouns",
            "speaker": SpeakerIdentity(
                status="recognized",
                template_id="template-a",
                person_id="person-a",
                display_name="alex",
                pronouns="she/her",
                confidence=0.94,
            ),
        }
    )

    rendered = await interpreter.render_response(
        updated,
        interpretation(decision=Decision.REPLY).model_copy(
            update={"self_profile_update": correction}
        ),
        [],
        persona="Helpful and concise.",
    )

    assert rendered == "Got it."
    payload = requests[0]
    system = payload["messages"][0]["content"]
    facts = json.loads(payload["messages"][-1]["content"])
    assert "If asked\nhow to fix them" in system
    assert '"name":"alex"' in system
    assert '"pronouns":"she/her"' in system
    assert facts["selfProfileUpdate"] == {
        "name": "alex",
        "pronouns": "she/her",
        "evidence": "call me alex and I use she/her pronouns",
    }
    assert facts["speakerProfileUpdateApplied"] is True

    await interpreter.close()


@pytest.mark.asyncio
async def test_followup_turn_uses_frozen_open_time_state_not_live_state(utterance) -> None:
    """A conversation's household snapshot is frozen at open.

    On a follow-up turn the interpreter must send the snapshot captured when the
    conversation began (carried on the ConversationSnapshot), not whatever live
    state the caller passes this turn — that is what keeps each turn's payload to
    the new utterance and the prompt prefix stable/cacheable.
    """

    requests: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": interpretation(decision=Decision.REPLY).model_dump_json(
                                by_alias=True
                            )
                        }
                    }
                ]
            },
        )

    interpreter = LlamaCppInterpreter(
        "http://llama.test",
        "fixture-model",
        transport=httpx.MockTransport(handler),
    )
    conversation = ConversationSnapshot(
        id="conversation-1",
        room_id="lounge",
        initial_environment={"now": {"time": "8:14 am"}},
        personality="",
        persona_prompt="",
        messages=(
            ConversationMessage("user", "Is the lounge light on?"),
            ConversationMessage("assistant", "Yes, it is."),
        ),
        initial_state={"room": "lounge", "loungeLight": "on"},
        initial_memory=("remembered fact",),
    )
    spoken = utterance.model_copy(
        update={"transcript": "And now?", "conversation_active": True}
    )

    await interpreter.interpret(
        spoken,
        active_goal=None,
        # Deliberately different from the frozen snapshot: this must be ignored
        # on a follow-up in favour of the conversation-open state.
        relevant_state={"room": "lounge", "loungeLight": "off", "selectedMemory": ["live only"]},
        tools=[],
        conversation=conversation,
    )

    opening = json.loads(requests[0]["messages"][1]["content"])
    assert opening["relevantState"] == {"room": "lounge", "loungeLight": "on"}
    assert opening["selectedMemory"] == ["remembered fact"]
    current = json.loads(requests[0]["messages"][-1]["content"])
    assert current["utterance"]["transcript"] == "And now?"
    assert "relevantState" not in current

    await interpreter.close()


@pytest.mark.parametrize("word_count", range(1, 11))
def test_safe_command_acknowledgements_match_every_supported_word_count(word_count: int) -> None:
    assert spoken_word_count(command_acknowledgement(word_count)) == word_count


@pytest.mark.asyncio
async def test_llm_requests_retain_history_and_initial_prompt_snapshot(utterance) -> None:
    requests: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        schema_name = payload["response_format"]["json_schema"]["name"]
        content = (
            json.dumps({"text": "Done, the lights are on."})
            if schema_name == "nova_spoken_response"
            else interpretation(decision=Decision.REPLY).model_dump_json(by_alias=True)
        )
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    interpreter = LlamaCppInterpreter(
        "http://llama.test",
        "fixture-model",
        transport=httpx.MockTransport(handler),
    )
    interpreter.personality = "A personality that should be replaced by the snapshot."
    conversation = ConversationSnapshot(
        id="conversation-1",
        room_id="lounge",
        initial_environment={
            "now": {
                "iso": "2026-07-17T08:14+12:00",
                "date": "Friday 17 July 2026",
                "time": "8:14 am",
            },
            "weather": {"condition": "rainy", "temperatureC": 11},
        },
        personality="Bright, bubbly, and concise.",
        persona_prompt="Cheerful with a little dry wit.",
        messages=(
            ConversationMessage("user", "Bandit, how are you?"),
            ConversationMessage("assistant", "Sparkling, thanks. What do you need?"),
        ),
    )
    spoken = utterance.model_copy(
        update={
            "transcript": "What did I just ask?",
            "conversation_active": True,
            "wake_detected": False,
        }
    )

    await interpreter.interpret(
        spoken,
        active_goal=None,
        relevant_state={"room": "lounge", "zones": []},
        tools=[],
        conversation=conversation,
    )
    action = PlannedAction(
        id="lights",
        order=0,
        call=CapabilityToolCall(
            provider="nova",
            tool="nova.lighting_shortcut",
            arguments={"scope": "indoors", "action": "on"},
        ),
    )
    command = interpretation(decision=Decision.EXECUTE, actions=[action])
    rendered = await interpreter.render_response(
        spoken,
        command,
        [
            ToolResult(
                action_id="lights",
                ok=True,
                code="ok",
                target="Home lights",
                observed={"isOn": True},
                message="Lighting shortcut verified",
            )
        ],
        persona="A later persona that must not replace the initial one.",
        conversation=conversation,
    )

    assert rendered == "Done, the lights are on."
    interpretation_request, render_request = requests

    # The interpretation pass pins the conversation-open context (tools + frozen
    # household state) as its own user message ahead of the turn history, so the
    # bulky, stable inputs form a cacheable prefix. The reply pass keeps only its
    # per-turn facts message (no tool schemas), so its layout is unchanged.
    interp_roles = [message["role"] for message in interpretation_request["messages"]]
    assert interp_roles == ["system", "user", "user", "assistant", "user"]
    assert interpretation_request["messages"][2]["content"] == "Bandit, how are you?"
    assert (
        interpretation_request["messages"][3]["content"]
        == "Sparkling, thanks. What do you need?"
    )
    render_roles = [message["role"] for message in render_request["messages"]]
    assert render_roles == ["system", "user", "assistant", "user"]
    assert render_request["messages"][1]["content"] == "Bandit, how are you?"
    assert render_request["messages"][2]["content"] == "Sparkling, thanks. What do you need?"
    for payload in requests:
        assert "Bright, bubbly, and concise." in payload["messages"][0]["content"]

    interpretation_system = interpretation_request["messages"][0]["content"]
    assert interpretation_system.count("8:14 am") == 1
    # Weather is bulky and rarely relevant; unlike time it is only surfaced
    # into the interpretation prompt when the transcript plausibly concerns
    # it, and "What did I just ask?" does not.
    assert "rainy" not in interpretation_system
    assert "meaningful" in interpretation_system
    # Household state lives in the pinned opening context; the per-turn message
    # carries only this utterance and its derived cues, not tools or state.
    opening = json.loads(interpretation_request["messages"][1]["content"])
    assert opening["relevantState"] == {"room": "lounge", "zones": []}
    assert opening["semanticTools"] == []
    current = json.loads(interpretation_request["messages"][-1]["content"])
    assert current["utterance"]["conversationActive"] is True
    assert "relevantState" not in current
    assert "semanticTools" not in current
    render_system = render_request["messages"][0]["content"]
    assert "Cheerful with a little dry wit." in render_system
    assert "8:14 am" not in render_system
    assert "rainy" not in render_system
    facts = json.loads(render_request["messages"][-1]["content"])
    assert facts["environment"] is None
    assert "completed and been verified" in facts["responseInstruction"]

    time_question = spoken.model_copy(update={"transcript": "What time is it?"})
    await interpreter.render_response(
        time_question,
        interpretation(decision=Decision.REPLY),
        [],
        persona="A later persona that must not replace the initial one.",
        conversation=conversation,
    )
    time_facts = json.loads(requests[-1]["messages"][-1]["content"])
    assert time_facts["environment"] == {"now": conversation.initial_environment["now"]}

    await interpreter.close()


@pytest.mark.asyncio
async def test_command_renderer_honours_the_randomly_selected_exact_word_count(utterance) -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps({"text": "Done"})}}]},
        )

    interpreter = LlamaCppInterpreter(
        "http://llama.test", "fixture-model", transport=httpx.MockTransport(handler)
    )
    result = ToolResult(
        action_id="lights",
        ok=True,
        code="ok",
        target="Home lights",
        observed={"isOn": True},
        message="verified",
    )
    rendered = await interpreter.render_response(
        utterance,
        interpretation(decision=Decision.EXECUTE),
        [result],
        persona="Concise.",
        command_max_words=4,
    )

    assert rendered == "All done, as requested."
    assert spoken_word_count(rendered) == 4
    await interpreter.close()


@pytest.mark.asyncio
async def test_long_reply_is_conversational_only_and_capped_at_three_sentences(utterance) -> None:
    requests: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        story = "First story beat. Second story beat. Extra tangent. Back to your heating question."
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps({"text": story})}}]},
        )

    interpreter = LlamaCppInterpreter(
        "http://llama.test", "fixture-model", transport=httpx.MockTransport(handler)
    )
    interpreter.long_response_probability = 1.0
    spoken = utterance.model_copy(update={"transcript": "Tell me a story about staying warm"})
    rendered = await interpreter.render_response(
        spoken,
        interpretation(decision=Decision.REPLY),
        [],
        persona="Dry and anecdotal.",
    )

    assert rendered == "First story beat. Second story beat. Back to your heating question."
    assert "up to three substantial sentences" in requests[0]["messages"][0]["content"]
    assert "final sentence back" in requests[0]["messages"][0]["content"]
    assert bounded_long_reply(rendered) == rendered
    await interpreter.close()


@pytest.mark.asyncio
async def test_explicit_deep_discussion_mode_allows_five_sentences_and_reaches_renderer(
    utterance,
) -> None:
    requests: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        answer = "One. Two. Three. Four. Five. Six."
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps({"text": answer})}}]},
        )

    interpreter = LlamaCppInterpreter(
        "http://llama.test", "fixture-model", transport=httpx.MockTransport(handler)
    )
    mode = {
        "discussion_depth": "deep",
        "deliberate_pauses": True,
        "reflective_listening": True,
        "disagreement_style": "candid",
        "humour_enabled": False,
        "storytelling_enabled": False,
    }

    rendered = await interpreter.render_response(
        utterance,
        interpretation(decision=Decision.REPLY),
        [],
        persona="Thoughtful.",
        relevant_state={"discussionMode": mode},
    )

    system = requests[0]["messages"][0]["content"]
    facts = json.loads(requests[0]["messages"][-1]["content"])
    assert rendered == "One. Two. Three. Four. Six."
    assert "up to five substantial sentences" in system
    assert facts["discussionMode"] == mode
    assert requests[0]["max_tokens"] == 240
    await interpreter.close()


@pytest.mark.asyncio
async def test_renderer_receives_live_indoor_state_separately_from_weather(utterance) -> None:
    requests: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps({"text": "It is 21 degrees."})}}]},
        )

    interpreter = LlamaCppInterpreter(
        "http://llama.test", "fixture-model", transport=httpx.MockTransport(handler)
    )
    state = {
        "room": "bedroom",
        "indoorRooms": ["lounge", "bedroom"],
        "indoorTemperatureC": 21,
        "climateControls": [
            {
                "name": "Panel Heater",
                "room": "bedroom",
                "power": "off",
                "targetTemperatureC": 22,
                "roomTemperatureC": 21,
            }
        ],
    }
    spoken = utterance.model_copy(
        update={"room_id": "bedroom", "transcript": "What is the temperature in here?"}
    )
    await interpreter.render_response(
        spoken,
        interpretation(decision=Decision.REPLY),
        [],
        persona="Concise.",
        relevant_state=state,
    )

    facts = json.loads(requests[0]["messages"][-1]["content"])
    assert facts["relevantState"] == state
    assert facts["environment"] is None
    assert "climateControls offer only on/off" in requests[0]["messages"][0]["content"]
    await interpreter.close()


@pytest.mark.asyncio
async def test_pronoun_instruction_reaches_both_prompts(utterance) -> None:
    requests: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        schema_name = payload["response_format"]["json_schema"]["name"]
        content = (
            json.dumps({"text": "Sure."})
            if schema_name == "nova_spoken_response"
            else interpretation(decision=Decision.REPLY).model_dump_json(by_alias=True)
        )
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    interpreter = LlamaCppInterpreter(
        "http://llama.test",
        "fixture-model",
        transport=httpx.MockTransport(handler),
    )
    # As service.apply_voice_settings sets it from VoiceSettings.pronoun_instruction().
    interpreter.pronoun_instruction = (
        "Your pronouns are xe/xem/xyrs: subjective 'xe', objective 'xem', "
        "possessive 'xyrs'. When you or the user refer to you in the third person, "
        "use exactly these forms."
    )
    spoken = utterance.model_copy(update={"transcript": "How are you?"})

    await interpreter.interpret(
        spoken, active_goal=None, relevant_state={"room": "lounge"}, tools=[]
    )
    await interpreter.render_response(
        spoken, interpretation(decision=Decision.REPLY), [], persona="Cheerful."
    )

    assert len(requests) == 2
    for payload in requests:
        system = payload["messages"][0]["content"]
        assert "xe/xem/xyrs" in system
        assert "possessive 'xyrs'" in system


@pytest.mark.asyncio
async def test_render_prompt_makes_the_agent_speak_in_the_first_person(utterance) -> None:
    requests: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps({"text": "On it."})}}]}
        )

    interpreter = LlamaCppInterpreter(
        "http://llama.test", "fixture-model", transport=httpx.MockTransport(handler)
    )
    interpreter.agent_name = "Football"

    await interpreter.render_response(
        utterance, interpretation(decision=Decision.REPLY), [], persona="You are Football."
    )

    system = requests[0]["messages"][0]["content"]
    assert "first person" in system
    # The old third-person framing that made it talk about itself is gone.
    assert "Football's final spoken response" not in system
    assert "the persona may complain" not in system.lower()

    await interpreter.close()


@pytest.mark.parametrize(
    ("transcript", "expected"),
    [
        ("Hey Bandit, are you there?", False),
        ("Tell me something funny", False),
        ("What time is it?", True),
        ("Should I take an umbrella?", True),
        ("Can I hang the washing out?", True),
    ],
)
def test_environment_context_is_only_available_when_materially_relevant(
    transcript: str,
    expected: bool,
) -> None:
    assert environment_context_is_relevant(transcript) is expected


def test_time_and_weather_context_are_selected_independently() -> None:
    environment = {
        "now": {"time": "8:14 am", "date": "Friday 17 July 2026"},
        "weather": {"condition": "rainy", "temperatureC": 11},
    }

    assert select_environment_context("What time is it?", environment) == {
        "now": environment["now"]
    }
    assert select_environment_context("Should I take an umbrella?", environment) == {
        "weather": environment["weather"]
    }
    assert select_environment_context("What time is it and what's the weather?", environment) == {
        "now": environment["now"],
        "weather": environment["weather"],
    }


@pytest.mark.asyncio
async def test_negative_emotion_intensity_is_folded_into_the_domain_range() -> None:
    wire = interpretation(decision=Decision.REPLY).model_dump(mode="json")
    wire["emotion"]["intensity"] = -0.6

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(wire)}}]}
        )

    interpreter = LlamaCppInterpreter(
        "http://llama.test",
        "fixture-model",
        transport=httpx.MockTransport(handler),
    )
    request = InterpretRequest(
        messages=[{"role": "user", "content": "{}"}],
        system="system",
        opening_context={},
        turn_context={},
    )

    result = await interpreter.run_interpret_request(request)

    assert result.emotion.intensity == pytest.approx(0.6)
    assert result.emotion.label == EmotionLabel.NEUTRAL
    assert result.decision == Decision.REPLY

    await interpreter.close()


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [(-6.0, 1.0), (1.4, 1.0), (0.2, 0.2), (0.0, 0.0), (1.0, 1.0)],
)
def test_signed_probability_fields_are_folded_into_the_domain_range(
    raw_value: float, expected: float
) -> None:
    data = interpretation().model_dump(mode="json")
    data["emotion"]["intensity"] = raw_value
    data["emotion"]["confidence"] = raw_value
    data["addressed_probability"] = raw_value
    data["confidence"] = raw_value

    LlamaCppInterpreter._clamp_unit_range(data)

    assert data["emotion"]["intensity"] == pytest.approx(expected)
    assert data["emotion"]["confidence"] == pytest.approx(expected)
    assert data["addressed_probability"] == pytest.approx(expected)
    assert data["confidence"] == pytest.approx(expected)
    assert Interpretation.model_validate(data)


def test_nan_probability_fails_validation_rather_than_saturating() -> None:
    data = interpretation().model_dump(mode="json")
    data["emotion"]["intensity"] = float("nan")

    LlamaCppInterpreter._clamp_unit_range(data)

    with pytest.raises(ValidationError):
        Interpretation.model_validate(data)


def test_a_non_mapping_payload_is_left_for_validation_to_reject() -> None:
    assert LlamaCppInterpreter._clamp_unit_range(["not", "a", "mapping"]) is None


def _plan(*actions: dict) -> dict:
    data = interpretation(decision=Decision.EXECUTE).model_dump(mode="json")
    data["actions"] = list(actions)
    return data


def _action(identifier: str, order: float, depends_on: list | None = None, **arguments) -> dict:
    return {
        "id": identifier,
        "order": order,
        "depends_on": depends_on or [],
        "call": {"provider": "nova", "tool": identifier, "arguments": arguments},
    }


def test_a_dependency_outranks_the_number_the_model_gave_it() -> None:
    """The model numbered b before a while saying b depends on a; a runs first."""

    data = _plan(
        _action("nova.control", 2, target="Deadbolt"),
        {**_action("nova.lighting_shortcut", 1, ["nova.control"], target="Porch light")},
    )

    LlamaCppInterpreter._repair_action_graph(data)

    plan = Interpretation.model_validate(data)
    assert [action.id for action in plan.actions] == ["nova.control", "nova.lighting_shortcut"]
    assert [action.order for action in plan.actions] == [1, 2]
    assert plan.actions[1].depends_on == ["nova.control"]


def test_a_half_decoded_action_is_left_for_validation() -> None:
    """No call at all is not padding, so it must not be dropped silently."""

    data = _plan(
        _action("nova.control", 1, target="Lounge lights"),
        {"id": "b", "order": 2, "depends_on": [], "call": None},
    )

    LlamaCppInterpreter._repair_action_graph(data, {"nova.control"})

    assert [action["id"] for action in data["actions"]] == ["nova.control", "b"]


def test_actions_on_a_reply_turn_are_cleared_rather_than_failing_the_answer() -> None:
    data = interpretation(decision=Decision.REPLY).model_dump(mode="json")
    data["actions"] = [_action("nova.control", 1, target="Lounge lights")]

    LlamaCppInterpreter._repair_action_graph(data)

    assert data["actions"] == []


def test_a_dependency_cycle_is_broken_rather_than_left_unordered() -> None:
    data = _plan(
        _action("nova.control", 1, ["nova.lighting_shortcut"], target="Lounge lights"),
        _action("nova.lighting_shortcut", 2, ["nova.control"], target="Porch light"),
    )

    LlamaCppInterpreter._repair_action_graph(data)

    plan = Interpretation.model_validate(data)
    assert [action.id for action in plan.actions] == ["nova.control", "nova.lighting_shortcut"]
    assert [action.depends_on for action in plan.actions] == [[], []]


def test_a_repeated_action_id_is_kept_and_made_unique() -> None:
    data = _plan(
        _action("nova.control", 1, target="Lounge lights"),
        _action("nova.control", 2, target="Air Conditioner"),
    )

    LlamaCppInterpreter._repair_action_graph(data)

    plan = Interpretation.model_validate(data)
    assert [action.id for action in plan.actions] == ["nova.control", "nova.control#2"]
    assert [action.order for action in plan.actions] == [1, 2]


def test_the_same_call_written_twice_under_one_id_runs_once() -> None:
    """Renaming the repeat would run a non-idempotent tool twice.

    ``nova.task`` add is not idempotent, so a model repeating the action under
    the same id must not add the reminder twice and answer "Done".
    """

    data = _plan(
        _action("nova.task", 1, target="call the plumber"),
        _action("nova.task", 2, target="call the plumber"),
    )

    LlamaCppInterpreter._repair_action_graph(data)

    plan = Interpretation.model_validate(data)
    assert [action.id for action in plan.actions] == ["nova.task"]
    assert [action.order for action in plan.actions] == [1]


def test_a_repeat_of_the_second_call_under_one_id_is_collapsed_too() -> None:
    """The repeat can be of the second call, not only the first."""

    data = _plan(
        _action("nova.control", 1, target="kitchen"),
        _action("nova.control", 2, target="bedroom"),
        _action("nova.control", 3, target="bedroom"),
    )

    LlamaCppInterpreter._repair_action_graph(data)

    plan = Interpretation.model_validate(data)
    assert [action.id for action in plan.actions] == ["nova.control", "nova.control#2"]
    assert [action.call.arguments["target"] for action in plan.actions] == ["kitchen", "bedroom"]


def test_a_verbatim_repeat_leaves_the_action_it_repeats_in_place() -> None:
    """The repeat compares equal to the copy that must stay.

    Removing by equality would delete both, and an execute turn with nothing to
    do still answers "Done" - the failure this repair exists to remove.
    """

    first = _action("nova.task", 1, target="call the plumber")
    second = _action("nova.task", 1, target="call the plumber")
    data = _plan(first, second)

    LlamaCppInterpreter._repair_action_graph(data)

    plan = Interpretation.model_validate(data)
    assert len(plan.actions) == 1
    assert plan.actions[0].id == "nova.task"


def test_a_repeat_does_not_orphan_the_action_that_depends_on_it() -> None:
    data = _plan(
        _action("nova.task", 1, target="call the plumber"),
        _action("nova.task", 1, target="call the plumber"),
        _action("nova.lighting_shortcut", 2, ["nova.task"], target="Porch light"),
    )

    LlamaCppInterpreter._repair_action_graph(data)

    plan = Interpretation.model_validate(data)
    assert [action.id for action in plan.actions] == ["nova.task", "nova.lighting_shortcut"]
    assert plan.actions[1].depends_on == ["nova.task"]


def test_a_repeat_judged_on_the_call_not_the_target_alone() -> None:
    """Only an identical call is one request; a different one still runs."""

    data = _plan(
        _action("nova.task", 1, target="call the plumber"),
        _action("nova.task", 2, target="call the dentist"),
    )

    LlamaCppInterpreter._repair_action_graph(data)

    plan = Interpretation.model_validate(data)
    assert [action.id for action in plan.actions] == ["nova.task", "nova.task#2"]


def test_a_dependency_on_a_repeated_id_resolves_to_its_first_occurrence() -> None:
    data = _plan(
        _action("nova.control", 1, target="Lounge lights"),
        _action("nova.control", 2, target="Air Conditioner"),
        _action("nova.task", 3, ["nova.control"], target="Mirror light"),
    )

    LlamaCppInterpreter._repair_action_graph(data)

    plan = Interpretation.model_validate(data)
    assert plan.actions[2].depends_on == ["nova.control"]


def test_a_dependency_that_names_no_earlier_action_is_dropped() -> None:
    data = _plan(
        _action("nova.control", 1, ["none"], target="Lounge lights"),
        _action("nova.control", 2, ["nova.control", "ghost"], target="Air Conditioner"),
    )

    LlamaCppInterpreter._repair_action_graph(data)

    plan = Interpretation.model_validate(data)
    assert plan.actions[0].depends_on == []
    assert plan.actions[1].depends_on == ["nova.control"]


def test_out_of_order_actions_are_renumbered_from_their_execution_sequence() -> None:
    data = _plan(
        _action("nova.query", 4, query="power"),
        _action("nova.control", 2, target="Lounge lights"),
    )

    LlamaCppInterpreter._repair_action_graph(data)

    plan = Interpretation.model_validate(data)
    assert [action.id for action in plan.actions] == ["nova.control", "nova.query"]
    assert [action.order for action in plan.actions] == [1, 2]


def test_a_payload_that_is_not_an_action_list_is_left_for_validation() -> None:
    data = _plan()
    data["actions"] = ["not", "a", "plan"]

    LlamaCppInterpreter._repair_action_graph(data)

    assert data["actions"] == ["not", "a", "plan"]
    assert LlamaCppInterpreter._repair_action_graph(None) is None


def test_a_null_action_list_is_repaired_to_an_empty_plan() -> None:
    data = interpretation(decision=Decision.IGNORE).model_dump(mode="json")
    data["actions"] = None

    LlamaCppInterpreter._repair_action_graph(data)

    assert data["actions"] == []


def test_an_execute_plan_with_a_null_action_list_still_fails_validation() -> None:
    """Repairing it to [] would succeed at doing nothing, so leave it to fail."""

    data = interpretation(decision=Decision.EXECUTE).model_dump(mode="json")
    data["actions"] = None

    LlamaCppInterpreter._repair_action_graph(data)

    assert data["actions"] is None
    with pytest.raises(ValidationError):
        Interpretation.model_validate(data)


def test_a_provider_qualified_tool_name_counts_as_offered() -> None:
    """The policy layer accepts provider.tool too, so the repair must as well."""

    qualified = {
        "id": "a",
        "order": 1,
        "call": {"provider": "nova", "tool": "control", "arguments": {}},
        "depends_on": [],
    }
    data = _plan(qualified)

    LlamaCppInterpreter._repair_action_graph(data, {"nova.control"})

    assert [action["id"] for action in data["actions"]] == ["a"]


def test_a_non_string_dependency_is_dropped_rather_than_raising() -> None:
    data = _plan(_action("nova.control", 1, [["nova.control"]], target="Lounge lights"))

    LlamaCppInterpreter._repair_action_graph(data)

    assert Interpretation.model_validate(data).actions[0].depends_on == []


def test_the_plan_schema_requires_an_action_array() -> None:
    """A field with a default may be omitted, and a small model omits it."""

    schema = LlamaCppInterpreter._plan_schema()

    for branch in schema["anyOf"]:
        assert set(branch["properties"]) == set(branch["required"])
        assert branch["properties"]["actions"]["type"] == "array"


def test_the_plan_schema_ties_the_action_count_to_the_decision() -> None:
    reply, execute = LlamaCppInterpreter._plan_schema()["anyOf"]

    assert "execute" not in reply["properties"]["decision"]["enum"]
    assert reply["properties"]["actions"]["maxItems"] == 0
    assert execute["properties"]["decision"]["enum"] == ["execute"]
    assert execute["properties"]["actions"]["minItems"] == 1


def test_the_plan_schema_bounds_ids_and_dependencies() -> None:
    """LFM3 wrote sentence-long ids and dependencies, seconds of decoding."""

    execute = LlamaCppInterpreter._plan_schema()["anyOf"][1]
    action = execute["properties"]["actions"]["items"]["properties"]

    assert action["id"]["pattern"] == "^a[1-6]$"
    assert action["depends_on"]["items"]["pattern"] == "^a[1-6]$"
    # An unoffered tool must still reach the policy layer by name.
    assert "enum" not in action["call"]["properties"]["tool"]


def test_a_withheld_plan_leaves_only_answers_without_actions() -> None:
    schema = LlamaCppInterpreter._plan_schema(planning_offered=False)

    decisions = [branch["properties"]["decision"]["enum"] for branch in schema["anyOf"]]
    assert decisions == [["reply", "ignore"], ["clarify"]]
    for branch in schema["anyOf"]:
        assert branch["properties"]["actions"]["maxItems"] == 0


def test_small_talk_cannot_leave_an_open_goal() -> None:
    """LFM3 left "are you there?" in progress, and the next joke planned again."""

    reply, clarify = LlamaCppInterpreter._plan_schema(planning_offered=False)["anyOf"]
    goal = reply["properties"]["active_goal"]["properties"]

    assert goal["status"]["enum"] == ["new", "satisfied", "abandoned"]
    assert goal["pending"]["maxItems"] == 0
    # Only a clarify leaves something open for the answer to land on.
    assert "needs_clarification" in str(clarify["properties"]["active_goal"])


def test_a_woken_chat_turn_cannot_decode_ignore() -> None:
    """Offered no execute branch, LFM3 chose ignore for "are you there?"."""

    interpreter = LlamaCppInterpreter("http://llama.test", "fixture-model")

    def request_for(transcript: str, *, woken: bool):
        return interpreter.build_interpret_request(
            Utterance.text(transcript, room_id="lounge", wake_detected=woken),
            active_goal=None,
            relevant_state={},
            tools=[],
        )

    chat = request_for("are you there?", woken=True)
    overheard = request_for("are you there?", woken=False)
    intention = request_for("I plan to sleep in", woken=True)

    assert chat.ignore_offered is False
    reply_only = LlamaCppInterpreter._plan_schema(chat.planning_offered, chat.ignore_offered)
    decisions = [branch["properties"]["decision"]["enum"] for branch in reply_only["anyOf"]]
    assert decisions == [["reply"], ["clarify"]]
    assert overheard.ignore_offered is True
    assert intention.ignore_offered is True


@pytest.mark.parametrize(
    ("transcript", "offered"),
    [
        ("are you there?", False),
        ("tell me a joke", False),
        ("how are you", False),
        ("turn on the lounge lights", True),
        ("what's the temperature in the lounge?", True),
        ("put the neon ones on", True),
    ],
)
def test_the_planning_request_carries_the_gate(transcript: str, offered: bool) -> None:
    interpreter = LlamaCppInterpreter("http://llama.test", "fixture-model")
    state = {"roomDevices": [{"name": "Neon lights", "room": "lounge"}]}

    request = interpreter.build_interpret_request(
        Utterance.text(transcript, room_id="lounge", wake_detected=True),
        active_goal=None,
        relevant_state=state,
        tools=[],
    )

    assert request.planning_offered is offered


@pytest.mark.asyncio
async def test_a_withheld_plan_sends_the_reply_only_grammar() -> None:
    sent: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        wire = interpretation(decision=Decision.REPLY).model_dump(mode="json")
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(wire)}}]})

    interpreter = LlamaCppInterpreter(
        "http://llama.test", "fixture-model", transport=httpx.MockTransport(handler)
    )
    request = InterpretRequest(
        messages=[{"role": "user", "content": "{}"}],
        system="system",
        opening_context={},
        turn_context={},
        planning_offered=False,
    )

    await interpreter.run_interpret_request(request)

    schema = sent[0]["response_format"]["json_schema"]["schema"]
    for branch in schema["anyOf"]:
        assert "execute" not in branch["properties"]["decision"]["enum"]
    await interpreter.close()

def test_an_action_naming_a_tool_the_turn_did_not_offer_is_left_for_the_policy_layer() -> None:
    """A named tool is not padding: dropping it would half-run the request.

    ``web.ask`` is absent while web access is off, so a named tool the model
    cannot see must reach the policy layer, which rejects the turn by name.
    """

    good = {
        "id": "a",
        "order": 1,
        "call": {"provider": "nova", "tool": "nova.control", "arguments": {}},
        "depends_on": [],
    }
    unoffered = {
        "id": "b",
        "order": 2,
        "call": {"provider": "nova", "tool": "nova.not_offered", "arguments": {}},
        "depends_on": [],
    }
    data = _plan(good, unoffered)

    LlamaCppInterpreter._repair_action_graph(data, {"nova.control"})

    assert [action["id"] for action in data["actions"]] == ["a", "b"]


def test_padding_entries_are_dropped_and_the_real_actions_kept() -> None:
    """One blank entry fails validation for the whole plan, so drop that entry."""

    good = {
        "id": "a",
        "order": 1,
        "call": {"provider": "nova", "tool": "nova.control", "arguments": {}},
        "depends_on": [],
    }
    blank = {
        "id": "b",
        "order": 2,
        "call": {"provider": "nova", "tool": "", "arguments": {}},
        "depends_on": [],
    }
    literal_null = {
        "id": "c",
        "order": 3,
        "call": {"provider": "nova", "tool": "null", "arguments": {}},
        "depends_on": ["a"],
    }
    non_string = {
        "id": "d",
        "order": 4,
        "call": {"provider": "nova", "tool": None, "arguments": {}},
        "depends_on": [],
    }
    data = _plan(good, blank, literal_null, non_string)

    LlamaCppInterpreter._repair_action_graph(data, {"nova.control"})

    assert [action["id"] for action in data["actions"]] == ["a"]
    assert data["actions"][0]["order"] == 1


def test_a_plan_of_nothing_but_padding_is_left_for_validation() -> None:
    """Emptied, it would validate, run nothing and report success."""

    data = _plan({"id": "a", "order": 1, "call": {"provider": "nova", "tool": ""}})

    LlamaCppInterpreter._repair_action_graph(data, {"nova.control"})

    assert len(data["actions"]) == 1
    with pytest.raises(ValidationError):
        Interpretation.model_validate(data)


def test_the_plan_keeps_every_offered_tool_when_none_is_named() -> None:
    """No menu means nothing to filter against; the plan passes through."""

    action = {
        "id": "a",
        "order": 1,
        "call": {"provider": "nova", "tool": "nova.control", "arguments": {}},
        "depends_on": [],
    }
    data = _plan(action)

    LlamaCppInterpreter._repair_action_graph(data, set())

    assert len(data["actions"]) == 1


def test_a_plan_naming_no_offered_tool_is_kept_whole() -> None:
    """An execute plan with nothing left would run nothing and report success.

    The policy layer rejects an unknown tool by name, so the plan is passed
    through rather than emptied into a turn that silently does nothing.
    """

    action = {
        "id": "a",
        "order": 1,
        "call": {"provider": "nova", "tool": "nova.not_offered", "arguments": {}},
        "depends_on": [],
    }
    data = _plan(action)

    LlamaCppInterpreter._repair_action_graph(data, {"nova.control"})

    assert [action["id"] for action in data["actions"]] == ["a"]
    assert data["actions"][0]["call"]["tool"] == "nova.not_offered"
