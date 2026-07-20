from __future__ import annotations

import json

import httpx
import pytest
from conftest import interpretation

from nova_voice.audio.conversation import ConversationMessage, ConversationSnapshot
from nova_voice.domain import (
    CapabilityToolCall,
    Decision,
    PlannedAction,
    SelfProfileUpdate,
    SpeakerIdentity,
    ToolResult,
)
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
    assert '"call me Addie"' in SYSTEM_PROMPT
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
                                    "name": "Adeline",
                                    "pronouns": "she/her",
                                    "evidence": (
                                        "my name is Adeline and my pronouns are she her"
                                    ),
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
        update={
            "transcript": (
                "By the way, my name is Adeline and my pronouns are she her"
            )
        }
    )

    update = await interpreter.extract_self_profile_update(spoken)

    assert update == SelfProfileUpdate(
        name="Adeline",
        pronouns="she/her",
        evidence="my name is Adeline and my pronouns are she her",
    )
    assert len(requests) == 1
    payload = requests[0]
    assert payload["response_format"]["json_schema"]["name"] == (
        "nova_identity_disclosure"
    )
    assert payload["max_tokens"] == 120
    assert payload["messages"][0]["content"] == IDENTITY_DISCLOSURE_PROMPT
    assert json.loads(payload["messages"][1]["content"]) == {
        "transcript": spoken.transcript
    }
    assert len(payload["messages"]) == 2

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
        name="Adeline",
        pronouns="she/her",
        evidence="call me Adeline and I use she/her pronouns",
    )
    updated = utterance.model_copy(
        update={
            "transcript": "Call me Adeline and I use she/her pronouns",
            "speaker": SpeakerIdentity(
                status="recognized",
                template_id="template-a",
                person_id="person-a",
                display_name="Adeline",
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
    assert facts["selfProfileUpdate"] == {
        "name": "Adeline",
        "pronouns": "she/her",
        "evidence": "call me Adeline and I use she/her pronouns",
    }
    assert facts["speakerProfileUpdateApplied"] is True

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
    for payload in requests:
        roles = [message["role"] for message in payload["messages"]]
        assert roles == ["system", "user", "assistant", "user"]
        assert payload["messages"][1]["content"] == "Bandit, how are you?"
        assert payload["messages"][2]["content"] == "Sparkling, thanks. What do you need?"
        system = payload["messages"][0]["content"]
        assert "Bright, bubbly, and concise." in system
    interpretation_system = interpretation_request["messages"][0]["content"]
    assert interpretation_system.count("8:14 am") == 1
    assert "rainy" in interpretation_system
    assert "meaningful" in interpretation_system
    current = json.loads(interpretation_request["messages"][-1]["content"])
    assert current["utterance"]["conversationActive"] is True
    assert current["relevantState"] == {"room": "lounge", "zones": []}
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
