from __future__ import annotations

import json

import httpx
import pytest
from conftest import interpretation

from nova_voice.audio.conversation import ConversationMessage, ConversationSnapshot
from nova_voice.domain import CapabilityToolCall, Decision, PlannedAction, ToolResult
from nova_voice.interpretation.llama_cpp import (
    LlamaCppInterpreter,
    environment_context_is_relevant,
    select_environment_context,
)


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
