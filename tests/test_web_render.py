from __future__ import annotations

import json

import httpx
import pytest
from conftest import interpretation

from nova_voice.domain import CapabilityToolCall, Decision, PlannedAction, ToolResult
from nova_voice.interpretation.llama_cpp import LlamaCppInterpreter


def _web_interpretation():
    action = PlannedAction(
        id="w1",
        order=0,
        call=CapabilityToolCall(provider="web", tool="web.ask", arguments={"query": "who won"}),
    )
    return interpretation(decision=Decision.EXECUTE, actions=[action])


@pytest.mark.asyncio
async def test_render_relays_web_answer_within_budget(utterance) -> None:
    captured: dict = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        captured.update(payload)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": json.dumps({"text": "The Crusaders won last night."})}}
                ]
            },
        )

    interpreter = LlamaCppInterpreter(
        "http://llama.test", "fixture-model", transport=httpx.MockTransport(handler)
    )
    interpreter.web_answer_max_sentences = 2
    web_result = ToolResult(
        action_id="w1",
        ok=True,
        code="ok",
        target="web",
        observed={
            "backend": "gemini",
            "answer": "The Crusaders won last night.",
            "sources": ["rnz.co.nz"],
        },
        message="Web answer retrieved",
    )

    rendered = await interpreter.render_response(
        utterance,
        _web_interpretation(),
        [web_result],
        persona="fixture persona",
    )
    await interpreter.close()

    assert rendered == "The Crusaders won last night."
    system = captured["messages"][0]["content"]
    facts = captured["messages"][-1]["content"]
    # The web relay branch was taken (not the command-acknowledgement branch):
    # the relay instruction rides in facts.responseInstruction, and the system
    # prompt carries the web sentence budget rather than the terse default.
    assert "A web lookup answered" in facts
    assert "at most 2 natural spoken sentences" in system
    assert "at most 10 spoken words" not in system
    # Token budget is sized to the sentence budget, not the terse 80-token cap.
    assert captured["max_tokens"] == 50 * 2 + 60
