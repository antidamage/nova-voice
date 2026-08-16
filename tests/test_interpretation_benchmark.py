"""The text-only interpretation benchmark isolates what it claims to measure."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
from conftest import interpretation

from nova_voice.api import create_app
from nova_voice.config import Settings
from nova_voice.interpretation.base import InterpretRequest


class _Interpreter:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def build_interpret_request(self, utterance, **kwargs) -> InterpretRequest:
        assert kwargs["tools"] == []
        assert kwargs["relevant_state"] == {}
        assert kwargs["active_goal"] is None
        assert kwargs["conversation"] is None
        return InterpretRequest(
            messages=[
                {"role": "system", "content": "benchmark rules"},
                {"role": "user", "content": utterance.transcript},
            ],
            system="benchmark rules",
            opening_context={
                "semanticTools": [],
                "relevantState": {},
                "selectedMemory": [],
            },
            turn_context={"utterance": {"transcript": utterance.transcript}},
        )

    async def interpret(self, utterance, **kwargs):
        self.calls.append({"utterance": utterance, **kwargs})
        return interpretation()


class _Router:
    def __init__(self) -> None:
        self.entries = [
            {
                "workload": "interpret",
                "at": "2026-08-16T22:00:00Z",
                "spoken": "local",
                "companion": {"text": "private", "elapsedMs": 1200.0},
                "local": {"text": "private", "elapsedMs": 600.0},
            }
        ]

    def comparisons(self) -> list[dict]:
        # The first call captures the previous entry. The second represents the
        # comparison produced by this benchmark sample.
        if not hasattr(self, "read_once"):
            self.read_once = True
            return []
        return self.entries


async def _post(settings: Settings, service) -> httpx.Response:
    app = create_app(settings, service=service)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="https://voice.test") as client:
        return await client.post(
            "/v1/test/interpretation",
            json={"transcript": "How are you feeling today", "room": "benchmark"},
        )


async def test_tool_free_benchmark_is_gated_by_the_test_harness_switch() -> None:
    service = SimpleNamespace(
        interpreter=_Interpreter(), companion_router=_Router(), companion_sessions=None
    )

    response = await _post(Settings(test_harness_enabled=False), service)

    assert response.status_code == 404


async def test_tool_free_benchmark_reports_arm_and_prompt_structure_without_answers() -> None:
    interpreter = _Interpreter()
    service = SimpleNamespace(
        interpreter=interpreter, companion_router=_Router(), companion_sessions=None
    )

    response = await _post(Settings(test_harness_enabled=True), service)

    assert response.status_code == 200
    payload = response.json()
    assert payload["armsMs"] == {"companion": 1200.0, "local": 600.0}
    assert payload["winner"] == "local"
    assert payload["input"]["toolsOffered"] == 0
    assert payload["input"]["stateFields"] == 0
    assert payload["input"]["memoryEntries"] == 0
    assert payload["input"]["historyMessages"] == 0
    assert payload["input"]["localPromptBytes"] > 0
    assert payload["input"]["companionPayloadBytes"] > 0
    assert "private" not in response.text
    assert interpreter.calls[0]["tools"] == []
    assert interpreter.calls[0]["relevant_state"] == {}
    assert interpreter.calls[0]["utterance"].dry_run is True
    assert payload["result"]["actions"] == 0
