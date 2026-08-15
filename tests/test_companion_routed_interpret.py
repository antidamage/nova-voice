"""Routing the planning pass — the one that decides what happens in the house.

`interpret` is the only routed workload whose failure mode is a *wrong action*
rather than a slower or blander sentence, so schema validation is not the whole
check here. A shape-valid plan can still name a tool that does not exist; the
action would then be planned, spoken about, and only refused at execution —
after the assistant had already said it was doing it.

So every action is checked against the exact catalogue the turn offered. The
load-bearing assertion is that **one bad action discards the whole plan**: a
plan with a clause removed is not the plan the model made, and "turn the lights
off and lock the door" minus one clause still reads as success.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from nova_voice.companion.routed import RoutedInterpreter
from nova_voice.companion.router import CompanionWorkloadRouter
from nova_voice.companion.session import CompanionSessionManager
from nova_voice.domain import Interpretation, Utterance
from nova_voice.interpretation.base import InterpretRequest, Interpreter

pytestmark = pytest.mark.asyncio

TOOLS = [
    {"type": "function", "function": {"name": "nova.set_light", "parameters": {}}},
    {"type": "function", "function": {"name": "nova.set_climate", "parameters": {}}},
]


def _utterance(text: str = "turn the kitchen light off") -> Utterance:
    now = datetime.now(UTC)
    return Utterance(
        id="utterance-1",
        satellite_id="test",
        room_id="kitchen",
        started_at=now,
        ended_at=now,
        transcript=text,
        wake_detected=True,
    )


def _wire(*tool_names: str) -> dict:
    """A companion's answer, in the shape it actually arrives in."""

    return {
        "emotion": {"label": "neutral", "confidence": 0.8, "intensity": 0.1, "evidence": []},
        "speech_act": "directive",
        "addressed_probability": 0.98,
        "decision": "execute",
        "confidence": 0.9,
        "active_goal": {"summary": "lights off", "status": "in_progress", "pending": []},
        "actions": [
            {
                "id": f"companion-{index}",
                "order": index,
                "depends_on": [],
                "call": {
                    "provider": name.split(".")[0],
                    "tool": name.split(".", 1)[1],
                    "arguments": {},
                },
            }
            for index, name in enumerate(tool_names)
        ],
        "response_plan": {
            "acknowledgement_style": "concise",
            "pre_action_speech": None,
            "requires_post_tool_rendering": True,
        },
        "self_profile_update": None,
    }


class _Backend(Interpreter):
    def __init__(self) -> None:
        self.local_runs = 0
        self.builds = 0

    def build_interpret_request(self, utterance, **_kwargs):
        self.builds += 1
        return InterpretRequest(
            messages=[{"role": "system", "content": "plan carefully"}],
            system="plan carefully",
            opening_context={"semanticTools": TOOLS, "relevantState": {}, "selectedMemory": []},
            turn_context={"utterance": {"transcript": utterance.transcript}},
        )

    async def run_interpret_request(self, request):
        self.local_runs += 1
        return Interpretation.model_validate(_wire("nova.set_light"))

    async def interpret(self, utterance, **_kwargs):  # pragma: no cover - routed path only
        raise AssertionError("the routed path should not call interpret()")


class _Companion:
    def __init__(self, result: object) -> None:
        self.result = result
        self.payloads: list[dict] = []

    def install(self, router: CompanionWorkloadRouter) -> None:
        async def run(workload, payload, local, *, parse=None, **_kwargs):
            self.payloads.append(payload)
            value = parse(self.result) if parse is not None else self.result
            if value is None:
                return SimpleNamespace(
                    source="local", value=await local(), reason="invalid result", elapsed_ms=0.0
                )
            return SimpleNamespace(
                source="companion", value=value, reason="accepted", elapsed_ms=1.0
            )

        router.run = run  # type: ignore[method-assign]


def _routed(backend: _Backend, companion: _Companion) -> RoutedInterpreter:
    router = CompanionWorkloadRouter(CompanionSessionManager(), enabled=True)
    companion.install(router)
    return RoutedInterpreter(backend, router)


async def _interpret(interpreter: RoutedInterpreter) -> Interpretation:
    return await interpreter.interpret(
        _utterance(), active_goal=None, relevant_state={}, tools=TOOLS
    )


async def test_a_valid_plan_from_the_companion_is_used() -> None:
    backend = _Backend()
    result = await _interpret(_routed(backend, _Companion(_wire("nova.set_light"))))

    assert result.decision == "execute"
    assert [action.call.tool for action in result.actions] == ["set_light"]
    assert backend.local_runs == 0


async def test_a_plan_naming_a_tool_that_was_not_offered_is_discarded() -> None:
    """Shape-valid and still unusable.

    `nova.unlock_door` parses perfectly; it simply was not in the catalogue
    this turn sent. Executing it would fail, but only after the assistant had
    announced it.
    """

    backend = _Backend()
    companion = _Companion(_wire("nova.unlock_door"))

    result = await _interpret(_routed(backend, companion))

    assert backend.local_runs == 1
    assert [action.call.tool for action in result.actions] == ["set_light"]


async def test_one_bad_action_discards_the_whole_plan() -> None:
    """Not just the offending clause.

    A plan with a clause silently removed is not the plan the model made, and
    the reply pass would still describe the turn as done.
    """

    backend = _Backend()
    companion = _Companion(_wire("nova.set_light", "nova.unlock_door"))

    result = await _interpret(_routed(backend, companion))

    assert backend.local_runs == 1
    # The local plan, whole — not the companion's first action kept.
    assert len(result.actions) == 1
    assert result.actions[0].call.tool == "set_light"


async def test_a_conversational_plan_needs_no_tools() -> None:
    """The common turn: no actions, so nothing to validate against."""

    backend = _Backend()
    plan = _wire()
    plan["decision"] = "reply"
    plan["active_goal"]["summary"] = "chat"

    result = await _interpret(_routed(backend, _Companion(plan)))

    assert result.decision == "reply"
    assert result.actions == []
    assert backend.local_runs == 0


async def test_a_malformed_plan_falls_back_locally() -> None:
    backend = _Backend()
    result = await _interpret(_routed(backend, _Companion({"decision": "execute"})))

    assert backend.local_runs == 1
    assert result.decision == "execute"


async def test_the_request_is_built_once_for_both_paths() -> None:
    backend = _Backend()
    await _interpret(_routed(backend, _Companion({"nonsense": True})))

    assert backend.builds == 1
    assert backend.local_runs == 1


async def test_the_turn_and_its_tools_are_offered_to_the_companion() -> None:
    """A planner cannot call a tool it was not shown, so the catalogue travels
    with the job — and it is the same catalogue the answer is checked against."""

    backend = _Backend()
    companion = _Companion(_wire())

    await _interpret(_routed(backend, companion))

    payload = companion.payloads[0]
    assert payload["turn"]["utterance"]["transcript"] == "turn the kitchen light off"
    names = [tool["function"]["name"] for tool in payload["semanticTools"]]
    assert names == ["nova.set_light", "nova.set_climate"]
    assert payload["instructions"] == "plan carefully"


async def test_an_interpreter_with_no_routable_prompt_is_untouched() -> None:
    class _Plain(Interpreter):
        def __init__(self) -> None:
            self.calls = 0

        async def interpret(self, utterance, **_kwargs):
            self.calls += 1
            return Interpretation.model_validate(_wire())

    backend = _Plain()
    router = CompanionWorkloadRouter(CompanionSessionManager(), enabled=True)
    companion = _Companion(_wire("nova.unlock_door"))
    companion.install(router)

    await RoutedInterpreter(backend, router).interpret(
        _utterance(), active_goal=None, relevant_state={}, tools=TOOLS
    )

    assert backend.calls == 1
    assert companion.payloads == []
