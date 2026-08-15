"""Routing the reply pass — the first workload inside the spoken turn.

`render_response` is routed by wrapping the interpreter rather than by editing
its three call sites, because the part that matters happens *after* generation:
a confirmed command is answered in exactly the requested number of words, and a
model that overruns is replaced by a canned acknowledgement. Three copies of
that rule would have been three chances for the phone's replies to drift from
Iridium's.

The assertions here are mostly about that contract surviving the trip.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from nova_voice.companion.routed import RoutedInterpreter, companion_render_payload
from nova_voice.companion.router import CompanionWorkloadRouter
from nova_voice.companion.session import CompanionSessionManager
from nova_voice.domain import (
    ActiveGoal,
    Emotion,
    EmotionLabel,
    GoalStatus,
    Interpretation,
    ResponsePlan,
    SpeechAct,
    Utterance,
)
from nova_voice.interpretation.base import Interpreter, RenderRequest

pytestmark = pytest.mark.asyncio


def _utterance(text: str = "turn the kitchen light on") -> Utterance:
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


def _interpretation() -> Interpretation:
    return Interpretation(
        emotion=Emotion(label=EmotionLabel.NEUTRAL, confidence=0.8, intensity=0.2),
        speech_act=SpeechAct.DIRECTIVE,
        addressed_probability=0.99,
        decision="execute",
        confidence=0.95,
        active_goal=ActiveGoal(summary="light on", status=GoalStatus.IN_PROGRESS, pending=[]),
        actions=[],
        response_plan=ResponsePlan(),
    )


class _Backend(Interpreter):
    """An interpreter with a routable reply prompt and a counted local path."""

    def __init__(self, *, request: RenderRequest | None, local_text: str = "local reply") -> None:
        self._request = request
        self.local_text = local_text
        self.builds = 0
        self.local_runs = 0
        self.unrouted_calls = 0

    async def interpret(self, utterance, **_kwargs):  # pragma: no cover - not exercised
        raise AssertionError("interpret is not routed here")

    def build_render_request(self, *_args, **_kwargs):
        self.builds += 1
        return self._request

    async def run_render_request(self, request):
        self.local_runs += 1
        return self.finalize_rendered(request, self.local_text)

    async def render_response(self, *_args, **_kwargs):
        self.unrouted_calls += 1
        return self.local_text

    def finalize_rendered(self, request, rendered):
        """The bit of the real contract these tests care about."""

        rendered = rendered.strip()
        if request.all_succeeded and request.command_max_words is not None:
            if len(rendered.split()) != request.command_max_words:
                return "Done."
        return rendered or None


def _request(**overrides) -> RenderRequest:
    fields = {
        "messages": [{"role": "system", "content": "be brief"}],
        "system": "be brief",
        "facts": {"utterance": "turn the kitchen light on", "responseInstruction": "one word"},
        "history": [],
    }
    fields.update(overrides)
    return RenderRequest(**fields)


class _Companion:
    """Stands in for an accepting companion, leaving the router's rules intact."""

    def __init__(self, result: object) -> None:
        self.result = result
        self.payloads: list[dict] = []

    def install(self, router: CompanionWorkloadRouter) -> CompanionWorkloadRouter:
        async def run(workload, payload, local, *, parse=None, **_kwargs):
            self.payloads.append(payload)
            value = self.result
            if parse is not None:
                value = parse(value) if isinstance(value, dict) else None
            if value is None:
                return SimpleNamespace(
                    source="local", value=await local(), reason="invalid result", elapsed_ms=0.0
                )
            return SimpleNamespace(
                source="companion", value=value, reason="accepted", elapsed_ms=1.0
            )

        router.run = run  # type: ignore[method-assign]
        return router


def _routed(backend: _Backend, companion: _Companion | None) -> RoutedInterpreter:
    router = CompanionWorkloadRouter(CompanionSessionManager(), enabled=True)
    if companion is not None:
        companion.install(router)
    return RoutedInterpreter(backend, router)


async def _render(interpreter: RoutedInterpreter, **overrides):
    kwargs = {"persona": "warm", "command_max_words": None}
    kwargs.update(overrides)
    return await interpreter.render_response(
        _utterance(), _interpretation(), [], **kwargs  # type: ignore[arg-type]
    )


async def test_a_companion_reply_is_used() -> None:
    backend = _Backend(request=_request())
    companion = _Companion({"text": "Sure thing."})

    assert await _render(_routed(backend, companion)) == "Sure thing."
    assert backend.local_runs == 0


async def test_a_companion_reply_is_held_to_the_word_budget() -> None:
    """The contract is enforced wherever the words came from.

    A confirmed command is answered in exactly one word. A phone that writes a
    friendly sentence instead gets the same canned acknowledgement the local
    model would have been given — otherwise routing would quietly change what
    the house says back.
    """

    backend = _Backend(request=_request(all_succeeded=True, command_max_words=1))
    companion = _Companion({"text": "I have turned the kitchen light on for you."})

    assert await _render(_routed(backend, companion)) == "Done."


async def test_an_empty_companion_reply_is_not_spoken() -> None:
    backend = _Backend(request=_request())
    companion = _Companion({"text": "   "})

    assert await _render(_routed(backend, companion)) is None


async def test_a_malformed_companion_result_falls_back_locally() -> None:
    backend = _Backend(request=_request())
    companion = _Companion({"reply": "wrong key"})

    assert await _render(_routed(backend, companion)) == "local reply"
    assert backend.local_runs == 1


async def test_the_request_is_built_once_for_both_paths() -> None:
    """Rebuilding for the fallback would not be harmless.

    The long-form branch is chosen by a coin flip inside the builder, so a
    second build could answer a different question from the one the phone was
    asked — and then hold the answer to the wrong word budget.
    """

    backend = _Backend(request=_request())
    companion = _Companion({"reply": "wrong key"})

    await _render(_routed(backend, companion))

    assert backend.builds == 1
    assert backend.local_runs == 1


async def test_a_backend_with_no_routable_prompt_is_untouched() -> None:
    """Deterministic and test interpreters must behave exactly as before."""

    backend = _Backend(request=None)
    companion = _Companion({"text": "should not be used"})

    assert await _render(_routed(backend, companion)) == "local reply"
    assert backend.unrouted_calls == 1
    assert companion.payloads == []


async def test_instructions_are_sent_whole_and_elastic_facts_are_trimmed() -> None:
    """Trim the household snapshot, never the rules.

    A reply generated without the instructions is not a cheaper reply, it is an
    off-contract one — which is exactly the failure this whole design exists to
    avoid.
    """

    instructions = "line one\n" * 200
    request = _request(
        system=instructions,
        facts={
            "responseInstruction": "one word",
            "relevantState": {"lights": [f"light-{index}" for index in range(400)]},
            "selectedMemory": [f"memory {index}" for index in range(400)],
        },
    )

    payload = companion_render_payload(request, context_tokens=4096)

    assert payload["instructions"] == instructions
    assert payload["facts"]["responseInstruction"] == "one word"
    assert len(payload["facts"]["relevantState"]["lights"]) < 400
    assert len(payload["facts"]["selectedMemory"]) < 400
