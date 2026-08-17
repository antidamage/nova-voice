"""The all-workload benchmark endpoint measures the real routed paths.

The tool-free interpretation harness could only exercise one of the five
workloads, in its easiest shape. These cover the endpoint that reaches the
other four and the tool-bearing shape of the first — and, more importantly,
that it cannot do anything to the house while doing so.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from conftest import interpretation

from nova_voice.api import create_app
from nova_voice.config import Settings

CATALOGUE = [
    {"name": "lights.set", "description": "Set a light"},
    {"name": "climate.set", "description": "Set a heater"},
]


class _Interpreter:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def interpret(self, utterance, **kwargs):
        self.calls.append({"workload": "interpret", "utterance": utterance, **kwargs})
        return interpretation()

    async def render_response(self, utterance, result, results, **kwargs):
        self.calls.append(
            {
                "workload": "render_response",
                "utterance": utterance,
                "results": results,
                **kwargs,
            }
        )
        return "The lounge light is on."

    async def classify_icon(self, name, icons):
        self.calls.append({"workload": "classify_icon", "name": name, "icons": icons})
        return icons[0]


class _Router:
    """Records one fresh comparison per sample, like the real router."""

    def __init__(self, workload: str) -> None:
        self.workload = workload
        self.reads = 0
        self.ran: list[tuple[str, dict]] = []

    def comparisons(self) -> list[dict]:
        self.reads += 1
        if self.reads == 1:
            return []
        return [
            {
                "workload": self.workload,
                "at": "2026-08-17T04:00:00Z",
                "spoken": "companion",
                "companion": {"elapsedMs": 900.0, "reason": None},
                "local": {"elapsedMs": 2400.0},
            }
        ]

    async def run(self, workload, payload, local, parse=None):
        self.ran.append((workload, payload))
        return SimpleNamespace(
            value=await local(), source="companion", reason=None, elapsed_ms=900.0
        )


def _service(workload: str, interpreter: _Interpreter | None = None) -> SimpleNamespace:
    service = SimpleNamespace(
        interpreter=interpreter or _Interpreter(),
        companion_router=_Router(workload),
        companion_sessions=None,
    )

    async def confirm(utterance, pending):
        service.interpreter.calls.append(
            {"workload": "confirm_objective", "pending": pending}
        )
        return SimpleNamespace(
            all_confirmed=False,
            items=[
                SimpleNamespace(
                    target=entry["target"], confirmed=False, reason="still off"
                )
                for entry in pending
            ],
        )

    async def profile(utterance):
        service.interpreter.calls.append({"workload": "extract_self_profile_update"})
        return SimpleNamespace(name="Adeline", pronouns=None, evidence="I'm Adeline")

    service._routed_confirm_objective = confirm
    service._routed_self_profile_update = profile
    return service


async def _post(service, body: dict, *, enabled: bool = True) -> httpx.Response:
    app = create_app(Settings(test_harness_enabled=enabled), service=service)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://voice.test"
    ) as client:
        return await client.post("/v1/test/workload", json=body)


async def test_the_endpoint_is_gated_by_the_test_harness_switch() -> None:
    response = await _post(
        _service("interpret"),
        {"workload": "interpret", "transcript": "hello"},
        enabled=False,
    )
    assert response.status_code == 404


async def test_unknown_workloads_are_rejected_by_the_schema() -> None:
    response = await _post(_service("interpret"), {"workload": "speak_aloud"})
    assert response.status_code == 422


async def test_interpretation_receives_the_offered_tool_catalogue() -> None:
    service = _service("interpret")
    response = await _post(
        service,
        {
            "workload": "interpret",
            "transcript": "turn the lounge light on",
            "tools": CATALOGUE,
            "state": {"lounge": "off"},
        },
    )

    assert response.status_code == 200
    call = service.interpreter.calls[0]
    assert call["tools"] == CATALOGUE
    assert call["relevant_state"] == {"lounge": "off"}
    assert response.json()["input"]["toolsOffered"] == 2


@pytest.mark.parametrize(
    "workload",
    [
        "interpret",
        "render_response",
        "confirm_objective",
        "extract_self_profile_update",
        "classify_icon",
    ],
)
async def test_every_utterance_is_marked_dry_run(workload: str) -> None:
    """The single most important property of this endpoint.

    A benchmark that ran a plan for real would turn a measurement into a
    household action, and the plan for this migration forbids exactly that.
    """

    service = _service(workload)
    response = await _post(
        service,
        {
            "workload": workload,
            "transcript": "turn the lounge light on",
            "tools": CATALOGUE,
            "name": "Water the plants",
            "icons": ["droplet", "leaf"],
            "pending": [{"target": "lounge", "objective": "on", "observed": {}}],
        },
    )

    assert response.status_code == 200
    for call in service.interpreter.calls:
        if "utterance" in call:
            assert call["utterance"].dry_run is True


async def test_rendering_is_given_supplied_results_and_never_executes() -> None:
    service = _service("render_response")
    response = await _post(
        service,
        {
            "workload": "render_response",
            "transcript": "did the light come on",
            "tool_results": [
                {
                    "action_id": "a0",
                    "ok": True,
                    "code": "ok",
                    "target": "lounge",
                    "message": "on",
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["result"] == {"text": "The lounge light is on."}
    call = next(
        entry for entry in service.interpreter.calls if entry["workload"] == "render_response"
    )
    # Supplied as fixture data, not produced by running anything.
    assert [result.target for result in call["results"]] == ["lounge"]


async def test_both_arms_are_reported_for_the_matching_workload() -> None:
    response = await _post(
        _service("interpret"), {"workload": "interpret", "transcript": "hello"}
    )

    body = response.json()
    assert body["armsMs"] == {"companion": 900.0, "local": 2400.0}
    assert body["winner"] == "companion"


async def test_a_comparison_for_another_workload_is_not_claimed() -> None:
    """Recency alone is not identity.

    Two workloads can be in flight at once on a live system. Attributing
    another pass's timings to this sample would silently corrupt every number
    in the report.
    """

    response = await _post(
        _service("classify_icon"),
        {
            "workload": "interpret",
            "transcript": "hello",
        },
    )

    assert response.json()["armsMs"] is None


async def test_icon_answers_outside_the_offered_vocabulary_are_discarded() -> None:
    class _Inventing(_Interpreter):
        async def classify_icon(self, name, icons):
            return "rocket"

    service = _service("classify_icon", _Inventing())
    response = await _post(
        service,
        {
            "workload": "classify_icon",
            "name": "Water the plants",
            "icons": ["droplet", "leaf"],
        },
    )

    assert response.json()["result"] == {"icon": None}


async def test_icon_requires_a_name_and_a_vocabulary() -> None:
    response = await _post(
        _service("classify_icon"), {"workload": "classify_icon", "icons": ["droplet"]}
    )
    assert response.status_code == 400


async def test_interpretation_actions_are_reported_in_execution_order() -> None:
    response = await _post(
        _service("interpret"), {"workload": "interpret", "transcript": "hello"}
    )
    result = response.json()["result"]
    assert "actions" in result
    assert isinstance(result["actions"], list)
