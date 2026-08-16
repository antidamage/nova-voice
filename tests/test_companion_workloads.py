"""The workload table is the contract; these tests are what keep it honest.

The rule being enforced: a workload may be routable only if it has a local
implementation to fall back to and a result schema to validate against. A route
that cannot resolve locally is not a route — it is an outage that waits for the
phone to be in someone's pocket.
"""

from __future__ import annotations

import inspect

import pytest
from pydantic import BaseModel

from nova_voice.companion.protocol import CompanionWorkload
from nova_voice.companion.router import DEFAULT_ROUTES
from nova_voice.companion.workloads import WORKLOADS, parse_result, spec
from nova_voice.interpretation.base import Interpreter

WORKLOAD_NAMES = sorted(WORKLOADS)


def _declared_workloads() -> set[str]:
    return set(CompanionWorkload.__args__)


def test_every_declared_workload_has_a_handler():
    """A workload nameable on the wire must be dispatchable."""

    assert _declared_workloads() == set(WORKLOADS)


def test_every_workload_has_a_route():
    """A configured route cannot name a workload with no handler, or vice versa."""

    assert set(DEFAULT_ROUTES) == set(WORKLOADS)


@pytest.mark.parametrize("workload", WORKLOAD_NAMES)
def test_workload_result_model_is_a_strict_model(workload):
    entry = spec(workload)
    assert issubclass(entry.result_model, BaseModel)
    assert entry.result_schema.endswith(".v1")


@pytest.mark.parametrize("workload", WORKLOAD_NAMES)
def test_workload_has_a_real_local_implementation(workload):
    """The fallback target must exist on the Interpreter protocol."""

    entry = spec(workload)
    method = getattr(Interpreter, entry.local_method, None)
    assert method is not None, f"{workload} names a non-existent fallback"
    assert inspect.iscoroutinefunction(method)


@pytest.mark.parametrize("workload", WORKLOAD_NAMES)
def test_workload_deadline_is_bounded(workload):
    entry = spec(workload)
    assert 0 < entry.default_timeout_seconds <= 600


@pytest.mark.parametrize("workload", WORKLOAD_NAMES)
def test_hot_path_deadlines_leave_room_for_fallback(workload):
    """A hot-path pass must not spend the whole turn waiting on the phone.

    Whatever the companion fails to deliver still has to be generated locally
    and then spoken, so the remote budget has to be a fraction of the turn.
    """

    entry = spec(workload)
    if entry.hot_path:
        assert entry.default_timeout_seconds <= 45.0


def test_unknown_workload_raises_rather_than_defaulting():
    with pytest.raises(KeyError, match="no companion workload handler"):
        spec("summon_a_demon")


@pytest.mark.parametrize("workload", WORKLOAD_NAMES)
def test_garbage_results_are_discarded_not_raised(workload):
    """A malformed answer degrades to the local path; it never propagates."""

    assert parse_result(workload, {"nonsense": True}) is None
    assert parse_result(workload, "not even a mapping") is None
    assert parse_result(workload, None) is None


def test_valid_results_round_trip():
    # Field names are whatever ``model_json_schema()`` emits, because that is
    # the schema the local pass already constrains its sampler with — the
    # companion generates against the identical shape.
    verdict = parse_result("confirm_objective", {"items": [], "all_confirmed": True})
    assert verdict is not None and verdict.all_confirmed is True

    rendered = parse_result("render_response", {"text": "Done."})
    assert rendered is not None and rendered.text == "Done."

    icon = parse_result("classify_icon", {"icon": "pill"})
    assert icon is not None and icon.icon == "pill"


def test_render_response_rejects_an_empty_reply():
    """Empty, absent and whitespace must not be confusable for TTS."""

    assert parse_result("render_response", {"text": ""}) is None


def test_deterministic_managers_are_not_routable():
    """Guards the NPT-001 seam correction against being quietly undone.

    ResearchManager, BriefingManager and AutomationManager.draft() generate
    nothing today. Listing them as routable would create a route with no local
    fallback, so they must stay out until a local LLM stage exists.
    """

    for absent in ("research_synthesis", "briefing_composition", "automation_draft"):
        assert absent not in WORKLOADS
        assert absent not in _declared_workloads()
