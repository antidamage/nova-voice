"""The ANE benchmark's scoring, gates and evidence rendering.

Scoring is the part of a benchmark most worth testing: a lenient scorer reports
a passing migration that is not passing, and nobody notices until the house
does something wrong. These cover what must fail, not what should pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ops"))

from companion_ane_benchmark import (  # noqa: E402
    BYTES_PER_TOKEN,
    FIXTURES,
    SUITES,
    bandwidth_report,
    build_body,
    gate_report,
    load_catalogue,
    load_fixtures,
    quality_report,
    render_graph,
    render_markdown,
    score_confirm,
    score_icon,
    score_interpret_tool_bearing,
    score_interpret_tool_free,
    score_profile,
    score_render,
)

CATALOGUE = load_catalogue()
OFFERED = {tool["name"] for tool in CATALOGUE["tools"]}
VOCABULARY = set(CATALOGUE["icons"])


# -- fixtures -----------------------------------------------------------------


def test_every_suite_has_the_case_count_the_plan_requires() -> None:
    """The plan fixes these counts; a quietly shrunk suite is a weaker gate."""

    expected = {
        "interpret-tool-free.jsonl": 40,
        "interpret-tool-bearing.jsonl": 40,
        "render-response.jsonl": 30,
        "classify-icon.jsonl": 40,
        "self-profile.jsonl": 30,
        "confirm-objective.jsonl": 30,
    }
    for name, count in expected.items():
        assert len(load_fixtures(name)) == count, name


def test_fixture_ids_are_unique_within_each_suite() -> None:
    for path in FIXTURES.glob("*.jsonl"):
        ids = [case["id"] for case in load_fixtures(path.name)]
        assert len(ids) == len(set(ids)), path.name


def test_tool_bearing_expectations_only_name_offered_tools() -> None:
    """A fixture that expects an unoffered tool would gate on a fabrication."""

    for case in load_fixtures("interpret-tool-bearing.jsonl"):
        for action in case["expect"].get("actions", []):
            assert action["tool"] in OFFERED, case["id"]


def test_icon_expectations_stay_inside_the_vocabulary() -> None:
    for case in load_fixtures("classify-icon.jsonl"):
        icon = case["expect"]["icon"]
        assert icon is None or icon in VOCABULARY, case["id"]


def test_negative_cases_exist_in_every_suite_that_can_fabricate() -> None:
    """A suite of only positive cases cannot detect over-eagerness."""

    tool_bearing = load_fixtures("interpret-tool-bearing.jsonl")
    assert sum(1 for c in tool_bearing if not c["expect"].get("actions")) >= 10
    profile = load_fixtures("self-profile.jsonl")
    assert sum(1 for c in profile if c["expect"] is None) >= 10
    render = load_fixtures("render-response.jsonl")
    assert sum(1 for c in render if c["expect"].get("mustNotClaimSuccess")) >= 8


# -- request construction -----------------------------------------------------


def test_tool_free_bodies_offer_no_tools() -> None:
    body = build_body(
        "interpret_tool_free", {"transcript": "hello"}, CATALOGUE
    )
    assert body["workload"] == "interpret"
    assert not body.get("tools")


def test_tool_bearing_bodies_offer_the_whole_catalogue_and_state() -> None:
    body = build_body(
        "interpret_tool_bearing", {"transcript": "lights on"}, CATALOGUE
    )
    assert body["tools"] == CATALOGUE["tools"]
    assert body["state"] == CATALOGUE["state"]


@pytest.mark.parametrize("suite,fixture,workload", SUITES)
def test_every_fixture_builds_a_body_for_its_declared_workload(
    suite, fixture, workload
) -> None:
    """Every case in every suite must produce a request the endpoint accepts.

    Catches the ordinary drift of a fixture gaining a field the builder does
    not read — which would silently benchmark a different question from the one
    the fixture was written to ask.
    """

    for case in load_fixtures(fixture):
        body = build_body(suite, case, CATALOGUE)
        assert body["workload"] == workload, case["id"]
        # The builder must consume what the fixture supplies, or the case is
        # not being asked as written.
        if suite == "confirm_objective":
            assert body["pending"] == case["pending"], case["id"]
        if suite == "classify_icon":
            assert body["name"] == case["name"], case["id"]
        if suite == "render_response":
            assert body["tool_results"] == case.get("tool_results", []), case["id"]


# -- interpretation scoring ---------------------------------------------------


def test_tool_free_case_that_plans_an_action_fails() -> None:
    """Nothing was offered, so any action is invented."""

    case = {"expect": {"decision": "reply", "speechAct": "question"}}
    result = {
        "decision": "reply",
        "speechAct": "question",
        "actions": [{"tool": "lights.set", "arguments": {}}],
    }
    scored = score_interpret_tool_free(case, result)
    assert not scored["pass"]
    assert scored["unofferedTool"]


def test_tool_free_wrong_decision_fails_even_with_the_right_speech_act() -> None:
    case = {"expect": {"decision": "ignore", "speechAct": "third_party"}}
    result = {"decision": "reply", "speechAct": "third_party", "actions": []}
    assert not score_interpret_tool_free(case, result)["pass"]


def test_an_unoffered_tool_fails_the_whole_case() -> None:
    case = {
        "expect": {
            "decision": "execute",
            "actions": [{"tool": "lights.set", "arguments": {"target": "lounge"}}],
        }
    }
    result = {
        "decision": "execute",
        "actions": [
            {"tool": "lights.set", "arguments": {"target": "lounge"}},
            {"tool": "locks.set", "arguments": {"target": "front_door"}},
        ],
    }
    scored = score_interpret_tool_bearing(case, result, OFFERED)
    assert not scored["pass"]
    assert scored["unofferedNames"] == ["locks.set"]


def test_a_wrong_argument_fails_even_with_the_right_tool() -> None:
    case = {
        "expect": {
            "decision": "execute",
            "actions": [
                {"tool": "lights.set", "arguments": {"target": "lounge", "state": "on"}}
            ],
        }
    }
    result = {
        "decision": "execute",
        "actions": [
            {"tool": "lights.set", "arguments": {"target": "kitchen", "state": "on"}}
        ],
    }
    assert not score_interpret_tool_bearing(case, result, OFFERED)["pass"]


def test_numeric_arguments_tolerate_device_rounding() -> None:
    """49 where 50 was asked is the device rounding, not a planning error."""

    case = {
        "expect": {
            "decision": "execute",
            "actions": [
                {"tool": "lights.set", "arguments": {"target": "lounge", "brightness": 50}}
            ],
        }
    }
    result = {
        "decision": "execute",
        "actions": [
            {"tool": "lights.set", "arguments": {"target": "lounge", "brightness": 49}}
        ],
    }
    assert score_interpret_tool_bearing(case, result, OFFERED)["pass"]
    far = {
        "decision": "execute",
        "actions": [
            {"tool": "lights.set", "arguments": {"target": "lounge", "brightness": 80}}
        ],
    }
    assert not score_interpret_tool_bearing(case, far, OFFERED)["pass"]


def test_extra_actions_beyond_the_expectation_fail() -> None:
    case = {"expect": {"decision": "clarify", "actions": []}}
    result = {
        "decision": "clarify",
        "actions": [{"tool": "lights.set", "arguments": {"target": "lounge"}}],
    }
    assert not score_interpret_tool_bearing(case, result, OFFERED)["pass"]


# -- icon scoring -------------------------------------------------------------


def test_an_icon_outside_the_offered_vocabulary_fails() -> None:
    case = {"expect": {"icon": "droplet"}}
    scored = score_icon(case, {"icon": "rocket"}, VOCABULARY)
    assert not scored["pass"]
    assert scored["outsideVocabulary"]


def test_null_is_a_correct_icon_answer_when_nothing_fits() -> None:
    case = {"expect": {"icon": None}}
    assert score_icon(case, {"icon": None}, VOCABULARY)["pass"]
    assert not score_icon(case, {"icon": "droplet"}, VOCABULARY)["pass"]


# -- profile scoring ----------------------------------------------------------


def test_inventing_a_profile_from_a_passing_remark_is_a_false_positive() -> None:
    """"I'm cold" is the failure that would rename the household."""

    case = {"expect": None}
    scored = score_profile(case, {"name": "Cold", "pronouns": None, "evidence": "I'm cold"})
    assert not scored["pass"]
    assert scored["falsePositive"]


def test_a_missed_disclosure_is_not_counted_as_a_false_positive() -> None:
    case = {"expect": {"name": "Adeline"}}
    scored = score_profile(case, None)
    assert not scored["pass"]
    assert not scored["falsePositive"]


def test_pronoun_spacing_does_not_change_the_verdict() -> None:
    case = {"expect": {"pronouns": "they/them"}}
    assert score_profile(case, {"name": None, "pronouns": "they / them", "evidence": "x"})[
        "pass"
    ]


# -- confirmation scoring -----------------------------------------------------


def test_a_verdict_that_contradicts_its_own_items_fails() -> None:
    """all_confirmed is derivable; disagreement means the judgement is broken."""

    case = {
        "expect": {"allConfirmed": False, "confirmed": {"lounge_light": False}},
    }
    result = {
        "allConfirmed": True,
        "items": [{"target": "lounge_light", "confirmed": False, "reason": "off"}],
    }
    scored = score_confirm(case, result)
    assert not scored["pass"]
    assert not scored["internallyConsistent"]


def test_a_verdict_missing_a_target_fails() -> None:
    case = {
        "expect": {
            "allConfirmed": False,
            "confirmed": {"lounge_light": True, "kitchen_light": False},
        }
    }
    result = {
        "allConfirmed": False,
        "items": [{"target": "lounge_light", "confirmed": True, "reason": "on"}],
    }
    scored = score_confirm(case, result)
    assert not scored["pass"]
    assert not scored["coversEveryTarget"]


def test_a_fully_correct_multi_target_verdict_passes() -> None:
    case = {
        "expect": {
            "allConfirmed": False,
            "confirmed": {"lounge_light": True, "kitchen_light": False},
        }
    }
    result = {
        "allConfirmed": False,
        "items": [
            {"target": "lounge_light", "confirmed": True, "reason": "on"},
            {"target": "kitchen_light", "confirmed": False, "reason": "off"},
        ],
    }
    assert score_confirm(case, result)["pass"]


# -- rendering scoring --------------------------------------------------------


def test_claiming_success_after_a_failure_fails() -> None:
    """The worst possible reply: telling someone a light is on when it is off."""

    case = {"expect": {"maxWords": 20, "mustNotClaimSuccess": True}}
    scored = score_render(case, {"text": "The lounge light is on."})
    assert not scored["pass"]
    assert scored["falseSuccessClaim"]


def test_an_honest_failure_reply_passes() -> None:
    case = {"expect": {"maxWords": 20, "mustNotClaimSuccess": True}}
    assert score_render(case, {"text": "I could not reach the lounge light."})["pass"]


def test_exceeding_the_word_budget_fails() -> None:
    case = {"expect": {"maxWords": 5, "mustNotClaimSuccess": False}}
    assert not score_render(case, {"text": " ".join(["word"] * 12)})["pass"]


def test_fabricating_data_that_was_never_returned_fails() -> None:
    case = {
        "expect": {
            "maxWords": 30,
            "mustNotClaimSuccess": True,
            "mustNotFabricate": ["degrees"],
        }
    }
    scored = score_render(case, {"text": "It is about fifteen degrees out."})
    assert not scored["pass"]
    assert scored["fabricated"] == ["degrees"]


def test_an_empty_reply_fails_rather_than_passing_vacuously() -> None:
    case = {"expect": {"maxWords": 20, "mustNotClaimSuccess": True}}
    assert not score_render(case, {"text": "   "})["pass"]


# -- aggregation and gates ----------------------------------------------------


def _samples(suite: str, passes: int, total: int, **flags) -> list[dict]:
    return [
        {
            "suite": suite,
            "id": f"{suite}-{index}",
            "result": {"x": 1},
            "score": {"pass": index < passes, **(flags if index >= passes else {})},
        }
        for index in range(total)
    ]


def test_one_unoffered_tool_fails_the_run_regardless_of_the_average() -> None:
    """39 of 40 correct is a 97.5% rate and still a failed run."""

    samples = _samples("interpret_tool_bearing", 39, 40, unofferedTool=True)
    gates = gate_report(quality_report(samples), {}, [])
    assert gates["interpretToolAccuracy"]["pass"]
    assert not gates["zeroUnofferedTools"]["pass"]


def test_one_profile_false_positive_fails_the_run() -> None:
    samples = _samples("extract_self_profile_update", 29, 30, falsePositive=True)
    gates = gate_report(quality_report(samples), {}, [])
    assert not gates["zeroProfileFalsePositives"]["pass"]


def test_one_false_success_claim_fails_the_run() -> None:
    samples = _samples("render_response", 29, 30, falseSuccessClaim=True)
    gates = gate_report(quality_report(samples), {}, [])
    assert not gates["zeroFalseSuccessClaims"]["pass"]


def test_accuracy_gates_use_the_thresholds_the_plan_states() -> None:
    samples = _samples("classify_icon", 36, 40)
    gates = gate_report(quality_report(samples), {}, [])
    assert gates["iconAccuracy"]["gate"] == 0.90
    assert gates["iconAccuracy"]["pass"]
    assert not gate_report(quality_report(_samples("classify_icon", 35, 40)), {}, [])[
        "iconAccuracy"
    ]["pass"]


def test_sustained_throughput_is_read_from_the_last_minute_not_the_average() -> None:
    """Averaging a ten-minute run hides exactly the decay it exists to find."""

    sustained = [
        {
            "contextShape": "2k",
            "companionAllMs": {"median": 30.0},
            "companionLastMinuteMs": {"median": 50.0},
        }
    ]
    gates = gate_report({}, {}, sustained)
    assert gates["sustained2kDecode"]["value"] == 20.0
    assert not gates["sustained2kDecode"]["pass"]


def test_latency_gates_are_reported_in_seconds() -> None:
    gates = gate_report({}, {"interpret_tool_free": {"median": 19000.0, "p95": 26000.0}}, [])
    assert gates["toolFreeMedianSeconds"]["pass"]
    assert not gates["toolFreeP95Seconds"]["pass"]


# -- bandwidth ----------------------------------------------------------------


def test_bandwidth_is_derived_from_measured_throughput() -> None:
    report = bandwidth_report(30.0)
    assert report["bytesPerToken"] == BYTES_PER_TOKEN
    assert report["effectiveBytesPerSecond"] == round(BYTES_PER_TOKEN * 30.0)
    assert report["withinBudget"]


def test_a_throughput_over_the_budget_is_flagged_rather_than_reported_flat() -> None:
    report = bandwidth_report(60.0)
    assert not report["withinBudget"]
    assert report["fractionOfBudget"] > 1


def test_the_ceiling_matches_the_plans_arithmetic() -> None:
    assert bandwidth_report(1.0)["ceilingTokensPerSecond"] == pytest.approx(40.8, abs=0.2)


# -- evidence rendering -------------------------------------------------------


def _report() -> dict:
    samples = _samples("interpret_tool_free", 39, 40)
    quality = quality_report(samples)
    return {
        "capturedAt": "2026-08-17T00:00:00+00:00",
        "latencyMs": {
            "interpret_tool_free": {
                "companion": {"median": 1800.0, "p95": 2400.0},
                "local": {"median": 900.0, "p95": 1100.0},
            }
        },
        "quality": quality,
        "gates": gate_report(quality, {}, []),
        "sustained": [
            {
                "contextShape": "2k",
                "companionFirstMinuteMs": {"median": 30.0},
                "companionLastMinuteMs": {"median": 44.0},
            }
        ],
        "bandwidth": bandwidth_report(25.0),
        "thermal": {"peak": "fair"},
    }


def test_the_graph_is_a_valid_png_and_byte_deterministic() -> None:
    """A graph that differs between runs cannot be reviewed as evidence."""

    first = render_graph(_report())
    assert first.startswith(b"\x89PNG\r\n\x1a\n")
    assert render_graph(_report()) == first


def test_the_report_labels_the_arms_as_different_models() -> None:
    """The single most misreadable thing here, so it is asserted rather than trusted."""

    text = render_markdown(_report()).lower()
    assert "different models" in text
    assert "gemma 4 e2b" in text
    assert "llama.cpp" in text
    assert "not" in text and "hardware" in text


def test_the_report_records_that_nothing_was_spoken_or_executed() -> None:
    text = render_markdown(_report()).lower()
    assert "no audio was captured" in text
    assert "no tool was executed" in text
