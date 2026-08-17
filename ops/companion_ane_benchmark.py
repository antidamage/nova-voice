"""Stage 7 of the ANE migration: quality and latency, phone against Iridium.

Runs all five configurable LLM workloads over text-only fixtures, interleaving
the two arms, and reports latency phases, throughput, quality pass rates,
thermal retention and effective memory bandwidth.

Produces no voice output, captures no audio, and executes no tool. Every
utterance is marked dry-run by the endpoint it calls, and rendering cases are
given fixture tool *results* rather than running anything to obtain them.

**The two arms are different models on different hardware.** The phone runs
Gemma 4 E2B through Core ML on the Neural Engine at an 8,192-token window;
Iridium runs its own llama.cpp model. This is a routing decision — "should this
pass go to the phone?" — and not a same-model hardware comparison. Every output
of this harness is labelled accordingly, because a reader who forgets it will
draw a conclusion the data cannot support.

Run on the voice host with the companion app foregrounded::

    /opt/nova-voice/venv/bin/python ops/companion_ane_benchmark.py \\
        --host https://127.0.0.1:8766 \\
        --cert /tmp/vs-tls/client.crt --key /tmp/vs-tls/client.key \\
        --out docs/evidence
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import sys
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deterministic_png import Canvas  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "companion-ane"

# From the compiled chunk shapes, recorded in nova-companion's
# docs/ANE-MIGRATION-PLAN.md. Held here so effective bandwidth can be derived
# from measured throughput rather than asserted.
DECODE_WEIGHT_BYTES = 1_141_558_592
KV_TRAFFIC_BYTES = 177_800_000
SIDECAR_BYTES = 10_500
BYTES_PER_TOKEN = DECODE_WEIGHT_BYTES + KV_TRAFFIC_BYTES + SIDECAR_BYTES
# 76.8 GB/s shared-system roofline less 30% reserved for iOS, audio and UI.
BANDWIDTH_BUDGET_BYTES = 53.8 * 1_000_000_000

GATES = {
    "warm2kDecodeP50": 30.0,
    "sustained2kDecode": 22.0,
    "warm8kDecodeP50": 14.0,
    "sustained8kDecode": 10.0,
    "toolFreeMedianSeconds": 20.0,
    "toolFreeP95Seconds": 25.0,
    "interpretToolFreeAccuracy": 0.95,
    "confirmAccuracy": 0.95,
    "interpretToolAccuracy": 0.90,
    "iconAccuracy": 0.90,
    "profileRecall": 0.90,
}


def load_fixtures(name: str) -> list[dict[str, Any]]:
    path = FIXTURES / name
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entries.append(json.loads(line))
    return entries


def load_catalogue() -> dict[str, Any]:
    return json.loads((FIXTURES / "catalogue.json").read_text(encoding="utf-8"))


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return round(ordered[index], 1)


def summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    median = statistics.median(values)
    return {
        "n": len(values),
        "median": round(median, 1),
        "p50": round(median, 1),
        "p95": percentile(values, 0.95),
        "min": round(min(values), 1),
        "max": round(max(values), 1),
        "mad": round(statistics.median([abs(v - median) for v in values]), 1),
    }


# -- scoring ------------------------------------------------------------------
#
# Scoring is deliberately separate from the endpoint that produced the answers.
# A system that grades itself reports what it intended rather than what it did.


def score_interpret_tool_free(case: dict, result: dict | None) -> dict[str, Any]:
    if not result:
        return {"pass": False, "reason": "no result"}
    expected = case["expect"]
    decision_ok = result.get("decision") == expected["decision"]
    act_ok = result.get("speechAct") == expected["speechAct"]
    # A tool-free case that plans an action has fabricated a tool from an empty
    # catalogue, which is a harder failure than a wrong label.
    unoffered = bool(result.get("actions"))
    return {
        "pass": decision_ok and act_ok and not unoffered,
        "decisionOk": decision_ok,
        "speechActOk": act_ok,
        "unofferedTool": unoffered,
    }


def score_interpret_tool_bearing(
    case: dict, result: dict | None, offered: set[str]
) -> dict[str, Any]:
    if not result:
        return {"pass": False, "reason": "no result", "unofferedTool": False}
    expected = case["expect"]
    actions = result.get("actions") or []
    named = [action.get("tool") for action in actions]
    unoffered = [tool for tool in named if tool not in offered]
    decision_ok = result.get("decision") == expected["decision"]

    wanted = expected.get("actions", [])
    tools_ok = named == [action["tool"] for action in wanted]
    arguments_ok = True
    for produced, want in zip(actions, wanted, strict=False):
        got = produced.get("arguments") or {}
        for key, value in (want.get("arguments") or {}).items():
            actual = got.get(key)
            if isinstance(value, str) and isinstance(actual, str):
                if actual.strip().lower() != value.strip().lower():
                    arguments_ok = False
            elif isinstance(value, (int, float)) and isinstance(actual, (int, float)):
                # Devices and models both round. An exact-equality test here
                # would fail on 49 vs 50 and teach nothing.
                if abs(float(actual) - float(value)) > 1:
                    arguments_ok = False
            elif actual != value:
                arguments_ok = False
    if len(actions) != len(wanted):
        tools_ok = False
        arguments_ok = False

    return {
        "pass": decision_ok and tools_ok and arguments_ok and not unoffered,
        "decisionOk": decision_ok,
        "toolsOk": tools_ok,
        "argumentsOk": arguments_ok,
        "unofferedTool": bool(unoffered),
        "unofferedNames": unoffered,
    }


def score_icon(case: dict, result: dict | None, vocabulary: set[str]) -> dict[str, Any]:
    icon = (result or {}).get("icon")
    expected = case["expect"]["icon"]
    outside = icon is not None and icon not in vocabulary
    return {
        "pass": icon == expected and not outside,
        "outsideVocabulary": outside,
        "got": icon,
    }


def score_profile(case: dict, result: dict | None) -> dict[str, Any]:
    expected = case["expect"]
    if expected is None:
        # A false positive here renames the household from a passing remark,
        # so it is counted separately and gated at zero rather than averaged in.
        return {"pass": result is None, "falsePositive": result is not None}
    if not result:
        return {"pass": False, "falsePositive": False, "missed": True}
    name_ok = expected.get("name") is None or (
        (result.get("name") or "").strip().lower() == expected["name"].lower()
    )
    pronouns_ok = expected.get("pronouns") is None or (
        (result.get("pronouns") or "").replace(" ", "").lower()
        == expected["pronouns"].replace(" ", "").lower()
    )
    return {
        "pass": name_ok and pronouns_ok,
        "falsePositive": False,
        "nameOk": name_ok,
        "pronounsOk": pronouns_ok,
    }


def score_confirm(case: dict, result: dict | None) -> dict[str, Any]:
    if not result:
        return {"pass": False, "reason": "no result"}
    expected = case["expect"]
    items = {item["target"]: item["confirmed"] for item in result.get("items", [])}
    per_target = all(
        items.get(target) == value for target, value in expected["confirmed"].items()
    )
    covered = set(items) == set(expected["confirmed"])
    overall_ok = result.get("allConfirmed") == expected["allConfirmed"]
    # all_confirmed is derivable. If it disagrees with the per-target verdicts
    # the whole judgement is internally inconsistent, whatever else matched.
    consistent = result.get("allConfirmed") == all(items.values())
    return {
        "pass": per_target and covered and overall_ok and consistent,
        "perTargetOk": per_target,
        "coversEveryTarget": covered,
        "allConfirmedOk": overall_ok,
        "internallyConsistent": consistent,
    }


def score_render(case: dict, result: dict | None) -> dict[str, Any]:
    text = ((result or {}).get("text") or "").strip()
    if not text:
        return {"pass": False, "reason": "no reply"}
    expected = case["expect"]
    words = len(text.split())
    within_budget = words <= expected["maxWords"]

    lowered = text.lower()
    # A crude but honest proxy: when nothing succeeded, a reply in the past
    # tense about the thing working is a false success claim. Flagged rather
    # than judged clever, and every flagged case is listed in the report for a
    # human to read.
    claims_success = any(
        phrase in lowered
        for phrase in (
            "is on",
            "is off",
            "turned on",
            "turned off",
            "switched on",
            "switched off",
            "done",
            "all set",
            "sorted",
        )
    )
    false_success = bool(expected.get("mustNotClaimSuccess")) and claims_success
    missing = [
        token
        for token in expected.get("mustMention", [])
        if token.lower() not in lowered
    ]
    fabricated = [
        token
        for token in expected.get("mustNotFabricate", [])
        if token.lower() in lowered
    ]
    return {
        "pass": within_budget and not false_success and not missing and not fabricated,
        "words": words,
        "withinWordBudget": within_budget,
        "falseSuccessClaim": false_success,
        "missingRequired": missing,
        "fabricated": fabricated,
        "text": text,
    }


# -- running ------------------------------------------------------------------


async def post_workload(
    client: httpx.AsyncClient, body: dict[str, Any]
) -> dict[str, Any]:
    response = await client.post("/v1/test/workload", json=body)
    response.raise_for_status()
    return response.json()


async def set_routes(client: httpx.AsyncClient, routes: dict[str, str]) -> None:
    response = await client.post("/v1/companion/routing", json={"routes": routes})
    response.raise_for_status()


def build_body(workload: str, case: dict, catalogue: dict) -> dict[str, Any]:
    if workload == "interpret_tool_free":
        return {"workload": "interpret", "transcript": case["transcript"]}
    if workload == "interpret_tool_bearing":
        return {
            "workload": "interpret",
            "transcript": case["transcript"],
            "tools": catalogue["tools"],
            "state": catalogue["state"],
        }
    if workload == "classify_icon":
        return {
            "workload": "classify_icon",
            "name": case["name"],
            "icons": catalogue["icons"],
        }
    if workload == "extract_self_profile_update":
        return {
            "workload": "extract_self_profile_update",
            "transcript": case["transcript"],
        }
    if workload == "confirm_objective":
        return {
            "workload": "confirm_objective",
            "transcript": case["transcript"],
            "pending": case["pending"],
        }
    if workload == "render_response":
        return {
            "workload": "render_response",
            "transcript": case["transcript"],
            "tool_results": case.get("tool_results", []),
            "interpretation": case.get("interpretation"),
            "state": case.get("state", {}),
        }
    raise ValueError(f"unknown fixture workload: {workload}")


SUITES = [
    ("interpret_tool_free", "interpret-tool-free.jsonl", "interpret"),
    ("interpret_tool_bearing", "interpret-tool-bearing.jsonl", "interpret"),
    ("render_response", "render-response.jsonl", "render_response"),
    ("classify_icon", "classify-icon.jsonl", "classify_icon"),
    ("extract_self_profile_update", "self-profile.jsonl", "extract_self_profile_update"),
    ("confirm_objective", "confirm-objective.jsonl", "confirm_objective"),
]


def score_case(
    suite: str, case: dict, result: dict | None, catalogue: dict
) -> dict[str, Any]:
    if suite == "interpret_tool_free":
        return score_interpret_tool_free(case, result)
    if suite == "interpret_tool_bearing":
        offered = {tool["name"] for tool in catalogue["tools"]}
        return score_interpret_tool_bearing(case, result, offered)
    if suite == "classify_icon":
        return score_icon(case, result, set(catalogue["icons"]))
    if suite == "extract_self_profile_update":
        return score_profile(case, result)
    if suite == "confirm_objective":
        return score_confirm(case, result)
    if suite == "render_response":
        return score_render(case, result)
    raise ValueError(suite)


async def run_suite(
    client: httpx.AsyncClient,
    suite: str,
    fixture_file: str,
    catalogue: dict,
    *,
    pause: float,
) -> list[dict[str, Any]]:
    cases = load_fixtures(fixture_file)
    samples = []
    for index, case in enumerate(cases):
        body = build_body(suite, case, catalogue)
        started = time.perf_counter()
        payload = await post_workload(client, body)
        wall_ms = round((time.perf_counter() - started) * 1000, 1)
        scored = score_case(suite, case, payload.get("result"), catalogue)
        samples.append(
            {
                "suite": suite,
                "id": case["id"],
                "index": index,
                "wallMs": wall_ms,
                "elapsedMs": payload.get("elapsedMs"),
                "armsMs": payload.get("armsMs"),
                "winner": payload.get("winner"),
                "companionReason": payload.get("companionReason"),
                "input": payload.get("input"),
                "result": payload.get("result"),
                "score": scored,
            }
        )
        print(
            f"  {suite:28} {case['id']:6} "
            f"{'pass' if scored.get('pass') else 'FAIL':4} "
            f"{wall_ms:8.1f}ms winner={payload.get('winner')}",
            flush=True,
        )
        if pause:
            await asyncio.sleep(pause)
    return samples


async def sustained_run(
    client: httpx.AsyncClient,
    catalogue: dict,
    *,
    seconds: float,
    long_context: bool,
) -> dict[str, Any]:
    """Hammer one shape for `seconds` and report throughput decay.

    Ten minutes rather than a burst because thermal behaviour is the whole
    reason the ANE was chosen over the GPU, and a burst measurement cannot see
    it. Reported as first-minute against last-minute so a decline is visible
    rather than averaged away.
    """

    case = {"transcript": "Turn the lounge light on"}
    body = build_body("interpret_tool_bearing", case, catalogue)
    if long_context:
        # Pad the elastic part of the prompt toward the 8K window without
        # touching instructions, tools or the turn itself.
        body["history"] = [
            {"role": "user", "content": "Earlier we talked about the house. " * 12}
            for _ in range(24)
        ]
    started = time.perf_counter()
    samples: list[dict[str, Any]] = []
    while time.perf_counter() - started < seconds:
        at = round(time.perf_counter() - started, 1)
        payload = await post_workload(client, body)
        samples.append(
            {
                "atSeconds": at,
                "elapsedMs": payload.get("elapsedMs"),
                "armsMs": payload.get("armsMs"),
                "winner": payload.get("winner"),
            }
        )
    def arm(entries: Iterable[dict], key: str) -> list[float]:
        values = []
        for entry in entries:
            arms = entry.get("armsMs") or {}
            value = arms.get(key)
            if isinstance(value, (int, float)):
                values.append(float(value))
        return values

    first = [s for s in samples if s["atSeconds"] <= 60]
    last = [s for s in samples if s["atSeconds"] >= seconds - 60]
    return {
        "contextShape": "8k" if long_context else "2k",
        "durationSeconds": seconds,
        "samples": samples,
        "companionFirstMinuteMs": summary(arm(first, "companion")),
        "companionLastMinuteMs": summary(arm(last, "companion")),
        "companionAllMs": summary(arm(samples, "companion")),
        "localAllMs": summary(arm(samples, "local")),
    }


def bandwidth_report(tokens_per_second: float | None) -> dict[str, Any]:
    if not tokens_per_second:
        return {"tokensPerSecond": None}
    effective = BYTES_PER_TOKEN * tokens_per_second
    return {
        "tokensPerSecond": round(tokens_per_second, 2),
        "bytesPerToken": BYTES_PER_TOKEN,
        "effectiveBytesPerSecond": round(effective),
        "budgetBytesPerSecond": round(BANDWIDTH_BUDGET_BYTES),
        "fractionOfBudget": round(effective / BANDWIDTH_BUDGET_BYTES, 3),
        "withinBudget": effective <= BANDWIDTH_BUDGET_BYTES,
        "ceilingTokensPerSecond": round(
            BANDWIDTH_BUDGET_BYTES / BYTES_PER_TOKEN, 1
        ),
    }


def quality_report(samples: list[dict]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for suite, _, _ in SUITES:
        selected = [s for s in samples if s["suite"] == suite]
        if not selected:
            continue
        passed = sum(1 for s in selected if s["score"].get("pass"))
        entry = {
            "n": len(selected),
            "passed": passed,
            "rate": round(passed / len(selected), 3),
            "failures": [
                {"id": s["id"], "score": s["score"]}
                for s in selected
                if not s["score"].get("pass")
            ],
        }
        entry["unofferedTools"] = sum(
            1 for s in selected if s["score"].get("unofferedTool")
        )
        entry["outsideVocabulary"] = sum(
            1 for s in selected if s["score"].get("outsideVocabulary")
        )
        entry["falsePositives"] = sum(
            1 for s in selected if s["score"].get("falsePositive")
        )
        entry["falseSuccessClaims"] = sum(
            1 for s in selected if s["score"].get("falseSuccessClaim")
        )
        entry["schemaValid"] = sum(1 for s in selected if s["result"] is not None)
        report[suite] = entry
    return report


def gate_report(quality: dict, latency: dict, sustained: list[dict]) -> dict[str, Any]:
    def rate(suite: str) -> float:
        return (quality.get(suite) or {}).get("rate", 0.0)

    def sustained_tps(shape: str) -> float | None:
        for run in sustained:
            if run["contextShape"] == shape:
                median = (run.get("companionLastMinuteMs") or {}).get("median")
                if median:
                    return 1000.0 / median
        return None

    gates = {
        "interpretToolFreeAccuracy": (
            rate("interpret_tool_free"),
            GATES["interpretToolFreeAccuracy"],
        ),
        "interpretToolAccuracy": (
            rate("interpret_tool_bearing"),
            GATES["interpretToolAccuracy"],
        ),
        "iconAccuracy": (rate("classify_icon"), GATES["iconAccuracy"]),
        "profileRecall": (
            rate("extract_self_profile_update"),
            GATES["profileRecall"],
        ),
        "confirmAccuracy": (rate("confirm_objective"), GATES["confirmAccuracy"]),
    }
    results = {
        name: {"value": round(value, 3), "gate": gate, "pass": value >= gate}
        for name, (value, gate) in gates.items()
    }
    # Absolute, not proportional: one unoffered tool or one fabricated profile
    # is a failed run whatever the averages say.
    results["zeroUnofferedTools"] = {
        "value": sum(
            (entry.get("unofferedTools") or 0) for entry in quality.values()
        ),
        "gate": 0,
        "pass": all(not entry.get("unofferedTools") for entry in quality.values()),
    }
    results["zeroProfileFalsePositives"] = {
        "value": (quality.get("extract_self_profile_update") or {}).get(
            "falsePositives", 0
        ),
        "gate": 0,
        "pass": not (quality.get("extract_self_profile_update") or {}).get(
            "falsePositives"
        ),
    }
    results["zeroFalseSuccessClaims"] = {
        "value": (quality.get("render_response") or {}).get("falseSuccessClaims", 0),
        "gate": 0,
        "pass": not (quality.get("render_response") or {}).get("falseSuccessClaims"),
    }

    tool_free = latency.get("interpret_tool_free") or {}
    if tool_free.get("median") is not None:
        results["toolFreeMedianSeconds"] = {
            "value": round(tool_free["median"] / 1000, 2),
            "gate": GATES["toolFreeMedianSeconds"],
            "pass": tool_free["median"] / 1000 <= GATES["toolFreeMedianSeconds"],
        }
    if tool_free.get("p95") is not None:
        results["toolFreeP95Seconds"] = {
            "value": round(tool_free["p95"] / 1000, 2),
            "gate": GATES["toolFreeP95Seconds"],
            "pass": tool_free["p95"] / 1000 <= GATES["toolFreeP95Seconds"],
        }
    for shape, key in (("2k", "sustained2kDecode"), ("8k", "sustained8kDecode")):
        tps = sustained_tps(shape)
        if tps is not None:
            results[key] = {
                "value": round(tps, 2),
                "gate": GATES[key],
                "pass": tps >= GATES[key],
                "note": "derived from end-to-end pass latency, not raw decode",
            }
    return results


def render_graph(report: dict[str, Any]) -> bytes:
    """The evidence graph. Labels both arms as different models on purpose."""

    suite_count = len([name for name, _, _ in SUITES if name in report["latencyMs"]])
    quality_count = len(report.get("quality") or {})
    sustained_count = len(report.get("sustained") or [])
    # Sized to its content rather than fixed. A fixed canvas either crops a full
    # six-suite run or leaves a third of the image blank on a partial one, and
    # both make the evidence harder to read than it needs to be.
    height = (
        190
        + suite_count * 30
        + 60
        + quality_count * 22
        + 60
        + sustained_count * 34
        + 90
    )
    width = 1000
    canvas = Canvas(width, height, (252, 252, 253))
    ink = (24, 26, 32)
    muted = (110, 116, 128)
    phone = (36, 98, 204)
    server = (196, 108, 32)
    good = (32, 140, 84)
    bad = (192, 48, 48)

    canvas.text(24, 22, "NOVA COMPANION: CORE ML / ANE VS IRIDIUM", ink, 2)
    canvas.text(
        24,
        52,
        "PHONE GEMMA 4 E2B COREML ANE 8K   SERVER IRIDIUM LLAMA.CPP",
        muted,
    )
    canvas.text(
        24,
        68,
        "DIFFERENT MODELS ON DIFFERENT HARDWARE - ROUTING, NOT A HARDWARE BENCHMARK",
        muted,
    )

    # Latency by suite, both arms.
    top = 100
    canvas.text(24, top, "MEDIAN LATENCY BY WORKLOAD (MS)", ink)
    latency = report["latencyMs"]
    suites = [name for name, _, _ in SUITES if name in latency]
    peak = max(
        [
            value
            for name in suites
            for value in (
                (latency[name].get("companion") or {}).get("median"),
                (latency[name].get("local") or {}).get("median"),
            )
            if value
        ]
        or [1]
    )
    row = top + 22
    for name in suites:
        canvas.text(24, row + 6, name.upper()[:30], ink)
        for arm, colour, offset in (
            ("companion", phone, 0),
            ("local", server, 10),
        ):
            value = (latency[name].get(arm) or {}).get("median")
            if not value:
                continue
            span = int(560 * value / peak)
            canvas.rect(300, row + offset, span, 8, colour)
            canvas.text(300 + span + 6, row + offset + 1, f"{value:.0f}", muted)
        row += 30
    canvas.text(300, row, "PHONE", phone)
    canvas.text(380, row, "IRIDIUM", server)

    # Quality pass rates against their gates.
    top = row + 30
    canvas.text(24, top, "QUALITY PASS RATE VS GATE", ink)
    row = top + 22
    for name in suites:
        entry = report["quality"].get(name)
        if not entry:
            continue
        rate = entry["rate"]
        gate = {
            "interpret_tool_free": GATES["interpretToolFreeAccuracy"],
            "interpret_tool_bearing": GATES["interpretToolAccuracy"],
            "classify_icon": GATES["iconAccuracy"],
            "extract_self_profile_update": GATES["profileRecall"],
            "confirm_objective": GATES["confirmAccuracy"],
        }.get(name)
        canvas.text(24, row + 2, name.upper()[:30], ink)
        canvas.rect(300, row, 400, 10, (228, 230, 236))
        canvas.rect(300, row, int(400 * rate), 10, good if not gate or rate >= gate else bad)
        if gate:
            canvas.vline(300 + int(400 * gate), row - 3, 16, ink)
        canvas.text(712, row + 2, f"{rate * 100:.0f}%", muted)
        row += 22

    # Burst against sustained, which is the thermal question.
    top = row + 20
    canvas.text(24, top, "BURST VS SUSTAINED (PHONE, MEDIAN MS)", ink)
    row = top + 22
    for run in report.get("sustained", []):
        first = (run.get("companionFirstMinuteMs") or {}).get("median")
        last = (run.get("companionLastMinuteMs") or {}).get("median")
        canvas.text(24, row + 2, f"{run['contextShape'].upper()} CONTEXT", ink)
        for value, colour, offset in ((first, phone, 0), (last, server, 12)):
            if not value:
                continue
            span = int(400 * value / max(first or 1, last or 1))
            canvas.rect(300, row + offset, span, 9, colour)
            canvas.text(300 + span + 6, row + offset + 1, f"{value:.0f}", muted)
        row += 34
    canvas.text(300, row, "FIRST MINUTE", phone)
    canvas.text(420, row, "LAST MINUTE", server)

    # Bandwidth and thermal retention.
    row += 26
    band = report.get("bandwidth") or {}
    if band.get("tokensPerSecond"):
        fraction = band["fractionOfBudget"]
        canvas.text(24, row, "EFFECTIVE BANDWIDTH VS 53.8 GB/S BUDGET", ink)
        canvas.rect(300, row, 400, 10, (228, 230, 236))
        canvas.rect(
            300, row, int(400 * min(1.0, fraction)), 10, good if fraction <= 1 else bad
        )
        canvas.text(712, row + 2, f"{fraction * 100:.0f}%", muted)
        row += 22
    thermal = report.get("thermal") or {}
    if thermal:
        canvas.text(
            24,
            row,
            f"THERMAL PEAK: {str(thermal.get('peak', 'unknown')).upper()}",
            bad if thermal.get("peak") in {"serious", "critical"} else ink,
        )
    return canvas.to_png()


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Nova companion: Core ML / ANE against Iridium",
        "",
        f"Captured {report['capturedAt']}.",
        "",
        "**These are two different models on two different machines.** The phone "
        "runs Gemma 4 E2B through Core ML on the Neural Engine at an 8,192-token "
        "window; Iridium runs its own llama.cpp model. Read every number below as "
        "an answer to \"should this pass be routed to the phone?\" and not as a "
        "hardware comparison.",
        "",
        "No audio was captured, nothing was spoken, and no tool was executed.",
        "",
        "## Gates",
        "",
        "| Gate | Value | Threshold | Result |",
        "| --- | ---: | ---: | --- |",
    ]
    for name, entry in report["gates"].items():
        lines.append(
            f"| {name} | {entry['value']} | {entry['gate']} | "
            f"{'pass' if entry['pass'] else '**FAIL**'} |"
        )

    lines += [
        "",
        "## Latency by workload",
        "",
        "| Workload | Phone median | Iridium median | Phone p95 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, entry in report["latencyMs"].items():
        companion = entry.get("companion") or {}
        local = entry.get("local") or {}
        lines.append(
            f"| {name} | {companion.get('median', '-')} | "
            f"{local.get('median', '-')} | {companion.get('p95', '-')} |"
        )

    lines += [
        "",
        "## Quality",
        "",
        "| Workload | n | Passed | Rate | Schema-valid |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, entry in report["quality"].items():
        lines.append(
            f"| {name} | {entry['n']} | {entry['passed']} | "
            f"{entry['rate'] * 100:.0f}% | {entry['schemaValid']}/{entry['n']} |"
        )

    band = report.get("bandwidth") or {}
    if band.get("tokensPerSecond"):
        lines += [
            "",
            "## Bandwidth",
            "",
            f"- Bytes per token: {band['bytesPerToken']:,}",
            f"- Measured: {band['tokensPerSecond']} tok/s",
            f"- Effective: {band['effectiveBytesPerSecond'] / 1e9:.1f} GB/s "
            f"({band['fractionOfBudget'] * 100:.0f}% of the 53.8 GB/s budget)",
            f"- Ceiling at budget: {band['ceilingTokensPerSecond']} tok/s",
        ]

    failures = [
        (suite, failure)
        for suite, entry in report["quality"].items()
        for failure in entry["failures"]
    ]
    if failures:
        lines += ["", "## Failures", ""]
        for suite, failure in failures:
            lines.append(f"- `{suite}` `{failure['id']}`: {failure['score']}")
    return "\n".join(lines) + "\n"


async def collect(args: argparse.Namespace) -> dict[str, Any]:
    catalogue = load_catalogue()
    async with httpx.AsyncClient(
        base_url=args.host, verify=False, cert=(args.cert, args.key), timeout=300
    ) as client:
        status = (await client.get("/v1/companion/status")).json()
        original = {name: entry["mode"] for name, entry in status["routes"].items()}
        try:
            # Every workload on both arms, so each sample carries a matched pair
            # rather than being compared against a different run's numbers.
            await set_routes(client, {name: "both" for name in original})

            cold: dict[str, Any] = {}
            if args.cold:
                # The very first pass after a launch pays model warm-up. Kept
                # and reported rather than discarded, because a user's first
                # request of the day pays it too.
                started = time.perf_counter()
                cold_payload = await post_workload(
                    client,
                    build_body(
                        "interpret_tool_free",
                        {"transcript": "Hello there"},
                        catalogue,
                    ),
                )
                cold = {
                    "wallMs": round((time.perf_counter() - started) * 1000, 1),
                    "armsMs": cold_payload.get("armsMs"),
                }

            # Warm-up, excluded from every summary below.
            await post_workload(
                client,
                build_body(
                    "interpret_tool_free", {"transcript": "Warm up"}, catalogue
                ),
            )

            samples: list[dict[str, Any]] = []
            for _ in range(args.repeats):
                for suite, fixture_file, _ in SUITES:
                    samples += await run_suite(
                        client,
                        suite,
                        fixture_file,
                        catalogue,
                        pause=args.pause_seconds,
                    )

            sustained = []
            if args.sustained_seconds > 0:
                for long_context in (False, True):
                    print(
                        f"  sustained {'8k' if long_context else '2k'} for "
                        f"{args.sustained_seconds}s",
                        flush=True,
                    )
                    sustained.append(
                        await sustained_run(
                            client,
                            catalogue,
                            seconds=args.sustained_seconds,
                            long_context=long_context,
                        )
                    )
        finally:
            await set_routes(client, original)

        final_status = (await client.get("/v1/companion/status")).json()

    latency: dict[str, Any] = {}
    for suite, _, _ in SUITES:
        selected = [s for s in samples if s["suite"] == suite]
        latency[suite] = {
            arm: summary(
                [
                    float(s["armsMs"][arm])
                    for s in selected
                    if isinstance((s.get("armsMs") or {}).get(arm), (int, float))
                ]
            )
            for arm in ("companion", "local")
        }
    tool_free = latency.get("interpret_tool_free", {}).get("companion", {})

    quality = quality_report(samples)
    gates = gate_report(
        quality, {"interpret_tool_free": tool_free}, sustained
    )
    sustained_median = None
    for run in sustained:
        if run["contextShape"] == "2k":
            sustained_median = (run.get("companionAllMs") or {}).get("median")
    return {
        "capturedAt": datetime.now(UTC).isoformat(),
        "method": {
            "endpoint": "/v1/test/workload",
            "audio": False,
            "spoken": False,
            "toolsExecuted": False,
            "dryRun": True,
            "bothArmsPerSample": True,
            "repeats": args.repeats,
            "warmupExcluded": True,
            "coldStartMeasured": bool(args.cold),
            "sustainedSeconds": args.sustained_seconds,
        },
        "arms": {
            "companion": {
                "model": "gemma-4-e2b",
                "runtime": "coreml/ane",
                "contextTokens": 8192,
                "identifier": "coreml/gemma-4-e2b-ane-8k@9838f7c",
            },
            "local": {"host": "iridium", "runtime": "llama.cpp"},
            "note": "Different models on different hardware; a routing comparison only.",
        },
        "companion": {
            key: status.get(key)
            for key in ("identity", "tier", "tierReason", "appVersion", "osVersion")
        },
        "coldStart": cold,
        "latencyMs": latency,
        "quality": quality,
        "gates": gates,
        "sustained": sustained,
        "bandwidth": bandwidth_report(
            1000.0 / sustained_median if sustained_median else None
        ),
        "thermal": {"peak": (final_status.get("telemetry") or {}).get("thermalState")},
        "samples": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="https://127.0.0.1:8766")
    parser.add_argument("--cert", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--pause-seconds", type=float, default=0.0)
    parser.add_argument(
        "--sustained-seconds",
        type=float,
        default=600.0,
        help="Ten minutes by default: thermal decay is the reason the ANE was chosen.",
    )
    parser.add_argument("--cold", action="store_true", default=True)
    parser.add_argument("--no-cold", dest="cold", action="store_false")
    parser.add_argument("--out", default="docs/evidence")
    args = parser.parse_args()

    report = asyncio.run(collect(args))
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"companion-ane-{stamp}.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (out / f"companion-ane-{stamp}.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    (out / f"companion-ane-{stamp}.png").write_bytes(render_graph(report))
    print(f"\nwrote companion-ane-{stamp}.{{json,md,png}} to {out}")
    failed = [name for name, entry in report["gates"].items() if not entry["pass"]]
    print("gates failed: " + (", ".join(failed) if failed else "none"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
