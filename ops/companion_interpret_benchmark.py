"""Benchmark the companion's tool-free interpretation pass without audio.

This harness deliberately calls ``/v1/test/interpretation`` rather than the
full voice-turn endpoint. It therefore measures only the real routed planner:
no STT, TTS, provider refresh, household state, memory, conversation history,
tool schemas, policy, execution, or background identity extraction.

The run is sequential and starts with an excluded warm-up. Measured scenarios
are interleaved by repeat instead of exhausting one phrase at a time, which
reduces cache order and thermal drift. Raw samples are retained alongside
median, p95, min, max and median absolute deviation; no single run is promoted
to "the" result.

Run on the voice host while the paid companion app is foregrounded::

    /opt/nova-voice/venv/bin/python ops/companion_interpret_benchmark.py \
        --host https://127.0.0.1:8766 \
        --cert /tmp/vs-tls/client.crt --key /tmp/vs-tls/client.key \
        --repeats 5 --out /tmp
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

SCENARIOS = [
    ("social_short", "Hello there"),
    ("question_short", "How are you feeling today"),
    ("observation", "The rain sounds peaceful tonight"),
    ("self_intention", "I might rearrange the lounge tomorrow"),
]


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return round(ordered[index], 1)


def summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "median": None, "p95": None, "min": None, "max": None, "mad": None}
    median = statistics.median(values)
    deviations = [abs(value - median) for value in values]
    return {
        "n": len(values),
        "median": round(median, 1),
        "p95": percentile(values, 0.95),
        "min": round(min(values), 1),
        "max": round(max(values), 1),
        "mad": round(statistics.median(deviations), 1),
    }


def summarise(samples: list[dict[str, Any]]) -> dict[str, Any]:
    groups = {"all": samples}
    groups.update(
        {
            scenario: [sample for sample in samples if sample["scenario"] == scenario]
            for scenario, _ in SCENARIOS
        }
    )
    report: dict[str, Any] = {}
    for group, selected in groups.items():
        report[group] = {
            arm: summary(
                [
                    float(sample["armsMs"][arm])
                    for sample in selected
                    if isinstance((sample.get("armsMs") or {}).get(arm), (int, float))
                ]
            )
            for arm in ("companion", "local")
        }
        ratios = [
            float(sample["armsMs"]["companion"]) / float(sample["armsMs"]["local"])
            for sample in selected
            if isinstance((sample.get("armsMs") or {}).get("companion"), (int, float))
            and isinstance((sample.get("armsMs") or {}).get("local"), (int, float))
            and float(sample["armsMs"]["local"]) > 0
        ]
        report[group]["companionToLocalRatio"] = summary(ratios)
        report[group]["missingCompanion"] = sum(
            not isinstance((sample.get("armsMs") or {}).get("companion"), (int, float))
            for sample in selected
        )
    return report


def counter_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, int]:
    before_counter = (before.get("counters") or {}).get("interpret") or {}
    after_counter = (after.get("counters") or {}).get("interpret") or {}
    return {
        key: int(after_counter.get(key, 0)) - int(before_counter.get(key, 0))
        for key in ("offered", "accepted", "rejected", "completed", "failed")
    }


async def set_routes(client: httpx.AsyncClient, routes: dict[str, str]) -> None:
    response = await client.post("/v1/companion/routing", json={"routes": routes})
    response.raise_for_status()


async def collect(args: argparse.Namespace) -> dict[str, Any]:
    async with httpx.AsyncClient(
        base_url=args.host,
        verify=False,
        cert=(args.cert, args.key),
        timeout=120,
    ) as client:
        status_response = await client.get("/v1/companion/status")
        status_response.raise_for_status()
        initial_status = status_response.json()
        original_routes = {
            name: details["mode"] for name, details in initial_status["routes"].items()
        }
        benchmark_routes = {name: "local" for name in original_routes}
        benchmark_routes["interpret"] = "both"

        samples: list[dict[str, Any]] = []
        warmup: dict[str, Any] | None = None
        try:
            await set_routes(client, benchmark_routes)
            warmup_response = await client.post(
                "/v1/test/interpretation",
                json={"transcript": "Warm up the interpretation engines"},
            )
            warmup_response.raise_for_status()
            warmup = warmup_response.json()

            sequence = 0
            for repeat in range(args.repeats):
                # Rotate the starting scenario each repeat. This Latin-square
                # shape avoids making one phrase systematically cold and one
                # systematically hot while keeping the run deterministic.
                offset = repeat % len(SCENARIOS)
                ordered = SCENARIOS[offset:] + SCENARIOS[:offset]
                for scenario, transcript in ordered:
                    before_response = await client.get("/v1/companion/status")
                    before_response.raise_for_status()
                    before = before_response.json()
                    response = await client.post(
                        "/v1/test/interpretation",
                        json={"transcript": transcript},
                    )
                    response.raise_for_status()
                    payload = response.json()
                    after_response = await client.get("/v1/companion/status")
                    after_response.raise_for_status()
                    after = after_response.json()
                    sequence += 1
                    sample = {
                        "sequence": sequence,
                        "repeat": repeat,
                        "scenario": scenario,
                        "armsMs": payload.get("armsMs"),
                        "elapsedMs": payload.get("elapsedMs"),
                        "winner": payload.get("winner"),
                        "input": payload.get("input"),
                        "result": payload.get("result"),
                        "routeCounterDelta": counter_delta(before, after),
                        "device": {
                            "tierBefore": before.get("tier"),
                            "tierReasonBefore": before.get("tierReason"),
                            "telemetryAgeBefore": before.get("telemetryAgeSeconds"),
                            "tierAfter": after.get("tier"),
                            "tierReasonAfter": after.get("tierReason"),
                        },
                    }
                    samples.append(sample)
                    arms = sample.get("armsMs") or {}
                    print(
                        f"{sequence:02d} r{repeat} {scenario:16} "
                        f"phone={str(arms.get('companion')):>8}ms "
                        f"server={str(arms.get('local')):>8}ms "
                        f"tier={sample['device']['tierAfter']}",
                        flush=True,
                    )
                    if args.pause_seconds:
                        await asyncio.sleep(args.pause_seconds)
        finally:
            await set_routes(client, original_routes)

    return {
        "capturedAt": datetime.now(UTC).isoformat(),
        "method": {
            "endpoint": "/v1/test/interpretation",
            "audio": False,
            "tools": False,
            "state": False,
            "memory": False,
            "history": False,
            "execution": False,
            "sequential": True,
            "warmupExcluded": True,
            "repeats": args.repeats,
            "pauseSeconds": args.pause_seconds,
        },
        "companion": {
            key: initial_status.get(key)
            for key in ("identity", "locality", "tier", "tierReason", "appVersion", "osVersion")
        },
        "warmup": warmup,
        "summaryMs": summarise(samples),
        "samples": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="https://127.0.0.1:8766")
    parser.add_argument("--cert", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--pause-seconds", type=float, default=1.0)
    parser.add_argument("--out", default="docs/evidence")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.pause_seconds < 0:
        parser.error("--pause-seconds cannot be negative")

    report = asyncio.run(collect(args))
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = Path(args.out) / f"companion-interpret-tool-free-{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {output}")
    print(json.dumps(report["summaryMs"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
