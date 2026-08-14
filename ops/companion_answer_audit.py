"""Capture spoken answers next to the household's own readings, for auditing.

Latency measurement cannot see the failure this exists for: a turn that is fast,
well-formed and confidently *wrong*. The specimen case is an assistant that
reports the same temperature indoors and outdoors — plausible prose, valid
structure, and contradicted by the household's own sensors.

So this asks factual questions, records exactly what was said, and records the
dashboard's ground truth for the same moment. It deliberately makes **no
judgement** itself: it produces the evidence and leaves the reading to whoever
looks, because a heuristic that decides what counts as agreement would be the
thing most likely to hide a discrepancy.

Runs dry — nothing in the house changes.
"""

from __future__ import annotations

import argparse
import array
import asyncio
import io
import json
import wave
from datetime import UTC, datetime
from pathlib import Path

import httpx

SAMPLE_RATE = 16_000

# Questions whose answers are checkable against a sensor or a clock. Commands
# are excluded on purpose: "turn on the lights" has no factual claim to be
# wrong about.
QUESTIONS = [
    "What is the temperature in here",
    "What is the temperature outside",
    "Tell me the weather",
    "Tell me the time",
    "Are the lounge lights on",
    "Is the heater on",
    "How warm is it in the bedroom",
]


def to_pcm16(data: bytes) -> bytes:
    with wave.open(io.BytesIO(data), "rb") as handle:
        frames = handle.readframes(handle.getnframes())
        channels, rate = handle.getnchannels(), handle.getframerate()
    if channels > 1:
        samples = array.array("h")
        samples.frombytes(frames)
        frames = array.array(
            "h",
            [
                sum(samples[index : index + channels]) // channels
                for index in range(0, len(samples), channels)
            ],
        ).tobytes()
    if rate != SAMPLE_RATE:
        # Box-average each output sample's whole source window. Dropping every
        # Nth sample instead aliases everything above the new Nyquist back into
        # the speech band, and the TTS engine's 32 kHz output has plenty up
        # there: it still sounds fine to a human and transcribes badly, turning
        # short words into other short words ("heater" -> "hero"). Measuring
        # the stack with that audio measures the harness instead.
        samples = array.array("h")
        samples.frombytes(frames)
        count = int(len(samples) * SAMPLE_RATE / rate)
        ratio = rate / SAMPLE_RATE
        resampled = array.array("h", bytes(count * 2))
        for index in range(count):
            start = int(index * ratio)
            end = max(start + 1, int((index + 1) * ratio))
            window = samples[start:end]
            resampled[index] = sum(window) // len(window) if window else 0
        frames = resampled.tobytes()
    return bytes(200 * SAMPLE_RATE // 1000 * 2) + frames + bytes(600 * SAMPLE_RATE // 1000 * 2)


def ground_truth(state: dict) -> dict:
    """The household's own readings, flattened to what the questions ask about."""

    weather = state.get("weather") or {}
    zones = state.get("zones") or []

    def zone_summary(zone: dict) -> dict:
        return {
            "name": zone.get("name"),
            "indoorTemperatureC": zone.get("indoorTemperatureC"),
            "lightsOn": zone.get("lightsOn"),
            "climate": (zone.get("climate") or {}).get("state")
            if isinstance(zone.get("climate"), dict)
            else None,
        }

    return {
        "capturedAt": datetime.now(UTC).isoformat(),
        "outdoor": {
            "condition": weather.get("condition"),
            "temperatureC": weather.get("temperature"),
        },
        "zones": [zone_summary(zone) for zone in zones if isinstance(zone, dict)],
    }


async def collect(args) -> dict:
    async with httpx.AsyncClient(base_url=args.dashboard, timeout=30) as dashboard:
        response = await dashboard.get("/api/state")
        response.raise_for_status()
        truth = ground_truth(response.json())

    answers = []
    async with httpx.AsyncClient(
        base_url=args.host, verify=False, timeout=300, cert=(args.cert, args.key)
    ) as client:
        for question in QUESTIONS:
            clip = await client.post("/v1/voices/preview", json={"text": question})
            clip.raise_for_status()
            turn = await client.post(
                "/v1/test/turn",
                params={
                    "satellite_id": "audit",
                    "room_id": args.room,
                    "dry_run": True,
                    "wake_detected": True,
                },
                headers={"content-type": "audio/l16; rate=16000"},
                content=to_pcm16(clip.content),
            )
            turn.raise_for_status()
            payload = turn.json()
            answers.append(
                {
                    "asked": question,
                    "heard": payload.get("transcript"),
                    "said": payload.get("responseText"),
                    "decision": (payload.get("interpretation") or {}).get("decision"),
                    "dropped": bool(payload.get("dropped")),
                    "requests": [
                        item.get("path") for item in (payload.get("dryRunRequests") or [])
                    ],
                }
            )
            print(f"  asked : {question}")
            print(f"  heard : {payload.get('transcript')}")
            print(f"  said  : {payload.get('responseText')!r}\n", flush=True)

    return {"groundTruth": truth, "answers": answers}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="https://127.0.0.1:8766")
    parser.add_argument("--dashboard", default="http://nova.local")
    parser.add_argument("--cert", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--room", default="lounge")
    parser.add_argument("--out", default="/tmp/baseline-out")
    args = parser.parse_args()

    report = asyncio.run(collect(args))

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"answer-audit-{stamp}.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"\nwrote {path}")
    print("\n=== household ground truth ===")
    print(json.dumps(report["groundTruth"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
